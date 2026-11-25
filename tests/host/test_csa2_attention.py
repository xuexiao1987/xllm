# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real tensor/cache execution against full-history and pinned-internal oracles."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from xllm.python.attention import csa2_attention
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2CompressedInputs,
    Csa2EagerMetadata,
    csa2_commit_compressed,
    csa2_eager_attention,
    csa2_gather_index,
    csa2_indexer_scores,
    csa2_pool_compressor,
    csa2_prepare_ring,
    csa2_quantize_index_rows,
    csa2_select_candidates,
    csa2_topk,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    target = os.environ.get("XLLM_CSA2_TEST_DEVICE", "cpu")
    if target.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(target)
    return torch.device(target)


def _oracle(
    q: torch.Tensor, kv: torch.Tensor, main: torch.Tensor, ratio: int, sinks: torch.Tensor, window: int
) -> torch.Tensor:
    # Full-history per-token reference: no block tables, ring writes, or chunk
    # indexing. A zero-valued sink is an extra softmax column.
    outputs = []
    for position, query in enumerate(q.double()):
        visible = kv[max(0, position - window + 1) : position + 1].double()
        if ratio:
            visible = torch.cat((visible, main[: (position + 1) // ratio].double()))
        logits = query @ visible.T * 0.5
        probs = torch.softmax(torch.cat((logits, sinks.double().unsqueeze(-1)), dim=-1), dim=-1)
        outputs.append(probs[:, :-1] @ visible)
    return torch.stack(outputs).to(q.dtype)


def _run_chunks(
    q: torch.Tensor,
    kv: torch.Tensor,
    main: torch.Tensor,
    ratio: int,
    sinks: torch.Tensor,
    chunks: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    # Two physical ring blocks in deliberately non-monotonic order. The two
    # other blocks are poison guards and must never be written.
    swa = torch.full((4, 2, 4), float("nan"), device=device, dtype=kv.dtype)
    cache = LayerCache(None, None, swa=swa)
    table = torch.tensor([[3, 1]], device=device)
    key_blocks = (main.size(0) + 3) // 2
    key = torch.full((key_blocks, 2, 4), float("nan"), device=device, dtype=kv.dtype)
    key_table = torch.arange(key_blocks - 1, -1, -1, device=device).unsqueeze(0)
    for row in range(main.size(0)):
        key[key_table[0, row // 2], row % 2] = main[row]
    offset = 0
    outputs = []
    for length in chunks:
        positions = torch.arange(offset, offset + length, device=device)
        slots = table[0, positions // 2 % 2] * 2 + positions % 2
        metadata = Csa2EagerMetadata((length,), (offset + length,), table, slots)
        # All committed/future rows are offered deliberately; the attention
        # path must mask future rows separately for each query in the chunk.
        picks = torch.arange(main.size(0) + 2, device=device).expand(length, -1)
        compressed = Csa2CompressedInputs(key, key_table, picks, ratio) if ratio else None
        outputs.append(
            csa2_eager_attention(
                q[offset : offset + length],
                kv[offset : offset + length],
                cache,
                metadata,
                window_size=4,
                scale=0.5,
                sinks=sinks,
                compressed=compressed,
            )
        )
        offset += length
    assert offset == q.size(0)
    assert torch.isnan(swa[[0, 2]]).all()
    for position in range(offset - 4, offset):
        block = table[0, position // 2 % 2]
        torch.testing.assert_close(swa[block, position % 2], kv[position])
    return torch.cat(outputs), swa


@pytest.mark.parametrize("ratio", [0, 1, 2, 4])
@pytest.mark.parametrize("chunks", [(13,), (3, 7, 3), (1,) * 13, (0, 5, 0, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunking_and_ring_wrap_match_full_history(
    device: torch.device, ratio: int, chunks: tuple[int, ...], dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(731)
    q = torch.randn(13, 2, 4, generator=generator).to(dtype)
    kv = torch.randn(13, 4, generator=generator).to(dtype)
    main = torch.randn(13 // max(ratio, 1), 4, generator=generator).to(dtype)
    sinks = torch.tensor([0.3, -0.4])
    expected = _oracle(q, kv, main, ratio, sinks, 4)
    actual, _ = _run_chunks(q.to(device), kv.to(device), main.to(device), ratio, sinks.to(device), chunks)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.015 if dtype == torch.bfloat16 else 2e-5, atol=2e-6)


def test_request_isolation_and_empty_request(device: torch.device) -> None:
    swa = torch.full((4, 2, 4), float("nan"), device=device)
    cache = LayerCache(None, None, swa=swa)
    table = torch.tensor([[3, 1], [0, 2], [-1, -1]], device=device)
    q = torch.ones(7, 1, 4, device=device)
    kv = torch.cat((torch.ones(5, 4), torch.full((2, 4), -2.0))).to(device)
    slots = torch.tensor([6, 7, 2, 3, 6, 0, 1], device=device)
    metadata = Csa2EagerMetadata((5, 2, 0), (5, 2, 0), table, slots)
    actual = csa2_eager_attention(q, kv, cache, metadata, window_size=4, scale=0.5, sinks=q.new_zeros(1))
    expected = torch.cat(
        [
            _oracle(q[:5].cpu(), kv[:5].cpu(), kv[:0].cpu(), 0, torch.zeros(1), 4),
            _oracle(q[5:].cpu(), kv[5:].cpu(), kv[:0].cpu(), 0, torch.zeros(1), 4),
        ]
    )
    torch.testing.assert_close(actual.cpu(), expected)
    assert torch.isnan(swa[2]).all()


def test_long_prefill_crosses_query_tiles(device: torch.device) -> None:
    generator = torch.Generator().manual_seed(911)
    q = torch.randn(145, 2, 4, generator=generator)
    kv = torch.randn(145, 4, generator=generator)
    main = torch.randn(36, 4, generator=generator)
    sinks = torch.tensor([1.0, -1.0])
    expected = _oracle(q, kv, main, 4, sinks, 4)
    for chunks in [(145,), (1, 64, 80)]:
        actual, _ = _run_chunks(q.to(device), kv.to(device), main.to(device), 4, sinks.to(device), chunks)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("missing", ["history", "compressed"])
def test_missing_visible_cache_fails_before_swa_commit(device: torch.device, missing: str) -> None:
    swa = torch.ones(2, 2, 4, device=device)
    cache = LayerCache(None, None, swa=swa)
    table = torch.tensor([[-1, 1] if missing == "history" else [0, 1]], device=device)
    metadata = Csa2EagerMetadata((1,), (5,), table, torch.tensor([0], device=device))
    key = swa.new_zeros(1, 2, 4)
    compressed = Csa2CompressedInputs(key, table.new_tensor([[-1]]), table.new_tensor([[0]]), 2)
    with pytest.raises(RuntimeError, match="no assigned physical block"):
        csa2_eager_attention(
            swa.new_ones(1, 1, 4),
            swa.new_full((1, 4), 9),
            cache,
            metadata,
            window_size=4,
            scale=1,
            sinks=swa.new_zeros(1),
            compressed=compressed if missing == "compressed" else None,
        )
    torch.testing.assert_close(swa, torch.ones_like(swa))


@pytest.mark.parametrize("sink", [-1000.0, 0.0, 1000.0])
def test_sink_only_adds_normalization_mass(device: torch.device, sink: float) -> None:
    q = torch.zeros(1, 1, 4, device=device)
    kv = torch.ones(1, 4, device=device)
    cache = LayerCache(None, None, swa=torch.full((1, 4, 4), float("nan"), device=device))
    metadata = Csa2EagerMetadata((1,), (1,), torch.tensor([[0]], device=device), torch.tensor([0], device=device))
    actual = csa2_eager_attention(q, kv, cache, metadata, window_size=4, scale=1, sinks=q.new_tensor([sink]))
    torch.testing.assert_close(actual, q.new_full((1, 1, 4), torch.sigmoid(torch.tensor(-sink)).item()))
    assert torch.isfinite(actual).all()


def test_window_one_does_not_read_expired_history(device: torch.device) -> None:
    cache = LayerCache(None, None, swa=torch.full((1, 1, 4), float("nan"), device=device))
    metadata = Csa2EagerMetadata((1,), (2,), torch.tensor([[-1, 0]], device=device), torch.tensor([0], device=device))
    q = torch.zeros(1, 1, 4, device=device)
    actual = csa2_eager_attention(q, q.new_ones(1, 4), cache, metadata, window_size=1, scale=1, sinks=q.new_zeros(1))
    torch.testing.assert_close(actual, q.new_full(q.shape, 0.5))


def test_topk_visibility_candidates_ties_and_padding(device: torch.device) -> None:
    scores = torch.tensor([[0.0, 0, 999, 999], [7, 8, 9, 10], [0, 0, 0, 0]], device=device)
    lengths = torch.tensor([2, 4, 0], device=device)
    candidates = torch.tensor([[1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)
    torch.testing.assert_close(
        csa2_topk(scores, lengths, 6, candidates).cpu(),
        torch.tensor([[0, 1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1], [-1] * 6], dtype=torch.int32),
    )
    assert csa2_topk(scores[:, :0], lengths * 0, 3).eq(-1).all()


def test_candidate_pool_keeps_partial_block_without_future_leak(device: torch.device) -> None:
    scores = torch.tensor([[9.0, 8, 7, 6, 0, 1000], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], device=device)
    lengths = torch.tensor([5, 6, 0], device=device)
    actual = csa2_select_candidates(scores, lengths, 2, 2)
    torch.testing.assert_close(
        actual.cpu(), torch.tensor([[1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1], [0] * 6], dtype=torch.bool)
    )
    assert csa2_select_candidates(scores[:, :0], lengths * 0, 2, 2).shape == (3, 0)


def test_indexer_and_topk_match_pinned_internal(device: torch.device) -> None:
    fixture = json.loads((Path(__file__).parent / "data/csa2_indexer.json").read_text())
    q, k, weights = (torch.tensor(fixture[name], device=device) for name in ("q", "k", "weights"))
    scores = csa2_indexer_scores(q, k, weights)
    torch.testing.assert_close(scores.cpu(), torch.tensor(fixture["scores"]), rtol=1e-5, atol=1e-6)
    committed = torch.tensor(fixture["committed"], device=device)
    actual = csa2_topk(scores, committed, 3)
    torch.testing.assert_close(actual.cpu(), torch.tensor(fixture["picks"], dtype=torch.int32))


def test_long_indexer_bounds_products_and_preserves_head_reduction(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise multiple query tiles with host-sized tensors.
    monkeypatch.setattr(csa2_attention, "_INDEXER_PRODUCT_ELEMENTS", 16 << 20)
    generator = torch.Generator().manual_seed(731)
    q = torch.randint(-2, 3, (65, 8, 4), generator=generator).float().to(device)
    key = torch.randint(-2, 3, (32769, 4), generator=generator).float().to(device)
    weights = torch.randint(-2, 3, (65, 8), generator=generator).float().to(device)
    expected = q.new_zeros((q.size(0), key.size(0)))
    for head in range(q.size(1)):
        expected += (q[:, head] @ key.T).relu() * weights[:, head, None]
    product_sizes = []
    einsum = torch.einsum

    def record_product(equation: str, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        result = einsum(equation, query, keys)
        product_sizes.append(result.numel())
        return result

    monkeypatch.setattr(torch, "einsum", record_product)
    actual = csa2_indexer_scores(q, key, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(product_sizes) > 1
    assert max(product_sizes) <= 16 << 20


def test_empty_batch_and_negative_write_slots(device: torch.device) -> None:
    swa = torch.full((1, 4, 4), 7.0, device=device)
    cache = LayerCache(None, None, swa=swa)
    table = torch.zeros(0, 1, dtype=torch.int64, device=device)
    empty = Csa2EagerMetadata((), (), table, torch.zeros(0, dtype=torch.int64, device=device))
    result = csa2_eager_attention(
        swa.new_empty(0, 1, 4), swa.new_empty(0, 4), cache, empty, window_size=4, scale=1, sinks=swa.new_zeros(1)
    )
    assert result.shape == (0, 1, 4)
    metadata = Csa2EagerMetadata((1,), (1,), table.new_zeros(1, 1), table.new_tensor([-1]))
    csa2_eager_attention(
        swa.new_ones(1, 1, 4), swa.new_ones(1, 4), cache, metadata, window_size=4, scale=1, sinks=swa.new_zeros(1)
    )
    torch.testing.assert_close(swa, swa.new_full(swa.shape, 7.0))


@pytest.mark.parametrize("lengths,contexts", [((2,), (1,)), ((-1,), (1,)), ((1, 2), (1,))])
def test_invalid_query_metadata_rejected(lengths: tuple[int, ...], contexts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Csa2EagerMetadata(lengths, contexts, torch.zeros(1, 1), torch.zeros(1))


@pytest.mark.parametrize("ratio", [0, 3, -1])
def test_invalid_compressed_ratio_rejected(ratio: int) -> None:
    with pytest.raises(ValueError, match="ratio"):
        Csa2CompressedInputs(torch.zeros(1, 2, 4), torch.zeros(1, 1), torch.zeros(1, 1), ratio)


def _compressor_cache(device: torch.device) -> LayerCache:
    return LayerCache(
        key=torch.full((40, 2, 4), float("nan"), device=device),
        value=None,
        index=torch.full((40, 2, 3), -99, dtype=torch.int8, device=device),
        indexer_scale=torch.full((40, 2, 1), float("nan"), dtype=torch.float16, device=device),
        swa=torch.full((4, 16, 4), float("nan"), device=device),
        compress_kv_state=torch.full((4, 32, 4), float("nan"), device=device),
        compress_score_state=torch.full((4, 32, 4), float("nan"), device=device),
    )


def _pool_oracle(kv: torch.Tensor, gates: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return kv
    rows = []
    for start in range(0, kv.size(0) - ratio + 1, ratio):
        # Full history; no persistent state, block table, chunk or commit logic.
        weights = torch.softmax(gates[start : start + ratio].double(), dim=0)
        rows.append((kv[start : start + ratio].double() * weights).sum(dim=0))
    return torch.stack(rows).to(kv.dtype) if rows else kv[:0]


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("chunks", [(72,), (1, 35, 36), (3, 1, 5, 63), (1,) * 72, (0, 3, 0, 69)])
def test_compressor_partial_groups_and_state_wrap(device: torch.device, ratio: int, chunks: tuple[int, ...]) -> None:
    g = torch.Generator().manual_seed(415)
    kv = torch.randn(72, 4, generator=g)
    gates = torch.randn(72, 4, generator=g) * 3
    expected = _pool_oracle(kv, gates, ratio)
    kv, gates = kv.to(device), gates.to(device)
    cache = _compressor_cache(device)
    table = torch.tensor([[3, 1]], device=device)
    outputs, ids, starts = [], [], []
    offset = 0
    for length in chunks:
        metadata = Csa2EagerMetadata((length,), (offset + length,), table, table.new_full((length,), -1))
        pooled = csa2_pool_compressor(
            kv[offset : offset + length],
            gates[offset : offset + length] if ratio > 1 else None,
            cache,
            metadata,
            ratio,
        )
        assert pooled.counts == ((offset + length) // ratio - offset // ratio,)
        outputs.append(pooled.values)
        ids.append(pooled.row_ids)
        starts.append(pooled.positions)
        offset += length
    torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch.cat(ids).cpu(), torch.arange(72 // ratio))
    torch.testing.assert_close(torch.cat(starts).cpu(), torch.arange(72 // ratio) * ratio)
    if ratio == 1:
        assert torch.isnan(cache.compress_kv_state).all()
        assert torch.isnan(cache.compress_score_state).all()
    else:
        for position in range(40, 72):
            if (position + 1) % ratio:
                block = table[0, position // 16 % 2]
                torch.testing.assert_close(cache.compress_kv_state[block, position % 32], kv[position])
                torch.testing.assert_close(cache.compress_score_state[block, position % 32], gates[position])
    assert torch.isnan(cache.compress_kv_state[[0, 2]]).all()
    assert torch.isnan(cache.swa).all()


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_compressor_commit_and_attention_chunk_pipeline(device: torch.device, ratio: int) -> None:
    g = torch.Generator().manual_seed(99)
    q, swa_kv, projected, gates = [torch.randn(shape, generator=g) for shape in [(13, 2, 4), (13, 4), (13, 4), (13, 4)]]
    full_main = _pool_oracle(projected, gates, ratio)
    sinks = torch.tensor([0.4, -0.8])
    expected = _oracle(q, swa_kv, full_main, ratio, sinks, 4)
    q, swa_kv, projected, gates, sinks = [t.to(device) for t in (q, swa_kv, projected, gates, sinks)]
    cache = _compressor_cache(device)
    table = torch.tensor([[3, 1]], device=device)
    key_table = torch.arange(39, -1, -1, device=device).unsqueeze(0)
    index_table = torch.arange(40, device=device).unsqueeze(0)
    outputs = []
    offset = 0
    for length in (1, 5, 3, 4):
        positions = torch.arange(offset, offset + length, device=device)
        slots = table[0, positions // 16 % 2] * 16 + positions % 16
        metadata = Csa2EagerMetadata((length,), (offset + length,), table, slots)
        pooled = csa2_pool_compressor(
            projected[offset : offset + length],
            gates[offset : offset + length] if ratio > 1 else None,
            cache,
            metadata,
            ratio,
        )
        # Operands are already in the cache coordinate system here. This
        # verifies pool/write/read/attention, not model normalization or RoPE.
        csa2_commit_compressed(cache, key_table, index_table, pooled, pooled.values, pooled.values[:, :3])
        committed = (offset + length) // ratio
        index_rows = csa2_gather_index(cache, index_table, 0, committed)
        raw_index = full_main[:committed, :3]
        if committed:
            row_scales = cache.indexer_scale.view(-1, 1)[:committed].float().cpu()
            assert ((index_rows.cpu() - raw_index).abs() <= row_scales * 0.501 + 1e-6).all()
        scores = csa2_indexer_scores(
            torch.ones(length, 1, 3, device=device), index_rows, torch.ones(length, 1, device=device)
        )
        picks = csa2_topk(scores, (positions + 1) // ratio, 13)
        compressed = Csa2CompressedInputs(cache.key, key_table, picks, ratio)
        outputs.append(
            csa2_eager_attention(
                q[offset : offset + length],
                swa_kv[offset : offset + length],
                cache,
                metadata,
                window_size=4,
                scale=0.5,
                sinks=sinks,
                compressed=compressed,
            )
        )
        offset += length
    torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=2e-5, atol=2e-6)


def test_compressor_two_requests_and_empty_request(device: torch.device) -> None:
    cache = _compressor_cache(device)
    table = torch.tensor([[3, 1], [0, 2], [-1, -1]], device=device)
    for lengths, contexts, data in [
        ((1, 3, 0), (1, 3, 0), [1.0, 10.0, 20.0, 30.0]),
        ((3, 1, 0), (4, 4, 0), [2.0, 3.0, 4.0, 40.0]),
    ]:
        kv = torch.tensor(data, device=device).unsqueeze(-1).expand(-1, 4).contiguous()
        metadata = Csa2EagerMetadata(lengths, contexts, table, table.new_full((len(data),), -1))
        result = csa2_pool_compressor(kv, torch.zeros_like(kv), cache, metadata, 2)
        expected = [15.0] if contexts[0] == 1 else [1.5, 3.5, 35.0]
        torch.testing.assert_close(result.values.cpu(), torch.tensor(expected).unsqueeze(-1).expand(-1, 4))
        assert result.counts == ((0, 1, 0) if contexts[0] == 1 else (2, 1, 0))


def test_index_quantization_zero_scale_and_round_to_even(device: torch.device) -> None:
    values = torch.tensor([[127.0, 0.5, 1.5, -2.5], [0.0, 0, 0, 0]], device=device)
    codes, scales = csa2_quantize_index_rows(values)
    torch.testing.assert_close(codes.cpu(), torch.tensor([[127, 0, 2, -2], [0, 0, 0, 0]], dtype=torch.int8))
    torch.testing.assert_close(scales.cpu(), torch.tensor([1.0, 2.0**-24], dtype=torch.float16), rtol=0, atol=0)


@pytest.mark.parametrize("bad_plane", ["key", "index", "scale"])
def test_compressed_commit_rejects_missing_plane_before_any_write(device: torch.device, bad_plane: str) -> None:
    cache = _compressor_cache(device)
    table = torch.tensor([[3, 1]], device=device)
    metadata = Csa2EagerMetadata((2,), (2,), table, table.new_full((2,), -1))
    projected = torch.ones(2, 4, device=device)
    pooled = csa2_pool_compressor(projected, None, cache, metadata, 1)
    key_table = table.new_tensor([[-1 if bad_plane == "key" else 0]])
    index_table = table.new_tensor([[-1 if bad_plane == "index" else 0]])
    target = replace(cache, indexer_scale=None) if bad_plane == "scale" else cache
    with pytest.raises((RuntimeError, ValueError)):
        csa2_commit_compressed(target, key_table, index_table, pooled, projected, projected[:, :3])
    assert torch.isnan(cache.key).all()
    assert cache.index.eq(-99).all()
    assert torch.isnan(cache.indexer_scale).all()


def test_compressor_empty_batch_and_partial_group_are_inert(device: torch.device) -> None:
    cache = _compressor_cache(device)
    empty = Csa2EagerMetadata(
        (), (), torch.empty(0, 1, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device)
    )
    result = csa2_pool_compressor(torch.empty(0, 4, device=device), torch.empty(0, 4, device=device), cache, empty, 2)
    assert result.values.shape == (0, 4) and result.counts == ()
    assert torch.isnan(cache.compress_kv_state).all()
    metadata = Csa2EagerMetadata(
        (1,), (1,), empty.swa_block_table.new_tensor([[3, 1]]), empty.slot_mapping.new_tensor([-1])
    )
    result = csa2_pool_compressor(torch.ones(1, 4, device=device), torch.zeros(1, 4, device=device), cache, metadata, 2)
    assert result.values.shape == (0, 4) and result.counts == (0,)
    csa2_commit_compressed(
        cache,
        empty.swa_block_table.new_tensor([[-1]]),
        empty.swa_block_table.new_tensor([[-1]]),
        result,
        result.values,
        result.values[:, :3],
    )
    assert torch.isnan(cache.key).all()
    assert cache.index.eq(-99).all()


@pytest.mark.parametrize("ratio", [0, 3, -1])
def test_invalid_pool_ratio_rejected(ratio: int) -> None:
    metadata = Csa2EagerMetadata((1,), (1,), torch.zeros(1, 1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64))
    with pytest.raises(ValueError, match="ratio"):
        csa2_pool_compressor(torch.ones(1, 4), None, LayerCache(None, None), metadata, ratio)


def test_packed_compressor_state_preserves_aliases(device: torch.device) -> None:
    packed = torch.full((4, 32, 8), float("nan"), device=device)
    cache = replace(_compressor_cache(device), compress_kv_state=packed[..., :4], compress_score_state=packed[..., 4:])
    assert not cache.compress_kv_state.is_contiguous()
    generator = torch.Generator().manual_seed(83)
    kv = torch.randn(72, 4, generator=generator)
    gates = torch.randn(72, 4, generator=generator)
    expected = _pool_oracle(kv, gates, 2)
    kv, gates = kv.to(device), gates.to(device)
    table = torch.tensor([[3, 1]], device=device)
    outputs = []
    offset = 0
    for length in (1, 35, 36):
        metadata = Csa2EagerMetadata((length,), (offset + length,), table, table.new_full((length,), -1))
        outputs.append(
            csa2_pool_compressor(
                kv[offset : offset + length], gates[offset : offset + length], cache, metadata, 2
            ).values
        )
        offset += length
    torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=2e-5, atol=2e-6)
    for position in range(40, 72, 2):
        block = table[0, position // 16 % 2]
        torch.testing.assert_close(packed[block, position % 32, :4], kv[position])
        torch.testing.assert_close(packed[block, position % 32, 4:], gates[position])
    assert torch.isnan(packed[[0, 2]]).all()


def test_bf16_main_and_int8_index_deployment_dimensions(device: torch.device) -> None:
    cache = LayerCache(
        key=torch.full((2, 16, 512), float("nan"), dtype=torch.bfloat16, device=device),
        value=None,
        index=torch.full((2, 16, 128), -99, dtype=torch.int8, device=device),
        indexer_scale=torch.full((2, 16, 1), float("nan"), dtype=torch.float16, device=device),
        swa=torch.full((2, 128, 512), float("nan"), dtype=torch.bfloat16, device=device),
        compress_kv_state=torch.full((2, 32, 512), float("nan"), device=device),
        compress_score_state=torch.full((2, 32, 512), float("nan"), device=device),
    )
    generator = torch.Generator().manual_seed(16)
    projected = torch.randn(6, 512, generator=generator).to(device)
    gates = torch.randn(6, 512, generator=generator).to(device)
    table = torch.tensor([[1]], device=device)
    offset = 0
    for length in (3, 3):
        metadata = Csa2EagerMetadata((length,), (offset + length,), table, table.new_full((length,), -1))
        rows = csa2_pool_compressor(
            projected[offset : offset + length], gates[offset : offset + length], cache, metadata, 2
        )
        main_k = rows.values.bfloat16()
        index_k = main_k[:, :128].contiguous()
        csa2_commit_compressed(cache, table, table.new_tensor([[0]]), rows, main_k, index_k)
        torch.testing.assert_close(cache.key[1, rows.row_ids], main_k, rtol=0, atol=0)
        actual = csa2_gather_index(cache, table.new_tensor([[0]]), 0, (offset + length) // 2)[rows.row_ids]
        scales = cache.indexer_scale[0, rows.row_ids].float()
        assert ((actual - index_k.float()).abs() <= scales * 0.501 + 1e-6).all()
        offset += length
    assert torch.isnan(cache.key[0]).all()
    assert cache.index[1].eq(-99).all()


@pytest.mark.parametrize("length", [1, 13])
def test_ring_plan_reuses_indices_without_reusing_layer_values(device: torch.device, length: int) -> None:
    generator = torch.Generator().manual_seed(391)
    previous = 7
    query = torch.randn(previous + length, 2, 4, generator=generator)
    table = torch.tensor([[3, 1]], device=device)
    positions = torch.arange(previous, previous + length, device=device)
    slots = table[0].index_select(0, positions // 2 % 2) * 2 + positions % 2
    # A masked final write must neither replace an earlier writer nor write
    # to the last physical row through a negative index.
    slots[-1] = -1
    metadata = Csa2EagerMetadata((length,), (previous + length,), table, slots)
    plan = csa2_prepare_ring(metadata, 2, 4, 4)
    for _ in range(2):
        values = torch.randn(previous + length, 4, generator=generator)
        swa = torch.full((4, 2, 4), float("nan"), device=device)
        for position in range(previous - 4, previous):
            block = int(table[0, position // 2 % 2])
            swa[block, position % 2] = values[position].to(device)
        expected_cache = swa.cpu().clone()
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                expected_cache.view(-1, 4)[slot] = values[previous + row]
        expected = _oracle(query, values, values[:0], 0, torch.zeros(2), 4)[previous:]
        actual = csa2_eager_attention(
            query[previous:].to(device),
            values[previous:].to(device),
            LayerCache(None, None, swa=swa),
            metadata,
            window_size=4,
            scale=0.5,
            sinks=torch.zeros(2, device=device),
            ring_plan=plan,
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(swa.cpu(), expected_cache, rtol=0, atol=0, equal_nan=True)
