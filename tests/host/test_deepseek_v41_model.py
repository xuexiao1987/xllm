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

"""Full text model execution through real projections, backend and checkpoints."""

import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_backend import Csa2AttentionBackend
from xllm.python.layers.rotary_embedding import apply_interleaved_rotary
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM
from xllm.python.models.deepseek_v41_engram import EngramLayout, engram_hash_multipliers


class _State(dict[str, torch.Tensor]):
    def has(self, name: str) -> bool:
        return name in self

    def get_tensor(self, name: str) -> torch.Tensor:
        return self[name]


@dataclass
class _Metadata:
    q_seq_lens_host: torch.Tensor
    kv_seq_lens_host: torch.Tensor
    multi_block_tables: tuple[torch.Tensor, ...]
    prepared_attention_state: object | None = None
    engram_token_history: torch.Tensor | None = None
    dp_execution_token_counts: tuple[int, ...] = ()


@pytest.fixture(scope="module")
def device() -> torch.device:
    name = os.environ.get("XLLM_V41_MODEL_TEST_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(name)
    return torch.device(name)


def _config(device: torch.device, dtype: torch.dtype) -> dict:
    return dict(
        model_type="deepseek_v41",
        device=str(device),
        dtype=dtype,
        num_hidden_layers=6,
        hidden_size=4,
        vocab_size=16,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        o_lora_rank=2,
        o_groups=1,
        index_n_heads=2,
        index_head_dim=4,
        qk_rope_head_dim=2,
        max_position_embeddings=64,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4],
        candidate_source_layer_id=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_topk=2,
        window_size=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        moe_intermediate_size=4,
        n_routed_experts=2,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
    )


def _checkpoint(model: DeepseekV41ForCausalLM) -> _State:
    generator = torch.Generator().manual_seed(411)
    state = _State()
    dtype = model.model.embed_tokens.weight.dtype
    g = model.cfg.checkpoint

    def _random(shape: tuple[int, ...], dt: torch.dtype = dtype) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.15).to(dt)

    state["embed.weight"] = _random((g.vocab_size, g.hidden_size))
    state["head.weight"] = _random((g.vocab_size, g.hidden_size))
    state["norm.weight"] = torch.ones(g.hidden_size, dtype=dtype)
    for layer in model.model.layers:
        prefix = f"layers.{layer.layer_id}."
        for spec in layer.self_attn.weight_specs:
            if spec.quantized:
                state[spec.source] = torch.randint(-8, 9, spec.shape, generator=generator, dtype=torch.int8)
            elif spec.source.endswith(".weight_scale"):
                state[spec.source] = torch.full(spec.shape, 1 / 128)
            elif spec.source.endswith(".weight_offset"):
                state[spec.source] = torch.zeros(spec.shape)
            else:
                state[spec.source] = (
                    torch.ones(spec.shape, dtype=dtype) if "norm.weight" in spec.source else _random(spec.shape)
                )
        for name in ("attn_norm", "ffn_norm"):
            state[prefix + name + ".weight"] = torch.ones(g.hidden_size, dtype=dtype)
        for name, parameter in layer.hc.named_parameters():
            state[prefix + name] = _random(tuple(parameter.shape), torch.float32)
        moe = layer.mlp.cfg if hasattr(layer.mlp, "cfg") else layer.mlp.config
        state[prefix + "ffn.gate.weight"] = _random((moe.n_routed_experts, g.hidden_size))
        state[prefix + "ffn.gate.bias"] = _random((moe.n_routed_experts,), torch.float32)
        for part in [*(f"experts.{i}" for i in range(moe.n_routed_experts)), "shared_experts"]:
            for name in ("w1", "w2", "w3"):
                key = prefix + f"ffn.{part}.{name}"
                shape = (
                    (g.hidden_size, moe.moe_intermediate_size)
                    if name == "w2"
                    else (moe.moe_intermediate_size, g.hidden_size)
                )
                if g.quant_method == "ascend":
                    state[key + ".weight"] = torch.randint(-8, 9, shape, generator=generator, dtype=torch.int8)
                    state[key + ".weight_scale"] = torch.full((shape[0], 1), 1 / 128)
                else:
                    state[key + ".weight"] = _random(shape)
        if layer.engram is not None:
            embed = model.model.engram_embeddings[str(layer.layer_id)]
            state[prefix + "engram.embed.weight"] = torch.randint(
                -4, 5, (embed.num_embeddings, embed.dim), generator=generator, dtype=torch.int8
            )
            state[prefix + "engram.embed.scale"] = torch.full((embed.num_embeddings, embed.dim // 32), 1 / 64)
            for name, parameter in layer.engram.named_parameters():
                state[prefix + "engram." + name] = _random(tuple(parameter.shape))
    return state


def _caches(
    model: DeepseekV41ForCausalLM, tokens_per_sequence: int = 16
) -> tuple[list[LayerCache], tuple[torch.Tensor, ...]]:
    weight = model.model.embed_tokens.weight
    device, dtype = weight.device, weight.dtype
    head_dim, index_dim = model.cfg.checkpoint.head_dim, model.cfg.checkpoint.index_head_dim
    blocks_per_sequence = (tokens_per_sequence + 1) // 2
    num_blocks = 2 * blocks_per_sequence
    caches = []
    for layer in model.model.layers:
        full = layer.self_attn.spec.mode == "full"
        state = full and layer.self_attn.spec.ratio > 1
        caches.append(
            LayerCache(
                key=torch.full((num_blocks, 2, head_dim), float("nan"), device=device, dtype=dtype) if full else None,
                value=None,
                index=torch.full((num_blocks, 2, index_dim), -99, device=device, dtype=torch.int8) if full else None,
                indexer_scale=torch.full((num_blocks, 2, 1), float("nan"), device=device, dtype=torch.float16)
                if full
                else None,
                swa=torch.full((4, 4, head_dim), float("nan"), device=device, dtype=dtype),
                compress_kv_state=torch.full((4, 32, head_dim), float("nan"), device=device) if state else None,
                compress_score_state=torch.full((4, 32, head_dim), float("nan"), device=device) if state else None,
            )
        )
    tables = (
        torch.tensor([[2, 0], [3, 1]], device=device),
        torch.arange(num_blocks, device=device).reshape(2, blocks_per_sequence).flip(1),
        torch.arange(num_blocks, device=device).reshape(2, blocks_per_sequence),
    )
    return caches, tables


def _run(model: DeepseekV41ForCausalLM, chunks: tuple[int, ...]) -> torch.Tensor:
    caches, tables = _caches(model, max(16, sum(chunks)))
    device = model.model.embed_tokens.weight.device
    backend = Csa2AttentionBackend(model.cfg.backend)
    backend.bind_kv_caches(caches)
    output = []
    offset = 0
    for length in chunks:
        tokens = torch.arange(offset, offset + length, device=device) % 16
        positions = torch.arange(offset, offset + length, device=device)
        metadata = _Metadata(torch.tensor([length, 0]), torch.tensor([offset + length, 0]), tables)
        if model.model.engram_hash is not None:
            width = model.model.engram_hash.layout.max_ngram_size - 1
            previous = torch.arange(offset - 1, offset - width - 1, -1, device=device)
            metadata.engram_token_history = torch.stack(
                (torch.where(previous >= 0, previous % 16, -1), previous.new_full((width,), -1))
            )
        backend.prepare(metadata)
        with forward_context(ForwardContext(backend, device, metadata, caches)):
            hidden = model.model(tokens, positions)
            output.append(model.compute_logits(hidden, None))
        offset += length
    return torch.cat(output).cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loaded_model_prefill_chunk_decode_agree(device: torch.device, dtype: torch.dtype) -> None:
    model = DeepseekV41ForCausalLM(_config(device, dtype))
    state = _checkpoint(model)
    model.load_weights([state], 0, 1)
    expected = _run(model, (13,))
    for chunks in [(1,) * 13, (3, 7, 3), (0, 3, 0, 10)]:
        actual = _run(model, chunks)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.02 if dtype == torch.bfloat16 else 5e-5,
            atol=0.01 if dtype == torch.bfloat16 else 2e-6,
        )
    assert expected.shape == (13, 16)
    assert not any(p.requires_grad for p in model.parameters())


def test_reuse_cannot_consume_previous_forward_picks(device: torch.device) -> None:
    model = DeepseekV41ForCausalLM(_config(device, torch.float32))
    model.load_weights([_checkpoint(model)], 0, 1)
    caches, tables = _caches(model)
    backend = Csa2AttentionBackend(model.cfg.backend)
    backend.bind_kv_caches(caches)
    for _ in range(2):
        metadata = _Metadata(torch.tensor([1, 0]), torch.tensor([1, 0]), tables)
        backend.prepare(metadata)
        with forward_context(ForwardContext(backend, device, metadata, caches)):
            with pytest.raises(RuntimeError, match="before its KV source"):
                model.model.layers[2].self_attn(
                    torch.ones(1, 4, device=device),
                    torch.zeros(1, dtype=torch.int64, device=device),
                    model.model.compress_rotary.cos_sin_cache,
                )
            result = model.model(torch.tensor([2], device=device), torch.tensor([0], device=device))
            assert torch.isfinite(result).all()


def test_partial_rotary_matches_complex_reference(device: torch.device) -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 10
    angle = torch.tensor([0.1, 0.3, 0.8])
    table = torch.stack((angle.cos(), angle.sin()), dim=-1)
    expected = x.clone()
    z = torch.view_as_complex(x[..., 2:].contiguous()) * torch.polar(torch.ones(3), angle)[:, None]
    expected[..., 2:] = torch.view_as_real(z)
    actual = apply_interleaved_rotary(x.to(device), torch.arange(3, device=device), table.to(device))
    torch.testing.assert_close(actual.cpu(), expected)
    inverse = apply_interleaved_rotary(actual, torch.arange(3, device=device), table.to(device), inverse=True)
    torch.testing.assert_close(inverse.cpu(), x)


def test_full_model_with_engram_history(device: torch.device, tmp_path: Path) -> None:
    config = _config(device, torch.float32)
    config.update(
        model_path=str(tmp_path),
        engram_layer_ids=[1, 4],
        engram_num_embeddings=[64, 192],
        engram_max_ngram_size=4,
        engram_vocab_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=16,
        engram_pad_token_id=2,
    )
    layout = EngramLayout.from_dict(config)
    torch.save(
        dict(token_map=torch.arange(16), multipliers=engram_hash_multipliers(layout)),
        tmp_path / "engram_hash_state.bin",
    )
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    _assert_dummy_preserves_cache(model)
    expected = _run(model, (13,))
    for chunks in [(1,) * 13, (3, 7, 3)]:
        torch.testing.assert_close(_run(model, chunks), expected, rtol=5e-5, atol=2e-6)


def test_model_executor_dispatch_and_prefill_decode(device: torch.device) -> None:
    from xllm.python.model_executor.executor import ModelExecutor

    config = _config(device, torch.float32)
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    expected = _run(model, (13,))
    executor = ModelExecutor(model, config, max_seqs_per_batch=2)
    caches, tables = _caches(model)
    executor.bind_kv_caches(caches)
    outputs = []
    offset = 0
    for length in (3, 7, 1, 1, 1):
        metadata = _Metadata(
            torch.tensor([length, 0]), torch.tensor([offset + length, 0]), tuple(t.cpu() for t in tables)
        )
        metadata.is_prefill = offset == 0
        metadata.is_chunked_prefill = offset > 0 and length > 1
        metadata.is_spec_verify = False
        tokens = torch.arange(offset, offset + length, device=device)
        hidden = executor.execute(tokens, tokens, metadata)
        outputs.append(model.compute_logits(hidden, None))
        offset += length
    torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=5e-5, atol=2e-6)


def test_batched_requests_reorder_without_cache_leakage(device: torch.device) -> None:
    model = DeepseekV41ForCausalLM(_config(device, torch.float32))
    model.load_weights([_checkpoint(model)], 0, 1)
    tokens = (torch.arange(7, device=device), (torch.arange(11, device=device) + 3) % 16)

    def _requests(rounds: tuple[tuple[tuple[int, int], ...], ...]) -> tuple[torch.Tensor, ...]:
        caches, tables = _caches(model)
        backend = Csa2AttentionBackend(model.cfg.backend)
        backend.bind_kv_caches(caches)
        offsets = [0, 0]
        results: list[list[torch.Tensor]] = [[], []]
        for requests in rounds:
            ids, lengths = zip(*requests, strict=True)
            contexts = [offsets[i] + length for i, length in requests]
            metadata = _Metadata(torch.tensor(lengths), torch.tensor(contexts), tuple(t[list(ids)] for t in tables))
            inputs = torch.cat([tokens[i][offsets[i] : end] for (i, _), end in zip(requests, contexts, strict=True)])
            positions = torch.cat(
                [torch.arange(offsets[i], end, device=device) for i, end in zip(ids, contexts, strict=True)]
            )
            backend.prepare(metadata)
            with forward_context(ForwardContext(backend, device, metadata, caches)):
                logits = model.compute_logits(model.model(inputs, positions), None)
            for i, output, end in zip(ids, logits.split(lengths), contexts, strict=True):
                results[i].append(output)
                offsets[i] = end
        return tuple(torch.cat(rows).cpu() for rows in results)

    expected = _requests((((0, 7),), ((1, 11),)))
    actual = _requests((((0, 3), (1, 2)), ((1, 3), (0, 4)), ((1, 6),)))
    for reference, batched in zip(expected, actual, strict=True):
        torch.testing.assert_close(batched, reference, rtol=5e-5, atol=2e-6)


def _assert_dummy_preserves_cache(model: DeepseekV41ForCausalLM) -> None:
    from xllm.python.model_executor.executor import ModelExecutor

    weight = model.model.embed_tokens.weight
    executor = ModelExecutor(model, model.cfg.source, max_seqs_per_batch=2)
    caches, _ = _caches(model)
    before = [[None if value is None else value.clone() for value in _cache_tensors(cache)] for cache in caches]
    executor.bind_kv_caches(caches)
    metadata = _Metadata(torch.tensor([1]), torch.tensor([1]), ())
    metadata.is_dummy = True
    metadata.is_prefill = False
    metadata.is_chunked_prefill = False
    metadata.is_spec_verify = False
    result = executor.execute(
        torch.ones(1, dtype=torch.int32, device=weight.device),
        torch.zeros(1, dtype=torch.int32, device=weight.device),
        metadata,
    )
    assert torch.isfinite(result).all()
    for cache, original in zip(caches, before, strict=True):
        for value, expected in zip(_cache_tensors(cache), original, strict=True):
            if value is not None:
                torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)


def _cache_tensors(cache: LayerCache) -> tuple[torch.Tensor | None, ...]:
    return (cache.key, cache.index, cache.indexer_scale, cache.swa, cache.compress_kv_state, cache.compress_score_state)


def test_empty_dp_dummy_does_not_write_cache(device: torch.device) -> None:
    model = DeepseekV41ForCausalLM(_config(device, torch.float32))
    model.load_weights([_checkpoint(model)], 0, 1)
    _assert_dummy_preserves_cache(model)


@pytest.mark.parametrize(
    "setting", [{"cp_size": 2}, {"kv_split_size": 2}, {"layerwise_split_size": 2}, {"enable_graph": True}]
)
def test_model_rejects_unsupported_execution_before_allocation(setting: dict) -> None:
    config = _config(torch.device("cpu"), torch.bfloat16)
    config.update(setting)
    with pytest.raises(NotImplementedError):
        DeepseekV41ForCausalLM(config)


@pytest.mark.parametrize("first_chunk", [1, 3])
def test_native_short_swa_tables_grow_with_context(device: torch.device, first_chunk: int) -> None:
    config = _config(device, torch.float32)
    config["window_size"] = 8
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    expected = _run(model, (first_chunk + 2,))
    caches, tables = _caches(model)
    backend = Csa2AttentionBackend(model.cfg.backend)
    backend.bind_kv_caches(caches)
    outputs = []
    offset = 0
    for query in (first_chunk, 1, 1):
        context = offset + query
        native_tables = (tables[0][:, : (context + 3) // 4], tables[1], tables[2])
        metadata = _Metadata(torch.tensor([query, 0]), torch.tensor([context, 0]), native_tables)
        backend.prepare(metadata)
        positions = torch.arange(offset, context, device=device)
        with forward_context(ForwardContext(backend, device, metadata, caches)):
            outputs.append(model.compute_logits(model.model(positions % 16, positions), None))
        offset = context
    torch.testing.assert_close(torch.cat(outputs).cpu(), expected, atol=2e-5, rtol=2e-5)


def test_native_swa_mixed_long_and_short_requests(device: torch.device) -> None:
    config = _config(device, torch.float32)
    config["window_size"] = 8
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    expected_long = _run(model, (15,))
    expected_short = _run(model, (2,))
    caches, tables = _caches(model)
    backend = Csa2AttentionBackend(model.cfg.backend)
    backend.bind_kv_caches(caches)
    caches = [
        replace(
            cache,
            swa=cache.swa.new_full((6, 4, cache.swa.size(-1)), float("nan")),
            compress_kv_state=cache.compress_kv_state.new_full((6, 32, cache.swa.size(-1)), float("nan"))
            if cache.compress_kv_state is not None
            else None,
            compress_score_state=cache.compress_score_state.new_full((6, 32, cache.swa.size(-1)), float("nan"))
            if cache.compress_score_state is not None
            else None,
        )
        for cache in caches
    ]
    backend.bind_kv_caches(caches)
    # Native tables retain logical columns, including expired placeholders;
    # the short request has only one allocated page despite batch-wide padding.
    swa_table = torch.tensor([[0, 1, 2, 3], [4, -1, -1, -1]], device=device)
    for queries, contexts, positions in [
        ([13, 0], [13, 0], list(range(13))),
        ([1, 1], [14, 1], [13, 0]),
        ([1, 1], [15, 2], [14, 1]),
    ]:
        if queries[1]:
            swa_table[0, 0] = -1
        metadata = _Metadata(torch.tensor(queries), torch.tensor(contexts), (swa_table, *tables[1:]))
        backend.prepare(metadata)
        pos = torch.tensor(positions, device=device)
        with forward_context(ForwardContext(backend, device, metadata, caches)):
            output = model.compute_logits(model.model(pos % 16, pos), None).cpu()
        if queries[1]:
            torch.testing.assert_close(output[0], expected_long[contexts[0] - 1], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(output[1], expected_short[contexts[1] - 1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("candidate", [1, 2, 6, -2])
def test_unsupported_candidate_source_rejected_before_model_allocation(candidate: int) -> None:
    from xllm.python.models.deepseek_v41 import DeepseekV41Config

    config = _config(torch.device("cpu"), torch.float32)
    config["candidate_source_layer_id"] = candidate
    with pytest.raises(ValueError, match="candidate source"):
        DeepseekV41Config.from_dict(config)


def test_native_manager_identity_is_checked_before_model_allocation() -> None:
    from xllm.python.models.deepseek_v41 import DeepseekV41Config

    config = _config(torch.device("cpu"), torch.float32)
    config["csa2_manager_ratios"] = [0, 1, 2]
    with pytest.raises(ValueError, match="native manager layout"):
        DeepseekV41Config.from_dict(config)


def test_ring_metadata_is_shared_per_geometry_and_replaced_per_forward(device: torch.device) -> None:
    model = DeepseekV41ForCausalLM(_config(device, torch.float32))
    caches, tables = _caches(model)
    # A distinct layer allocation capacity requires independent validation,
    # even though its logical table matches the other layers.
    caches[-1] = replace(caches[-1], swa=torch.full((5, 4, 4), float("nan"), device=device))
    backend = Csa2AttentionBackend(model.cfg.backend)
    backend.bind_kv_caches(caches)
    first = _Metadata(torch.tensor([1, 0]), torch.tensor([1, 0]), tables)
    backend.prepare(first)
    old = first.prepared_attention_state
    assert old.layers[0] is old.layers[1]
    assert old.rings[0] is old.rings[1]
    assert old.rings[-1] is not old.rings[0]
    second = _Metadata(torch.tensor([1, 0]), torch.tensor([2, 0]), tables)
    backend.prepare(second)
    new = second.prepared_attention_state
    assert new.rings[0] is not old.rings[0]
    assert new.layers[0].context_lens == (2, 0)
    assert old.layers[0].context_lens == (1, 0)


@pytest.mark.parametrize("length", [65, 129])
def test_index_tiles_preserve_full_model_chunk_and_ring_results(device: torch.device, length: int) -> None:
    config = _config(device, torch.float32)
    config["max_position_embeddings"] = 256
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    expected = _run(model, (length,))
    for chunks in ((64, length - 64), (1,) * length):
        torch.testing.assert_close(_run(model, chunks), expected, rtol=5e-5, atol=2e-6)
