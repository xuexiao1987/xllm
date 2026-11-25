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

"""CSA2 eager attention arithmetic over executor-owned paged caches.

The model supplies projected/rotated Q and KV, sinks, and compressed-row
indices. This module owns neither weights nor persistent storage. It is an
explicit eager implementation with a fused SMLA tail for supported NPU shapes.
Selection happens before execution; device failures propagate. Model
projection/normalization/RoPE and backend dispatch remain separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_npu import Csa2SmlaState, csa2_smla_attention

_INDEXER_PRODUCT_ELEMENTS = 128 << 20


@dataclass(frozen=True, slots=True)
class Csa2EagerMetadata:
    """Packed query lengths and the executor's SWA ring write/read mapping.

    Context lengths include the current chunk. Tables contain physical block
    IDs; slots contain physical row IDs in packed-query order. The executor
    guarantees assigned blocks for visible history and disjoint live request
    allocations. Negative slots represent rows that must not be committed.
    """

    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    swa_block_table: torch.Tensor
    slot_mapping: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.query_lens) != len(self.context_lens):
            raise ValueError("CSA2 query/context batch sizes must match")
        if any(q < 0 or c < q for q, c in zip(self.query_lens, self.context_lens, strict=True)):
            raise ValueError("CSA2 requires 0 <= query length <= context length")
        if self.swa_block_table.ndim != 2 or self.swa_block_table.size(0) != len(self.query_lens):
            raise ValueError("CSA2 SWA block table must have one row per request")
        if self.swa_block_table.size(1) == 0:
            raise ValueError("CSA2 SWA ring needs at least one block column")
        if self.slot_mapping.ndim != 1 or self.slot_mapping.numel() != sum(self.query_lens):
            raise ValueError("CSA2 slot mapping must have one entry per query")


@dataclass(frozen=True, slots=True)
class Csa2CompressedInputs:
    """Non-owning view of the Full layer's KV and Full/Reindex layer's picks.

    Indices are logical compressed-row positions, packed [tokens, topk], with
    -1 padding. They have no SWA/chunk offset. Each query can see only complete
    compression groups through its own position, even after a chunk commits
    newer groups. Reuse passes the same view within that forward.
    """

    key: torch.Tensor
    block_table: torch.Tensor
    indices: torch.Tensor
    ratio: int

    def __post_init__(self) -> None:
        if self.ratio not in (1, 2, 4):
            raise ValueError("CSA2 compressed inputs require ratio 1, 2 or 4")
        if self.key.ndim != 3 or min(self.key.shape) == 0:
            raise ValueError("CSA2 compressed cache must be nonempty [blocks, rows, dim]")
        if self.block_table.ndim != 2 or self.block_table.size(1) == 0 or self.indices.ndim != 2:
            raise ValueError("CSA2 compressed tables and picks must be two-dimensional")


def csa2_indexer_scores(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Score [query, key]: per-head ReLU precedes weighted head reduction."""
    # A fixed query tile still creates multi-GB per-head intermediates at long
    # contexts. Target 512 MiB for each product tensor, with one query minimum.
    tile_size = max(1, _INDEXER_PRODUCT_ELEMENTS // max(1, q.size(1) * index_k.size(0)))
    q, index_k, weights = q.float(), index_k.float(), weights.float()
    if q.size(0) <= tile_size:
        products = torch.einsum("shd,td->sht", q, index_k)
        return (products.relu() * weights.unsqueeze(-1)).sum(dim=1)
    scores = q.new_empty((q.size(0), index_k.size(0)))
    for start in range(0, q.size(0), tile_size):
        end = min(start + tile_size, q.size(0))
        products = torch.einsum("shd,td->sht", q[start:end], index_k)
        scores[start:end] = (products.relu() * weights[start:end].unsqueeze(-1)).sum(dim=1)
    return scores


def csa2_select_candidates(
    scores: torch.Tensor, committed: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Select a block pool, retaining the newest (possibly partial) block.

    Visibility is imposed here so future rows cannot influence block scores.
    Equal scores prefer lower block IDs, making the pool independent of the
    batch/chunk width and device topk tie ordering.
    """
    if block_size <= 0 or topk_blocks <= 0:
        raise ValueError("CSA2 candidate block size and budget must be positive")
    width = scores.size(-1)
    visible = torch.arange(width, device=scores.device) < committed.unsqueeze(-1)
    scores = scores.masked_fill(~visible, float("-inf"))
    if width == 0:
        return visible
    padded = torch.nn.functional.pad(scores, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = torch.div(committed - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(torch.arange(blocks.size(-1), device=scores.device) == last.unsqueeze(-1), float("inf"))
    picks = torch.argsort(blocks, dim=-1, descending=True, stable=True)[..., :topk_blocks]
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, picks, blocks.gather(-1, picks) > float("-inf"))
    expanded = keep.unsqueeze(-1).expand(*keep.shape, block_size).flatten(-2)[..., :width]
    return expanded & visible


def csa2_topk(
    scores: torch.Tensor, committed: torch.Tensor, topk: int, candidates: torch.Tensor | None = None
) -> torch.Tensor:
    """Select visible finite scores, break ties by position, return sorted picks."""
    if topk <= 0:
        raise ValueError("CSA2 index topk must be positive")
    visible = torch.arange(scores.size(-1), device=scores.device) < committed.unsqueeze(-1)
    if candidates is not None:
        visible = visible & candidates
    scores = scores.masked_fill(~visible, float("-inf"))
    picks = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :topk].sort(dim=-1).values
    valid = torch.isfinite(scores.gather(-1, picks))
    picks = torch.where(valid, picks, -1).to(torch.int32)
    return torch.nn.functional.pad(picks, (0, topk - picks.size(-1)), value=-1)


def _gather_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    # CSA2 only gathers along dimension zero. Keep multidimensional picks out
    # of the generic NPU Index/AICPU path, including empty compressor groups.
    return values.index_select(0, rows.reshape(-1)).reshape((*rows.shape, *values.shape[1:]))


def _paged_slots(
    table: torch.Tensor, seq: int, rows: torch.Tensor, block_size: int, num_blocks: int, *, ring: bool
) -> torch.Tensor:
    cols = torch.div(rows.clamp_min(0), block_size, rounding_mode="floor")
    valid = rows >= 0
    if ring:
        cols = cols % table.size(1)
    blocks = _gather_rows(table[seq], cols.clamp_max(table.size(1) - 1)).long()
    missing = (cols >= table.size(1)) | (blocks < 0) | (blocks >= num_blocks)
    # Missing visible cache is a metadata/allocation error, not zero-valued KV.
    # This eager boundary check synchronizes, including before fused SMLA.
    # Graph integration must establish the invariant outside capture.
    if bool((valid & missing).any()):
        raise RuntimeError("CSA2 visible cache row has no assigned physical block")
    slots = blocks.clamp(0, num_blocks - 1) * block_size + rows.clamp_min(0) % block_size
    return torch.where(valid, slots, -1)


def _gather_paged(
    cache: torch.Tensor, table: torch.Tensor, seq: int, rows: torch.Tensor, *, ring: bool
) -> torch.Tensor:
    slots = _paged_slots(table, seq, rows, cache.size(1), cache.size(0), ring=ring)
    gathered = _gather_rows(cache.view(-1, cache.size(-1)), slots.clamp_min(0))
    return torch.where((slots >= 0).unsqueeze(-1), gathered, 0)


@dataclass(frozen=True, slots=True)
class Csa2RingPlan:
    """Forward-local physical row indices, shared by identically paged layers.

    The plan contains no KV values. Allocation tables and query geometry stay
    fixed during a forward; each layer still reads and writes its own cache.
    """

    history_slots: tuple[torch.Tensor, ...]
    write_slots: torch.Tensor
    write_rows: torch.Tensor


def csa2_prepare_ring(metadata: Csa2EagerMetadata, block_size: int, num_blocks: int, window_size: int) -> Csa2RingPlan:
    history = []
    for seq, (length, context) in enumerate(zip(metadata.query_lens, metadata.context_lens, strict=True)):
        previous = context - length
        positions = previous - window_size + torch.arange(window_size, device=metadata.swa_block_table.device)
        # The oldest candidate precedes every query's inclusive window.
        positions = torch.where(positions >= max(0, previous - window_size + 1), positions, -1)
        if length == 0:
            positions = torch.full_like(positions, -1)
        history.append(_paged_slots(metadata.swa_block_table, seq, positions, block_size, num_blocks, ring=True))
    write_slots, write_rows = _last_row_indices(metadata.slot_mapping)
    return Csa2RingPlan(tuple(history), write_slots, write_rows)


def _last_row_indices(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # A long chunk can wrap the ring multiple times. Select the last writer
    # before index_copy_, whose duplicate-index behavior is undefined.
    order = torch.argsort(slots, stable=True)
    sorted_slots = slots.index_select(0, order)
    last = torch.ones_like(sorted_slots, dtype=torch.bool)
    last[:-1] = sorted_slots[:-1] != sorted_slots[1:]
    kept = (last & (sorted_slots >= 0)).nonzero().flatten()
    return sorted_slots.index_select(0, kept).long(), order.index_select(0, kept)


def _scatter_last_rows(cache: torch.Tensor, slots: torch.Tensor, kv: torch.Tensor) -> None:
    write_slots, write_rows = _last_row_indices(slots)
    cache.view(-1, cache.size(-1)).index_copy_(0, write_slots, kv.index_select(0, write_rows))


def csa2_eager_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    cache: LayerCache,
    metadata: Csa2EagerMetadata,
    *,
    window_size: int,
    scale: float,
    sinks: torch.Tensor,
    compressed: Csa2CompressedInputs | None = None,
    smla_state: Csa2SmlaState | None = None,
    ring_plan: Csa2RingPlan | None = None,
) -> torch.Tensor:
    """Run causal SWA plus selected compressed KV, then commit the SWA ring.

    Q is [tokens, heads, dim], KV [tokens, dim], both already rotated. KV is
    shared between heads and acts as both K and V. A sink adds normalization
    mass only. Invalid rows are zeroed before any arithmetic (0 * NaN is NaN).
    No graph capture, CP or noncausal draft-block semantics are implied here.
    """
    swa = cache.swa
    if swa is None or swa.ndim != 3 or min(swa.shape) == 0:
        raise ValueError("CSA2 requires a nonempty executor-owned SWA cache")
    tokens = sum(metadata.query_lens)
    if q.ndim != 3 or q.size(0) != tokens or kv.shape != (tokens, q.size(-1)):
        raise ValueError("CSA2 projected Q/KV must match packed query geometry")
    if sinks.shape != (q.size(1),) or swa.size(-1) != q.size(-1):
        raise ValueError("CSA2 sink/head and cache dimensions must match Q")
    if window_size <= 0:
        raise ValueError("CSA2 attention window must be positive")
    # Native logical tables grow with the context; a new request need not own
    # pages for history that does not exist. Visible rows are checked below.
    required_rows = min(window_size, max(metadata.context_lens, default=0))
    if metadata.swa_block_table.size(1) * swa.size(1) < required_rows:
        raise ValueError("CSA2 ring capacity must cover the visible attention window")
    if compressed is not None and (
        compressed.indices.size(0) != tokens
        or compressed.block_table.size(0) != len(metadata.query_lens)
        or compressed.key.size(-1) != q.size(-1)
    ):
        raise ValueError("CSA2 compressed inputs must match packed queries and cache dimensions")
    if ring_plan is None:
        ring_plan = csa2_prepare_ring(metadata, swa.size(1), swa.size(0), window_size)
    output = torch.empty_like(q)
    use_smla = (
        q.device.type in ("npu", "privateuseone")
        and q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and swa.dtype == torch.bfloat16
        and q.size(-1) == 512
        and window_size == 128
        and torch.npu.get_device_name(q.device).startswith("Ascend910_93")
        and (
            compressed is None
            or (
                compressed.ratio in (1, 2)
                and compressed.indices.size(1) <= 1024
                and compressed.key.dtype == torch.bfloat16
                and compressed.key.size(1) == 128
            )
        )
    )
    offset = 0
    for seq, (length, context) in enumerate(zip(metadata.query_lens, metadata.context_lens, strict=True)):
        if length == 0:
            continue
        previous = context - length
        slots = ring_plan.history_slots[seq]
        history = _gather_rows(swa.view(-1, swa.size(-1)), slots.clamp_min(0))
        history = torch.where((slots >= 0).unsqueeze(-1), history, 0)
        window_kv = torch.cat((history, kv[offset : offset + length]), dim=0)
        if use_smla:
            cpicks = None
            if compressed is not None:
                positions = previous + torch.arange(length, device=q.device)
                cpicks = compressed.indices[offset : offset + length].long()
                visible = (cpicks >= 0) & (cpicks < ((positions + 1) // compressed.ratio).unsqueeze(-1))
                cpicks = torch.where(visible, cpicks, -1)
                # Retain the eager missing-block invariant before passing
                # unchecked physical tables to a fused device kernel.
                _paged_slots(
                    compressed.block_table,
                    seq,
                    cpicks,
                    compressed.key.size(1),
                    compressed.key.size(0),
                    ring=False,
                )
            output[offset : offset + length] = csa2_smla_attention(
                q[offset : offset + length],
                window_kv,
                context,
                sinks,
                scale,
                key=compressed.key if compressed is not None else None,
                table=compressed.block_table[seq : seq + 1] if compressed is not None else None,
                indices=cpicks,
                ratio=compressed.ratio if compressed is not None else 0,
                state=smla_state,
            )
            offset += length
            continue
        pick_width = window_size + (compressed.indices.size(1) if compressed is not None else 0)
        tile_size = max(1, min(64, (8 << 20) // (2 * pick_width * q.size(-1) * 4)))
        for start in range(0, length, tile_size):
            end = min(start + tile_size, length)
            positions = previous + torch.arange(start, end, device=q.device)
            window_positions = positions.unsqueeze(-1) - window_size + 1 + torch.arange(window_size, device=q.device)
            valid = window_positions >= 0
            picks = (window_positions - previous + window_size).clamp_min(0)
            picked = torch.where(valid.unsqueeze(-1), _gather_rows(window_kv, picks), 0)
            if compressed is not None:
                cpicks = compressed.indices[offset + start : offset + end].long()
                cvalid = (cpicks >= 0) & (cpicks < ((positions + 1) // compressed.ratio).unsqueeze(-1))
                crows = _gather_paged(
                    compressed.key, compressed.block_table, seq, torch.where(cvalid, cpicks, -1), ring=False
                )
                picked = torch.cat((picked, torch.where(cvalid.unsqueeze(-1), crows, 0)), dim=1)
                valid = torch.cat((valid, cvalid), dim=1)
            picked = picked.float()
            scores = torch.einsum("shd,snd->shn", q[offset + start : offset + end].float(), picked) * scale
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            # Include the sink in the stabilization max, not only the final
            # denominator: a large learned sink must not overflow exp().
            maximum = torch.maximum(scores.amax(dim=-1), sinks.float())
            probs = torch.exp(scores - maximum.unsqueeze(-1))
            denominator = probs.sum(dim=-1) + torch.exp(sinks.float() - maximum)
            value = torch.einsum("shn,snd->shd", probs, picked) / denominator.unsqueeze(-1)
            output[offset + start : offset + end] = value.to(q.dtype)
        offset += length
    # Defer even the first write until every request has consumed its history.
    swa.view(-1, swa.size(-1)).index_copy_(0, ring_plan.write_slots, kv.index_select(0, ring_plan.write_rows))
    return output


@dataclass(frozen=True, slots=True)
class Csa2PooledRows:
    """Completed groups in request order, before model-owned RMSNorm/RoPE.

    ``positions`` are absolute group starts for RoPE; ``row_ids`` are logical
    compressed-cache positions. ``counts`` keeps zero-query requests in the
    batch so the cache writer can use the original request block tables.
    """

    values: torch.Tensor
    positions: torch.Tensor
    row_ids: torch.Tensor
    counts: tuple[int, ...]


def csa2_pool_compressor(
    kv: torch.Tensor,
    scores: torch.Tensor | None,
    cache: LayerCache,
    metadata: Csa2EagerMetadata,
    ratio: int,
) -> Csa2PooledRows:
    """Pool projected KV and preserve partial groups in executor state rings.

    Ratio > 1 consumes FP32 model projections and FP32 state, with a separate
    softmax gate per latent channel. The model casts the pooled result to its
    activation dtype, applies RMSNorm, derives index K from that pre-RoPE
    latent, then rotates main/index K before committing them. Ratio 1 returns
    projected rows directly and uses neither scores nor compressor state.
    """
    if ratio not in (1, 2, 4):
        raise ValueError("CSA2 compressor ratio must be 1, 2 or 4")
    if kv.ndim != 2 or kv.size(0) != sum(metadata.query_lens):
        raise ValueError("CSA2 compressor projections must match packed queries")
    kv_state, score_state = cache.compress_kv_state, cache.compress_score_state
    if ratio == 1:
        if scores is not None:
            raise ValueError("CSA2 ratio 1 has no gate projection")
    else:
        if scores is None or scores.shape != kv.shape or scores.dtype != torch.float32 or kv.dtype != torch.float32:
            raise ValueError("CSA2 gated projections must be matching FP32 tensors")
        if cache.swa is None or kv_state is None or score_state is None:
            raise ValueError("CSA2 gated compression requires executor-owned state and SWA layout")
        if (
            kv_state.ndim != 3
            or kv_state.shape != score_state.shape
            or kv_state.size(0) != cache.swa.size(0)
            or kv_state.size(1) < ratio
            or kv_state.size(2) != kv.size(1)
            or kv_state.dtype != torch.float32
            or score_state.dtype != torch.float32
        ):
            raise ValueError("CSA2 compressor state must be matching FP32 [SWA blocks, state rows, latent dim]")

    values, starts, row_ids = [], [], []
    park_slots, park_values, park_scores = [], [], []
    counts = []
    offset = 0
    for seq, (length, context) in enumerate(zip(metadata.query_lens, metadata.context_lens, strict=True)):
        previous = context - length
        groups = torch.arange(previous // ratio, context // ratio, device=kv.device)
        counts.append(groups.numel())
        row_ids.append(groups)
        starts.append(groups * ratio)
        current = kv[offset : offset + length]
        if ratio == 1:
            values.append(current)
        elif length:
            assert kv_state is not None and score_state is not None and cache.swa is not None and scores is not None
            state_rows, block_size = kv_state.size(1), cache.swa.size(1)
            gates = scores[offset : offset + length]
            members = groups.unsqueeze(-1) * ratio + torch.arange(ratio, device=kv.device)
            from_history = members < previous
            # Only residual members address state. Current/future projections
            # must not read an expired or still-unassigned state block.
            history_positions = torch.where(from_history, members, -1)
            history_swa_slots = _paged_slots(
                metadata.swa_block_table, seq, history_positions, block_size, kv_state.size(0), ring=True
            )
            history_slots = (history_swa_slots.clamp_min(0) // block_size) * state_rows + members % state_rows
            current_rows = (members - previous).clamp(0, length - 1)
            member_values = torch.where(
                from_history.unsqueeze(-1),
                _gather_rows(kv_state.view(-1, kv.size(1)), history_slots),
                _gather_rows(current, current_rows),
            )
            member_scores = torch.where(
                from_history.unsqueeze(-1),
                _gather_rows(score_state.view(-1, kv.size(1)), history_slots),
                _gather_rows(gates, current_rows),
            )
            values.append((member_values * member_scores.softmax(dim=1)).sum(dim=1))
            positions = previous + torch.arange(length, device=kv.device)
            # Keep the trailing state-ring span, including previous complete
            # groups' nonfinal members, matching internal residual retention.
            park = ((positions + 1) % ratio != 0) & (positions >= context - state_rows)
            parked_rows = park.nonzero().flatten()
            selected = positions.index_select(0, parked_rows)
            swa_slots = _paged_slots(metadata.swa_block_table, seq, selected, block_size, kv_state.size(0), ring=True)
            park_slots.append((swa_slots // block_size) * state_rows + selected % state_rows)
            park_values.append(current.index_select(0, parked_rows))
            park_scores.append(gates.index_select(0, parked_rows))
        else:
            values.append(kv[:0])
        offset += length
    # All old residual reads finish before any request's state is modified.
    if park_slots:
        assert kv_state is not None and score_state is not None
        slots = torch.cat(park_slots)
        _scatter_last_rows(kv_state, slots, torch.cat(park_values))
        _scatter_last_rows(score_state, slots, torch.cat(park_scores))
    empty_rows = torch.empty(0, dtype=torch.int64, device=kv.device)
    return Csa2PooledRows(
        torch.cat(values) if values else kv[:0],
        torch.cat(starts) if starts else empty_rows,
        torch.cat(row_ids) if row_ids else empty_rows,
        tuple(counts),
    )


def csa2_quantize_index_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 with FP16 row scales and round-to-even codes.

    Round the scale to FP16 before dividing, matching the stored scale plane
    and the indexer query convention. Zero rows use the smallest FP16 subnormal.
    """
    scale = (x.float().abs().amax(dim=-1) / 127).half().float().clamp_min(2.0**-24)
    codes = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


def csa2_commit_compressed(
    cache: LayerCache,
    key_table: torch.Tensor,
    index_table: torch.Tensor,
    pooled: Csa2PooledRows,
    rotated_key: torch.Tensor,
    rotated_index_key: torch.Tensor,
) -> None:
    """Commit model-normalized/rotated main KV and quantized index K.

    Main and index block tables may have different physical layouts. Resolve
    every destination before the first mutation; missing storage is an error,
    never a reason to skip an index or scale-plane write.
    """
    key, index, scales = cache.key, cache.index, cache.indexer_scale
    if key is None or index is None or scales is None:
        raise ValueError("CSA2 compressed commit requires key, index and scale planes")
    if (
        key.ndim != 3
        or index.ndim != 3
        or scales.shape != (*index.shape[:2], 1)
        or min(key.shape) == 0
        or min(index.shape) == 0
        or index.dtype != torch.int8
        or scales.dtype != torch.float16
    ):
        raise ValueError("CSA2 requires paged key, INT8 index and FP16 per-row scale storage")
    total = sum(pooled.counts)
    if (
        rotated_key.shape != (total, key.size(-1))
        or rotated_key.dtype != key.dtype
        or rotated_index_key.shape != (total, index.size(-1))
        or pooled.row_ids.shape != (total,)
        or key_table.ndim != 2
        or index_table.ndim != 2
        or key_table.size(0) != len(pooled.counts)
        or index_table.size(0) != len(pooled.counts)
        or key_table.size(1) == 0
        or index_table.size(1) == 0
    ):
        raise ValueError("CSA2 compressed commit tensors must match completed groups and cache layout")
    key_slots, index_slots = [], []
    offset = 0
    for seq, count in enumerate(pooled.counts):
        rows = pooled.row_ids[offset : offset + count]
        key_slots.append(_paged_slots(key_table, seq, rows, key.size(1), key.size(0), ring=False))
        index_slots.append(_paged_slots(index_table, seq, rows, index.size(1), index.size(0), ring=False))
        offset += count
    if not key_slots:
        return
    codes, row_scales = csa2_quantize_index_rows(rotated_index_key)
    key_slot, index_slot = torch.cat(key_slots), torch.cat(index_slots)
    _scatter_last_rows(key, key_slot, rotated_key)
    _scatter_last_rows(index, index_slot, codes)
    _scatter_last_rows(scales, index_slot, row_scales.unsqueeze(-1))


def csa2_gather_index(cache: LayerCache, table: torch.Tensor, seq: int, count: int) -> torch.Tensor:
    """Read a request's committed index K in logical order and dequantize."""
    if cache.index is None or cache.indexer_scale is None:
        raise ValueError("CSA2 index reads require both code and scale planes")
    rows = torch.arange(count, device=cache.index.device)
    codes = _gather_paged(cache.index, table, seq, rows, ring=False)
    scales = _gather_paged(cache.indexer_scale, table, seq, rows, ring=False)
    return codes.float() * scales.float()
