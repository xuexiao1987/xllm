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

"""CSA2 layer orchestration with model-owned projections and forward-owned state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2CompressedInputs,
    Csa2EagerMetadata,
    Csa2RingPlan,
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
from xllm.python.attention.csa2_npu import Csa2SmlaState
from xllm.python.layers.rotary_embedding import apply_interleaved_rotary
from xllm.python.model_executor.forward_context import get_forward_context


@dataclass(frozen=True, slots=True)
class Csa2LayerSpec:
    layer_id: int
    mode: str
    ratio: int
    kv_source: int


@dataclass(frozen=True, slots=True)
class Csa2BackendConfig:
    layers: tuple[Csa2LayerSpec, ...]
    window_size: int
    index_topk: int
    candidate_source: int
    candidate_blocks: int
    candidate_block_size: int
    # Native manager ordering: SWA, C2, C1. Ratio 4 uses the C4 manager.
    manager_ratios: tuple[int, ...] = (0, 2, 1)


class Csa2Projections(Protocol):
    spec: Csa2LayerSpec
    scale: float
    attn_sink: torch.Tensor

    def compress_kv_score(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def compress_normalize(self, pooled: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...
    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor: ...
    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor: ...
    def indexer_weights(self, hidden: torch.Tensor) -> torch.Tensor: ...


@dataclass(slots=True)
class Csa2ForwardState:
    layers: tuple[Csa2EagerMetadata, ...]
    key_tables: tuple[torch.Tensor | None, ...]
    # Shared group state is created for each ForwardContext; it is not retained
    # by the model or the backend between batches.
    is_dummy: bool = False
    source: int = -1
    picks: torch.Tensor | None = None
    candidates: list[torch.Tensor] = field(default_factory=list)
    smla: Csa2SmlaState = field(default_factory=Csa2SmlaState)
    rings: tuple[Csa2RingPlan, ...] = ()


def _select_sequence_indices(
    query: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    visible: torch.Tensor,
    config: Csa2BackendConfig,
    *,
    create_candidates: bool,
    candidates: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute one sequence's picks/pool without publishing forward state.

    The outer 64-query tile bounds score/selection matrices; indexer_scores
    separately bounds its per-head products. Candidate-source selection itself
    is unrestricted; only consuming layers apply the previously published pool.
    """
    picks, pools = [], []
    for start in range(0, query.size(0), 64):
        end = min(start + 64, query.size(0))
        score = csa2_indexer_scores(query[start:end], index_k, weights[start:end])
        if create_candidates:
            pools.append(
                csa2_select_candidates(score, visible[start:end], config.candidate_blocks, config.candidate_block_size)
            )
        pool = candidates[start:end] if candidates is not None else None
        picks.append(csa2_topk(score, visible[start:end], config.index_topk, pool))
    indices = torch.cat(picks) if picks else torch.empty((0, config.index_topk), dtype=torch.int32, device=query.device)
    pool = None
    if create_candidates:
        pool = torch.cat(pools) if pools else torch.empty((0, index_k.size(0)), dtype=torch.bool, device=query.device)
    return indices, pool


class Csa2AttentionBackend(AttentionBackend):
    def __init__(self, config: Csa2BackendConfig) -> None:
        self.config = config
        self._caches: list[LayerCache] = []

    @property
    def uses_executor_cp_context(self) -> bool:
        return False

    @property
    def num_kv_blocks(self) -> int:
        return self._caches[0].swa.size(0)

    @property
    def page_size(self) -> int:
        return self._caches[0].swa.size(1)

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        if len(kv_caches) != len(self.config.layers) or any(c.swa is None for c in kv_caches):
            raise ValueError("CSA2 requires one bound SWA cache per model layer")
        self._caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        if graph_mode:
            raise NotImplementedError("CSA2 eager backend cannot capture ACL graphs")
        if getattr(metadata, "is_dummy", False):
            # Empty DP ranks still execute MoE collectives, but own no request
            # blocks. Never route the framework's dummy slot 0 into live KV.
            metadata.prepared_attention_state = Csa2ForwardState((), (), is_dummy=True)
            return
        q_lengths = tuple(int(n) for n in metadata.q_seq_lens_host.tolist())
        contexts = tuple(int(n) for n in metadata.kv_seq_lens_host.tolist())
        tables = [table.to(self._caches[0].swa.device) for table in metadata.multi_block_tables]
        if len(tables) != len(self.config.manager_ratios):
            raise ValueError("CSA2 cache-manager tables do not match the allocation layout")
        swa_table = tables[self.config.manager_ratios.index(0)]
        if swa_table is None:
            raise ValueError("CSA2 requires the SWA manager table")
        layers, key_tables, rings = [], [], []
        prepared: dict[tuple[int, int], tuple[Csa2EagerMetadata, Csa2RingPlan]] = {}
        for spec, cache in zip(self.config.layers, self._caches, strict=True):
            block_size, num_blocks = cache.swa.size(1), cache.swa.size(0)
            geometry = (block_size, num_blocks)
            if geometry not in prepared:
                slots = []
                for seq, (query, context) in enumerate(zip(q_lengths, contexts, strict=True)):
                    positions = torch.arange(context - query, context, device=swa_table.device)
                    blocks = swa_table[seq].index_select(0, positions // block_size % swa_table.size(1))
                    slots.append(torch.where(blocks >= 0, blocks * block_size + positions % block_size, -1))
                slot_mapping = torch.cat(slots) if slots else swa_table.new_empty(0)
                local = Csa2EagerMetadata(q_lengths, contexts, swa_table, slot_mapping)
                prepared[geometry] = (local, csa2_prepare_ring(local, block_size, num_blocks, self.config.window_size))
            local, ring = prepared[geometry]
            layers.append(local)
            rings.append(ring)
            key_tables.append(tables[self.config.manager_ratios.index(spec.ratio)] if spec.mode != "swa" else None)
        metadata.prepared_attention_state = Csa2ForwardState(tuple(layers), tuple(key_tables), rings=tuple(rings))

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Csa2Projections) -> torch.Tensor:
        raise RuntimeError("CSA2 attention requires its projected latent context")

    def execute_csa2(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        hidden: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rope: torch.Tensor,
        layer: Csa2Projections,
    ) -> torch.Tensor:
        context = get_forward_context()
        state = context.metadata.prepared_attention_state
        if not isinstance(state, Csa2ForwardState):
            raise RuntimeError("CSA2 forward metadata was not prepared")
        if state.is_dummy:
            return torch.zeros_like(q)
        spec = layer.spec
        local = state.layers[spec.layer_id]
        compressed = None
        if spec.mode == "full":
            cache = self._caches[spec.layer_id]
            table = state.key_tables[spec.layer_id]
            projected, scores = layer.compress_kv_score(hidden)
            pooled = csa2_pool_compressor(projected, scores, cache, local, spec.ratio)
            latent = layer.compress_normalize(pooled.values, hidden.dtype)
            index_key = apply_interleaved_rotary(layer.indexer_key(latent), pooled.positions, rope)
            key = apply_interleaved_rotary(latent, pooled.positions, rope)
            csa2_commit_compressed(cache, table, table, pooled, key, index_key)
            state.source, state.picks, state.candidates = spec.layer_id, None, []
        if spec.mode != "swa":
            if state.source != spec.kv_source:
                raise RuntimeError("CSA2 consuming layer ran before its KV source in this forward")
            source_cache = self._caches[state.source]
            table = state.key_tables[state.source]
            if spec.mode in ("full", "reindex"):
                query = apply_interleaved_rotary(layer.indexer_query(qr), positions, rope)
                codes, scales = csa2_quantize_index_rows(query)
                query = codes.float() * scales.float().unsqueeze(-1)
                weights = layer.indexer_weights(hidden)
                picks, pools = [], []
                offset = 0
                for seq, (length, total) in enumerate(zip(local.query_lens, local.context_lens, strict=True)):
                    index_k = csa2_gather_index(source_cache, table, seq, total // spec.ratio)
                    visible = (positions[offset : offset + length] + 1) // spec.ratio
                    candidates = None
                    if length and spec.layer_id > self.config.candidate_source >= 0:
                        if not state.candidates:
                            raise RuntimeError("CSA2 candidate source has not run in this forward")
                        candidates = state.candidates[seq]
                    indices, pool = _select_sequence_indices(
                        query[offset : offset + length],
                        index_k,
                        weights[offset : offset + length],
                        visible,
                        self.config,
                        create_candidates=spec.layer_id == self.config.candidate_source,
                        candidates=candidates,
                    )
                    picks.append(indices)
                    if pool is not None:
                        pools.append(pool)
                    offset += length
                state.picks = (
                    torch.cat(picks)
                    if picks
                    else torch.empty((0, self.config.index_topk), dtype=torch.int32, device=q.device)
                )
                if spec.layer_id == self.config.candidate_source:
                    state.candidates = pools
            if state.picks is None:
                raise RuntimeError("CSA2 reuse requires top-k from this forward")
            compressed = Csa2CompressedInputs(source_cache.key, table, state.picks, spec.ratio)
        return csa2_eager_attention(
            q,
            kv,
            self._caches[spec.layer_id],
            local,
            window_size=self.config.window_size,
            scale=layer.scale,
            sinks=layer.attn_sink,
            compressed=compressed,
            smla_state=state.smla,
            ring_plan=state.rings[spec.layer_id],
        )
