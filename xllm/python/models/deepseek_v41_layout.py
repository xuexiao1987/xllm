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

"""Device-independent DeepSeek V4.1 CSA2 layer ownership.

The model constructs this plan once. Attention and cache integration consume
its source-layer assignments; importing it does not initialize device kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

CSA2_MODE_SWA = "swa"
CSA2_MODE_FULL = "full"
CSA2_MODE_REINDEX = "reindex"
CSA2_MODE_REUSE = "reuse"


@dataclass(frozen=True, slots=True)
class CSA2LayerMode:
    """Static CSA2 assignment for one layer."""

    layer_id: int
    mode: str
    # Main-KV compression ratio: 2 for encoder CSA2 layers, 1 for decoder
    # main-KV layers, 0 for pure SWA layers.
    compress_ratio: int
    # Layer whose main-KV / index caches this layer reads (-1 for SWA layers,
    # self for Full layers, the group's kv source for Reindex/Reuse layers).
    main_kv_source_layer: int

    @property
    def is_csa2(self) -> bool:
        return self.mode != CSA2_MODE_SWA

    @property
    def is_full(self) -> bool:
        return self.mode == CSA2_MODE_FULL

    @property
    def is_reindex(self) -> bool:
        return self.mode == CSA2_MODE_REINDEX

    @property
    def is_reuse(self) -> bool:
        return self.mode == CSA2_MODE_REUSE


class CSA2LayerPlan:
    """Static per-layer CSA2 assignment, built once at model init.

    ``kv_source_layer_ids`` are the
    Full layers that own a Compressor and publish the group's compressed KV
    and index K; ``index_source_layer_ids`` additionally lists the Reindex
    layers that re-score the shared K with their own indexer Q; every other
    layer after the nearest preceding kv source reuses the shared main KV and
    top-k. Layers before the first kv source (or with compress ratio 0) are
    pure SWA. For the reference config (``kv=[2,8,14,20]``,
    ``index=[2,8,14,20,24,28,32,36]``) this reproduces the documented
    assignment: layers 0-1 SWA, 2-19 encoder groups (Full at 2/8/14, Reuse
    otherwise), 20-39 decoder group (Full at 20, Reindex at 24/28/32/36,
    Reuse otherwise).

    Every KV-source layer compresses its own attention input, rather than
    capturing the hidden state from a different layer.
    """

    def __init__(self, modes: Sequence[CSA2LayerMode]) -> None:
        self._modes = tuple(modes)

    def mode(self, layer_id: int) -> CSA2LayerMode:
        if layer_id < 0 or layer_id >= len(self._modes):
            raise IndexError(f"layer {layer_id} outside the CSA2 plan's {len(self._modes)} layers")
        return self._modes[layer_id]

    def __len__(self) -> int:
        return len(self._modes)

    @property
    def manager_ratios(self) -> tuple[int, ...]:
        """Native V1 pool identity/order: SWA, then present C2 and C1 groups."""
        return (0,) + tuple(r for r in (2, 1) if any(m.is_full and m.compress_ratio == r for m in self._modes))

    def validate_candidate_source(self, source: int) -> None:
        if source < -1:
            raise ValueError("CSA2 candidate source must be -1 or an index-producing layer")
        if source == -1:
            return
        if source >= len(self) or self.mode(source).mode not in (CSA2_MODE_FULL, CSA2_MODE_REINDEX):
            raise ValueError("CSA2 candidate source must be an index-producing layer")
        owner = self.mode(source).main_kv_source_layer
        if any(m.is_csa2 and m.main_kv_source_layer != owner for m in self._modes[source + 1 :]):
            raise ValueError("CSA2 candidate source cannot cross KV groups")

    @classmethod
    def build(
        cls,
        n_layers: int,
        compress_ratios: Sequence[int],
        kv_source_layer_ids: Sequence[int],
        index_source_layer_ids: Sequence[int],
    ) -> CSA2LayerPlan:
        """Validate the config-derived inputs and derive the per-layer modes."""
        ratios = parse_compress_ratios(compress_ratios, n_layers)
        kv_sources = [int(source) for source in kv_source_layer_ids]
        index_sources = [int(source) for source in index_source_layer_ids]
        for sources in (kv_sources, index_sources):
            if any(a >= b for a, b in zip(sources, sources[1:])):
                raise ValueError("CSA2 source layers must be strictly increasing")
        kv_source_set = set(kv_sources)
        index_source_set = set(index_sources)
        for source in kv_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        for source in index_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"index source layer {source} is outside the model's {n_layers} layers")
        non_indexed = [source for source in kv_sources if source not in index_source_set]
        if non_indexed:
            raise ValueError(f"kv source layers must also be index sources: {non_indexed}")
        for source in index_sources:
            if not any(kv_source <= source for kv_source in kv_sources):
                raise ValueError(f"index source layer {source} has no preceding kv source layer")
        for source in kv_sources:
            if ratios[source] == 0:
                raise ValueError(f"kv source layer {source} must have a positive compress ratio")

        modes: list[CSA2LayerMode] = []
        for layer_id in range(n_layers):
            ratio = ratios[layer_id]
            kv_source = max((source for source in kv_sources if source <= layer_id), default=-1)
            if ratio == 0 or kv_source < 0:
                mode = CSA2_MODE_SWA
                main_kv_source = -1
            elif layer_id in kv_source_set:
                mode = CSA2_MODE_FULL
                main_kv_source = layer_id
            elif layer_id in index_source_set:
                mode = CSA2_MODE_REINDEX
                main_kv_source = kv_source
            else:
                mode = CSA2_MODE_REUSE
                main_kv_source = kv_source
            if main_kv_source >= 0 and ratio != ratios[main_kv_source]:
                raise ValueError("CSA2 consumers must use their source compression ratio")
            if layer_id in index_source_set and mode == CSA2_MODE_SWA:
                raise ValueError("CSA2 index source must have a positive compression ratio")
            modes.append(
                CSA2LayerMode(
                    layer_id=layer_id,
                    mode=mode,
                    compress_ratio=ratio,
                    main_kv_source_layer=main_kv_source,
                )
            )
        return cls(modes)


def parse_compress_ratios(raw: Sequence[int], n_layers: int) -> list[int]:
    """Parse V4.1 ``compress_ratios`` preserving ratio 0 / 1 / 2.

    V4 normalized every ``ratio <= 1`` to 1, which erases V4.1's distinction
    between pure-SWA layers (ratio 0) and decoder main-KV layers (ratio 1).
    The checkpoint config carries ``n_layers + num_nextn_predict_layers``
    entries (trailing zeros for the DSpark draft blocks); the main model
    consumes the first ``n_layers``. The native V1 model supports C1/C2;
    component arithmetic supporting C4 does not enable a C4 model layout.
    """
    if n_layers < 0:
        raise ValueError(f"n_layers must be nonnegative, got {n_layers}")
    ratios = [int(ratio) for ratio in raw]
    for ratio in ratios:
        if ratio not in (0, 1, 2):
            raise ValueError(f"unsupported DeepSeek-V4.1 compression ratio: {ratio}")
    if len(ratios) < n_layers:
        raise ValueError("compress_ratios must cover every model layer")
    return ratios[:n_layers]
