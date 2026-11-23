# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SMLA execution over validated, non-owning CSA2 cache views."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class _SmlaInputs:
    operands: dict[str, torch.Tensor | None]
    metadata: torch.Tensor
    original_table: torch.Tensor


@dataclass(slots=True)
class Csa2SmlaState:
    """Metadata shared only within one forward; contains no Q or KV values."""

    _entries: dict[tuple[torch.device, int, int, int, int, int, int], _SmlaInputs] = field(default_factory=dict)


def csa2_smla_attention(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    context: int,
    sinks: torch.Tensor,
    scale: float,
    *,
    key: torch.Tensor | None,
    table: torch.Tensor | None,
    indices: torch.Tensor | None,
    ratio: int,
    state: Csa2SmlaState | None = None,
) -> torch.Tensor:
    """Execute one packed sequence without mutating any persistent cache.

    The caller validates visible physical blocks and masks compressed picks
    per query. The window contains 128 history rows followed by this chunk;
    visible lengths exclude the zero padding before the beginning of a request.
    """
    from xllm.python import kernels

    length, heads, dim = q.shape
    window = 128
    has_compressed = indices is not None and indices.size(1) > 0
    ratio = ratio if has_compressed else 0
    width = ((indices.size(1) + 511) // 512) * 512 if has_compressed else 0
    capacity = table.size(1) * key.size(1) if has_compressed else 0
    cache_key = (q.device, length, context, heads, ratio, width, capacity)
    cached = state._entries.get(cache_key) if state is not None else None
    attributes = dict(
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    if cached is None:
        options = dict(dtype=torch.int32, device=q.device)
        starts = torch.tensor([0, length], **options)
        visible = torch.arange(context - length + 1, context + 1, **options).clamp(max=window).view(-1, 1)
        inputs = dict(
            cu_seqlens_q=starts,
            seqused_ori_kv=torch.tensor([window_kv.size(0)], **options),
            seqused_cmp_kv=torch.tensor([context // ratio], **options) if ratio else None,
            cmp_residual_kv=torch.tensor([context % ratio], **options) if ratio else None,
            ori_topk_length=visible,
        )
        metadata = kernels.smla_va_metadata(
            heads,
            1,
            dim,
            **inputs,
            **attributes,
            batch_size=1,
            max_seqlen_q=length,
            max_seqlen_ori_kv=window_kv.size(0),
            max_seqlen_cmp_kv=capacity,
            ori_topk=0,
            cmp_topk=width,
            has_ori_kv=True,
            has_cmp_kv=has_compressed,
        )
        original_blocks = (window_kv.size(0) + window - 1) // window
        cached = _SmlaInputs(inputs, metadata, torch.arange(original_blocks, **options).view(1, -1))
        if state is not None:
            state._entries[cache_key] = cached
    original = torch.nn.functional.pad(window_kv, (0, 0, 0, -window_kv.size(0) % window))
    original = original.view(-1, window, 1, dim)
    picks = None
    if has_compressed:
        picks = torch.nn.functional.pad(indices, (0, width - indices.size(1)), value=-1)
        picks = picks.to(torch.int32).unsqueeze(1).contiguous()
    output, _ = kernels.smla_va(
        q.contiguous(),
        ori_kv=original,
        cmp_kv=key.view(*key.shape[:2], 1, dim) if has_compressed else None,
        ori_block_table=cached.original_table,
        cmp_block_table=table.to(torch.int32).contiguous() if has_compressed else None,
        cmp_sparse_indices=picks,
        sinks=sinks.float(),
        metadata=cached.metadata,
        softmax_scale=scale,
        **cached.operands,
        **attributes,
    )
    return output
