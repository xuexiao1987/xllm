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

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["position"])
def csa2_decode_compress_kernel(
    raw: tl.tensor,
    scores: tl.tensor,
    norm_weight: tl.tensor,
    rope: tl.tensor,
    state: tl.tensor,
    main: tl.tensor,
    state_table: tl.tensor,
    main_table: tl.tensor,
    latent: tl.tensor,
    positions: tl.tensor,
    slots: tl.tensor,
    position: int,
    state_columns: int,
    norm_eps: float,
    RATIO: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SAVE_TAIL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
) -> None:
    """Pool one decode row, normalize with BF16 round points, and commit KV."""
    cols = tl.arange(0, HEAD_DIM)
    current = tl.load(raw + cols).to(tl.float32)
    if RATIO == 2:
        gate = tl.load(scores + cols)
        if position % 2 == 1:
            prior = position - 1
            page = tl.load(state_table + (prior // 128) % state_columns)
            row = page.to(tl.int64) * 32 + prior % 32
            previous = tl.load(state + row * (2 * HEAD_DIM) + cols)
            previous_gate = tl.load(state + row * (2 * HEAD_DIM) + HEAD_DIM + cols)
            maximum = tl.maximum(previous_gate, gate)
            first = tl.exp(previous_gate - maximum)
            second = tl.exp(gate - maximum)
            denominator = first + second
            current = previous * (first / denominator) + current * (second / denominator)
        if SAVE_TAIL or position % 2 == 0:
            page = tl.load(state_table + (position // 128) % state_columns)
            row = page.to(tl.int64) * 32 + position % 32
            tl.store(state + row * (2 * HEAD_DIM) + cols, tl.load(raw + cols))
            tl.store(state + row * (2 * HEAD_DIM) + HEAD_DIM + cols, gate)
    if RATIO == 1 or position % 2 == 1:
        # Preserve both trained BF16 round points before RMSNorm and RoPE.
        rounded = current.to(tl.bfloat16).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(rounded * rounded) / HEAD_DIM + norm_eps)
        weight = tl.load(norm_weight + cols).to(tl.float32)
        normalized = (rounded * inverse * weight).to(tl.bfloat16)
        tl.store(latent + cols, normalized)
        logical = position // RATIO
        start = logical * RATIO
        page = tl.load(main_table + logical // 128)
        target = page.to(tl.int64) * 128 + logical % 128
        tl.store(positions, start)
        tl.store(slots, target)
        rotated = normalized.to(tl.float32)
        if ROPE_DIM > 0:
            in_tail = cols >= HEAD_DIM - ROPE_DIM
            relative = cols - (HEAD_DIM - ROPE_DIM)
            coefficient = start * (2 * ROPE_DIM) + tl.maximum(relative // 2 * 2, 0)
            cosine = tl.load(rope + coefficient, mask=in_tail, other=1)
            sine = tl.load(rope + coefficient + ROPE_DIM, mask=in_tail, other=0)
            other = tl.gather(rotated, cols ^ 1, axis=0)
            signed = tl.where(cols % 2 == 0, -other, other)
            rotated = rotated * cosine + signed * sine
        tl.store(main + target * HEAD_DIM + cols, rotated.to(tl.bfloat16))
