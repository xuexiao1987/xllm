# Copyright 2025-2026 The xLLM Authors.
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

"""NEOX-style rotary embedding cache in the layout the fused
``xllm_ops.fused_qk_norm_rope`` kernel expects. Pure table construction — no op
dispatch dependency."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def _expand_half_rope_cos_sin(
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a compact half-width cache row for partial-rotary kernels."""
    half = cos_sin.size(-1) // 2
    cos = cos_sin[..., :half].repeat_interleave(2, dim=-1).contiguous()
    sin = cos_sin[..., half:].repeat_interleave(2, dim=-1).contiguous()
    return cos, sin


class RotaryEmbedding(nn.Module):
    """Holds the NEOX-style RoPE cos/sin cache in the exact layout the fused
    ``xllm_ops.fused_qk_norm_rope`` kernel expects.

    ``cos_sin_cache`` is a single ``[max_position, head_dim]`` tensor whose first
    ``head_dim/2`` columns are ``cos(freqs)`` and last ``head_dim/2`` are
    ``sin(freqs)`` (``freqs = outer(positions, inv_freq)``). This matches C++
    ``MRotaryEmbedding::precomputed_cos_sin_cache()`` so both paths use identical
    rotary tables. Built on ``device`` in the model dtype to match C++ exactly.
    """

    cos_sin_cache: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        rope_scaling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        attention_factor = 1.0
        scaling = rope_scaling or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type == "yarn":
            factor = float(scaling["factor"])
            original_max = int(scaling["original_max_position_embeddings"])
            if factor <= 0 or original_max <= 0:
                raise ValueError("YaRN requires positive scaling and original context length")
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))

            def correction(rotations: float) -> float:
                return head_dim * math.log(original_max / (rotations * 2 * math.pi)) / (2 * math.log(rope_theta))

            low = max(math.floor(correction(beta_fast)), 0)
            high = min(math.ceil(correction(beta_slow)), head_dim - 1)
            denominator = high - low if high != low else 0.001
            ramp = ((torch.arange(head_dim // 2, dtype=torch.float32, device=device) - low) / denominator).clamp(0, 1)
            inv_freq = inv_freq * ((1 - ramp) + ramp / factor)
            attention_factor = float(scaling.get("attention_factor") or (0.1 * math.log(factor) + 1.0))
        elif rope_type != "default":
            raise NotImplementedError(f"unsupported Qwen RoPE type: {rope_type}")
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_position, head_dim/2]
        cos_half = freqs.cos() * attention_factor  # [max_position, head_dim/2]
        sin_half = freqs.sin() * attention_factor
        if dtype is not None:
            cos_half = cos_half.to(dtype)
            sin_half = sin_half.to(dtype)

        device_type = torch.device(device).type if device else "cpu"
        cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1)
        self.register_buffer("cos_sin_cache", cos_sin_cache.contiguous(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index into the cache and return per-token (cos, sin).

        Returns cos/sin as ``[1, num_tokens, 1, head_dim]`` via index + view
        (zero allocation on the hot path).
        """
        num_tokens = positions.size(0)
        cos_half = self.cos_sin_cache[positions]
        cos = cos_half.view(1, num_tokens, 1, self.head_dim)
        sin = cos_half.view(1, num_tokens, 1, self.head_dim)
        return cos, sin


def gather_half_rope_cos_sin(cos_sin_cache: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-token ``(cos_half, sin_half)`` rows from a cache built by
    :class:`RotaryEmbedding` (first half cos, second half sin)."""
    cos_sin = cos_sin_cache[positions]
    half = cos_sin.size(-1) // 2
    return cos_sin[..., :half], cos_sin[..., half:]


def apply_rotary_half(x: torch.Tensor, half_cos: torch.Tensor, half_sin: torch.Tensor) -> torch.Tensor:
    """NEOX rotate-half RoPE from half-width cos/sin rows.

    ``x``: [num_tokens, num_heads, head_dim]; ``half_cos``/``half_sin``:
    [num_tokens, head_dim // 2] (as returned by :func:`gather_half_rope_cos_sin`),
    broadcast over heads.
    """
    cos = half_cos.unsqueeze(1)
    sin = half_sin.unsqueeze(1)
    half = half_cos.size(-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@dataclass(frozen=True, slots=True)
class InterleavedRotaryInputs:
    """Immutable coefficients for one forward's positions, table and dtype."""

    rows: torch.Tensor
    expanded: tuple[torch.Tensor, torch.Tensor] | None


def prepare_interleaved_rotary(
    positions: torch.Tensor, cos_sin_cache: torch.Tensor, dtype: torch.dtype
) -> InterleavedRotaryInputs:
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).long()).float()
    expanded = None
    if rows.device.type in ("npu", "privateuseone") and dtype == torch.bfloat16 and rows.size(-1) == 64:
        expanded = _expand_half_rope_cos_sin(rows.to(dtype))
    return InterleavedRotaryInputs(rows, expanded)


def apply_interleaved_rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    inverse: bool = False,
    prepared: InterleavedRotaryInputs | None = None,
) -> torch.Tensor:
    """Rotate adjacent pairs in the trailing RoPE dimensions.

    The compact table contains [cos, sin] half rows; the first tensor axis is
    tokens and optional middle axes (heads) share each token's rotation.
    The NPU BF16 kernel consumes BF16 coefficients; the portable path keeps
    coefficients and intermediate arithmetic in FP32.
    """
    half = cos_sin_cache.size(-1) // 2
    rows = (
        prepared.rows if prepared is not None else cos_sin_cache.index_select(0, positions.reshape(-1).long()).float()
    )
    if (
        x.device.type in ("npu", "privateuseone")
        and x.dtype == torch.bfloat16
        and x.ndim in (2, 3)
        and x.numel() > 0
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.size(-1) == 64
        and 64 <= x.size(-1) <= 1024
        and x.size(-1) % 2 == 0
    ):
        from xllm.python import kernels

        cos, sin = (
            prepared.expanded
            if prepared is not None and prepared.expanded is not None
            else _expand_half_rope_cos_sin(rows.to(x.dtype))
        )
        # The base kernel is in-place; this public helper retains its
        # out-of-place contract, including for non-contiguous input views.
        output = x.clone(memory_format=torch.contiguous_format)
        kernels.npu_inplace_partial_rotary_mul(
            output.unsqueeze(1) if output.ndim == 2 else output,
            cos,
            sin,
            x.size(-1) - 64,
            64,
            inverse=inverse,
        )
        return output
    cos, sin = rows[..., :half], rows[..., half:]
    if inverse:
        sin = -sin
    for _ in range(x.ndim - 2):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    prefix, tail = x[..., : x.size(-1) - 2 * half], x[..., x.size(-1) - 2 * half :].float()
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((prefix, rotated.to(x.dtype)), dim=-1)


def _deepseek_yarn_inv_freq(
    factor: float,
    dim: int,
    theta: float,
    beta_fast: int,
    beta_slow: int,
    original_context: int,
    device: torch.device,
) -> torch.Tensor:
    # Preserve DeepSeek's interpolation/extrapolation operation order; folding
    # into one multiplication can change the final BF16 cache by one ULP.
    frequencies = theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    extrapolation = 1.0 / frequencies
    interpolation = 1.0 / (factor * frequencies)
    low = max(math.floor(dim * math.log(original_context / (beta_fast * 2 * math.pi)) / (2 * math.log(theta))), 0)
    high = min(math.ceil(dim * math.log(original_context / (beta_slow * 2 * math.pi)) / (2 * math.log(theta))), dim - 1)
    if low == high:
        high += 0.001
    mask = 1 - ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
    return interpolation * (1 - mask) + extrapolation * mask


class DeepseekRotaryEmbedding(nn.Module):
    """Compact-cache equivalent of C++ DeepseekRotaryEmbedding.

    C++ keeps cache length and YaRN's old-context length as separate inputs.
    The generic Python DeepseekYarnRotaryEmbedding derives cache length as
    ``old_context * factor``, which cannot represent the native DSV4 call.
    """

    _cache_lock = threading.Lock()
    _cache_by_descriptor: dict[tuple[Any, ...], torch.Tensor] = {}

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        theta: float,
        beta_fast: int,
        beta_slow: int,
        old_context_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        descriptor = (
            rotary_dim,
            max_position_embeddings,
            scaling_factor,
            theta,
            beta_fast,
            beta_slow,
            old_context_len,
            dtype,
            torch.device(device),
        )
        with self._cache_lock:
            cache = self._cache_by_descriptor.get(descriptor)
            if cache is None:
                cache = self._build_cache(
                    rotary_dim,
                    max_position_embeddings,
                    scaling_factor,
                    theta,
                    beta_fast,
                    beta_slow,
                    old_context_len,
                    dtype,
                    device,
                )
                self._cache_by_descriptor[descriptor] = cache
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _build_cache(
        rotary_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        theta: float,
        beta_fast: int,
        beta_slow: int,
        old_context_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Match C++ rotary::apply_deepseek_yarn_rope_scaling() and
        # create_cos_sin_tensor(): build the cache in CPU float32, then perform
        # one final transfer/conversion to the model device and dtype. Computing
        # pow/cos/sin directly on NPU can change BF16 cache entries by one ULP.
        cpu = torch.device("cpu")
        inv_freq = _deepseek_yarn_inv_freq(
            scaling_factor,
            rotary_dim,
            theta,
            beta_fast,
            beta_slow,
            old_context_len,
            cpu,
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=cpu)
        freqs = torch.outer(positions, inv_freq)
        # Keep one value per frequency. Call sites repeat_interleave to the
        # same [position, rotary_dim] interleaved layout C++ stores directly.
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device=device, dtype=dtype)
        return cache.contiguous()
