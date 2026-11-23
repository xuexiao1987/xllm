# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""V4.1 single-pass mHC with portable math and Ascend A3 VA kernels.

Each sublayer collapses with the *previous* sublayer's pre coefficients and
produces new coefficients for the next sublayer. V4 kernels that collapse with
the current coefficients cannot replace this operation without changing it.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from scripts.logger import logger
from xllm.python.model_loader import ModelWeightLoader


class DeepseekV41HyperConnection(nn.Module):
    """Model-owned FP32 mixing weights and explicit residual recombination."""

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if min(hidden_size, hc_mult, sinkhorn_iters) <= 0 or min(norm_eps, hc_eps) <= 0:
            raise ValueError("mHC dimensions, iterations and epsilons must be positive")
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.sinkhorn_iters = sinkhorn_iters
        self._fused_pre = None
        self._fused_post = None
        if device.type in ("npu", "privateuseone") and hc_mult == 4:
            from xllm.python.kernels_npu.hyper_connection import hc_post_va, hc_pre_va

            if torch.npu.get_device_name(device).startswith("Ascend910_93"):
                self._fused_pre = hc_pre_va
                self._fused_post = hc_post_va
                logger.info("V4.1 mHC VA kernels: pre=%s post=%s", hc_pre_va is not None, hc_post_va is not None)
        mixes = (2 + hc_mult) * hc_mult
        for part in ("attn", "ffn"):
            for suffix, shape in (("fn", (mixes, hc_mult * hidden_size)), ("base", (mixes,)), ("scale", (3,))):
                self.register_parameter(
                    f"hc_{part}_{suffix}",
                    nn.Parameter(torch.empty(shape, dtype=torch.float32, device=device), requires_grad=False),
                )

    @torch.no_grad()
    def load_weights(self, loader: ModelWeightLoader, layer_id: int) -> None:
        for name, parameter in self.named_parameters():
            key = f"layers.{layer_id}.{name}"
            tensor = loader.get_tensor(key)
            if tensor.dtype != torch.float32 or tensor.shape != parameter.shape:
                raise ValueError(f"{key} must be FP32 with shape {tuple(parameter.shape)}")
            parameter.copy_(tensor)

    def forward(
        self, hidden: torch.Tensor, previous_pre: torch.Tensor | None, *, part: Literal["attn", "ffn"]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return collapsed input, next pre mix, this post mix and residual matrix."""
        if part not in ("attn", "ffn"):
            raise ValueError(f"unknown mHC sublayer {part!r}")
        if self._fused_pre is not None and hidden.dtype == torch.bfloat16 and hidden.numel() > 0:
            if previous_pre is None:
                previous_pre = torch.zeros(hidden.shape[:2], dtype=torch.float32, device=hidden.device)
                previous_pre[:, 0] = 1.0
            # VA returns the next pre last; the model carries it to the next
            # sublayer. V4's hc_pre instead collapses with its own new mix.
            collapsed, post, combination, pre = self._fused_pre(
                hidden.contiguous(),
                getattr(self, f"hc_{part}_fn"),
                getattr(self, f"hc_{part}_scale"),
                getattr(self, f"hc_{part}_base"),
                previous_pre.contiguous(),
                self.hc_mult,
                self.sinkhorn_iters,
                self.norm_eps,
                self.hc_eps,
            )
            return collapsed, pre, post, combination
        pre, post, combination = self._mixes(
            hidden, getattr(self, f"hc_{part}_fn"), getattr(self, f"hc_{part}_scale"), getattr(self, f"hc_{part}_base")
        )
        # First block uses stream zero; the final model merge must instead use
        # the last FFN's learned pre, via collapse(), not a mean over streams.
        collapsed = hidden[:, 0] if previous_pre is None else self.collapse(hidden, previous_pre)
        return collapsed, pre, post, combination

    def _mixes(
        self, hidden: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = hidden.flatten(1).float()
        projected = flattened.new_empty((flattened.shape[0], weight.shape[0]))
        # Keep GEMM geometry independent of prefill chunk size. Use one tile,
        # not a full padded batch; zeros also avoid undefined padding values.
        tile = flattened.new_zeros((128, flattened.shape[1]))
        for start in range(0, flattened.shape[0], 128):
            count = min(128, flattened.shape[0] - start)
            tile[:count].copy_(flattened[start : start + count])
            projected[start : start + count].copy_(F.linear(tile, weight)[:count])
        mixes = projected * torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mult = self.hc_mult
        pre = torch.sigmoid(mixes[:, :mult] * scale[0] + base[:mult]) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[:, mult : 2 * mult] * scale[1] + base[mult : 2 * mult])
        combination = (mixes[:, 2 * mult :] * scale[2] + base[2 * mult :]).unflatten(-1, (mult, mult))
        combination = torch.softmax(combination, dim=-1) + self.hc_eps
        combination = combination / (combination.sum(dim=1, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            combination = combination / (combination.sum(dim=-1, keepdim=True) + self.hc_eps)
            combination = combination / (combination.sum(dim=1, keepdim=True) + self.hc_eps)
        return pre, post, combination

    @staticmethod
    def collapse(hidden: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Sum in a fixed FP32 order before converting back to activation dtype."""
        contributions = (pre[:, :, None] * hidden.float()).unbind(dim=1)
        output = contributions[0]
        for contribution in contributions[1:]:
            output = output + contribution
        return output.to(hidden.dtype)

    def expand(
        self, output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, combination: torch.Tensor
    ) -> torch.Tensor:
        """combination[input stream, output stream] mixes the residual."""
        if (
            self._fused_post is not None
            and output.dtype == torch.bfloat16
            and output.numel() > 0
            and output.size(-1) % 64 == 0
        ):
            return self._fused_post(
                output.contiguous().unsqueeze(0),
                residual.contiguous().unsqueeze(0),
                post.contiguous().unsqueeze(0),
                combination.contiguous().unsqueeze(0),
            ).squeeze(0)
        value = post[:, :, None] * output.float()[:, None, :]
        value = value + torch.einsum("tkj,tkd->tjd", combination.float(), residual.float())
        return value.to(output.dtype)
