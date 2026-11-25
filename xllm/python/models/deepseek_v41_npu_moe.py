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

"""V4.1 W8A8 text experts on the base NPU grouped-MoE execution path."""

from __future__ import annotations

import torch

from scripts.logger import logger
from xllm.python import distributed, kernels
from xllm.python.model_executor.forward_context import get_execution_buffer, get_forward_context_or_none
from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v4 import DeepseekV4MoE
from xllm.python.models.deepseek_v32 import DeepseekV3MLP
from xllm.python.models.deepseek_v41_moe import (
    DeepseekV41MoEConfig,
    load_v41_w8a8_weights,
)


class DeepseekV41SharedMLP(DeepseekV3MLP):
    """Preserve W8A8 INT32 accumulation through the clamped activation."""

    def __init__(
        self,
        cfg: DeepseekV41MoEConfig,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
        skip_tp_reduce: bool = False,
        tp_override: int | None = None,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__(cfg, intermediate_size, dtype, device, skip_tp_reduce, tp_override, swiglu_limit)
        self._use_fused_activation = (
            device.type in ("npu", "privateuseone")
            and dtype == torch.bfloat16
            and 0.0 < swiglu_limit < 1_000_000.0
            and torch.npu.get_device_name(device).startswith("Ascend910_93")
        )
        logger.info("V4.1 shared expert fused clamped activation: %s", self._use_fused_activation)

    def forward(self, x: torch.Tensor, tp_reduce_add: torch.Tensor | None = None) -> torch.Tensor:
        if not self._use_fused_activation or x.numel() == 0:
            return super().forward(x, tp_reduce_add)
        x_int8, per_token = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        rows = gate_up.shape[0]
        if get_forward_context_or_none() is None:
            group_index = torch.tensor([rows], dtype=torch.int64, device=x.device)
        else:
            group_index = get_execution_buffer(
                ("V41_SHARED_ACT_GROUP_INDEX", x.device, rows),
                lambda: torch.tensor([rows], dtype=torch.int64, device=x.device),
            )
        # The grouped V2 call honors clamp_limit. The dense variant can ignore
        # it; casting the INT32 GEMM to BF16 first also changes quantized codes.
        act_int8, act_scale = kernels.dequant_swiglu_quant(
            gate_up,
            self.gate_up_proj.weight_scale.reshape(1, -1),
            per_token,
            group_index=group_index,
            swiglu_mode=1,
            clamp_limit=self.swiglu_limit,
        )
        out = self.down_proj.forward_quantized(act_int8, act_scale)
        return self._reduce_output(out, tp_reduce_add)


class DeepseekV41W8A8MoE(DeepseekV4MoE):
    """Reuse DP gather/scatter, orthogonal TP/EP, quantization and grouped GEMMs.

    This is the portable base W8A8 path. V4.1 MC2/MegaMoE and multimodal
    routing are separate work; this class does not select those algorithms.
    """

    _SHARED_MLP_CLS = DeepseekV41SharedMLP

    def __init__(
        self, config: DeepseekV41MoEConfig, layer_id: int, *, dtype: torch.dtype, device: torch.device
    ) -> None:
        if not config.norm_topk_prob or config.n_activated_experts <= 1:
            raise ValueError("V4.1 base W8A8 routing requires normalized top-k with k > 1")
        if dtype != torch.bfloat16:
            raise ValueError("V4.1 W8A8 execution requires BF16 activations")
        super().__init__(config, layer_id, dtype, device)

    def load_weights(self, loader: ModelWeightLoader, layer_id: int) -> None:
        """Load the standard model.layers.<id>.mlp ownership and pack once."""
        load_v41_w8a8_weights(loader, self, self.cfg, layer_id)
        self.process_weights_after_loading()

    def _reduce_moe_outputs(self, routed_out: torch.Tensor, shared_out: torch.Tensor) -> torch.Tensor:
        if self.cfg.ep_size <= 1:
            return super()._reduce_moe_outputs(routed_out, shared_out)
        # EP sums independently quantized expert contributions. Keep this sum
        # in FP32 across both collective dimensions, then round once before
        # adding the shared expert, in the model's activation dtype.
        accumulator = routed_out.to(torch.promote_types(routed_out.dtype, torch.float32))
        if self.moe_tp_size > 1:
            distributed.moe_tp_all_reduce(accumulator)
        distributed.moe_ep_all_reduce(accumulator)
        routed_out.copy_(accumulator)
        if self.moe_tp_size > 1:
            distributed.moe_tp_all_reduce(shared_out)
        return routed_out + shared_out
