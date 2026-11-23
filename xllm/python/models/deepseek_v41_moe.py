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

"""V4.1 text MoE configuration, routing and eager floating-point execution.

The eager backend is selected at construction for unquantized checkpoints.
It is not an error fallback for W8A8 and does not support graph capture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v4_checkpoint import (
    find_checkpoint_prefix,
    load_deepseek_moe_experts,
    require_checkpoint_key,
    require_shape,
    resolve_mlp_projection_names,
)


@dataclass(frozen=True)
class DeepseekV41MoEConfig:
    hidden_size: int = 5120
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    scoring_func: str = "sqrtsoftplus"
    n_hash_layers: int = 0
    tp_size: int = 1
    tp_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "n_activated_experts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for axis in ("tp", "moe_tp", "ep", "dp"):
            size, rank = getattr(self, axis + "_size"), getattr(self, axis + "_rank")
            if (
                isinstance(size, bool)
                or isinstance(rank, bool)
                or not isinstance(size, int)
                or not isinstance(rank, int)
                or size <= 0
                or not 0 <= rank < size
            ):
                raise ValueError(f"invalid {axis} topology")
        if self.n_routed_experts % self.ep_size or self.moe_intermediate_size % self.moe_tp_size:
            raise ValueError("Expert count must divide EP and intermediate width must divide MoE TP")
        if self.n_activated_experts > self.n_routed_experts or self.n_shared_experts != 1:
            raise ValueError("V4.1 requires top-k within expert count and one shared expert")
        if self.n_hash_layers != 0 or self.scoring_func != "sqrtsoftplus":
            raise ValueError("V4.1 requires sqrtsoftplus routing without hash layers")
        if not isinstance(self.norm_topk_prob, bool):
            raise ValueError("norm_topk_prob must be boolean")
        if (
            not math.isfinite(self.routed_scaling_factor)
            or self.routed_scaling_factor <= 0
            or not math.isfinite(self.swiglu_limit)
        ):
            raise ValueError("invalid routed scaling or SwiGLU limit")

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> DeepseekV41MoEConfig:
        text = config.get("text_config", config)
        if not isinstance(text, Mapping):
            raise ValueError("text_config must be an object")
        fields = {}
        for name in cls.__dataclass_fields__:
            # Parallel geometry is injected by the engine, outside HF text_config.
            source = (
                config
                if name
                in ("tp_size", "tp_rank", "moe_tp_size", "moe_tp_rank", "ep_size", "ep_rank", "dp_size", "dp_rank")
                else text
            )
            if name in source:
                fields[name] = source[name]
        if "moe_intermediate_size" not in fields and "intermediate_size" in text:
            fields["moe_intermediate_size"] = text["intermediate_size"]
        if "n_activated_experts" not in fields and "num_experts_per_tok" in text:
            fields["n_activated_experts"] = text["num_experts_per_tok"]
        return cls(**fields)


def _v41_moe_checkpoint_prefix(loader: ModelWeightLoader, layer_id: int, first_expert: int) -> str:
    prefix = find_checkpoint_prefix(
        loader,
        (f"layers.{layer_id}.ffn.", f"layers.{layer_id}.mlp."),
        (f"experts.{first_expert}.w1.weight", f"experts.{first_expert}.gate_proj.weight"),
    )
    if prefix is None:
        raise KeyError(f"V4.1 layer {layer_id} local expert weights not found")
    return prefix


def _validate_v41_w8a8_experts(loader: ModelWeightLoader, config: DeepseekV41MoEConfig, prefix: str) -> None:
    """Reject unsupported packed/asymmetric exports before the shared base loader."""
    count = config.n_routed_experts // config.ep_size
    start = count * config.ep_rank
    sources = [prefix + f"experts.{i}." for i in range(start, start + count)] + [prefix + "shared_experts."]
    for source in sources:
        gate, up, down = resolve_mlp_projection_names(loader, source)
        for name, shape in (
            (gate, (config.moe_intermediate_size, config.hidden_size)),
            (up, (config.moe_intermediate_size, config.hidden_size)),
            (down, (config.hidden_size, config.moe_intermediate_size)),
        ):
            key = source + name
            weight = loader.get_tensor(key + ".weight")
            require_shape(weight, shape, key + ".weight")
            if weight.dtype != torch.int8:
                raise ValueError(f"{key} requires W8A8 INT8 weights")
            scale = loader.get_tensor(key + ".weight_scale")
            require_shape(scale, (shape[0], 1), key + ".weight_scale")
            if scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"{key} requires floating point per-output scales")
            if loader.has(key + ".weight_offset"):
                offset = loader.get_tensor(key + ".weight_offset")
                require_shape(offset, (shape[0], 1), key + ".weight_offset")
                if bool(torch.count_nonzero(offset)):
                    raise ValueError(f"{key} requires symmetric zero-offset quantization")


@torch.no_grad()
def _load_v41_gate(loader: ModelWeightLoader, module: nn.Module, prefix: str, layer_id: int) -> None:
    gate = find_checkpoint_prefix(loader, (prefix + "gate.", f"layers.{layer_id}.gate."), ("weight",))
    if gate is None:
        raise KeyError(f"V4.1 layer {layer_id} routing weight not found")
    bias = require_checkpoint_key(loader, (gate + "bias", gate + "e_score_correction_bias"), "V4.1 routing bias")
    for source, target in ((gate + "weight", module.gate.weight), (bias, module.e_score_correction_bias)):
        tensor = loader.get_tensor(source)
        require_shape(tensor, tuple(target.shape), source)
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"{source} must be floating point")
        target.copy_(tensor)


@torch.no_grad()
def load_v41_w8a8_weights(
    loader: ModelWeightLoader, module: nn.Module, config: DeepseekV41MoEConfig, layer_id: int
) -> None:
    """Stage the complete V4.1 gate/routed/shared set before device packing."""
    prefix = _v41_moe_checkpoint_prefix(loader, layer_id, module.start_expert_id)
    _validate_v41_w8a8_experts(loader, config, prefix)
    _load_v41_gate(loader, module, prefix, layer_id)
    # The base dynamic linear allocates offsets with empty(). V4.1 permits
    # omitted symmetric offsets, so initialize both shared projections before
    # the base loader copies any explicitly supplied zeros.
    module.shared_experts.gate_up_proj.weight_offset.zero_()
    module.shared_experts.down_proj.weight_offset.zero_()
    load_deepseek_moe_experts(
        loader,
        module,
        prefix,
        f"model.layers.{layer_id}.",
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
    )


def v41_route(
    hidden: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, *, topk: int, normalize: bool, scaling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias selects experts; unbiased scores determine their contribution."""
    scores = F.softplus(F.linear(hidden.float(), gate.float())).sqrt()
    indices = (scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(1, indices)
    if normalize and topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scaling, indices


def _expert(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    limit: float,
    route_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    gate, up = F.linear(hidden, w1).float(), F.linear(hidden, w3).float()
    if 0 < limit < 1_000_000:
        gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
    activation = F.silu(gate) * up
    if route_weight is not None:
        activation = activation * route_weight[:, None]
    # Routing before the down projection preserves BF16 rounding semantics.
    return F.linear(activation.to(hidden.dtype), w2)


class DeepseekV41MoE(nn.Module):
    """Eager BF16/FP32 text experts, with explicit EP and orthogonal MoE TP."""

    def __init__(self, config: DeepseekV41MoEConfig, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if config.dp_size != 1:
            raise ValueError("V4.1 eager floating-point MoE currently requires DP=1")
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("V4.1 eager MoE requires BF16 or FP32 weights")
        self.config = config
        self.num_experts_per_rank = config.n_routed_experts // config.ep_size
        self.start_expert_id = config.ep_rank * self.num_experts_per_rank
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.gate.requires_grad_(False)
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32, device=device), requires_grad=False
        )
        inter = config.moe_intermediate_size // config.moe_tp_size
        for name, shape in (
            ("w1", (inter, config.hidden_size)),
            ("w2", (config.hidden_size, inter)),
            ("w3", (inter, config.hidden_size)),
        ):
            self.register_parameter(
                "experts_" + name,
                nn.Parameter(
                    torch.empty((self.num_experts_per_rank, *shape), dtype=dtype, device=device), requires_grad=False
                ),
            )
            self.register_parameter(
                "shared_" + name, nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
            )

    @torch.no_grad()
    def load_weights(self, loader: ModelWeightLoader, layer_id: int) -> None:
        config = self.config
        prefix = _v41_moe_checkpoint_prefix(loader, layer_id, self.start_expert_id)
        _load_v41_gate(loader, self, prefix, layer_id)
        for slot in range(self.num_experts_per_rank + 1):
            shared = slot == self.num_experts_per_rank
            source = prefix + ("shared_experts." if shared else f"experts.{self.start_expert_id + slot}.")
            gate, up, down = resolve_mlp_projection_names(loader, source)
            for name, target, dim in ((gate, "w1", 0), (down, "w2", 1), (up, "w3", 0)):
                tensor = loader.get_tensor(source + name + ".weight")
                expected = (
                    (config.moe_intermediate_size, config.hidden_size)
                    if dim == 0
                    else (config.hidden_size, config.moe_intermediate_size)
                )
                require_shape(tensor, expected, source + name)
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(
                        f"{source}{name} requires floating point weights; quantized checkpoints need their selected backend"
                    )
                shard = loader.shard(tensor, dim, world=config.moe_tp_size, rank=config.moe_tp_rank)
                parameter = getattr(self, "shared_" + target) if shared else getattr(self, "experts_" + target)[slot]
                parameter.copy_(shard)

    def _local_partials(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rank-local routed/shared contributions before group reductions."""
        config = self.config
        flat = hidden.reshape(-1, config.hidden_size)
        weights, indices = v41_route(
            flat,
            self.gate.weight,
            self.e_score_correction_bias,
            topk=config.n_activated_experts,
            normalize=config.norm_topk_prob,
            scaling=config.routed_scaling_factor,
        )
        routes = indices.flatten()
        local = torch.where(
            (routes >= self.start_expert_id) & (routes < self.start_expert_id + self.num_experts_per_rank)
        )[0]
        experts, order = routes[local].sort(stable=True)
        ordered = local[order]
        active, counts = torch.unique_consecutive(experts, return_counts=True)
        segments = torch.stack((active - self.start_expert_id, counts), dim=1).cpu().tolist()
        routed = torch.zeros_like(flat, dtype=torch.float32)
        offset = 0
        for expert, count in segments:
            selected = ordered[offset : offset + count]
            rows = selected // config.n_activated_experts
            output = _expert(
                flat[rows],
                self.experts_w1[expert],
                self.experts_w2[expert],
                self.experts_w3[expert],
                config.swiglu_limit,
                weights.flatten()[selected],
            )
            routed.index_add_(0, rows, output.float())
            offset += count
        shared = _expert(flat, self.shared_w1, self.shared_w2, self.shared_w3, config.swiglu_limit).float()
        return routed.reshape(hidden.shape), shared.reshape(hidden.shape)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        routed, shared = self._local_partials(hidden)
        config = self.config
        if config.ep_size > 1 or config.moe_tp_size > 1:
            from xllm.python import distributed

            if config.moe_tp_size > 1:
                distributed.moe_tp_all_reduce(routed)
                distributed.moe_tp_all_reduce(shared)
            if config.ep_size > 1:
                distributed.moe_ep_all_reduce(routed)
        return (routed + shared).to(hidden.dtype)
