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

"""Shared DeepSeek V4/V4.1 checkpoint loading; no accelerator imports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from xllm.python.model_loader import ModelWeightLoader


class DeepseekMoEWeights(Protocol):
    """Mutable local expert tensors required by the shared checkpoint loader.

    Tensor storage is resized during loading; packing and execution belong to
    the concrete MoE implementation. Shared projections use loader.copy_in.
    """

    moe_tp_size: int
    moe_tp_rank: int
    start_expert_id: int
    num_experts_per_rank: int
    w4a8_dynamic: bool
    shared_expert_gate_is_loaded: bool
    experts_w13: torch.Tensor
    experts_w2: torch.Tensor
    experts_w13_scale: torch.Tensor
    experts_w13_scale_second: torch.Tensor
    experts_w2_scale: torch.Tensor
    experts_w2_scale_second: torch.Tensor
    experts_w13_offset: torch.Tensor
    experts_w2_offset: torch.Tensor
    experts_w13_scale_bias: torch.Tensor
    experts_w2_scale_bias: torch.Tensor


def find_checkpoint_key(loader: ModelWeightLoader, candidates: Sequence[str]) -> str | None:
    """Return the first checkpoint key present in compatibility order."""
    return next((name for name in candidates if loader.has(name)), None)


def require_checkpoint_key(
    loader: ModelWeightLoader,
    candidates: Sequence[str],
    description: str,
) -> str:
    key = find_checkpoint_key(loader, candidates)
    if key is None:
        expected = ", ".join(candidates)
        raise KeyError(f"{description} not found; expected one of: {expected}")
    return key


def require_tensor(
    loader: ModelWeightLoader,
    key: str,
    description: str,
) -> torch.Tensor:
    """Return the tensor at a mandatory single checkpoint key."""
    return loader.get_tensor(require_checkpoint_key(loader, (key,), description))


def require_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    tensor_name: str,
) -> None:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"DeepSeek-V4 tensor {tensor_name} has shape {actual_shape}, expected {expected_shape}")


def find_checkpoint_prefix(
    loader: ModelWeightLoader,
    candidates: Sequence[str],
    probes: Sequence[str],
) -> str | None:
    """Return the first module prefix containing a supported probe tensor."""
    return next(
        (prefix for prefix in candidates if any(loader.has(prefix + probe) for probe in probes)),
        None,
    )


def _find_mlp_projection_names(loader: ModelWeightLoader, prefix: str) -> tuple[str, str, str] | None:
    return next(
        (names for names in _DSV4_MLP_NAME_SCHEMES if all(loader.has(prefix + name + ".weight") for name in names)),
        None,
    )


def resolve_mlp_projection_names(loader: ModelWeightLoader, checkpoint_prefix: str) -> tuple[str, str, str]:
    """Resolve native ``gate/up/down`` and legacy ``w1/w3/w2`` names."""
    names = _find_mlp_projection_names(loader, checkpoint_prefix)
    if names is not None:
        return names
    expected = " or ".join("/".join(names) for names in _DSV4_MLP_NAME_SCHEMES)
    raise KeyError(f"DeepSeek-V4 MLP projections not found under {checkpoint_prefix}; expected {expected}")


def resolve_dense_mlp_checkpoint(loader: ModelWeightLoader, checkpoint_prefix: str) -> tuple[str, tuple[str, str, str]]:
    """Resolve a complete dense MLP once: ffn before mlp, native before legacy.

    Incomplete triplets do not select a prefix. Quantization tensors remain
    mandatory at load time; they must not change the projection alias choice.
    """
    for suffix in ("ffn.", "mlp."):
        prefix = checkpoint_prefix + suffix
        names = _find_mlp_projection_names(loader, prefix)
        if names is not None:
            return prefix, names
    raise KeyError(f"DeepSeek-V4 dense MLP weights not found under {checkpoint_prefix}")


_DSV4_MLP_NAME_SCHEMES = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))


def load_deepseek_moe_experts(
    loader: ModelWeightLoader,
    mlp: DeepseekMoEWeights,
    source_prefix: str,
    pm: str,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    """Load local EP experts, MoE-TP shards and shared experts before packing."""
    # Per-expert w1+w3 -> fused w13, w2 -> w2 (int8 + scale).
    # EP: only load local experts [start_expert_id, start_expert_id + num_experts_per_rank).
    tp = mlp.moe_tp_size
    tp_rank = mlp.moe_tp_rank
    start = mlp.start_expert_id
    nepr = mlp.num_experts_per_rank
    w13 = mlp.experts_w13
    w2 = mlp.experts_w2
    w13_scale = mlp.experts_w13_scale
    w13_scale_second = mlp.experts_w13_scale_second
    w2_scale = mlp.experts_w2_scale
    w2_scale_second = mlp.experts_w2_scale_second
    w13_offset = mlp.experts_w13_offset
    w2_offset = mlp.experts_w2_offset
    w13_scale_bias = mlp.experts_w13_scale_bias
    w2_scale_bias = mlp.experts_w2_scale_bias

    def _shard_fused(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Shard w1/w3 independently, matching LOAD_MOE_FUSED_WEIGHT."""
        if tp > 1:
            first = loader.shard(first, dim=0, world=tp, rank=tp_rank)
            second = loader.shard(second, dim=0, world=tp, rank=tp_rank)
        return torch.cat([first, second], dim=0)

    probe = source_prefix + f"experts.{start}."
    gate_name, up_name, down_name = resolve_mlp_projection_names(loader, probe)
    probe_gate = loader.get_tensor(probe + gate_name + ".weight")
    if probe_gate.dtype != torch.int8 or probe_gate.dim() != 2 or probe_gate.shape[1] != hidden_size:
        raise ValueError(
            "DeepSeek-V4 expert gate weight must be a two-dimensional INT8 tensor "
            f"with input width {hidden_size}, got dtype={probe_gate.dtype}, "
            f"shape={tuple(probe_gate.shape)}"
        )
    if probe_gate.shape[0] == intermediate_size:
        is_w4a8 = False
    elif intermediate_size % 2 == 0 and probe_gate.shape[0] * 2 == intermediate_size:
        is_w4a8 = True
    else:
        raise ValueError(
            "DeepSeek-V4 expert gate weight rows must match either W8A8 or packed W4A8 layout: "
            f"rows={probe_gate.shape[0]}, intermediate_size={intermediate_size}"
        )
    if is_w4a8 and hidden_size % 2 != 0:
        raise ValueError(f"DeepSeek-V4 packed W4A8 requires an even hidden size, got {hidden_size}")
    mlp.w4a8_dynamic = is_w4a8
    device = w13.device
    if is_w4a8:
        w13.data = torch.empty(nepr, intermediate_size // tp, hidden_size, dtype=torch.int8, device=device)
        w2.data = torch.empty(nepr, hidden_size // 2, intermediate_size // tp, dtype=torch.int8, device=device)
        w13_scale.data = torch.empty(nepr, 2 * (intermediate_size // tp), 1, dtype=torch.float32, device=device)
        w2_scale.data = torch.empty(nepr, hidden_size, 1, dtype=torch.float32, device=device)
        w13_offset.data = torch.empty_like(w13_scale)
        w2_offset.data = torch.empty_like(w2_scale)
        w13_scale_bias.data = torch.empty(nepr, 2 * (intermediate_size // tp), 1, dtype=torch.float32, device=device)
        sb2 = require_tensor(
            loader,
            probe + down_name + ".scale_bias",
            f"DeepSeek-V4 W4A8 expert {start} down scale bias",
        )
        if tp > 1:
            sb2 = loader.shard(sb2, dim=1, world=tp, rank=tp_rank)
        w2_scale_bias.data = torch.empty(nepr, *sb2.shape, dtype=torch.float32, device=device)
    else:
        inter_local = intermediate_size // tp
        w13.data = torch.empty(nepr, 2 * inter_local, hidden_size, dtype=torch.int8, device=device)
        w2.data = torch.empty(nepr, hidden_size, inter_local, dtype=torch.int8, device=device)
        w13_scale.data = torch.empty(nepr, 2 * inter_local, 1, dtype=torch.float32, device=device)
        w2_scale.data = w2_scale.new_empty((nepr, hidden_size, 1))
        w13_offset.data = torch.zeros_like(w13_scale)
        w2_offset.data = torch.zeros_like(w2_scale, dtype=torch.float32)
    for local_idx in range(nepr):
        global_id = start + local_idx
        e = source_prefix + f"experts.{global_id}."
        gate_weight = probe_gate if local_idx == 0 else loader.get_tensor(e + gate_name + ".weight")
        up_weight = loader.get_tensor(e + up_name + ".weight")
        w2_j = loader.get_tensor(e + down_name + ".weight")
        packed_divisor = 2 if is_w4a8 else 1
        require_shape(
            gate_weight,
            (intermediate_size // packed_divisor, hidden_size),
            e + gate_name + ".weight",
        )
        require_shape(
            up_weight,
            (intermediate_size // packed_divisor, hidden_size),
            e + up_name + ".weight",
        )
        require_shape(
            w2_j,
            (hidden_size // packed_divisor, intermediate_size),
            e + down_name + ".weight",
        )
        if gate_weight.dtype != torch.int8 or up_weight.dtype != torch.int8 or w2_j.dtype != torch.int8:
            raise ValueError(f"DeepSeek-V4 expert weights under {e} must use INT8 storage")
        w13_j = _shard_fused(gate_weight, up_weight)
        if tp > 1:
            w2_j = loader.shard(w2_j, dim=1, world=tp, rank=tp_rank)
        w13[local_idx].copy_(w13_j.to(w13.dtype))
        w2[local_idx].copy_(w2_j.to(w2.dtype))
        # Legacy exports can share the gate scale with the up projection.
        gate_scale = require_tensor(
            loader,
            e + gate_name + ".weight_scale",
            f"DeepSeek-V4 expert {global_id} gate scale",
        )
        up_scale_key = (
            e + up_name + ".weight_scale"
            if loader.has(e + up_name + ".weight_scale")
            else e + gate_name + ".weight_scale"
        )
        s13 = _shard_fused(gate_scale, loader.get_tensor(up_scale_key))
        require_shape(s13, tuple(w13_scale[local_idx].shape), f"{e}gate/up weight_scale")
        w13_scale[local_idx].copy_(s13)
        if is_w4a8:
            gate_second_key = e + gate_name + ".weight_scale_second"
            if loader.has(gate_second_key):
                s1_second = loader.get_tensor(gate_second_key)
                up_second_key = e + up_name + ".weight_scale_second"
                s3_second = loader.get_tensor(up_second_key) if loader.has(up_second_key) else s1_second
                s13_second = _shard_fused(s1_second, s3_second)
                if w13_scale_second.numel() == 0:
                    w13_scale_second.data = torch.empty(nepr, *s13_second.shape, dtype=torch.float32, device=device)
                w13_scale_second[local_idx].copy_(s13_second)
            sb13 = _shard_fused(
                require_tensor(
                    loader,
                    e + gate_name + ".scale_bias",
                    f"DeepSeek-V4 W4A8 expert {global_id} gate scale bias",
                ),
                require_tensor(
                    loader,
                    e + up_name + ".scale_bias",
                    f"DeepSeek-V4 W4A8 expert {global_id} up scale bias",
                ),
            )
            require_shape(sb13, tuple(w13_scale_bias[local_idx].shape), f"{e}gate/up scale_bias")
            w13_scale_bias[local_idx].copy_(sb13)
        down_scale_key = e + down_name + ".weight_scale"
        down_scale = require_tensor(
            loader,
            down_scale_key,
            f"DeepSeek-V4 expert {global_id} down scale",
        )
        require_shape(down_scale, tuple(w2_scale[local_idx].shape), down_scale_key)
        w2_scale[local_idx].copy_(down_scale)
        if is_w4a8:
            down_second_key = e + down_name + ".weight_scale_second"
            if loader.has(down_second_key):
                s2_second = loader.get_tensor(down_second_key)
                if tp > 1:
                    s2_second = loader.shard(s2_second, dim=1, world=tp, rank=tp_rank)
                if w2_scale_second.numel() == 0:
                    w2_scale_second.data = torch.empty(nepr, *s2_second.shape, dtype=torch.float32, device=device)
                w2_scale_second[local_idx].copy_(s2_second)
            down_bias_key = e + down_name + ".scale_bias"
            s2_bias = require_tensor(
                loader,
                down_bias_key,
                f"DeepSeek-V4 W4A8 expert {global_id} down scale bias",
            )
            if tp > 1:
                s2_bias = loader.shard(s2_bias, dim=1, world=tp, rank=tp_rank)
            require_shape(s2_bias, tuple(w2_scale_bias[local_idx].shape), down_bias_key)
            w2_scale_bias[local_idx].copy_(s2_bias)
    # Shared experts: checkpoint has w1/w2/w3 (W8A8 dynamic), fuse w1+w3 -> gate_up_proj.
    se = source_prefix + "shared_experts."
    shared_gate_keys = (
        source_prefix + "shared_expert_gate.weight",
        source_prefix + "shared_experts_gate.weight",
    )
    shared_gate_key = find_checkpoint_key(loader, shared_gate_keys)
    if shared_gate_key is not None:
        loader.copy_in(
            pm + "mlp.shared_expert_gate.weight",
            loader.get_tensor(shared_gate_key),
        )
        mlp.shared_expert_gate_is_loaded = True
    if loader.has(se + "gate_proj.weight") or loader.has(se + "w1.weight"):
        shared_gate_name, shared_up_name, shared_down_name = resolve_mlp_projection_names(loader, se)
        se_w1 = loader.get_tensor(se + shared_gate_name + ".weight")
        se_w3 = loader.get_tensor(se + shared_up_name + ".weight")
        se_w13 = _shard_fused(se_w1, se_w3)
        loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight", se_w13)
        if loader.has(se + shared_gate_name + ".weight_scale"):
            s1 = loader.get_tensor(se + shared_gate_name + ".weight_scale")
            s3 = loader.get_tensor(se + shared_up_name + ".weight_scale")
            se_s13 = _shard_fused(s1, s3)
            loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight_scale", se_s13[: se_w13.size(0)])
        if loader.has(se + shared_gate_name + ".weight_offset"):
            o1 = loader.get_tensor(se + shared_gate_name + ".weight_offset")
            o3 = loader.get_tensor(se + shared_up_name + ".weight_offset")
            loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight_offset", _shard_fused(o1, o3)[: se_w13.size(0)])
        se_w2 = loader.get_tensor(se + shared_down_name + ".weight")
        if tp > 1:
            se_w2 = loader.shard(se_w2, dim=1, world=tp, rank=tp_rank)
        loader.copy_in(pm + "mlp.shared_experts.down_proj.weight", se_w2)
        if loader.has(se + shared_down_name + ".weight_scale"):
            loader.copy_in(
                pm + "mlp.shared_experts.down_proj.weight_scale",
                loader.get_tensor(se + shared_down_name + ".weight_scale"),
            )
        if loader.has(se + shared_down_name + ".weight_offset"):
            loader.copy_in(
                pm + "mlp.shared_experts.down_proj.weight_offset",
                loader.get_tensor(se + shared_down_name + ".weight_offset"),
            )
