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

"""Real tensor loading and expert computation, without device-kernel mocks."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v4_checkpoint import load_deepseek_moe_experts
from xllm.python.models.deepseek_v41 import create_v41_moe
from xllm.python.models.deepseek_v41_moe import (
    DeepseekV41MoE,
    DeepseekV41MoEConfig,
    _validate_v41_w8a8_experts,
    load_v41_w8a8_weights,
    v41_route,
)


class _State(dict[str, torch.Tensor]):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__(tensors)
        self.reads: list[str] = []

    def has(self, name: str) -> bool:
        return name in self

    def get_tensor(self, name: str) -> torch.Tensor:
        self.reads.append(name)
        return self[name]


def _reference(dtype: torch.dtype) -> tuple[DeepseekV41MoEConfig, _State, torch.Tensor, torch.Tensor]:
    fixture = json.loads((Path(__file__).parent / "data/deepseek_v41_moe.json").read_text())
    cfg = DeepseekV41MoEConfig.from_dict(fixture["config"])
    case = fixture["cases"][str(dtype)]
    weights = {k: torch.tensor(v, dtype=dtype) for k, v in case["weights"].items()}
    prefix = "layers.0.ffn."
    source = {
        prefix + "gate.weight": weights["gate.weight"],
        prefix + "gate.bias": torch.tensor(case["weights"]["e_score_correction_bias"]),
    }
    for name in ("w1", "w2", "w3"):
        for expert in range(4):
            source[prefix + f"experts.{expert}.{name}.weight"] = weights["experts_" + name][expert]
        source[prefix + f"shared_experts.{name}.weight"] = weights[f"shared_{name}.weight"]
    return cfg, _State(source), torch.tensor(case["hidden"], dtype=dtype), torch.tensor(case["output"], dtype=dtype)


def _model(cfg: DeepseekV41MoEConfig, state: _State, dtype: torch.dtype = torch.float32) -> DeepseekV41MoE:
    model = DeepseekV41MoE(cfg, dtype=dtype, device=torch.device("cpu"))
    # Attention TP is deliberately different: expert loading must use MoE TP.
    model.load_weights(ModelWeightLoader(model, [state], tp_size=8, tp_rank=7), 0)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_matches_pinned_internal(dtype: torch.dtype) -> None:
    cfg, state, hidden, expected = _reference(dtype)
    model = _model(cfg, state, dtype)
    torch.testing.assert_close(
        model(hidden),
        expected,
        rtol=0 if dtype == torch.bfloat16 else 1e-6,
        atol=0 if dtype == torch.bfloat16 else 1e-6,
    )
    assert model(hidden[:0]).shape == (0, 4)
    assert not any(p.requires_grad for p in model.parameters())


def test_ep_sums_only_routed_outputs_and_reads_only_local_experts() -> None:
    cfg, state, hidden, expected = _reference(torch.float32)
    partials = []
    for rank in (0, 1):
        state.reads.clear()
        model = _model(replace(cfg, ep_size=2, ep_rank=rank), state)
        assert all(not any(f"experts.{i}." in key for i in range(4) if i // 2 != rank) for key in state.reads)
        partials.append(model._local_partials(hidden))
    torch.testing.assert_close(partials[0][1], partials[1][1])
    torch.testing.assert_close(partials[0][0] + partials[1][0] + partials[0][1], expected)


def test_orthogonal_moe_tp_reassembles_both_partials() -> None:
    cfg, state, hidden, expected = _reference(torch.float32)
    partials = [_model(replace(cfg, moe_tp_size=2, moe_tp_rank=rank), state)._local_partials(hidden) for rank in (0, 1)]
    torch.testing.assert_close(sum(routed + shared for routed, shared in partials), expected)


@pytest.mark.parametrize("topk", [1, 2])
def test_bias_affects_selection_only_and_top1_is_not_renormalized(topk: int) -> None:
    gate = torch.tensor([[1.0], [2.0], [-1.0]])
    weights, indices = v41_route(
        torch.ones(1, 1), gate, torch.tensor([0.0, 0.0, 100.0]), topk=topk, normalize=True, scaling=1.5
    )
    assert indices[0, 0] == 2
    raw = [math.sqrt(math.log1p(math.exp(float(gate[i, 0])))) for i in indices[0]]
    expected = [value * 1.5 / (sum(raw) if topk > 1 else 1) for value in raw]
    torch.testing.assert_close(weights, torch.tensor([expected]))


@pytest.mark.parametrize(
    "update",
    [
        {"ep_size": 3},
        {"moe_tp_size": 3},
        {"n_hash_layers": 1},
        {"scoring_func": "sigmoid"},
        {"n_shared_experts": 2},
        {"norm_topk_prob": "true"},
    ],
)
def test_invalid_configuration_rejected(update: dict) -> None:
    cfg, _, _, _ = _reference(torch.float32)
    with pytest.raises(ValueError):
        replace(cfg, **update)


def test_native_and_nested_configuration_and_weight_failures() -> None:
    cfg, state, _, _ = _reference(torch.float32)
    nested = DeepseekV41MoEConfig.from_dict(
        {
            "text_config": {
                "hidden_size": 4,
                "moe_intermediate_size": 4,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
            },
            "tp_size": 8,
            "moe_tp_size": 2,
            "moe_tp_rank": 1,
        }
    )
    assert (nested.tp_size, nested.moe_tp_size, nested.moe_tp_rank) == (8, 2, 1)
    bad = _State(dict(state))
    del bad["layers.0.ffn.shared_experts.w1.weight"]
    with pytest.raises(KeyError):
        _model(cfg, bad)
    bad = _State(dict(state))
    bad["layers.0.ffn.experts.0.w1.weight"] = bad["layers.0.ffn.experts.0.w1.weight"].to(torch.int8)
    with pytest.raises(ValueError, match="quantized"):
        _model(cfg, bad)


def _quantized(
    w4: bool, moe_tp: int, rank: int, *, offsets: bool = True
) -> tuple[nn.Module, ModelWeightLoader, _State]:
    owner = nn.Module()
    owner.model = nn.Module()
    layer = nn.Module()
    mlp = nn.Module()
    layer.mlp = mlp
    owner.model.layers = nn.ModuleList([layer])
    mlp.moe_tp_size = moe_tp
    mlp.moe_tp_rank = rank
    mlp.start_expert_id = 2
    mlp.num_experts_per_rank = 2
    mlp.gate = nn.Linear(4, 4, bias=False)
    mlp.register_buffer("e_score_correction_bias", torch.empty(4))
    for name in ("experts_w13", "experts_w2"):
        mlp.register_parameter(name, nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False))
    for name in (
        "experts_w13_scale",
        "experts_w13_scale_second",
        "experts_w2_scale",
        "experts_w2_scale_second",
        "experts_w13_offset",
        "experts_w2_offset",
        "experts_w13_scale_bias",
        "experts_w2_scale_bias",
    ):
        mlp.register_buffer(name, torch.empty(0, dtype=torch.bfloat16 if name == "experts_w2_scale" else torch.float32))
    mlp.shared_experts = nn.Module()
    for name, shape in (("gate_up_proj", (8 // moe_tp, 4)), ("down_proj", (4, 4 // moe_tp))):
        linear = nn.Module()
        linear.register_buffer("weight", torch.empty(shape, dtype=torch.int8))
        linear.register_buffer("weight_scale", torch.empty(shape[0], 1))
        linear.register_buffer("weight_offset", torch.zeros(shape[0], 1))
        setattr(mlp.shared_experts, name, linear)
    source = {"layers.0.ffn.gate.weight": torch.eye(4), "layers.0.ffn.gate.bias": torch.arange(4, dtype=torch.float32)}
    for expert in (2, 3, "shared"):
        prefix = "layers.0.ffn." + ("shared_experts." if expert == "shared" else f"experts.{expert}.")
        packed = 2 if w4 and expert != "shared" else 1
        for i, (name, shape) in enumerate(
            (("w1", (4 // packed, 4)), ("w3", (4 // packed, 4)), ("w2", (4 // packed, 4)))
        ):
            source[prefix + name + ".weight"] = (torch.arange(shape[0] * shape[1]).reshape(shape) + i * 20).to(
                torch.int8
            )
            source[prefix + name + ".weight_scale"] = torch.arange(1, 5, dtype=torch.float32).view(4, 1) / 8
            if offsets:
                source[prefix + name + ".weight_offset"] = torch.zeros(4, 1)
            if packed == 2:
                source[prefix + name + ".scale_bias"] = torch.ones((4, 4 if name == "w2" else 1))
    state = _State(source)
    return mlp, ModelWeightLoader(owner, [state], tp_size=8, tp_rank=7), state


@pytest.mark.parametrize("offsets", [False, True])
def test_complete_w41_w8_loading_initializes_shared_offsets(offsets: bool) -> None:
    mlp, loader, _ = _quantized(False, 2, 1, offsets=offsets)
    cfg = DeepseekV41MoEConfig(
        hidden_size=4,
        moe_intermediate_size=4,
        n_routed_experts=4,
        n_activated_experts=2,
        ep_size=2,
        ep_rank=1,
        moe_tp_size=2,
        moe_tp_rank=1,
    )
    mlp.shared_experts.gate_up_proj.weight_offset.fill_(float("nan"))
    mlp.shared_experts.down_proj.weight_offset.fill_(float("nan"))
    load_v41_w8a8_weights(loader, mlp, cfg, 0)
    torch.testing.assert_close(mlp.gate.weight, torch.eye(4))
    torch.testing.assert_close(mlp.e_score_correction_bias, torch.arange(4, dtype=torch.float32))
    assert torch.count_nonzero(mlp.shared_experts.gate_up_proj.weight_offset) == 0
    assert torch.count_nonzero(mlp.shared_experts.down_proj.weight_offset) == 0


@pytest.mark.parametrize("w4", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_shared_base_loader_preserves_w8_w4_and_moe_tp(w4: bool, rank: int) -> None:
    mlp, loader, state = _quantized(w4, 2, rank)
    load_deepseek_moe_experts(loader, mlp, "layers.0.ffn.", "model.layers.0.", hidden_size=4, intermediate_size=4)
    divisor = 2 if w4 else 1
    rows = 2 // divisor
    for slot, expert in enumerate((2, 3)):
        prefix = f"layers.0.ffn.experts.{expert}."
        expected = torch.cat(
            [state[prefix + name + ".weight"][rank * rows : (rank + 1) * rows] for name in ("w1", "w3")]
        )
        torch.testing.assert_close(mlp.experts_w13[slot], expected)
        torch.testing.assert_close(mlp.experts_w2[slot], state[prefix + "w2.weight"][:, rank * 2 : (rank + 1) * 2])
    assert mlp.experts_w2_scale.dtype == (torch.float32 if w4 else torch.bfloat16)
    assert mlp.w4a8_dynamic == w4
    torch.testing.assert_close(
        mlp.shared_experts.gate_up_proj.weight,
        torch.cat(
            [state["layers.0.ffn.shared_experts." + n + ".weight"][rank * 2 : (rank + 1) * 2] for n in ("w1", "w3")]
        ),
    )


def test_w41_w8_requires_complete_symmetric_checkpoint() -> None:
    _, loader, state = _quantized(False, 1, 0)
    cfg = DeepseekV41MoEConfig(
        hidden_size=4, moe_intermediate_size=4, n_routed_experts=4, n_activated_experts=2, ep_size=2, ep_rank=1
    )
    _validate_v41_w8a8_experts(loader, cfg, "layers.0.ffn.")
    state["layers.0.ffn.experts.2.w1.weight_offset"].fill_(1)
    with pytest.raises(ValueError, match="symmetric"):
        _validate_v41_w8a8_experts(loader, cfg, "layers.0.ffn.")


def test_factory_does_not_dequantize_or_fallback_for_unsupported_backend() -> None:
    config = {"hidden_size": 4, "moe_intermediate_size": 4, "n_routed_experts": 4, "n_activated_experts": 2}
    assert isinstance(create_v41_moe(config, 0, dtype=torch.float32, device=torch.device("cpu")), DeepseekV41MoE)
    config["quantization_config"] = {"quant_method": "ascend"}
    with pytest.raises(ValueError, match="NPU runtime"):
        create_v41_moe(config, 0, dtype=torch.bfloat16, device=torch.device("cpu"))
    config["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="not implemented"):
        create_v41_moe(config, 0, dtype=torch.bfloat16, device=torch.device("cpu"))
