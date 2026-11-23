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

"""Real NPU runtime imports and CPU post-load packing; no mocked kernels."""

import os

import pytest
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v41_moe import DeepseekV41MoEConfig
from xllm.python.models.deepseek_v41_npu_moe import DeepseekV41SharedMLP, DeepseekV41W8A8MoE


class _State(dict[str, torch.Tensor]):
    def has(self, name: str) -> bool:
        return name in self

    def get_tensor(self, name: str) -> torch.Tensor:
        return self[name]


@pytest.mark.skipif(not os.environ.get("XLLM_V41_MODEL_TEST_DEVICE"), reason="requires an explicitly available NPU")
@pytest.mark.parametrize("rows", [1, 4, 129])
def test_v41_shared_expert_preserves_clamp_and_quantization(rows: int) -> None:
    from tests.core.kernels.npu.glm5_next_moe_cases import _activation_reference, _quantize_reference

    device = torch.device(os.environ["XLLM_V41_MODEL_TEST_DEVICE"])
    torch.npu.set_device(device)
    config = DeepseekV41MoEConfig(hidden_size=128, moe_intermediate_size=128)
    mlp = DeepseekV41SharedMLP(config, 128, torch.bfloat16, device, skip_tp_reduce=True, swiglu_limit=10.0)
    assert mlp._use_fused_activation
    identity = torch.eye(128, dtype=torch.int8)
    packed = torch.cat((identity, identity))
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(packed)
        mlp.gate_up_proj.weight_scale.fill_(1)
        mlp.gate_up_proj.weight_offset.zero_()
        mlp.down_proj.weight.copy_(identity)
        mlp.down_proj.weight_scale.fill_(1)
        mlp.down_proj.weight_offset.zero_()
        mlp.process_weights_after_loading()
        hidden = torch.tensor([-20, -5, -1, 1, 5, 20, -12, 12], dtype=torch.bfloat16).repeat(rows, 16)
        input_codes, input_scale = _quantize_reference(hidden.float())
        accumulated = input_codes.int() @ packed.t().int()
        dequantized = accumulated.float() * input_scale[:, None]
        activation_codes, activation_scale = _quantize_reference(_activation_reference(dequantized))
        expected = (activation_codes.float() * activation_scale[:, None]).to(torch.bfloat16)
        actual = mlp(hidden.to(device)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("offsets", [False, True])
def test_v41_w8a8_complete_cpu_load_and_post_pack(offsets: bool) -> None:
    config = DeepseekV41MoEConfig(
        hidden_size=4,
        moe_intermediate_size=4,
        n_routed_experts=4,
        n_activated_experts=2,
        tp_size=8,
        tp_rank=7,
        ep_size=2,
        ep_rank=1,
        moe_tp_size=2,
        moe_tp_rank=1,
    )
    mlp = DeepseekV41W8A8MoE(config, 0, dtype=torch.bfloat16, device=torch.device("cpu"))
    owner = nn.Module()
    owner.model = nn.Module()
    layer = nn.Module()
    layer.mlp = mlp
    owner.model.layers = nn.ModuleList([layer])
    state = _State(
        {"layers.0.ffn.gate.weight": torch.eye(4), "layers.0.ffn.gate.bias": torch.arange(4, dtype=torch.float32)}
    )
    for expert in ("experts.2", "experts.3", "shared_experts"):
        for offset, name in enumerate(("w1", "w3", "w2")):
            key = f"layers.0.ffn.{expert}.{name}"
            state[key + ".weight"] = (torch.arange(16).reshape(4, 4) + offset * 20).to(torch.int8)
            state[key + ".weight_scale"] = torch.full((4, 1), 0.25)
            if offsets:
                state[key + ".weight_offset"] = torch.zeros(4, 1)
    mlp.load_weights(ModelWeightLoader(owner, [state], tp_size=8, tp_rank=7), 0)
    for slot, expert in enumerate((2, 3)):
        prefix = f"layers.0.ffn.experts.{expert}."
        expected = torch.cat((state[prefix + "w1.weight"][2:], state[prefix + "w3.weight"][2:])).t().contiguous()
        torch.testing.assert_close(mlp.experts_w13[slot], expected)
        torch.testing.assert_close(mlp.experts_w2[slot], state[prefix + "w2.weight"][:, 2:].t().contiguous())
    assert mlp.experts_w2_scale.dtype == torch.bfloat16
    assert mlp.shared_experts.gate_up_proj._weight_is_transposed
    assert mlp.shared_experts.gate_up_proj.weight_offset.shape == (4,)
    assert torch.count_nonzero(mlp.shared_experts.gate_up_proj.weight_offset) == 0
    assert torch.count_nonzero(mlp.shared_experts.down_proj.weight_offset) == 0
    assert not mlp.hash_layer


@pytest.mark.parametrize("float_query", [False, True])
def test_v41_full_model_mixed_w8a8_cpu_load(float_query: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.host.test_deepseek_v41_model import _checkpoint, _config
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    def _cpu_weight_owner(config: dict, layer_id: int, *, dtype: torch.dtype, device: torch.device) -> nn.Module:
        # Select the real W8A8 weight owner without invoking the production NPU
        # execution factory. This test never runs a quantized forward on CPU.
        return DeepseekV41W8A8MoE(DeepseekV41MoEConfig.from_dict(config), layer_id, dtype=dtype, device=device)

    monkeypatch.setattr("xllm.python.models.deepseek_v41.create_v41_moe", _cpu_weight_owner)
    config = _config(torch.device("cpu"), torch.bfloat16)
    config["quantization_config"] = {"quant_method": "ascend"}
    if float_query:
        config["quant_description"] = {"layers.0.attn.wq_a.weight": "FLOAT"}
    model = DeepseekV41ForCausalLM(config)
    state = _checkpoint(model)
    model.load_weights([state], 0, 1)
    for layer in model.model.layers:
        projection = layer.self_attn.q_b_proj
        source = state[f"layers.{layer.layer_id}.attn.wq_b.weight"]
        torch.testing.assert_close(projection.weight, source.t().contiguous())
        assert projection.weight.dtype == torch.int8
        assert projection.weight_scale.ndim == 1
        assert layer.mlp.shared_experts.gate_up_proj._weight_is_transposed
    assert model.model.layers[0].self_attn.q_a_proj.weight.dtype == (torch.bfloat16 if float_query else torch.int8)


@pytest.mark.skipif(not os.environ.get("XLLM_V41_MODEL_TEST_DEVICE"), reason="requires an explicitly available NPU")
@pytest.mark.parametrize("float_query", [False, True])
def test_v41_full_w8a8_prefill_chunk_decode(float_query: bool) -> None:
    from tests.host.test_deepseek_v41_model import _checkpoint, _config, _run
    from xllm.python.models.deepseek_v41 import DeepseekV41ForCausalLM

    device = torch.device(os.environ["XLLM_V41_MODEL_TEST_DEVICE"])
    torch.npu.set_device(device)
    config = _config(device, torch.bfloat16)
    config.update(
        hidden_size=256,
        head_dim=128,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        index_head_dim=128,
        qk_rope_head_dim=64,
        moe_intermediate_size=128,
        quantization_config={"quant_method": "ascend"},
    )
    if float_query:
        config["quant_description"] = {"layers.0.attn.wq_a.weight": "FLOAT"}
    model = DeepseekV41ForCausalLM(config)
    model.load_weights([_checkpoint(model)], 0, 1)
    expected = _run(model, (13,))
    for chunks in ((1,) * 13, (3, 7, 3)):
        actual = _run(model, chunks)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)
