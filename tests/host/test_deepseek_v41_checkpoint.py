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

"""Actual CPU tensor loading for the V4.1 attention checkpoint contract."""

from collections.abc import KeysView

import pytest
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v41_checkpoint import (
    DeepseekV41CheckpointConfig,
    load_v41_attention_weights,
    load_v41_backbone_weights,
    v41_attention_weights,
)


def _config() -> dict:
    return {
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "ascend"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 8,
            "vocab_size": 12,
            "num_attention_heads": 4,
            "head_dim": 2,
            "q_lora_rank": 4,
            "o_lora_rank": 3,
            "o_groups": 2,
            "index_n_heads": 2,
            "index_head_dim": 2,
            "num_hidden_layers": 5,
            "compress_ratios": [0, 2, 2, 1, 1, 0],
            "kv_source_layer_ids": [1, 3],
            "index_source_layer_ids": [1, 3, 4],
        },
    }


class _TensorState:
    """In-memory checkpoint storage implementing the real loader's protocol."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def keys(self) -> KeysView[str]:
        return self.tensors.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _checkpoint() -> dict[str, torch.Tensor]:
    tensors = {}
    for name, shape, quantized in (
        ("wq_a", (4, 8), True),
        ("wq_b", (8, 4), True),
        ("wkv", (2, 8), True),
        ("wo_a", (6, 4), False),
        ("wo_b", (8, 6), False),
    ):
        dtype = torch.int8 if quantized else torch.bfloat16
        prefix = "layers.0.attn." + name
        tensors[prefix + ".weight"] = torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype)
        if quantized:
            tensors[prefix + ".weight_scale"] = torch.arange(1, shape[0] + 1, dtype=torch.float32).view(-1, 1)
            tensors[prefix + ".weight_offset"] = torch.zeros(shape[0], 1)
    tensors["layers.0.attn.q_norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    tensors["layers.0.attn.kv_norm.weight"] = torch.ones(2, dtype=torch.bfloat16)
    tensors["layers.0.attn.attn_sink"] = torch.arange(4, dtype=torch.float32)
    return tensors


def _model(tp_size: int, *, cp: bool = False, mixed: bool = False) -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    attn = nn.Module()
    layer.self_attn = attn
    for name, rows, cols, quantized in (
        ("q_a_proj", 4, 8, not mixed),
        ("q_b_proj", 8 if cp else 8 // tp_size, 4, True),
        ("kv_proj", 2, 8, not mixed),
        ("o_a_proj", 6 // tp_size, 4, False),
        ("o_b_proj", 8, 6 // tp_size, False),
    ):
        module = nn.Module()
        setattr(attn, name, module)
        module.register_buffer("weight", torch.zeros(rows, cols, dtype=torch.int8 if quantized else torch.bfloat16))
        if quantized:
            module.register_buffer("weight_scale", torch.zeros(rows, 1))
            module.register_buffer("weight_offset", torch.ones(rows, 1))
    for name, size in (("q_a_layernorm", 4), ("kv_a_layernorm", 2)):
        module = nn.Module()
        setattr(attn, name, module)
        module.register_buffer("weight", torch.zeros(size, dtype=torch.bfloat16))
    attn.register_buffer("attn_sink", torch.full((4 if cp else 4 // tp_size,), -1.0))
    return model


def test_nested_config_preserves_text_geometry_and_csa2_ownership() -> None:
    raw = _config()
    raw["hidden_size"] = 999
    cfg = DeepseekV41CheckpointConfig.from_dict(raw)
    assert cfg.hidden_size == 8
    assert len(cfg.layer_plan) == 5
    assert [cfg.layer_plan.mode(i).mode for i in range(5)] == ["swa", "full", "reuse", "full", "reindex"]
    raw["text_config"]["kv_source_layer_ids"].clear()
    assert cfg.layer_plan.mode(2).main_kv_source_layer == 1


def test_flattened_config_aliases_and_default_quantization() -> None:
    raw = _config()["text_config"]
    raw["n_layers"] = raw.pop("num_hidden_layers")
    raw["n_heads"] = raw.pop("num_attention_heads")
    cfg = DeepseekV41CheckpointConfig.from_dict(raw)
    assert cfg.n_heads == 4
    assert len(cfg.layer_plan) == 5
    assert not any(s.quantized for s in v41_attention_weights(cfg, 1))


@pytest.mark.parametrize(("field", "value"), [("hidden_size", 0), ("head_dim", True), ("o_groups", 3)])
def test_invalid_geometry_fails_at_config_boundary(field: str, value: object) -> None:
    raw = _config()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        DeepseekV41CheckpointConfig.from_dict(raw)


def test_unimplemented_fp8_is_explicit_not_treated_as_bf16() -> None:
    raw = _config()
    raw["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="ascend/bf16"):
        DeepseekV41CheckpointConfig.from_dict(raw)


def test_full_reindex_reuse_own_different_checkpoint_tensors() -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    names = [{s.source.split(".attn.")[1] for s in v41_attention_weights(cfg, i)} for i in range(5)]
    assert "compressor.wgate.weight" in names[1]
    assert "compressor.wkv.weight" in names[3]
    assert "compressor.wgate.weight" not in names[3]
    assert "indexer.wq_b.weight" in names[4]
    assert "indexer.wk.weight" not in names[4]
    assert "indexer.wk.weight" in names[1]
    for i in (0, 2):
        assert not any(n.startswith(("indexer.", "compressor.")) for n in names[i])


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("cp", [False, True])
def test_real_loader_preserves_tp_rows_columns_and_cp_replication(rank: int, cp: bool) -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    tensors = _checkpoint()
    model = _model(2, cp=cp)
    loader = ModelWeightLoader(model, [_TensorState(tensors)], tp_size=2, tp_rank=rank)
    load_v41_attention_weights(loader, v41_attention_weights(cfg, 0, tp_size=2, enable_dsa_cp=cp))
    attn = model.model.layers[0].self_attn
    q = tensors["layers.0.attn.wq_b.weight"]
    assert torch.equal(attn.q_b_proj.weight, q if cp else q[rank * 4 : (rank + 1) * 4])
    scale = tensors["layers.0.attn.wq_b.weight_scale"]
    assert torch.equal(attn.q_b_proj.weight_scale, scale if cp else scale[rank * 4 : (rank + 1) * 4])
    assert torch.equal(attn.q_a_proj.weight, tensors["layers.0.attn.wq_a.weight"])
    assert torch.equal(attn.kv_proj.weight, tensors["layers.0.attn.wkv.weight"])
    assert torch.equal(attn.o_a_proj.weight, tensors["layers.0.attn.wo_a.weight"][rank * 3 : (rank + 1) * 3])
    assert torch.equal(attn.o_b_proj.weight, tensors["layers.0.attn.wo_b.weight"][:, rank * 3 : (rank + 1) * 3])
    sink = tensors["layers.0.attn.attn_sink"]
    assert torch.equal(attn.attn_sink, sink if cp else sink[rank * 2 : (rank + 1) * 2])


def test_mixed_float_projections_keep_fractional_values() -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    tensors = _checkpoint()
    description = {}
    for name in ("wq_a", "wkv"):
        key = f"layers.0.attn.{name}.weight"
        tensors[key] = tensors[key].float().add(0.5).to(torch.bfloat16)
        description[key] = "FLOAT"
        del tensors[f"layers.0.attn.{name}.weight_scale"]
        del tensors[f"layers.0.attn.{name}.weight_offset"]
    model = _model(1, mixed=True)
    loader = ModelWeightLoader(model, [_TensorState(tensors)], tp_size=1, tp_rank=0)
    load_v41_attention_weights(loader, v41_attention_weights(cfg, 0, quant_description=description))
    assert torch.equal(model.model.layers[0].self_attn.q_a_proj.weight, tensors["layers.0.attn.wq_a.weight"])
    assert torch.equal(model.model.layers[0].self_attn.kv_proj.weight, tensors["layers.0.attn.wkv.weight"])


@pytest.mark.parametrize("failure", ["missing", "dtype", "shape", "scale"])
def test_invalid_checkpoint_fails_instead_of_casting_or_skipping(failure: str) -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    tensors = _checkpoint()
    key = "layers.0.attn.wq_a.weight"
    if failure == "missing":
        del tensors[key]
    elif failure == "dtype":
        tensors[key] = tensors[key].float()
    elif failure == "shape":
        tensors[key] = tensors[key].t()
    else:
        del tensors["layers.0.attn.wq_a.weight_scale"]
    loader = ModelWeightLoader(_model(1), [_TensorState(tensors)], tp_size=1, tp_rank=0)
    with pytest.raises((KeyError, ValueError)):
        load_v41_attention_weights(loader, v41_attention_weights(cfg, 0))


def test_absent_offset_and_sink_have_explicit_zero_defaults() -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    tensors = {k: v for k, v in _checkpoint().items() if not k.endswith(("weight_offset", "attn_sink"))}
    model = _model(1)
    load_v41_attention_weights(ModelWeightLoader(model, [_TensorState(tensors)], 1, 0), v41_attention_weights(cfg, 0))
    attn = model.model.layers[0].self_attn
    assert torch.count_nonzero(attn.attn_sink) == 0
    assert torch.count_nonzero(attn.q_a_proj.weight_offset) == 0


def test_legacy_sink_suffix_is_loaded() -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    tensors = _checkpoint()
    tensors["layers.0.attn.attn_sink.weight"] = tensors.pop("layers.0.attn.attn_sink")
    model = _model(1)
    load_v41_attention_weights(ModelWeightLoader(model, [_TensorState(tensors)], 1, 0), v41_attention_weights(cfg, 0))
    assert torch.equal(model.model.layers[0].self_attn.attn_sink, torch.arange(4, dtype=torch.float32))


def test_invalid_parallel_geometry_and_quantization_are_rejected() -> None:
    cfg = DeepseekV41CheckpointConfig.from_dict(_config())
    with pytest.raises(ValueError, match="divisible"):
        v41_attention_weights(cfg, 0, tp_size=3)
    with pytest.raises(ValueError, match="unsupported quantization"):
        v41_attention_weights(cfg, 0, quant_description={"layers.0.attn.wq_a.weight": "W4A8_DYNAMIC"})
    with pytest.raises(ValueError, match="floating point"):
        v41_attention_weights(cfg, 0, quant_description={"layers.0.attn.wo_a.weight": "W8A8_DYNAMIC"})


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("head_key", ["head.weight", "lm_head.weight", "model.head.weight", "model.lm_head.weight"])
def test_backbone_shards_and_replicated_norms(rank: int, head_key: str) -> None:
    config = DeepseekV41CheckpointConfig.from_dict(_config())
    module = nn.Module()
    module.model = nn.Module()
    module.model.embed_tokens = nn.Embedding(12, 4)
    module.model.norm = nn.LayerNorm(8, bias=False)
    module.model.layers = nn.ModuleList()
    module.lm_head = nn.Linear(8, 6, bias=False)
    embedding = torch.arange(96, dtype=torch.float32).view(12, 8)
    head = embedding + 100
    tensors = {"embed.weight": embedding, head_key: head, "norm.weight": torch.arange(8, dtype=torch.float32)}
    for i in range(5):
        layer = nn.Module()
        layer.input_layernorm = nn.LayerNorm(8, bias=False)
        layer.post_attention_layernorm = nn.LayerNorm(8, bias=False)
        module.model.layers.append(layer)
        tensors[f"layers.{i}.attn_norm.weight"] = torch.full((8,), float(i + 1))
        tensors[f"layers.{i}.ffn_norm.weight"] = torch.full((8,), float(i + 10))
    loader = ModelWeightLoader(module, [_TensorState(tensors)], tp_size=2, tp_rank=rank)
    load_v41_backbone_weights(loader, config)
    torch.testing.assert_close(module.model.embed_tokens.weight, embedding[:, rank * 4 : (rank + 1) * 4])
    torch.testing.assert_close(module.lm_head.weight, head[rank * 6 : (rank + 1) * 6])
    torch.testing.assert_close(module.model.norm.weight, tensors["norm.weight"])
    for i, layer in enumerate(module.model.layers):
        torch.testing.assert_close(layer.input_layernorm.weight, tensors[f"layers.{i}.attn_norm.weight"])
        torch.testing.assert_close(layer.post_attention_layernorm.weight, tensors[f"layers.{i}.ffn_norm.weight"])
    del tensors[head_key]
    with pytest.raises(KeyError):
        load_v41_backbone_weights(ModelWeightLoader(module, [_TensorState(tensors)], tp_size=2, tp_rank=rank), config)
