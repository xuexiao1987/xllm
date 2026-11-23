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

"""V4.1 text checkpoint geometry and loading, independent of NPU kernels.

Attention and backbone mappings share the base loader. Checkpoint
matrices remain in [output, input] order; device packing belongs to the model's
post-load step. Expert, Engram, mHC, vision and draft weights have their own loaders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v41_layout import CSA2LayerPlan


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class DeepseekV41CheckpointConfig:
    """Text dimensions needed before allocating attention and backbone parameters."""

    hidden_size: int
    vocab_size: int
    n_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    index_n_heads: int
    index_head_dim: int
    quant_method: str
    layer_plan: CSA2LayerPlan

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> DeepseekV41CheckpointConfig:
        """Accept the HF nested text config or the native flattened fields."""
        text = config.get("text_config", config)
        if not isinstance(text, Mapping):
            raise ValueError("text_config must be an object")
        model_type = text.get("model_type", config.get("model_type"))
        if model_type not in ("deepseek_v41", "deepseek_v41_text"):
            raise ValueError(f"expected a DeepSeek-V4.1 text config, got {model_type!r}")
        fields = {}
        for name, alias, default in (
            ("hidden_size", "hidden_size", 5120),
            ("vocab_size", "vocab_size", 129280),
            ("n_heads", "num_attention_heads", 64),
            ("head_dim", "head_dim", 512),
            ("q_lora_rank", "q_lora_rank", 1280),
            ("o_lora_rank", "o_lora_rank", 1024),
            ("o_groups", "o_groups", 8),
            ("index_n_heads", "index_n_heads", 32),
            ("index_head_dim", "index_head_dim", 128),
        ):
            value = text.get(name)
            if value is None:
                value = text.get(alias)
            fields[name] = _positive_int(default if value is None else value, name)
        n_layers = text.get("num_hidden_layers")
        if n_layers is None:
            n_layers = text.get("n_layers", 40)
        n_layers = _positive_int(n_layers, "n_layers")
        if fields["n_heads"] % fields["o_groups"]:
            raise ValueError("n_heads must be divisible by o_groups")
        quant = config.get("quantization_config") or text.get("quantization_config") or {}
        if not isinstance(quant, Mapping):
            raise ValueError("quantization_config must be an object")
        method = quant.get("quant_method", "none")
        if method not in ("ascend", "none"):
            raise ValueError(f"V4.1 attention checkpoint loading currently supports ascend/bf16, got {method!r}")
        return cls(
            **fields,
            quant_method=method,
            layer_plan=CSA2LayerPlan.build(
                n_layers,
                text.get("compress_ratios") or [],
                text.get("kv_source_layer_ids") or [],
                text.get("index_source_layer_ids") or [],
            ),
        )


@dataclass(frozen=True)
class V41CheckpointWeight:
    """One global source tensor and its pre-packing destination on a TP rank."""

    source: str
    destination: str
    shape: tuple[int, ...]
    shard_dim: int | None = None
    quantized: bool = False
    zero_if_missing: bool = False


def v41_attention_weights(
    config: DeepseekV41CheckpointConfig,
    layer_id: int,
    *,
    tp_size: int = 1,
    enable_dsa_cp: bool = False,
    quant_description: Mapping[str, str] | None = None,
) -> tuple[V41CheckpointWeight, ...]:
    """Resolve mixed precision and CSA2 ownership before allocating a layer.

    W8A8 exports without explicit FLOAT declarations retain dynamic INT8 Q/KV
    projections. Full layers own compressor and index K; Reindex owns index Q
    only; Reuse/SWA own neither. CP replicates Q and sink but still shards O.
    """
    tp_size = _positive_int(tp_size, "tp_size")
    if config.n_heads % tp_size or config.o_groups % tp_size:
        raise ValueError("attention heads and output groups must be divisible by tp_size")
    mode = config.layer_plan.mode(layer_id)
    source = f"layers.{layer_id}.attn."
    destination = f"model.layers.{layer_id}.self_attn."
    description = quant_description or {}
    specs = []
    q_dim = None if enable_dsa_cp else 0

    def _projection(
        name: str, target: str, shape: tuple[int, int], dim: int | None = None, *, dynamic: bool = False
    ) -> None:
        declared = description.get(source + name + ".weight")
        quantized = dynamic and config.quant_method == "ascend"
        if declared is not None:
            if declared not in ("FLOAT", "W8A8_DYNAMIC"):
                raise ValueError(f"unsupported quantization {declared!r} for {source}{name}.weight")
            if declared == "W8A8_DYNAMIC" and not quantized:
                raise ValueError(f"{source}{name}.weight must be floating point in this checkpoint format")
            quantized = declared == "W8A8_DYNAMIC"
        specs.append(
            V41CheckpointWeight(source + name + ".weight", destination + target + ".weight", shape, dim, quantized)
        )
        if quantized:
            for suffix in ("weight_scale", "weight_offset"):
                specs.append(
                    V41CheckpointWeight(
                        source + name + "." + suffix,
                        destination + target + "." + suffix,
                        (shape[0], 1),
                        dim,
                        zero_if_missing=suffix == "weight_offset",
                    )
                )

    _projection("wq_a", "q_a_proj", (config.q_lora_rank, config.hidden_size), dynamic=True)
    _projection("wq_b", "q_b_proj", (config.n_heads * config.head_dim, config.q_lora_rank), q_dim, dynamic=True)
    _projection("wkv", "kv_proj", (config.head_dim, config.hidden_size), dynamic=True)
    _projection(
        "wo_a",
        "o_a_proj",
        (config.o_groups * config.o_lora_rank, config.n_heads * config.head_dim // config.o_groups),
        0,
    )
    _projection("wo_b", "o_b_proj", (config.hidden_size, config.o_groups * config.o_lora_rank), 1)
    for name, target, size in (
        ("q_norm.weight", "q_a_layernorm.weight", config.q_lora_rank),
        ("kv_norm.weight", "kv_a_layernorm.weight", config.head_dim),
    ):
        specs.append(V41CheckpointWeight(source + name, destination + target, (size,)))
    specs.append(
        V41CheckpointWeight(
            source + "attn_sink", destination + "attn_sink", (config.n_heads,), q_dim, zero_if_missing=True
        )
    )
    if mode.is_full:
        _projection("compressor.wkv", "cmp_wkv", (config.head_dim, config.hidden_size))
        if mode.compress_ratio > 1:
            _projection("compressor.wgate", "cmp_wgate", (config.head_dim, config.hidden_size))
        specs.append(
            V41CheckpointWeight(source + "compressor.norm.weight", destination + "cmp_norm.weight", (config.head_dim,))
        )
    if mode.is_full or mode.is_reindex:
        _projection(
            "indexer.wq_b",
            "indexer.wq_b",
            (config.index_n_heads * config.index_head_dim, config.q_lora_rank),
            dynamic=True,
        )
        _projection("indexer.weights_proj", "indexer.weights_proj", (config.index_n_heads, config.hidden_size))
        if mode.is_full:
            _projection("indexer.wk", "indexer.wk", (config.index_head_dim, config.head_dim))
            specs.append(
                V41CheckpointWeight(
                    source + "indexer.k_norm.weight", destination + "indexer.k_norm.weight", (config.index_head_dim,)
                )
            )
    return tuple(specs)


def load_v41_attention_weights(loader: ModelWeightLoader, weights: tuple[V41CheckpointWeight, ...]) -> None:
    """Copy one layer through the base loader; no device packing or execution."""
    _load_v41_weights(loader, weights)


def load_v41_backbone_weights(loader: ModelWeightLoader, config: DeepseekV41CheckpointConfig) -> None:
    """Load hidden-sharded input embedding, vocab-sharded head and replicated norms.

    mHC weights are owned by DeepseekV41HyperConnection. Attention, Expert and
    Engram loading remain layer-owned; this does not initialize a complete LM.
    """
    if config.hidden_size % loader.tp_size or config.vocab_size % loader.tp_size:
        raise ValueError("V4.1 hidden and vocabulary sizes must be divisible by TP")
    weights = [
        V41CheckpointWeight("embed.weight", "model.embed_tokens.weight", (config.vocab_size, config.hidden_size), 1),
        V41CheckpointWeight("head.weight", "lm_head.weight", (config.vocab_size, config.hidden_size), 0),
        V41CheckpointWeight("norm.weight", "model.norm.weight", (config.hidden_size,)),
    ]
    for layer in range(len(config.layer_plan)):
        for source, target in (("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
            weights.append(
                V41CheckpointWeight(
                    f"layers.{layer}.{source}.weight", f"model.layers.{layer}.{target}.weight", (config.hidden_size,)
                )
            )
    _load_v41_weights(loader, tuple(weights))


def _load_v41_weights(loader: ModelWeightLoader, weights: tuple[V41CheckpointWeight, ...]) -> None:
    for spec in weights:
        source = spec.source
        if source == "head.weight":
            source = next(
                (
                    key
                    for key in ("lm_head.weight", "model.lm_head.weight", "model.head.weight", "head.weight")
                    if loader.has(key)
                ),
                source,
            )
        if source.endswith(".attn_sink") and not loader.has(source) and loader.has(source + ".weight"):
            source += ".weight"
        if spec.zero_if_missing and not loader.has(source):
            tensor = torch.zeros(spec.shape, dtype=torch.float32)
        else:
            tensor = loader.get_tensor(source)
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"{source} has shape {tuple(tensor.shape)}, expected {spec.shape}")
        if spec.quantized:
            if tensor.dtype not in (torch.int8, torch.uint8):
                raise ValueError(f"{source} requires INT8 storage, got {tensor.dtype}")
        elif tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"{source} requires floating point storage, got {tensor.dtype}")
        if spec.shard_dim is not None:
            tensor = loader.shard(tensor, spec.shard_dim)
        loader.copy_in(spec.destination, tensor)
