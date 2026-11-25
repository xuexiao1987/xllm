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

"""DeepSeek V4.1 text model: model-owned weights over the CSA2 backend."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn

from xllm.python import distributed
from xllm.python.attention.csa2_backend import Csa2AttentionBackend, Csa2BackendConfig, Csa2ForwardState, Csa2LayerSpec
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.rotary_embedding import (
    DeepseekRotaryEmbedding,
    InterleavedRotaryInputs,
    apply_interleaved_rotary,
    prepare_interleaved_rotary,
)
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event, wait_for_kv_layer
from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v41_checkpoint import (
    DeepseekV41CheckpointConfig,
    load_v41_attention_weights,
    load_v41_backbone_weights,
    v41_attention_weights,
)
from xllm.python.models.deepseek_v41_engram import (
    EngramEmbedding,
    EngramHasher,
    EngramLayout,
    EngramProjection,
    engram_token_windows,
    load_engram_gate_unrotate,
)
from xllm.python.models.deepseek_v41_mhc import DeepseekV41HyperConnection
from xllm.python.models.deepseek_v41_moe import DeepseekV41MoE, DeepseekV41MoEConfig


def create_v41_moe(
    config: Mapping[str, object], layer_id: int, *, dtype: torch.dtype, device: torch.device
) -> nn.Module:
    """Select the checkpoint's backend before allocation; never retry a failed path."""
    geometry = DeepseekV41MoEConfig.from_dict(config)
    text = config.get("text_config", config)
    quantization = config.get("quantization_config") or text.get("quantization_config") or {}
    if not isinstance(quantization, Mapping):
        raise ValueError("quantization_config must be an object")
    method = quantization.get("quant_method", "none")
    if method == "none":
        return DeepseekV41MoE(geometry, dtype=dtype, device=device)
    if method != "ascend":
        raise ValueError(f"V4.1 MoE checkpoint format {method!r} is not implemented")
    if device.type not in ("npu", "privateuseone"):
        raise ValueError("Ascend W8A8 MoE requires the initialized NPU runtime")
    from xllm.python.models.deepseek_v41_npu_moe import DeepseekV41W8A8MoE

    return DeepseekV41W8A8MoE(geometry, layer_id, dtype=dtype, device=device)


@dataclass(frozen=True)
class DeepseekV41Config:
    checkpoint: DeepseekV41CheckpointConfig
    source: dict
    tp_size: int
    tp_rank: int
    rms_norm_eps: float
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    rope_dim: int
    max_position: int
    rope_theta: float
    compress_theta: float
    rope_factor: float
    original_context: int
    beta_fast: int
    beta_slow: int
    backend: Csa2BackendConfig

    @classmethod
    def from_dict(cls, config: dict) -> DeepseekV41Config:
        if config.get("enable_graph", False) or str(config.get("python_graph_backend", "off")).lower() not in (
            "off",
            "none",
            "",
            "0",
        ):
            raise NotImplementedError("V4.1 CSA2 text execution currently requires graph disabled")
        geometry = DeepseekV41CheckpointConfig.from_dict(config)
        text = dict(config.get("text_config", config))
        merged = {**config, **text}
        # Execution topology belongs to the engine, not nested HF text_config.
        for key in ("tp_size", "tp_rank", "ep_size", "ep_rank", "moe_tp_size", "moe_tp_rank", "dp_size", "dp_rank"):
            if key in config:
                merged[key] = config[key]
        merged.pop("text_config", None)
        tp, rank = int(config.get("tp_size", 1)), int(config.get("tp_rank", 0))
        if (
            tp <= 0
            or not 0 <= rank < tp
            or any(n % tp for n in (geometry.n_heads, geometry.o_groups, geometry.hidden_size, geometry.vocab_size))
        ):
            raise ValueError("V4.1 TP must divide attention heads/groups, hidden size and vocabulary")
        if int(config.get("cp_size", 1)) != 1 or config.get("enable_dsa_cp", False):
            raise NotImplementedError("V4.1 eager text model requires CP=1")
        if int(config.get("kv_split_size", 0)) not in (0, 1) or int(config.get("layerwise_split_size", 1)) != 1:
            raise NotImplementedError("V4.1 eager text model requires replicated KV and no layerwise split")
        if int(text.get("first_k_dense_replace", 0)) or int(text.get("moe_layer_freq", 1)) != 1:
            raise ValueError("V4.1 text checkpoints require MoE in every decoder layer")
        specs = tuple(
            Csa2LayerSpec(
                i,
                geometry.layer_plan.mode(i).mode,
                geometry.layer_plan.mode(i).compress_ratio,
                geometry.layer_plan.mode(i).main_kv_source_layer,
            )
            for i in range(len(geometry.layer_plan))
        )
        ratios = geometry.layer_plan.manager_ratios
        native_ratios = config.get("csa2_manager_ratios")
        if native_ratios is not None and tuple(native_ratios) != ratios:
            raise ValueError("CSA2 native manager layout differs from the model layout")
        window = int(text.get("window_size", text.get("sliding_window", 128)))
        rope_dim = int(text.get("qk_rope_head_dim", 64))
        if window <= 0 or rope_dim <= 0 or rope_dim % 2 or rope_dim > min(geometry.head_dim, geometry.index_head_dim):
            raise ValueError("invalid V4.1 window or partial RoPE geometry")
        rope = text.get("rope_scaling") or {}
        backend = Csa2BackendConfig(
            specs,
            window,
            int(text.get("index_topk", 512)),
            int(text.get("candidate_source_layer_id", -1)),
            int(text.get("candidate_topk_blocks", 2048)),
            int(text.get("candidate_block_size", 8)),
            ratios,
        )
        geometry.layer_plan.validate_candidate_source(backend.candidate_source)
        return cls(
            geometry,
            merged,
            tp,
            rank,
            float(text.get("rms_norm_eps", 1e-20)),
            int(text.get("hc_mult", 4)),
            float(text.get("hc_eps", 1e-6)),
            int(text.get("hc_sinkhorn_iters", 20)),
            rope_dim,
            int(text.get("max_position_embeddings", 1048576)),
            float(text.get("rope_theta", 10000)),
            float(text.get("compress_rope_theta", 160000)),
            float(rope.get("factor", text.get("factor", 16))),
            int(
                rope.get(
                    "original_max_position_embeddings", text.get("rope_scaling_original_max_position_embeddings", 65536)
                )
            ),
            int(rope.get("beta_fast", text.get("beta_fast", 32))),
            int(rope.get("beta_slow", text.get("beta_slow", 1))),
            backend,
        )


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device), requires_grad=False)
        self._native_norm = None
        if device.type in ("npu", "privateuseone"):
            from xllm.python import kernels

            self._native_norm = kernels.rms_norm

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self._native_norm is not None and value.numel() > 0:
            if value.dtype == self.weight.dtype == torch.bfloat16:
                return self._native_norm(value, self.weight, self.eps)
            return self._native_norm(value.float(), self.weight.float(), self.eps).to(value.dtype)
        x = value.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(value.dtype)


class DeepseekV41Attention(Attention):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        description: dict[str, str],
    ) -> None:
        g = cfg.checkpoint
        super().__init__(g.n_heads // cfg.tp_size, 1, g.head_dim, g.head_dim**-0.5, cfg.backend.window_size, layer_id)
        self.cfg, self.spec = cfg, cfg.backend.layers[layer_id]
        self.csa2_backend_config = cfg.backend
        self.weight_specs = v41_attention_weights(g, layer_id, tp_size=cfg.tp_size, quant_description=description)
        self._quantized: list[nn.Module] = []
        prefix = f"model.layers.{layer_id}.self_attn."
        for spec in self.weight_specs:
            if not spec.destination.endswith(".weight") or len(spec.shape) != 2:
                continue
            path = spec.destination[len(prefix) : -len(".weight")]
            owner = self
            parts = path.split(".")
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    owner.add_module(part, nn.Module())
                owner = getattr(owner, part)
            output, input_size = spec.shape
            if spec.shard_dim == 0:
                output //= cfg.tp_size
            elif spec.shard_dim == 1:
                input_size //= cfg.tp_size
            if spec.quantized:
                from xllm.python.models.deepseek_v32 import W8A8DynamicLinear

                projection = W8A8DynamicLinear(input_size, output, device)
                self._quantized.append(projection)
            else:
                projection_dtype = torch.float32 if path in ("cmp_wkv", "cmp_wgate") and self.spec.ratio > 1 else dtype
                projection = nn.Linear(input_size, output, bias=False, dtype=projection_dtype, device=device)
            owner.add_module(parts[-1], projection)
        self.q_a_layernorm = _RMSNorm(g.q_lora_rank, cfg.rms_norm_eps, dtype, device)
        self.kv_a_layernorm = _RMSNorm(g.head_dim, cfg.rms_norm_eps, dtype, device)
        self.attn_sink = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32, device=device), requires_grad=False
        )
        if self.spec.mode == "full":
            self.cmp_norm = _RMSNorm(g.head_dim, cfg.rms_norm_eps, dtype, device)
            self.indexer.k_norm = _RMSNorm(g.index_head_dim, cfg.rms_norm_eps, dtype, device)
        self.requires_grad_(False)

    def process_weights_after_loading(self) -> None:
        for module in self._quantized:
            module.process_weights_after_loading()

    def compress_kv_score(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.spec.ratio == 1:
            return self.cmp_wkv(hidden), None
        return self.cmp_wkv(hidden.float()), self.cmp_wgate(hidden.float())

    def compress_normalize(self, pooled: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.cmp_norm(pooled.to(dtype))

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        return self.indexer.k_norm(self.indexer.wk(latent))

    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor:
        g = self.cfg.checkpoint
        return self.indexer.wq_b(qr).reshape(qr.size(0), g.index_n_heads, g.index_head_dim)

    def indexer_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        g = self.cfg.checkpoint
        return self.indexer.weights_proj(hidden) * (g.index_head_dim * g.index_n_heads) ** -0.5

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        rope: torch.Tensor,
        rotary_inputs: InterleavedRotaryInputs | None = None,
    ) -> torch.Tensor:
        backend = get_forward_context().attention_backend
        if not isinstance(backend, Csa2AttentionBackend):
            raise RuntimeError("V4.1 requires the CSA2 backend")
        qr = self.q_a_layernorm(self.q_a_proj(hidden))
        q = apply_interleaved_rotary(
            self.q_b_proj(qr).reshape(hidden.size(0), self.num_heads, self.head_dim),
            positions,
            rope,
            prepared=rotary_inputs,
        )
        kv = apply_interleaved_rotary(
            self.kv_a_layernorm(self.kv_proj(hidden)), positions, rope, prepared=rotary_inputs
        )
        output = backend.execute_csa2(q, kv, hidden, qr, positions, rope, self)
        output = apply_interleaved_rotary(output, positions, rope, inverse=True, prepared=rotary_inputs)
        groups = self.cfg.checkpoint.o_groups // self.cfg.tp_size
        weight = self.o_a_proj.weight.reshape(groups, self.cfg.checkpoint.o_lora_rank, -1)
        low = torch.einsum(
            "tgd,grd->tgr", output.reshape(hidden.size(0), groups, self.num_heads * self.head_dim // groups), weight
        )
        output = self.o_b_proj(low.flatten(1))
        if self.cfg.tp_size > 1:
            distributed.tp_all_reduce(output)
        return output


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        description: dict[str, str],
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = DeepseekV41Attention(cfg, layer_id, dtype, device, description)
        self.input_layernorm = _RMSNorm(cfg.checkpoint.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.post_attention_layernorm = _RMSNorm(cfg.checkpoint.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hc = DeepseekV41HyperConnection(
            hidden_size=cfg.checkpoint.hidden_size,
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            norm_eps=cfg.rms_norm_eps,
            hc_eps=cfg.hc_eps,
            device=device,
        )
        self.mlp = create_v41_moe(cfg.source, layer_id, dtype=dtype, device=device)
        self.engram: EngramProjection | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        pre: torch.Tensor | None,
        positions: torch.Tensor,
        rope: torch.Tensor,
        engram_rows: torch.Tensor | None,
        rotary_inputs: InterleavedRotaryInputs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.engram is not None:
            if engram_rows is None:
                raise RuntimeError("V4.1 Engram rows missing for this forward")
            hidden = self.engram(hidden, engram_rows)
        collapsed, next_pre, post, combination = self.hc(hidden, pre, part="attn")
        attention = self.self_attn(self.input_layernorm(collapsed), positions, rope, rotary_inputs)
        hidden = self.hc.expand(attention, hidden, post, combination)
        collapsed, pre, post, combination = self.hc(hidden, next_pre, part="ffn")
        hidden = self.hc.expand(self.mlp(self.post_attention_layernorm(collapsed)), hidden, post, combination)
        return hidden, pre


class _VocabHead(nn.Linear):
    def __init__(self, hidden: int, vocab: int, tp: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(hidden, vocab // tp, bias=False, dtype=dtype, device=device)
        self.tp_size = tp

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = super().forward(hidden)
        return distributed.tp_all_gather(logits, dim=-1, world_size=self.tp_size) if self.tp_size > 1 else logits


class DeepseekV41Model(nn.Module):
    def __init__(
        self, cfg: DeepseekV41Config, dtype: torch.dtype, device: torch.device, description: dict[str, str]
    ) -> None:
        super().__init__()
        self.cfg = cfg
        g = cfg.checkpoint
        self.embed_tokens = HiddenParallelEmbedding(
            g.vocab_size, g.hidden_size // cfg.tp_size, cfg.tp_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(cfg, i, dtype, device, description) for i in range(len(g.layer_plan))]
        )
        self.norm = _RMSNorm(g.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = DeepseekRotaryEmbedding(
            cfg.rope_dim,
            cfg.max_position,
            1.0,
            cfg.rope_theta,
            cfg.beta_fast,
            cfg.beta_slow,
            cfg.original_context,
            torch.float32,
            device,
        )
        self.compress_rotary = DeepseekRotaryEmbedding(
            cfg.rope_dim,
            cfg.max_position,
            cfg.rope_factor,
            cfg.compress_theta,
            cfg.beta_fast,
            cfg.beta_slow,
            cfg.original_context,
            torch.float32,
            device,
        )
        self.engram_hash: EngramHasher | None = None
        self.engram_embeddings = nn.ModuleDict()
        if cfg.source.get("engram_layer_ids"):
            layout = EngramLayout.from_dict(cfg.source)
            model_path = Path(cfg.source["model_path"])
            self.engram_hash = EngramHasher.from_checkpoint(layout, model_path, g.vocab_size).to(device)
            rotation = load_engram_gate_unrotate(model_path, cfg.source.get("engram_rotation_config") or {})
            for index, layer_id in enumerate(layout.layer_ids):
                self.engram_embeddings[str(layer_id)] = EngramEmbedding(
                    layout, index, tp_size=cfg.tp_size, tp_rank=cfg.tp_rank, device=device
                )
                self.layers[layer_id].engram = EngramProjection(
                    input_size=len(layout.head_sizes[index]) * layout.head_dim,
                    hidden_size=g.hidden_size,
                    hc_mult=cfg.hc_mult,
                    eps=cfg.rms_norm_eps,
                    dtype=dtype,
                    device=device,
                    gate_unrotate=rotation,
                )
        self.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        context = get_forward_context()
        state = context.metadata.prepared_attention_state
        if not isinstance(state, Csa2ForwardState):
            raise RuntimeError("V4.1 requires prepared CSA2 metadata")
        hidden = self.embed_tokens(input_ids).unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        hashes = None
        if self.engram_hash is not None and not state.is_dummy:
            lengths = state.layers[0]
            history = getattr(context.metadata, "engram_token_history", None)
            if history is None:
                if any(q != c for q, c in zip(lengths.query_lens, lengths.context_lens, strict=True)):
                    raise RuntimeError("V4.1 Engram decode/chunk requires request token history")
                history = input_ids.new_full((len(lengths.query_lens), self.engram_hash.layout.max_ngram_size - 1), -1)
            starts = input_ids.new_tensor((0, *lengths.query_lens)).cumsum(0)
            hashes = self.engram_hash(engram_token_windows(input_ids, starts, history.to(input_ids.device)))
        # Both tables share immutable packed positions throughout this forward.
        # Keep coefficients local, so the next batch cannot reuse stale rows.
        rotary_inputs = {
            "swa": prepare_interleaved_rotary(positions, self.rotary.cos_sin_cache, hidden.dtype),
            "compressed": prepare_interleaved_rotary(positions, self.compress_rotary.cos_sin_cache, hidden.dtype),
        }
        pre = None
        for layer in self.layers:
            wait_for_kv_layer(layer.layer_id)
            rows = None
            if layer.engram is not None and state.is_dummy:
                rows = hidden.new_zeros((input_ids.numel(), layer.engram.wkv.in_features))
            elif layer.engram is not None:
                index = self.engram_hash.layout.layer_ids.index(layer.layer_id)
                rows = self.engram_embeddings[str(layer.layer_id)](hashes[:, index])
                if self.cfg.tp_size > 1:
                    rows = distributed.tp_all_gather(rows, dim=1, world_size=self.cfg.tp_size)
            rope = (
                self.rotary.cos_sin_cache if layer.self_attn.spec.mode == "swa" else self.compress_rotary.cos_sin_cache
            )
            prepared = rotary_inputs["swa" if layer.self_attn.spec.mode == "swa" else "compressed"]
            hidden, pre = layer(hidden, pre, positions, rope, rows, prepared)
            record_layer_event(layer.layer_id)
        return self.norm(DeepseekV41HyperConnection.collapse(hidden, pre))


class DeepseekV41ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        if config.get("model_path"):
            path = Path(config["model_path"]) / "config.json"
            if path.is_file():
                # Preserve HF-only model geometry/rotation declarations while
                # native runtime topology and dtype remain authoritative.
                config = {**json.loads(path.read_text()), **config}
        self.cfg = DeepseekV41Config.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        description = config.get("quant_description") or {}
        if not description and config.get("model_path"):
            path = Path(config["model_path"]) / "quant_model_description.json"
            if path.is_file():
                description = json.loads(path.read_text())
        self.model = DeepseekV41Model(self.cfg, dtype, device, description)
        self.lm_head = _VocabHead(
            self.cfg.checkpoint.hidden_size, self.cfg.checkpoint.vocab_size, self.cfg.tp_size, dtype, device
        )
        self.requires_grad_(False)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        if (tp_size, tp_rank) != (self.cfg.tp_size, self.cfg.tp_rank):
            raise ValueError("V4.1 loading topology differs from constructed model")
        loader = ModelWeightLoader(self, state_dicts, tp_rank=tp_rank, tp_size=tp_size)
        load_v41_backbone_weights(loader, self.cfg.checkpoint)
        for layer in self.model.layers:
            load_v41_attention_weights(loader, layer.self_attn.weight_specs)
            layer.hc.load_weights(loader, layer.layer_id)
            layer.mlp.load_weights(loader, layer.layer_id)
            if layer.engram is not None:
                self.model.engram_embeddings[str(layer.layer_id)].load_weights(loader)
                layer.engram.load_weights(loader, layer.layer_id)
            layer.self_attn.process_weights_after_loading()
