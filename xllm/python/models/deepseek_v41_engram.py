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

"""V4.1 Engram hash, local INT8 lookup and gated residual projection.

The execution layer supplies token history and gathers TP head rows. This
module does not own scheduler slots, collectives, host staging or graph buffers.
Those lifetimes must not be inferred from the shape of a previous batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader


def _is_prime(value: int) -> bool:
    # Deterministic Miller-Rabin witnesses for values below 2**32.
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for witness in (2, 7, 61):
        if witness % value == 0:
            continue
        result = pow(witness, exponent, value)
        if result in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            result = result * result % value
            if result == value - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class EngramLayout:
    """Immutable checkpoint bucket order, shared by hashing and TP loading."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    compressed_vocab_size: int
    pad_token_id: int
    head_sizes: tuple[tuple[int, ...], ...]

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> EngramLayout:
        """Read the declared Engram geometry from nested or native text config."""
        text = config.get("text_config", config)
        if not isinstance(text, Mapping):
            raise ValueError("text_config must be an object")
        return cls.build(
            text["engram_layer_ids"],
            text["engram_num_embeddings"],
            **{
                name: text["engram_" + name]
                for name in (
                    "max_ngram_size",
                    "n_heads",
                    "head_dim",
                    "vocab_size",
                    "compressed_vocab_size",
                    "pad_token_id",
                )
            },
        )

    @classmethod
    def build(
        cls,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        *,
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> EngramLayout:
        layers, rows = tuple(layer_ids), tuple(num_embeddings)
        values = (*layers, *rows, max_ngram_size, n_heads, head_dim, vocab_size, compressed_vocab_size, pad_token_id)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError("Engram dimensions and layer ids must be integers")
        if not layers or len(layers) != len(rows) or len(set(layers)) != len(layers) or min(layers) < 0:
            raise ValueError("Engram requires unique nonnegative layers and one table size per layer")
        if max_ngram_size < 2 or min(n_heads, head_dim, vocab_size, compressed_vocab_size) <= 0 or pad_token_id < 0:
            raise ValueError("invalid Engram dimensions or padding token")
        if head_dim % 32 or min(rows) <= 0 or max(rows) > torch.iinfo(torch.int32).max:
            raise ValueError("Engram requires group-32 columns and positive int32 table row counts")
        if vocab_size >= 2**32:
            raise ValueError("Engram prime bucket seeds must be below 2**32")
        # Every head across all layers receives a distinct prime, in checkpoint
        # order. A rank owns complete buckets, not an equal fraction of rows.
        candidate = vocab_size
        sizes = []
        for count in rows:
            heads = []
            for _ in range((max_ngram_size - 1) * n_heads):
                while not _is_prime(candidate):
                    candidate += 1
                heads.append(candidate)
                candidate += 1
            if sum(heads) > count:
                raise ValueError("Engram prime buckets exceed the configured table size")
            sizes.append(tuple(heads))
        return cls(layers, rows, max_ngram_size, n_heads, head_dim, compressed_vocab_size, pad_token_id, tuple(sizes))


def engram_hash_multipliers(layout: EngramLayout) -> torch.Tensor:
    """Reproduce the training PCG64 stream; this is checkpoint data, not a seed choice."""
    bound = max(1, (np.iinfo(np.int64).max // layout.compressed_vocab_size) // 2)
    return torch.stack(
        [
            torch.from_numpy(
                np.random.default_rng(10007 * layer).integers(0, bound, size=layout.max_ngram_size) * 2 + 1
            )
            for layer in layout.layer_ids
        ]
    )


def engram_token_windows(input_ids: torch.Tensor, query_start_loc: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Build current-first windows for a packed batch, without crossing requests.

    ``history[R, max_ngram_size - 1]`` is newest-first at each chunk start;
    -1 denotes a sequence boundary/dead token. The caller resolves cache slots
    and validates token ids before calling this tensor-only path. Empty padded
    requests are allowed in the cumulative query lengths.
    """
    rows = torch.arange(input_ids.numel(), device=input_ids.device)
    request = torch.searchsorted(query_start_loc[1:], rows, right=True)
    starts = query_start_loc[request]
    shifts = torch.arange(history.shape[1] + 1, device=input_ids.device)
    source = rows[:, None] - shifts
    history_index = (starts[:, None] - source - 1).clamp(0, history.shape[1] - 1)
    return torch.where(
        source >= starts[:, None], input_ids[source.clamp_min(0)], history[request[:, None], history_index]
    )


class EngramHasher(nn.Module):
    """Hash explicit windows; history storage and mutation stay in the cache owner."""

    def __init__(self, layout: EngramLayout, token_map: torch.Tensor, multipliers: torch.Tensor | None = None) -> None:
        super().__init__()
        if token_map.device.type != "cpu" or token_map.dtype not in (torch.int32, torch.int64) or token_map.ndim != 1:
            raise ValueError("Engram token map must be a one-dimensional CPU integer tensor at initialization")
        if token_map.numel() <= layout.pad_token_id or int(token_map.min()) < 0:
            raise ValueError("Engram token map is missing padding or contains negative compressed ids")
        if int(token_map.max()) + 1 != layout.compressed_vocab_size:
            raise ValueError("Engram compressed vocabulary does not match the checkpoint")
        expected = engram_hash_multipliers(layout)
        if multipliers is not None and (
            multipliers.dtype != torch.int64 or not torch.equal(multipliers.cpu(), expected)
        ):
            raise ValueError("Engram sidecar multipliers do not match the configured layer order/vocabulary")
        self.layout = layout
        self.pad_id = int(token_map[layout.pad_token_id])
        self.register_buffer("token_map", token_map.to(torch.int64).clone(), persistent=False)
        primes = torch.tensor(layout.head_sizes, dtype=torch.int64)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", primes.cumsum(-1) - primes, persistent=False)
        self.register_buffer("multipliers", expected, persistent=False)

    @classmethod
    def from_checkpoint(cls, layout: EngramLayout, model_path: Path, vocabulary_size: int) -> EngramHasher:
        """Load mandatory tokenizer-derived state and validate it before serving."""
        state = torch.load(model_path / "engram_hash_state.bin", map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(state.get(key), torch.Tensor) for key in ("token_map", "multipliers")
        ):
            raise ValueError("Engram hash sidecar must contain token_map and multipliers tensors")
        if state["token_map"].numel() != vocabulary_size:
            raise ValueError("Engram hash sidecar does not cover the model vocabulary")
        return cls(layout, state["token_map"], state["multipliers"])

    def forward(self, token_windows: torch.Tensor) -> torch.Tensor:
        """Return [tokens, Engram layers, hash heads] int32 table row ids.

        A negative token breaks that n-gram and all older lookbacks. Padding
        still hashes into trained table rows; it never means a zero embedding.
        """
        compressed = self.token_map[token_windows.clamp_min(0)]
        blocked = (token_windows < 0).to(torch.int32).cumsum(-1) > 0
        compressed = torch.where(blocked, self.pad_id, compressed)
        rolling = torch.zeros(
            (token_windows.shape[0], len(self.layout.layer_ids)), device=token_windows.device, dtype=torch.int64
        )
        outputs = []
        for shift in range(self.layout.max_ngram_size):
            rolling = rolling ^ (compressed[:, shift, None] * self.multipliers[None, :, shift])
            if shift:
                start = (shift - 1) * self.layout.n_heads
                end = start + self.layout.n_heads
                outputs.append(rolling[:, :, None] % self.primes[None, :, start:end] + self.offsets[None, :, start:end])
        return torch.cat(outputs, dim=-1).to(torch.int32)


def load_engram_gate_unrotate(model_path: Path, rotation_config: Mapping[str, object]) -> torch.Tensor | None:
    """Load Q.T only for the supported export; keep exact repeated 32-wide blocks."""
    if not rotation_config.get("value_projection_rotated", False):
        if rotation_config:
            raise ValueError("unsupported Engram rotation configuration")
        return None
    if (
        rotation_config.get("value_basis") != "quarot_global"
        or rotation_config.get("key_and_gate_basis") != "original"
        or rotation_config.get("runtime_delta_rotation") is not False
    ):
        raise ValueError("Engram requires folded QuaRot values and original-basis keys/gates")
    from safetensors import safe_open

    with safe_open(model_path / "optional/quarot.safetensors", framework="pt", device="cpu") as weights:
        rotation = weights.get_tensor("global_rotation")
    if (
        rotation.ndim != 2
        or rotation.shape[0] != rotation.shape[1]
        or not rotation.shape[0]
        or rotation.dtype not in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("Engram global rotation must be a square floating point matrix")
    width = rotation.shape[0]
    if width > 32 and width % 32 == 0:
        block = rotation[:32, :32]
        # Inspect one block-row at a time instead of allocating a second full
        # hidden-size squared matrix just to detect the compact representation.
        repeated = True
        for start in range(0, width, 32):
            if (
                not torch.equal(rotation[start : start + 32, start : start + 32], block)
                or bool(torch.count_nonzero(rotation[start : start + 32, :start]))
                or bool(torch.count_nonzero(rotation[start : start + 32, start + 32 :]))
            ):
                repeated = False
                break
        if repeated:
            rotation = block
    return rotation.t().to(torch.float32).contiguous()


class EngramEmbedding(nn.Module):
    """One TP rank's complete head buckets, stored as INT8 with group-32 scales."""

    def __init__(
        self, layout: EngramLayout, layer_index: int, *, tp_size: int, tp_rank: int, device: torch.device
    ) -> None:
        super().__init__()
        if not 0 <= layer_index < len(layout.layer_ids):
            raise ValueError("invalid Engram layer index")
        sizes = layout.head_sizes[layer_index]
        if tp_size <= 0 or not 0 <= tp_rank < tp_size or len(sizes) % tp_size:
            raise ValueError("Engram TP must divide the hash head count and contain this rank")
        self.layer_id = layout.layer_ids[layer_index]
        self.num_embeddings = layout.num_embeddings[layer_index]
        self.dim = layout.head_dim
        self.head_count = len(sizes) // tp_size
        self.head_start = tp_rank * self.head_count
        self.row_start = sum(sizes[: self.head_start])
        self.row_count = sum(sizes[self.head_start : self.head_start + self.head_count])
        self.register_buffer("weight", torch.empty((self.row_count, self.dim), dtype=torch.int8, device=device))
        self.register_buffer(
            "weight_scale", torch.empty((self.row_count, self.dim // 32), dtype=torch.float32, device=device)
        )

    @torch.no_grad()
    def load_weights(self, loader: ModelWeightLoader, *, chunk_rows: int = 1 << 20) -> None:
        """Narrow mmap-backed tensors before copying or decoding any scale data."""
        if chunk_rows <= 0:
            raise ValueError("Engram load chunk size must be positive")
        prefix = f"layers.{self.layer_id}.engram.embed."
        table, scale = loader.get_tensor(prefix + "weight"), loader.get_tensor(prefix + "scale")
        if table.dtype not in (torch.int8, torch.uint8):
            raise ValueError("Engram INT8 table requires signed bytes or raw uint8 storage")
        if table.ndim != 2 or table.shape[1] != self.dim or table.shape[0] < self.num_embeddings:
            raise ValueError("Engram table shape does not match the configured row count and width")
        if scale.shape != (table.shape[0], self.dim // 32) or scale.dtype not in (
            torch.float32,
            torch.uint8,
            torch.float8_e8m0fnu,
        ):
            raise ValueError("Engram requires matching FP32 or UE8M0 group-32 scales")
        table = table.view(torch.int8)
        for offset in range(0, self.row_count, chunk_rows):
            loader.check_cancelled()
            count = min(chunk_rows, self.row_count - offset)
            scales = scale.narrow(0, self.row_start + offset, count)
            if scales.dtype != torch.float32:
                raw = scales.view(torch.uint8).to(torch.int32)
                if bool((raw == 255).any()):
                    raise ValueError("Engram UE8M0 scale contains invalid NaN byte 255")
                # UE8M0 byte zero encodes the float32 subnormal 2**-127.
                scales = torch.where(raw == 0, 0x00400000, raw << 23).contiguous().view(torch.float32)
            self.weight_scale.narrow(0, offset, count).copy_(scales)
            self.weight.narrow(0, offset, count).copy_(table.narrow(0, self.row_start + offset, count))

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Return this rank's [tokens, local heads, dim] rows for a TP gather."""
        local = hash_ids[:, self.head_start : self.head_start + self.head_count].to(torch.int64) - self.row_start
        values = self.weight.index_select(0, local.flatten()).float()
        scales = self.weight_scale.index_select(0, local.flatten()).repeat_interleave(32, dim=-1)
        return (values * scales).to(torch.bfloat16).reshape(hash_ids.shape[0], self.head_count, self.dim)


class EngramProjection(nn.Module):
    """Replicated projection and mHC gate consuming already gathered head rows."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        hc_mult: int,
        eps: float,
        dtype: torch.dtype,
        device: torch.device,
        gate_unrotate: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, hc_mult) <= 0 or eps <= 0:
            raise ValueError("Engram projection dimensions and normalization epsilon must be positive")
        if gate_unrotate is not None:
            if (
                gate_unrotate.ndim != 2
                or gate_unrotate.shape[0] != gate_unrotate.shape[1]
                or not gate_unrotate.shape[0]
                or hidden_size % gate_unrotate.shape[0]
            ):
                raise ValueError("Engram gate unrotation must be square and divide the hidden dimension")
            gate_unrotate = gate_unrotate.to(device=device, dtype=torch.float32).clone()
        self.eps = eps
        self.wkv = nn.Linear(input_size, hidden_size * (hc_mult + 1), bias=False, dtype=dtype, device=device)
        self.wkv.requires_grad_(False)
        self.q_weight = nn.Parameter(torch.empty(hc_mult, hidden_size, dtype=dtype, device=device), requires_grad=False)
        self.k_weight = nn.Parameter(torch.empty_like(self.q_weight), requires_grad=False)
        self.register_buffer("gate_unrotate", gate_unrotate, persistent=False)

    @torch.no_grad()
    def load_weights(self, loader: ModelWeightLoader, layer_id: int) -> None:
        prefix = f"layers.{layer_id}.engram."
        for name, parameter in (
            ("wkv.weight", self.wkv.weight),
            ("q_weight", self.q_weight),
            ("k_weight", self.k_weight),
        ):
            tensor = loader.get_tensor(prefix + name)
            if tensor.shape != parameter.shape or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"{prefix}{name} must be a floating point tensor of shape {tuple(parameter.shape)}")
            parameter.copy_(tensor)

    def forward(
        self, hidden_states: torch.Tensor, rows: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens, hc_mult, dim = hidden_states.shape
        kv = self.wkv(rows.flatten(1).to(self.wkv.weight.dtype))
        keys = kv[:, : hc_mult * dim].reshape(tokens, hc_mult, dim).float()
        value = kv[:, hc_mult * dim :].float()
        hidden = hidden_states.float()
        gate_hidden = hidden
        if self.gate_unrotate is not None:
            width = self.gate_unrotate.shape[0]
            gate_hidden = (hidden.reshape(-1, width) @ self.gate_unrotate).reshape_as(hidden)
        dot = (gate_hidden * self.q_weight.float() * self.k_weight.float() * keys).sum(-1)
        dot = (
            dot
            * torch.rsqrt(gate_hidden.square().mean(-1) + self.eps)
            * torch.rsqrt(keys.square().mean(-1) + self.eps)
            * dim**-0.5
        )
        gate_input = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -gate_input, gate_input))
        if token_mask is not None:
            gate = torch.where(token_mask[:, None], gate, 0.0)
        # Exported V is already rotated. Only the gate's Q uses unrotation.
        return (hidden + gate[:, :, None] * value[:, None, :]).to(hidden_states.dtype)
