# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable and backend-owned layers for Python model execution.

Simple layers use the active :mod:`python.kernels` package directly. Complex
models whose native fusion boundaries differ by device keep their lowering in
``layers/<device>/`` and select it once when the model is constructed.
"""

from importlib import import_module

# Import only the requested layer. Tensor-only layers must remain usable for
# host validation before native operators and the device runtime are loaded.
_LAYER_EXPORTS = {
    "Attention": "attention",
    "DFlash2GroupedConv": "dflash2_grouped_conv",
    "HiddenParallelEmbedding": "embedding",
    "FusedMoE": "fused_moe",
    "GatedMLP": "gated_mlp",
    "GemmaRMSNorm": "layernorm",
    "RMSNorm": "layernorm",
    "ColumnParallelLinear": "linear",
    "RowParallelLinear": "linear",
    "RotaryEmbedding": "rotary_embedding",
    "apply_rotary_half": "rotary_embedding",
    "gather_half_rope_cos_sin": "rotary_embedding",
}


def __getattr__(name: str) -> object:
    if name not in _LAYER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAYER_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


__all__ = [
    "Attention",
    "DFlash2GroupedConv",
    "FusedMoE",
    "GatedMLP",
    "RMSNorm",
    "GemmaRMSNorm",
    "RotaryEmbedding",
    "apply_rotary_half",
    "gather_half_rope_cos_sin",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "HiddenParallelEmbedding",
]
