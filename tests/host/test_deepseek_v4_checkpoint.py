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

"""Projection discovery owns prefix and alias precedence at the checkpoint boundary."""

from collections import Counter

import pytest

from xllm.python.models.deepseek_v4_checkpoint import resolve_dense_mlp_checkpoint, resolve_mlp_projection_names


class _Keys:
    def __init__(self, keys: set[str]) -> None:
        self.keys = keys
        self.probes: Counter[str] = Counter()

    def has(self, key: str) -> bool:
        self.probes[key] += 1
        return key in self.keys


def _weights(prefix: str, names: tuple[str, str, str]) -> set[str]:
    return {prefix + name + ".weight" for name in names}


@pytest.mark.parametrize("prefix", ["ffn.", "mlp."])
@pytest.mark.parametrize("names", [("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")])
def test_dense_checkpoint_aliases_resolve_once(prefix: str, names: tuple[str, str, str]) -> None:
    loader = _Keys(_weights("layers.0." + prefix, names))
    assert resolve_dense_mlp_checkpoint(loader, "layers.0.") == ("layers.0." + prefix, names)
    assert max(loader.probes.values()) == 1


def test_prefix_precedes_projection_alias_and_native_precedes_legacy() -> None:
    native, legacy = ("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")
    keys = _weights("l.ffn.", legacy) | _weights("l.mlp.", native)
    assert resolve_dense_mlp_checkpoint(_Keys(keys), "l.") == ("l.ffn.", legacy)
    keys |= _weights("l.ffn.", native)
    assert resolve_dense_mlp_checkpoint(_Keys(keys), "l.") == ("l.ffn.", native)


def test_incomplete_prefix_is_skipped_but_missing_triplet_preserves_error() -> None:
    names = ("gate_proj", "up_proj", "down_proj")
    keys = {"l.ffn.gate_proj.weight"} | _weights("l.mlp.", names)
    assert resolve_dense_mlp_checkpoint(_Keys(keys), "l.") == ("l.mlp.", names)
    with pytest.raises(KeyError) as error:
        resolve_dense_mlp_checkpoint(_Keys({"l.ffn.gate_proj.weight"}), "l.")
    assert error.value.args == ("DeepSeek-V4 dense MLP weights not found under l.",)
    with pytest.raises(KeyError, match="MLP projections not found under l.ffn."):
        resolve_mlp_projection_names(_Keys(set()), "l.ffn.")
