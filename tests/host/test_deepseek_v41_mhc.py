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

"""V4.1 mHC external pre-mix, checkpoint and numerical reference checks."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v41_mhc import DeepseekV41HyperConnection


class _State(dict[str, torch.Tensor]):
    def has(self, name: str) -> bool:
        return name in self

    def get_tensor(self, name: str) -> torch.Tensor:
        return self[name]


def _reference() -> tuple[DeepseekV41HyperConnection, dict]:
    data = json.loads((Path(__file__).parent / "data/deepseek_v41_mhc.json").read_text())
    model = DeepseekV41HyperConnection(
        hidden_size=3, hc_mult=4, sinkhorn_iters=20, norm_eps=1e-5, hc_eps=1e-6, device=torch.device("cpu")
    )
    state = _State({"layers.0." + key: torch.tensor(value) for key, value in data["weights"].items()})
    model.load_weights(ModelWeightLoader(nn.Module(), [state], tp_size=1, tp_rank=0), 0)
    return model, data


def test_matches_pinned_internal_reference() -> None:
    model, data = _reference()
    hidden, previous = torch.tensor(data["hidden"]), torch.tensor(data["previous_pre"])
    collapsed, pre, post, comb = model(hidden, previous, part="attn")
    for tensor, name in ((collapsed, "collapsed"), (pre, "pre"), (post, "post"), (comb, "combination")):
        torch.testing.assert_close(tensor, torch.tensor(data[name]))
    torch.testing.assert_close(model.expand(collapsed, hidden, post, comb), torch.tensor(data["expanded"]))
    torch.testing.assert_close(comb.sum(-1), torch.ones(2, 4), atol=2e-6, rtol=0)
    torch.testing.assert_close(comb.sum(1), torch.ones(2, 4), atol=2e-6, rtol=0)


def test_external_pre_mix_and_final_merge() -> None:
    model, data = _reference()
    hidden = torch.tensor(data["hidden"])
    first, attn_pre, post, comb = model(hidden, None, part="attn")
    torch.testing.assert_close(first, hidden[:, 0])
    expanded = model.expand(first, hidden, post, comb)
    ffn, ffn_pre, _, _ = model(expanded, attn_pre, part="ffn")
    torch.testing.assert_close(ffn, model.collapse(expanded, attn_pre))
    assert not torch.allclose(ffn, model.collapse(expanded, ffn_pre))
    final = model.collapse(expanded, ffn_pre)
    assert not torch.allclose(final, expanded.mean(1))


def test_residual_combination_orientation() -> None:
    residual = torch.tensor([[[1.0], [10.0], [100.0]]])
    # Each input routes to the next output; transpose would reverse this cycle.
    combination = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    model, _ = _reference()
    output = model.expand(torch.zeros(1, 1), residual, torch.ones(1, 3), combination)
    torch.testing.assert_close(output, torch.tensor([[[100.0], [1.0], [10.0]]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_boundary_and_empty_batch(dtype: torch.dtype) -> None:
    model, data = _reference()
    hidden = torch.tensor(data["hidden"]).repeat(70, 1, 1).to(dtype)
    previous = torch.tensor(data["previous_pre"]).repeat(70, 1)
    whole = model(hidden, previous, part="attn")
    left = model(hidden[:129], previous[:129], part="attn")
    right = model(hidden[129:], previous[129:], part="attn")
    for actual, first, last in zip(whole, left, right, strict=True):
        torch.testing.assert_close(actual, torch.cat((first, last)), rtol=0, atol=0)
    empty = model(hidden[:0], previous[:0], part="ffn")
    assert all(t.shape[0] == 0 for t in empty)
    assert whole[0].dtype == dtype


def test_invalid_mhc_weights_rejected() -> None:
    model, _ = _reference()
    state = _State(
        {
            "layers.0." + name: torch.zeros_like(parameter).to(torch.bfloat16)
            for name, parameter in model.named_parameters()
        }
    )
    with pytest.raises(ValueError, match="FP32"):
        model.load_weights(ModelWeightLoader(nn.Module(), [state], tp_size=1, tp_rank=0), 0)
    assert not any(p.requires_grad for p in model.parameters())
