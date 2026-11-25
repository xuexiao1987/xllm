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

"""Engram numerical contracts, including a pinned internal-reference hash vector."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v41_engram import (
    EngramEmbedding,
    EngramHasher,
    EngramLayout,
    EngramProjection,
    engram_hash_multipliers,
    engram_token_windows,
    load_engram_gate_unrotate,
)


class _State:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def _loader(tensors: dict[str, torch.Tensor]) -> ModelWeightLoader:
    return ModelWeightLoader(nn.Module(), [_State(tensors)], tp_rank=0, tp_size=1)


def _layout() -> EngramLayout:
    return EngramLayout.build(
        [1], [40], max_ngram_size=3, n_heads=2, head_dim=32, vocab_size=5, compressed_vocab_size=8, pad_token_id=2
    )


def test_hashes_match_pinned_internal_reference() -> None:
    fixture = json.loads((Path(__file__).parent / "data/deepseek_v41_engram_hash.json").read_text())
    layout = EngramLayout.build(
        [1, 14],
        [384006168, 384016682],
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        vocab_size=16000000,
        compressed_vocab_size=8,
        pad_token_id=2,
    )
    hasher = EngramHasher(layout, torch.arange(8), torch.tensor(fixture["multipliers"]))
    assert layout.head_sizes == tuple(tuple(row) for row in fixture["head_sizes"])
    windows = engram_token_windows(
        torch.tensor(fixture["inputs"]), torch.tensor(fixture["query_start_loc"]), torch.tensor(fixture["history"])
    )
    torch.testing.assert_close(hasher(windows), torch.tensor(fixture["hashes"], dtype=torch.int32))


def test_packed_chunk_and_decode_match_whole_prefill() -> None:
    hasher = EngramHasher(_layout(), torch.arange(8))
    full = hasher(
        engram_token_windows(torch.tensor([3, 4, 5, 6, 7, 1]), torch.tensor([0, 4, 6]), torch.full((2, 2), -1))
    )
    first = hasher(engram_token_windows(torch.tensor([3, 4, 7]), torch.tensor([0, 2, 3]), torch.full((2, 2), -1)))
    second = hasher(
        engram_token_windows(torch.tensor([5, 6, 1]), torch.tensor([0, 2, 3]), torch.tensor([[4, 3], [7, -1]]))
    )
    torch.testing.assert_close(first, full[[0, 1, 4]])
    torch.testing.assert_close(second, full[[2, 3, 5]])
    # Empty padded requests must not attach this decode token to another row.
    decode = hasher(
        engram_token_windows(torch.tensor([6]), torch.tensor([0, 0, 1, 1]), torch.tensor([[-1, -1], [5, 4], [-1, -1]]))
    )
    torch.testing.assert_close(decode, full[3:4])


def test_dead_history_cuts_off_older_tokens_and_input_mutation_isolated() -> None:
    token_map = torch.arange(8)
    hasher = EngramHasher(_layout(), token_map)
    token_map.fill_(0)
    result = hasher(torch.tensor([[5, -1, 3], [5, -1, 7], [5, 2, 2]]))
    torch.testing.assert_close(result[0], result[1])
    torch.testing.assert_close(result[0], result[2])
    assert hasher(
        engram_token_windows(
            torch.empty(0, dtype=torch.int64), torch.tensor([0]), torch.empty((0, 2), dtype=torch.int64)
        )
    ).shape == (0, 1, 4)


def test_incorrect_sidecar_rejected() -> None:
    with pytest.raises(ValueError, match="multipliers"):
        EngramHasher(_layout(), torch.arange(8), torch.ones((1, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match="vocabulary"):
        EngramHasher(_layout(), torch.arange(7))
    with pytest.raises(ValueError, match="negative"):
        EngramHasher(_layout(), torch.tensor([-1, 1, 2]))


@pytest.mark.parametrize("override", [{"n_heads": 0}, {"head_dim": 31}, {"max_ngram_size": 1}, {"vocab_size": 2**32}])
def test_invalid_layout_rejected(override: dict[str, int]) -> None:
    args = dict(max_ngram_size=3, n_heads=2, head_dim=32, vocab_size=5, compressed_vocab_size=8, pad_token_id=2)
    args.update(override)
    with pytest.raises(ValueError):
        EngramLayout.build([1], [40], **args)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("raw_bytes", [False, True])
def test_head_shards_load_and_reassemble_lookup(tp_size: int, raw_bytes: bool) -> None:
    table = (torch.arange(40 * 32).reshape(40, 32) % 256).to(torch.uint8).view(torch.int8)
    scales = torch.arange(1, 41, dtype=torch.float32).reshape(40, 1) / 8
    tensors = {
        "layers.1.engram.embed.weight": table.view(torch.uint8) if raw_bytes else table,
        "layers.1.engram.embed.scale": scales,
    }
    hashes = EngramHasher(_layout(), torch.arange(8))(torch.tensor([[3, 2, 1], [4, 3, 2]]))[:, 0]
    shards = []
    covered = []
    for rank in range(tp_size):
        embed = EngramEmbedding(_layout(), 0, tp_size=tp_size, tp_rank=rank, device=torch.device("cpu"))
        embed.load_weights(_loader(tensors), chunk_rows=3)
        shards.append(embed(hashes))
        covered.extend(range(embed.row_start, embed.row_start + embed.row_count))
    assert covered == list(range(36))  # Bucket sizes 5,7,11,13; four unused trailing rows.
    expected = (table.float() * scales).to(torch.bfloat16)[hashes.to(torch.int64)]
    torch.testing.assert_close(torch.cat(shards, dim=1), expected)


def test_ue8m0_decodes_zero_and_only_owned_rows() -> None:
    scale = torch.full((40, 1), 255, dtype=torch.uint8)
    scale[12:36] = 127
    scale[12] = 0
    scale[13] = 126
    scale[14] = 128
    embed = EngramEmbedding(_layout(), 0, tp_size=2, tp_rank=1, device=torch.device("cpu"))
    tensors = {
        "layers.1.engram.embed.weight": torch.ones((40, 32), dtype=torch.int8),
        "layers.1.engram.embed.scale": scale,
    }
    embed.load_weights(_loader(tensors), chunk_rows=2)
    torch.testing.assert_close(embed.weight_scale[:3, 0], torch.tensor([2**-127, 0.5, 2.0]), rtol=0, atol=0)
    scale[12] = 255
    with pytest.raises(ValueError, match="NaN"):
        embed.load_weights(_loader(tensors))


@pytest.mark.parametrize("bad", ["dtype", "rows", "width", "scale"])
def test_malformed_table_rejected(bad: str) -> None:
    table = torch.zeros((40, 32), dtype=torch.int8)
    scale = torch.ones((40, 1))
    if bad == "dtype":
        table = table.float()
    elif bad == "rows":
        table = table[:39]
    elif bad == "width":
        table = table[:, :31]
    else:
        scale = scale[:39]
    with pytest.raises(ValueError):
        EngramEmbedding(_layout(), 0, tp_size=1, tp_rank=0, device=torch.device("cpu")).load_weights(
            _loader({"layers.1.engram.embed.weight": table, "layers.1.engram.embed.scale": scale})
        )


def _projection(rotation: torch.Tensor | None = None) -> EngramProjection:
    projection = EngramProjection(
        input_size=1,
        hidden_size=2,
        hc_mult=2,
        eps=1e-6,
        dtype=torch.float32,
        device=torch.device("cpu"),
        gate_unrotate=rotation,
    )
    projection.load_weights(
        _loader(
            {
                "layers.1.engram.wkv.weight": torch.tensor([[1.0], [0.0], [-1.0], [0.0], [2.0], [3.0]]),
                "layers.1.engram.q_weight": torch.ones(2, 2),
                "layers.1.engram.k_weight": torch.ones(2, 2),
            }
        ),
        1,
    )
    return projection


def test_gate_sign_zero_clamp_and_mask() -> None:
    projection = _projection()
    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]])
    output = projection(hidden, torch.ones(3, 1, 1), torch.tensor([True, True, False]))
    dot = 1 / (0.5 + 1e-6) / math.sqrt(2)
    positive = 1 / (1 + math.exp(-math.sqrt(dot)))
    negative = 1 - positive
    torch.testing.assert_close(
        output[0], hidden[0] + torch.tensor([[positive * 2, positive * 3], [negative * 2, negative * 3]])
    )
    # Zero follows the positive clamped branch, not exactly sigmoid(0).
    zero = 1 / (1 + math.exp(-0.001))
    torch.testing.assert_close(output[1], torch.tensor([[zero * 2, zero * 3], [zero * 2, zero * 3]]))
    torch.testing.assert_close(output[2], hidden[2])


def test_quarot_unrotates_gate_but_not_value_and_repeated_batches_use_new_rows() -> None:
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    projection = _projection(rotation)
    rotated = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    expected_increment = _projection()(torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]), torch.ones(1, 1, 1)) - torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]]]
    )
    output = projection(rotated, torch.ones(1, 1, 1))
    torch.testing.assert_close(output - rotated, expected_increment)
    # Same row count cannot justify reusing old embeddings.
    torch.testing.assert_close(projection(rotated, torch.zeros(1, 1, 1)), rotated)
    assert not any(parameter.requires_grad for parameter in projection.parameters())


def test_bad_projection_weight_rejected() -> None:
    projection = _projection()
    with pytest.raises(ValueError, match="floating point"):
        projection.load_weights(_loader({"layers.1.engram.wkv.weight": torch.ones((6, 1), dtype=torch.int8)}), 1)


def test_hash_sidecar_is_mandatory_and_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EngramHasher.from_checkpoint(_layout(), tmp_path, 8)
    state = {"token_map": torch.arange(8), "multipliers": engram_hash_multipliers(_layout())}
    torch.save(state, tmp_path / "engram_hash_state.bin")
    loaded = EngramHasher.from_checkpoint(_layout(), tmp_path, 8)
    torch.testing.assert_close(
        loaded(torch.tensor([[3, 2, 1]])), EngramHasher(_layout(), torch.arange(8))(torch.tensor([[3, 2, 1]]))
    )
    with pytest.raises(ValueError, match="vocabulary"):
        EngramHasher.from_checkpoint(_layout(), tmp_path, 9)
    state["multipliers"] += 2
    torch.save(state, tmp_path / "engram_hash_state.bin")
    with pytest.raises(ValueError, match="multipliers"):
        EngramHasher.from_checkpoint(_layout(), tmp_path, 8)


@pytest.mark.parametrize("compact", [True, False])
def test_quarot_sidecar_transpose_and_exact_block_detection(tmp_path: Path, compact: bool) -> None:
    from safetensors.torch import save_file

    config = dict(
        value_projection_rotated=True,
        value_basis="quarot_global",
        key_and_gate_basis="original",
        runtime_delta_rotation=False,
    )
    assert load_engram_gate_unrotate(tmp_path, {}) is None
    with pytest.raises(FileNotFoundError):
        load_engram_gate_unrotate(tmp_path, config)
    block = torch.eye(32).roll(1, 0)
    rotation = torch.block_diag(block, block)
    if not compact:
        rotation = rotation.roll(1, 0)
    (tmp_path / "optional").mkdir()
    save_file({"global_rotation": rotation}, tmp_path / "optional/quarot.safetensors")
    actual = load_engram_gate_unrotate(tmp_path, config)
    torch.testing.assert_close(actual, block.t() if compact else rotation.t())
    config["runtime_delta_rotation"] = True
    with pytest.raises(ValueError, match="folded"):
        load_engram_gate_unrotate(tmp_path, config)


def test_lookup_projection_end_to_end_is_tp_invariant() -> None:
    layout = _layout()
    windows = engram_token_windows(torch.tensor([3, 4, 5]), torch.tensor([0, 3]), torch.full((1, 2), -1))
    hashes = EngramHasher(layout, torch.arange(8))(windows)[:, 0]
    source = {
        "layers.1.engram.embed.weight": (torch.arange(40 * 32).reshape(40, 32) % 127).to(torch.int8),
        "layers.1.engram.embed.scale": torch.ones(40, 1) / 100,
    }
    projection = EngramProjection(
        input_size=128, hidden_size=2, hc_mult=2, eps=1e-6, dtype=torch.float32, device=torch.device("cpu")
    )
    projection.load_weights(
        _loader(
            {
                "layers.1.engram.wkv.weight": torch.sin(torch.arange(6 * 128).reshape(6, 128)) / 10,
                "layers.1.engram.q_weight": torch.ones(2, 2),
                "layers.1.engram.k_weight": torch.ones(2, 2),
            }
        ),
        1,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 5
    outputs = []
    for tp in (1, 2, 4):
        rows = []
        for rank in range(tp):
            embed = EngramEmbedding(layout, 0, tp_size=tp, tp_rank=rank, device=torch.device("cpu"))
            embed.load_weights(_loader(source))
            rows.append(embed(hashes))
        outputs.append(projection(hidden, torch.cat(rows, dim=1)))
    for actual in outputs[1:]:
        torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)
