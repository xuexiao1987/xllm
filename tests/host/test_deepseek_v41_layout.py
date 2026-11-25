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

"""V4.1 cache ownership contracts independent of device initialization."""

from dataclasses import FrozenInstanceError

import pytest

from xllm.python.models.deepseek_v41_layout import (
    CSA2LayerPlan,
    parse_compress_ratios,
)


def test_reference_flash_layout_preserves_cache_ownership() -> None:
    # Fixed 40-layer checkpoint geometry, including its three draft entries.
    ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    plan = CSA2LayerPlan.build(40, ratios, [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36])
    assert len(plan) == 40
    assert [plan.mode(i).main_kv_source_layer for i in range(40)] == (
        [-1, -1] + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
    )
    assert [i for i in range(40) if plan.mode(i).is_full] == [2, 8, 14, 20]
    assert [i for i in range(40) if plan.mode(i).is_reindex] == [24, 28, 32, 36]
    assert [i for i in range(40) if not plan.mode(i).is_csa2] == [0, 1]
    assert sum(plan.mode(i).is_reuse for i in range(40)) == 30
    assert [plan.mode(i).compress_ratio for i in range(40)] == ratios[:40]


def test_zero_ratio_breaks_cache_consumption_without_changing_later_owner() -> None:
    plan = CSA2LayerPlan.build(6, [0, 2, 2, 0, 2, 2], [1], [1, 4])
    assert [plan.mode(i).mode for i in range(6)] == ["swa", "full", "reuse", "swa", "reindex", "reuse"]
    assert [plan.mode(i).main_kv_source_layer for i in range(6)] == [-1, 1, 1, -1, 1, 1]


def test_plan_copies_inputs() -> None:
    ratios = [0, 2, 2, 0, 0]
    sources = [1]
    plan = CSA2LayerPlan.build(5, ratios, sources, sources)
    ratios[1] = 0
    sources.clear()
    assert [plan.mode(i).mode for i in range(5)] == ["swa", "full", "reuse", "swa", "swa"]
    assert plan.mode(1).compress_ratio == 2
    with pytest.raises(FrozenInstanceError):
        plan.mode(1).main_kv_source_layer = 3


@pytest.mark.parametrize("ratio", [-1, 3, 4, 8, 128])
def test_invalid_ratio_is_rejected_even_in_draft_tail(ratio: int) -> None:
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4.1 compression ratio"):
        parse_compress_ratios([0, 2, ratio], 2)


@pytest.mark.parametrize(
    ("ratios", "kv_sources", "index_sources"),
    [
        ([1] * 6, [-1], [-1]),
        ([1] * 6, [6], [6]),
        ([1] * 6, [2], [2, 6]),
        ([1] * 6, [2], [4]),
        ([1] * 6, [4], [2, 4]),
        ([0] * 6, [2], [2]),
        ([0, 2], [4], [4]),
    ],
)
def test_invalid_cache_sources_are_rejected(ratios: list[int], kv_sources: list[int], index_sources: list[int]) -> None:
    with pytest.raises(ValueError):
        CSA2LayerPlan.build(6, ratios, kv_sources, index_sources)


def test_plan_rejects_out_of_range_access_and_negative_layer_count() -> None:
    plan = CSA2LayerPlan.build(2, [0, 0], [], [])
    for index in (-1, 2):
        with pytest.raises(IndexError):
            plan.mode(index)
    with pytest.raises(ValueError, match="n_layers"):
        CSA2LayerPlan.build(-1, [], [], [])


@pytest.mark.parametrize("sources", [[3, 1], [1, 1, 3]])
def test_source_order_and_duplicates_are_rejected(sources: list[int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CSA2LayerPlan.build(5, [0, 2, 2, 1, 1], sources, [1, 3, 4])
    with pytest.raises(ValueError, match="strictly increasing"):
        CSA2LayerPlan.build(5, [0, 2, 2, 1, 1], [1, 3], sources)


def test_incomplete_ratios_and_mismatched_consumers_are_rejected() -> None:
    with pytest.raises(ValueError, match="cover every"):
        CSA2LayerPlan.build(5, [0, 2, 2], [1], [1])
    with pytest.raises(ValueError, match="source compression ratio"):
        CSA2LayerPlan.build(3, [0, 2, 1], [1], [1])
