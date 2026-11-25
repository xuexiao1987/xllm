# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production-size attention against a full-history CPU float64 oracle."""

import os
from contextlib import nullcontext

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2CompressedInputs,
    Csa2EagerMetadata,
    csa2_eager_attention,
    csa2_prepare_ring,
)
from xllm.python.attention.csa2_npu import Csa2SmlaState


@pytest.fixture
def device() -> torch.device:
    target = torch.device(os.environ.get("XLLM_CSA2_TEST_DEVICE", "cpu"))
    if target.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(target)
    return target


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("previous,length", [(0, 1), (0, 5), (127, 1), (127, 3), (255, 1), (129, 129)])
def test_paged_attention_matches_full_history(
    previous: int, length: int, ratio: int, device: torch.device, smla_state: Csa2SmlaState
) -> None:
    generator = torch.Generator().manual_seed(73 + previous + length)
    total = previous + length
    query = torch.randn(length, 16, 512, generator=generator).to(torch.bfloat16)
    history = torch.randn(total, 512, generator=generator).to(torch.bfloat16)
    main = torch.randn(total // ratio if ratio else 0, 512, generator=generator).to(torch.bfloat16)
    sinks = torch.linspace(-1, 1, 16)
    scale = 512**-0.5
    keys = torch.cat((history, main)).double()
    positions = torch.arange(previous, total)
    logical = torch.arange(total)
    visible = (logical.unsqueeze(0) <= positions.unsqueeze(1)) & (logical.unsqueeze(0) > positions.unsqueeze(1) - 128)
    if ratio:
        visible = torch.cat((visible, torch.arange(main.size(0)) < ((positions + 1) // ratio).unsqueeze(1)), dim=1)
    scores = torch.einsum("thd,sd->ths", query.double(), keys) * scale
    scores = scores.masked_fill(~visible.unsqueeze(1), -torch.inf)
    scores = torch.cat((scores, sinks.double().view(1, -1, 1).expand(length, -1, -1)), dim=-1)
    expected = torch.einsum("ths,sd->thd", scores.softmax(-1)[..., :-1], keys).to(torch.bfloat16)

    swa = torch.full((3, 128, 512), torch.nan, dtype=torch.bfloat16)
    table = torch.tensor([[2, 0]], dtype=torch.int32)
    for position in range(max(0, previous - 256), previous):
        swa[table[0, position // 128 % 2], position % 128] = history[position]
    slots = table[0, positions // 128 % 2].long() * 128 + positions % 128
    before = swa.clone()
    cache = LayerCache(None, None, swa=swa.to(device))
    metadata = Csa2EagerMetadata((length,), (total,), table.to(device), slots.to(device))
    ring_plan = csa2_prepare_ring(metadata, 128, 3, 128)
    compressed = None
    if ratio:
        blocks = max(1, (main.size(0) + 127) // 128)
        key = torch.full((blocks + 1, 128, 512), torch.nan, dtype=torch.bfloat16)
        key_table = torch.arange(blocks, 0, -1, dtype=torch.int32).view(1, -1)
        for position, row in enumerate(main):
            key[key_table[0, position // 128], position % 128] = row
        # Includes future and out-of-capacity rows, which must be masked
        # before lookup. Width 513 exercises the second native sparse tile.
        picks = torch.arange(513, dtype=torch.int32).expand(length, -1)
        compressed = Csa2CompressedInputs(key.to(device), key_table.to(device), picks.to(device), ratio)
    profile = device.type == "npu" and (previous, length, ratio) == (127, 3, 2)
    capture = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    with capture as events:
        for _ in range(2):
            cache.swa.copy_(before.to(device))
            actual = csa2_eager_attention(
                query.to(device),
                history[previous:].to(device),
                cache,
                metadata,
                window_size=128,
                scale=scale,
                sinks=sinks.to(device),
                compressed=compressed,
                smla_state=smla_state,
                ring_plan=ring_plan,
            ).cpu()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    if profile:
        counts = {event.key: event.count for event in events.key_averages()}
        assert counts["xllm_ops::smla_va"] == 2
        assert counts["xllm_ops::smla_va_metadata"] == 1
    before.view(-1, 512).index_copy_(0, slots, history[previous:])
    torch.testing.assert_close(cache.swa.cpu(), before, rtol=0, atol=0, equal_nan=True)


@pytest.fixture(scope="module")
def smla_state() -> Csa2SmlaState:
    # Different contexts, query widths and compression ratios exercise the
    # metadata key; repeated calls exercise reuse without caching Q/KV values.
    return Csa2SmlaState()


def test_fused_geometry_rejects_missing_visible_compressed_page(device: torch.device) -> None:
    query = torch.zeros(1, 16, 512, dtype=torch.bfloat16, device=device)
    key = torch.zeros(1, 128, 512, dtype=torch.bfloat16, device=device)
    metadata = Csa2EagerMetadata(
        (1,),
        (1,),
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
    )
    compressed = Csa2CompressedInputs(
        key,
        torch.full((1, 1), -1, dtype=torch.int32, device=device),
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        1,
    )
    with pytest.raises(RuntimeError, match="no assigned physical block"):
        csa2_eager_attention(
            query,
            query[:, 0],
            LayerCache(None, None, swa=key.clone()),
            metadata,
            window_size=128,
            scale=512**-0.5,
            sinks=torch.zeros(16, device=device),
            compressed=compressed,
        )
