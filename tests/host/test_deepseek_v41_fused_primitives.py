# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model normalization and rotary operations against independent CPU math."""

import os

import pytest
import torch

from xllm.python.layers.rotary_embedding import apply_interleaved_rotary, prepare_interleaved_rotary
from xllm.python.models.deepseek_v41 import _RMSNorm


@pytest.fixture(scope="module")
def device() -> torch.device:
    target = os.environ.get("XLLM_V41_MODEL_TEST_DEVICE", "cpu")
    if target.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(target)
    return torch.device(target)


@pytest.mark.parametrize("rows", [1, 4, 129])
@pytest.mark.parametrize("width", [128, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_against_double_reference(device: torch.device, rows: int, width: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(817)
    hidden = (torch.randn(rows, width, generator=generator) * torch.logspace(-2, 2, rows)[:, None]).to(dtype)
    weight = torch.randn(width, generator=generator).to(dtype)
    norm = _RMSNorm(width, 1e-6, dtype, device)
    with torch.no_grad():
        norm.weight.copy_(weight.to(device))
    expected = hidden.double() / torch.sqrt(hidden.double().square().mean(-1, keepdim=True) + norm.eps)
    expected = (expected * weight.double()).to(dtype)
    actual = norm(hidden.to(device)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.008 if dtype == torch.bfloat16 else 2e-5, atol=2e-5)
    assert torch.equal(norm(torch.zeros_like(hidden, device=device)).cpu(), torch.zeros_like(hidden))


@pytest.mark.parametrize("rows", [1, 4, 129])
@pytest.mark.parametrize("heads", [None, 4])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("precompute", [False, True])
def test_partial_rotary_preserves_input_and_prefix(
    device: torch.device, rows: int, heads: int | None, inverse: bool, strided: bool, precompute: bool
) -> None:
    generator = torch.Generator().manual_seed(419)
    shape = (rows, 128) if heads is None else (rows, heads, 128)
    hidden = torch.randn(shape, generator=generator).to(torch.bfloat16)
    angles = torch.randn(rows + 7, 32, generator=generator)
    positions = torch.arange(rows) + 7
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    pairs = torch.view_as_complex(hidden[..., -64:].float().reshape(*shape[:-1], 32, 2).contiguous())
    angle = angles[positions] * (-1 if inverse else 1)
    phase = torch.polar(torch.ones_like(angle), angle)
    if heads is not None:
        phase = phase[:, None]
    expected = torch.view_as_real(pairs * phase).flatten(-2).to(hidden.dtype)
    value = hidden.to(device)
    if strided:
        backing = torch.zeros((*shape[:-1], 256), dtype=hidden.dtype, device=device)
        backing[..., ::2].copy_(value)
        value = backing[..., ::2]
        assert not value.is_contiguous()
    device_positions, device_table = positions.to(device), table.to(device)
    prepared = prepare_interleaved_rotary(device_positions, device_table, value.dtype) if precompute else None
    expanded_before = tuple(t.clone() for t in prepared.expanded) if prepared is not None and prepared.expanded else ()
    actual = apply_interleaved_rotary(value, device_positions, device_table, inverse=inverse, prepared=prepared)
    if prepared is not None:
        torch.testing.assert_close(prepared.rows.cpu(), table[positions], rtol=0, atol=0)
        for actual_coefficients, original_coefficients in zip(prepared.expanded or (), expanded_before, strict=True):
            torch.testing.assert_close(actual_coefficients, original_coefficients, rtol=0, atol=0)
    assert actual.data_ptr() != value.data_ptr()
    torch.testing.assert_close(value.cpu(), hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu()[..., :64], hidden[..., :64], rtol=0, atol=0)
    # The fused kernel rounds coefficients to BF16, matching the internal
    # runtime contract. Check against complex multiplication, not the kernel.
    torch.testing.assert_close(actual.cpu()[..., -64:], expected, rtol=2**-6, atol=2**-6)


def test_empty_normalization_and_rotary(device: torch.device) -> None:
    value = torch.empty((0, 128), dtype=torch.bfloat16, device=device)
    norm = _RMSNorm(128, 1e-20, value.dtype, device)
    assert norm(value).shape == value.shape
    table = torch.zeros((1, 64), device=device)
    positions = torch.empty(0, dtype=torch.int64, device=device)
    assert apply_interleaved_rotary(value, positions, table).shape == value.shape
