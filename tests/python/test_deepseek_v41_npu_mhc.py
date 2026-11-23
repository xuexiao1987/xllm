# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

"""Execute the real V4.1 VA kernels against the CPU model math."""

import pytest
import torch

from xllm.python.models.deepseek_v41_mhc import DeepseekV41HyperConnection


@pytest.mark.parametrize("hidden_size", [128, 5120])
@pytest.mark.parametrize("rows", [1, 4, 129])
@pytest.mark.parametrize("first", [False, True])
@torch.no_grad()
def test_npu_mhc_composition(hidden_size: int, rows: int, first: bool) -> None:
    torch.manual_seed(1234)
    params = dict(hidden_size=hidden_size, hc_mult=4, sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6)
    cpu = DeepseekV41HyperConnection(**params, device=torch.device("cpu"))
    npu = DeepseekV41HyperConnection(**params, device=torch.device("npu:0"))
    assert npu._fused_pre is not None and npu._fused_post is not None
    for name, value in cpu.named_parameters():
        value.normal_(std=0.02)
        getattr(npu, name).copy_(value)
    hidden = torch.randn(rows, 4, hidden_size, dtype=torch.bfloat16)
    previous = None if first else torch.sigmoid(torch.randn(rows, 4))
    device_previous = None if previous is None else previous.npu()
    expected = cpu(hidden, previous, part="attn")
    actual = npu(hidden.npu(), device_previous, part="attn")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        assert torch.isfinite(left).all()
        # BF16 output uses the existing VA-post operator test's 1% bound.
        # FP32 coefficients use a stricter bound; no passing-element fraction.
        tolerance = 1e-2 if index == 0 else 1e-5
        torch.testing.assert_close(left.cpu(), right, atol=tolerance, rtol=tolerance)
    expected_hidden = cpu.expand(expected[0], hidden, expected[2], expected[3])
    actual_hidden = npu.expand(actual[0], hidden.npu(), actual[2], actual[3])
    torch.testing.assert_close(actual_hidden.cpu(), expected_hidden, atol=1e-2, rtol=1e-2)
    # Exercise the next sublayer with the mix exported by this sublayer.
    expected_ffn = cpu(expected_hidden, expected[1], part="ffn")
    actual_ffn = npu(actual_hidden, actual[1], part="ffn")
    torch.testing.assert_close(actual_ffn[0].cpu(), expected_ffn[0], atol=1e-2, rtol=1e-2)
