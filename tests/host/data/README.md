<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# V4.1 numerical fixtures

The JSON fixtures were generated from internal commit
`5026beb058c166cdd108f7feddc7dbbec317485d`, independently of the new implementation.
Their `reference_path` identifies the source; inputs and outputs are embedded.

- `deepseek_v41_engram_hash.json`: original `EngramLayout`,
  `compute_hash_multipliers` and `EngramHashState.forward`, with no slot cache,
  explicit history, two packed requests, layer ids 1/14, real bucket geometry,
  and a small identity token map. It checks integer hashes exactly.
- `deepseek_v41_mhc.json`: original eager `DeepseekV41HyperConnection`
  `hc_mixes`, `hc_pre` and `hc_post`, with FP32 activations, four streams,
  hidden width three and 20 Sinkhorn iterations. All six parameter tensors and
  external pre coefficients are embedded. Accelerator dispatch was disabled
  while generating the fixture; no production implementation was replaced.
- `deepseek_v41_moe.json`: original eager `DeepseekV41MoE.forward` for FP32
  and BF16, four routed experts/top-2 and one shared expert. Includes source
  weights and expected output. Routing multiplies the clamped activation
  before casting and the down projection; BF16 comparison is exact.

The test suite loads these static expected values. It does not import, copy or
execute a second implementation of the algorithms as its runtime oracle.
