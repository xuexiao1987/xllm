/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <vector>

namespace xllm::layer {

struct DeepseekV41HashOptions {
  std::vector<int32_t> layer_ids;
  std::vector<int64_t> num_embeddings;
  int64_t max_ngram_size = 4;
  int64_t n_heads = 8;
  int64_t bucket_vocab_size = 0;
  int64_t compressed_vocab_size = 0;
};

struct DeepseekV41HashParameters {
  torch::Tensor multipliers;
  torch::Tensor primes;
  torch::Tensor offsets;
};

// Deterministic checkpoint parameters. PCG64 and bucket traversal must match
// training exactly; these values are not interchangeable random seeds.
DeepseekV41HashParameters build_deepseek_v41_hash_parameters(
    const DeepseekV41HashOptions& options);

}  // namespace xllm::layer
