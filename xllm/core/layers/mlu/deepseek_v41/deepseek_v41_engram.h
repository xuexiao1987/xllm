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

// Forward-scoped host inputs shared by all Engram layers. CPU int64 inputs
// retain their storage; device inputs are copied once before traversing layers.
class DeepseekV41EngramInput final {
 public:
  DeepseekV41EngramInput() = default;
  DeepseekV41EngramInput(const torch::Tensor& tokens,
                         const torch::Tensor& history);
  torch::Tensor tokens_;
  torch::Tensor history_;
};

// One layer's immutable trained hash parameters. Inputs use the base Batch
// Builder's nearest-token-first Engram history, not a private request cache.
class DeepseekV41EngramHash final {
 public:
  DeepseekV41EngramHash(const torch::Tensor& token_map,
                        const torch::Tensor& multipliers,
                        const torch::Tensor& primes,
                        const torch::Tensor& offsets,
                        int64_t pad_token_id,
                        int64_t compressed_vocab_size,
                        int64_t table_rows);
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& history,
                        const std::vector<int32_t>& query_lengths) const;

 private:
  torch::Tensor token_map_, multipliers_, primes_, offsets_;
  int64_t pad_token_id_;
  int64_t depth_;
  int64_t heads_;
};

// Runs after the table rows have been reduced across lookup ranks. This
// numerical boundary does not own a communicator, stream or request state.
torch::Tensor deepseek_v41_engram_project(const torch::Tensor& hidden,
                                          const torch::Tensor& rows,
                                          const torch::Tensor& tokens,
                                          const torch::Tensor& wkv,
                                          const torch::Tensor& q_weight,
                                          const torch::Tensor& k_weight,
                                          double eps);

}  // namespace xllm::layer
