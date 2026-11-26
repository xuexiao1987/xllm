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

namespace xllm::layer {

// Owns a host shard independently of the checkpoint mmap. Lookup writes raw
// bytes into caller-owned staging buffers; device transfer and collectives
// remain responsibilities of the Engram layer.
class DeepseekV41EngramTable final {
 public:
  DeepseekV41EngramTable(const torch::Tensor& weight,
                         const torch::Tensor& scale,
                         int64_t rank,
                         int64_t world_size);
  DeepseekV41EngramTable(const DeepseekV41EngramTable&) = delete;
  DeepseekV41EngramTable& operator=(const DeepseekV41EngramTable&) = delete;
  DeepseekV41EngramTable(DeepseekV41EngramTable&&) = default;
  DeepseekV41EngramTable& operator=(DeepseekV41EngramTable&&) = default;

  int64_t begin() const { return begin_; }
  int64_t end() const { return end_; }
  int64_t head_dim() const { return weight_.size(1); }
  int64_t scale_dim() const { return scale_.size(1); }

  // Non-owned hashes produce zero rows for the subsequent rank reduction.
  // Buffers may be reused between calls, including empty batches.
  void lookup_rows(const torch::Tensor& hashes,
                   torch::Tensor& weight_rows,
                   torch::Tensor& scale_rows) const;

 private:
  int64_t begin_;
  int64_t end_;
  torch::Tensor weight_;
  torch::Tensor scale_;
};

// Dequantize E4M3FN weights with UE8M0 scales. The lookup tensor is created
// once per device by the layer; no FP8 arithmetic is required on MLU.
torch::Tensor deepseek_v41_fp8_values(const torch::Device& device);
torch::Tensor dequantize_deepseek_v41_engram_rows(
    const torch::Tensor& weight_rows,
    const torch::Tensor& scale_rows,
    const torch::Tensor& fp8_values);

}  // namespace xllm::layer
