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

#include "core/framework/model/execution_batch_metadata.h"

namespace xllm::layer {

// Borrowed views from the source layer's base KV owner. State is packed
// [pages, 32, 2 * dim]; its page IDs come from the SWA block table.
struct Csa2CompressionCacheView {
  torch::Tensor main;
  torch::Tensor state;
  torch::Tensor main_block_table;
  torch::Tensor state_block_table;
};

struct Csa2CompressedRows {
  torch::Tensor latent;
  torch::Tensor positions;
  torch::Tensor slots;
};

// Eager compression of projected rows. Returns normalized, unrotated latent
// for index K and commits rotated main KV. Owns neither weights nor KV pages.
class Csa2Compressor final {
 public:
  Csa2Compressor(int64_t ratio,
                 int64_t block_size,
                 int64_t rope_dim,
                 double norm_eps,
                 bool save_full_tail = false);
  Csa2CompressedRows forward(const torch::Tensor& raw,
                             const torch::Tensor& scores,
                             const torch::Tensor& norm_weight,
                             const torch::Tensor& rope,
                             const ExecutionBatchMetadata& batch,
                             const Csa2CompressionCacheView& cache) const;

 private:
  int64_t ratio_;
  int64_t block_size_;
  int64_t rope_dim_;
  double norm_eps_;
  bool save_full_tail_;
};

}  // namespace xllm::layer
