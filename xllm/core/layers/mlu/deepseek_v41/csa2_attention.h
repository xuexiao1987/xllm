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
#include <optional>
#include <vector>

#include "core/framework/model/execution_batch_metadata.h"
#include "core/kv_cache/layout/csa2_cache_plan.h"

namespace xllm::layer {

enum class Csa2CandidateMode : int8_t { NONE, PRODUCE, CONSUME };

struct Csa2IndexOptions {
  int64_t ratio = 1;
  int64_t topk = 512;
  Csa2CandidateMode candidate_mode = Csa2CandidateMode::NONE;
  int64_t candidate_block_size = 0;
  int64_t candidate_topk_blocks = 0;
};

struct Csa2Selection {
  torch::Tensor indices;
  torch::Tensor candidates;
  // Nonzero only when the producer selected every block in this key span.
  int64_t complete_candidate_blocks = 0;
};

// Eager selection. The model owns cross-layer selections; this
// object owns no KV pages or mutable request state.
class Csa2Indexer final {
 public:
  explicit Csa2Indexer(Csa2IndexOptions options);
  Csa2Selection select(const torch::Tensor& query,
                       const torch::Tensor& weights,
                       const torch::Tensor& keys,
                       int64_t query_begin,
                       const torch::Tensor& candidates = {},
                       int64_t complete_candidate_blocks = 0) const;
  Csa2Selection select_decode(const torch::Tensor& query,
                              const torch::Tensor& weights,
                              const torch::Tensor& cache,
                              const torch::Tensor& table,
                              int64_t query_begin,
                              int64_t block_size,
                              const torch::Tensor& candidates = {},
                              int64_t complete_candidate_blocks = 0) const;

  void select_decode_batch(const torch::Tensor& query,
                           const torch::Tensor& weights,
                           const torch::Tensor& cache,
                           const torch::Tensor& table,
                           const std::vector<int32_t>& query_begin,
                           int64_t block_size,
                           std::vector<Csa2Selection>& selections) const;

 private:
  Csa2IndexOptions options_;
};

// Projected q and KV, after RoPE. Window is the old SWA tail concatenated
// with the current chunk, read before the ring is overwritten. Indices are
// logical compressed-token indices, with -1 denoting a masked selection.
torch::Tensor csa2_sparse_attention(const torch::Tensor& query,
                                    const torch::Tensor& window,
                                    const torch::Tensor& compressed,
                                    const torch::Tensor& indices,
                                    const torch::Tensor& sink,
                                    int64_t query_begin,
                                    int64_t window_size);

// Shallow views of the existing KV owner. For Reindex/Reuse, main and index
// reference the plan's Full source layer while SWA belongs to this layer.
struct Csa2LayerCacheView {
  torch::Tensor swa;
  torch::Tensor main;
  torch::Tensor index;
  torch::Tensor swa_block_table;
  torch::Tensor main_block_table;
};

// Eager projected attention with base batch metadata and caller-owned KV.
// Source compression/projections run before this boundary. The forward state
// is a vector per batch, passed between layers and discarded after the batch.
class Csa2Attention final {
 public:
  Csa2Attention(Csa2CachePlan::Mode mode,
                int64_t window_size,
                int64_t block_size,
                Csa2IndexOptions index_options);
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& kv,
                        const torch::Tensor& index_query,
                        const torch::Tensor& index_weights,
                        const torch::Tensor& sink,
                        const ExecutionBatchMetadata& batch,
                        const Csa2LayerCacheView& cache,
                        std::vector<Csa2Selection>& selections) const;

 private:
  Csa2CachePlan::Mode mode_;
  int64_t window_size_;
  int64_t block_size_;
  int64_t ratio_;
  int64_t topk_;
  std::optional<Csa2Indexer> indexer_;
};

}  // namespace xllm::layer
