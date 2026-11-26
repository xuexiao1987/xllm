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

#include "core/layers/mlu/deepseek_v41/csa2_compressor.h"

#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "core/kv_cache/layout/csa2_cache_plan.h"

#if defined(USE_MLU)
#include "core/kernels/mlu/mlu_ops_api.h"
#endif

namespace xllm::layer {
namespace {
// The state ring uses SWA page IDs but only 32 rows per page. Retain a
// strided packed view rather than flattening the separate KV/score slices.
torch::Tensor state_slots(const torch::Tensor& table,
                          int64_t seq,
                          int64_t begin,
                          int64_t end,
                          int64_t block_size) {
  const auto positions =
      torch::arange(begin, end, table.options().dtype(torch::kInt64));
  const auto columns =
      torch::floor_divide(positions, block_size).remainder(table.size(1));
  return table[seq].to(torch::kInt64).index_select(0, columns) *
             Csa2CachePlan::kStateRows +
         positions.remainder(Csa2CachePlan::kStateRows);
}
}  // namespace

Csa2Compressor::Csa2Compressor(int64_t ratio,
                               int64_t block_size,
                               int64_t rope_dim,
                               double norm_eps,
                               bool save_full_tail)
    : ratio_(ratio),
      block_size_(block_size),
      rope_dim_(rope_dim),
      norm_eps_(norm_eps),
      save_full_tail_(save_full_tail) {
  CHECK(ratio == 1 || ratio == 2);
  CHECK_GT(block_size, 0);
  CHECK_GE(rope_dim, 0);
  CHECK_EQ(rope_dim % 2, 0);
  CHECK_GT(norm_eps, 0);
  if (ratio == 2) {
    CHECK_EQ(block_size % Csa2CachePlan::kStateRows, 0);
  }
}

Csa2CompressedRows Csa2Compressor::forward(
    const torch::Tensor& raw,
    const torch::Tensor& scores,
    const torch::Tensor& norm_weight,
    const torch::Tensor& rope,
    const ExecutionBatchMetadata& batch,
    const Csa2CompressionCacheView& cache) const {
  CHECK_EQ(raw.dim(), 2);
  CHECK_GE(batch.num_reqs, 0);
  CHECK_GE(batch.num_tokens, 0);
  CHECK_LE(batch.num_tokens, raw.size(0));
  const int64_t dim = raw.size(1);
  CHECK_GT(dim, 0);
  CHECK_LE(rope_dim_, dim);
  CHECK_EQ(norm_weight.numel(), dim);
  Csa2CompressedRows result{
      torch::empty({0, dim}, raw.options().dtype(torch::kBFloat16)),
      torch::empty({0}, raw.options().dtype(torch::kInt64)),
      torch::empty({0}, raw.options().dtype(torch::kInt64))};
  if (batch.num_reqs == 0) {
    CHECK_EQ(batch.num_tokens, 0);
    return result;
  }
  CHECK_EQ(batch.num_scheduled_tokens.size(), batch.num_reqs);
  CHECK_EQ(batch.num_computed_tokens.size(), batch.num_reqs);
  CHECK_EQ(batch.query_start_loc.size(), batch.num_reqs + 1);
  CHECK_EQ(batch.query_start_loc.front(), 0);
  CHECK_EQ(batch.query_start_loc.back(), batch.num_tokens);
  for (int32_t seq = 0; seq < batch.num_reqs; ++seq) {
    CHECK_GE(batch.num_computed_tokens[seq], 0);
    CHECK_GE(batch.num_scheduled_tokens[seq], 0);
    CHECK_EQ(batch.query_start_loc[seq + 1] - batch.query_start_loc[seq],
             batch.num_scheduled_tokens[seq]);
  }
  if (batch.num_tokens == 0) {
    return result;
  }
  CHECK_EQ(cache.main.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cache.main.size(-1), dim);
  CHECK_EQ(cache.main_block_table.dim(), 2);
  CHECK_GE(cache.main_block_table.size(0), batch.num_reqs);
  if (rope_dim_ > 0) {
    CHECK_EQ(rope.dim(), 2);
    CHECK_EQ(rope.size(1), 2 * rope_dim_);
  }
  torch::Tensor state;
  if (ratio_ == 2) {
    CHECK_EQ(raw.scalar_type(), torch::kFloat32);
    CHECK_EQ(scores.scalar_type(), torch::kFloat32);
    CHECK(scores.sizes() == raw.sizes());
    CHECK_EQ(cache.state.scalar_type(), torch::kFloat32);
    CHECK_EQ(cache.state.dim(), 3);
    CHECK_EQ(cache.state.size(1), Csa2CachePlan::kStateRows);
    CHECK_EQ(cache.state.size(2), 2 * dim);
    CHECK_EQ(cache.state_block_table.dim(), 2);
    CHECK_GE(cache.state_block_table.size(0), batch.num_reqs);
    CHECK_GT(cache.state_block_table.size(1), 0);
    state = cache.state.view({-1, 2 * dim});
  } else {
    CHECK_EQ(raw.scalar_type(), torch::kBFloat16);
  }
#if defined(USE_MLU)
  const bool decode =
      raw.device().type() == torch::kPrivateUse1 && dim == 512 &&
      block_size_ == 128 && raw.size(0) == batch.num_reqs &&
      batch.is_prefilling.size() == batch.num_reqs &&
      std::all_of(batch.num_scheduled_tokens.begin(),
                  batch.num_scheduled_tokens.end(),
                  [](int32_t count) { return count == 1; }) &&
      std::none_of(batch.is_prefilling.begin(),
                   batch.is_prefilling.end(),
                   [](uint8_t prefill) { return prefill != 0; });
  if (decode) {
    const int64_t emitted = std::count_if(
        batch.num_computed_tokens.begin(),
        batch.num_computed_tokens.end(),
        [this](int32_t position) { return ratio_ == 1 || position % 2 == 1; });
    result.latent = torch::empty({emitted, dim}, cache.main.options());
    result.positions = torch::empty({emitted}, result.positions.options());
    result.slots = torch::empty({emitted}, result.slots.options());
    int64_t output_row = 0;
    for (int32_t seq = 0; seq < batch.num_reqs; ++seq) {
      const int64_t position = batch.num_computed_tokens[seq];
      const int64_t rows = ratio_ == 1 || position % 2 == 1 ? 1 : 0;
      kernel::mlu::csa2_decode_compress(
          raw[seq],
          ratio_ == 2 ? scores[seq] : raw[seq],
          norm_weight,
          rope_dim_ > 0 ? rope : norm_weight,
          ratio_ == 2 ? state : cache.main,
          cache.main,
          ratio_ == 2 ? cache.state_block_table[seq]
                      : cache.main_block_table[seq],
          cache.main_block_table[seq],
          position,
          ratio_,
          rope_dim_,
          norm_eps_,
          save_full_tail_,
          result.latent.slice(0, output_row, output_row + rows),
          result.positions.slice(0, output_row, output_row + rows),
          result.slots.slice(0, output_row, output_row + rows));
      output_row += rows;
    }
    return result;
  }
#endif
  std::vector<torch::Tensor> latents, positions, slots;
  latents.reserve(batch.num_reqs);
  positions.reserve(batch.num_reqs);
  slots.reserve(batch.num_reqs);
  for (int32_t seq = 0; seq < batch.num_reqs; ++seq) {
    const int64_t count = batch.num_scheduled_tokens[seq];
    if (count == 0) {
      continue;
    }
    const int64_t offset = batch.query_start_loc[seq];
    const int64_t begin = batch.num_computed_tokens[seq];
    const int64_t end = begin + count;
    const int64_t first = begin / ratio_;
    const int64_t stop = end / ratio_;
    if (stop > first) {
      const int64_t history = begin % ratio_;
      const int64_t current = (stop - first) * ratio_ - history;
      torch::Tensor values = raw.slice(0, offset, offset + current);
      torch::Tensor gates;
      if (ratio_ == 2) {
        gates = scores.slice(0, offset, offset + current);
        if (history != 0) {
          const auto prior = state.index_select(
              0,
              state_slots(
                  cache.state_block_table, seq, begin - 1, begin, block_size_));
          values = torch::cat({prior.slice(1, 0, dim), values}, 0);
          gates = torch::cat({prior.slice(1, dim), gates}, 0);
        }
        values = (values.view({-1, 2, dim}) *
                  torch::softmax(gates.view({-1, 2, dim}), 1))
                     .sum(1);
      }
      // Internal MLU rounds pooled BF16 before RMSNorm, and rounds latent
      // again before RoPE. Moving either cast changes the trained path.
      const auto rounded = values.to(torch::kBFloat16).to(torch::kFloat32);
      latents.emplace_back(
          (rounded * torch::rsqrt(rounded.square().mean(-1, true) + norm_eps_) *
           norm_weight.to(torch::kFloat32))
              .to(torch::kBFloat16));
      const auto logical = torch::arange(first, stop, result.slots.options());
      positions.emplace_back(logical * ratio_);
      slots.emplace_back(
          cache.main_block_table[seq]
                  .to(torch::kInt64)
                  .index_select(0, torch::floor_divide(logical, block_size_)) *
              block_size_ +
          logical.remainder(block_size_));
    }
    if (ratio_ == 2) {
      const int64_t tail =
          save_full_tail_ ? std::min<int64_t>(count, Csa2CachePlan::kStateRows)
                          : end % 2;
      const auto target = state_slots(
          cache.state_block_table, seq, end - tail, end, block_size_);
      state.index_copy_(
          0,
          target,
          torch::cat({raw.slice(0, offset + count - tail, offset + count),
                      scores.slice(0, offset + count - tail, offset + count)},
                     1));
    }
  }
  if (latents.empty()) {
    return result;
  }
  result.latent = torch::cat(latents, 0);
  result.positions = torch::cat(positions, 0);
  result.slots = torch::cat(slots, 0);
  torch::Tensor rotated = result.latent.clone();
  if (rope_dim_ > 0) {
    const auto coefficients = rope.index_select(0, result.positions);
    const auto cos = coefficients.slice(1, 0, rope_dim_, 2);
    const auto sin = coefficients.slice(1, rope_dim_, 2 * rope_dim_, 2);
    const auto pairs = result.latent.slice(1, dim - rope_dim_)
                           .to(torch::kFloat32)
                           .view({-1, rope_dim_ / 2, 2});
    rotated.slice(1, dim - rope_dim_, dim, 2)
        .copy_(pairs.select(2, 0) * cos - pairs.select(2, 1) * sin);
    rotated.slice(1, dim - rope_dim_ + 1, dim, 2)
        .copy_(pairs.select(2, 1) * cos + pairs.select(2, 0) * sin);
  }
  cache.main.view({-1, dim}).index_copy_(0, result.slots, rotated);
  return result;
}

}  // namespace xllm::layer
