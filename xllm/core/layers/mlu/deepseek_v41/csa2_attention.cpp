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

#include "core/layers/mlu/deepseek_v41/csa2_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(USE_MLU)
#include "core/kernels/mlu/mlu_ops_api.h"
#endif

namespace xllm::layer {
namespace {
torch::Tensor cache_slots(const torch::Tensor& table,
                          int64_t seq,
                          int64_t begin,
                          int64_t end,
                          int64_t block_size,
                          bool ring) {
  const torch::Tensor positions =
      torch::arange(begin, end, table.options().dtype(torch::kInt64));
  torch::Tensor columns = torch::floor_divide(positions, block_size);
  if (ring) {
    columns = columns.remainder(table.size(1));
  }
  return table[seq].to(torch::kInt64).index_select(0, columns) * block_size +
         positions.remainder(block_size);
}
}  // namespace

Csa2Indexer::Csa2Indexer(Csa2IndexOptions options) : options_(options) {
  CHECK(options.ratio == 1 || options.ratio == 2);
  CHECK_GT(options.topk, 0);
  if (options.candidate_mode != Csa2CandidateMode::NONE) {
    CHECK_GT(options.candidate_block_size, 0);
    CHECK_GT(options.candidate_topk_blocks, 0);
  }
}

Csa2Selection Csa2Indexer::select(const torch::Tensor& query,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& keys,
                                  int64_t query_begin,
                                  const torch::Tensor& candidates,
                                  int64_t complete_candidate_blocks) const {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(keys.dim(), 2);
  CHECK_EQ(query.size(0), weights.size(0));
  CHECK_EQ(query.size(1), weights.size(1));
  CHECK_GT(query.size(1), 0);
  CHECK_EQ(query.size(2), keys.size(1));
  CHECK_GE(query_begin, 0);
  const int64_t count = query.size(0);
  const int64_t length = keys.size(0);
  const auto ints = query.options().dtype(torch::kInt64);
  if (options_.candidate_mode == Csa2CandidateMode::CONSUME) {
    CHECK(candidates.defined());
    CHECK_EQ(candidates.dim(), 2);
    CHECK_EQ(candidates.size(0), count);
  }
  Csa2Selection result;
  if (count == 0 || length == 0) {
    result.indices =
        torch::empty({count, std::min(options_.topk, length)}, ints);
    if (options_.candidate_mode == Csa2CandidateMode::PRODUCE) {
      result.candidates = torch::empty({count, 0}, ints);
    }
    return result;
  }
  // Bound the per-head score product at long contexts. Even one row must fit
  // the supported model's index-head geometry; no chunk-sized score tensor.
  constexpr int64_t kScoreBudgetBytes = 128 * 1024 * 1024;
  const int64_t tile_rows = std::max<int64_t>(
      1,
      std::min<int64_t>(32, kScoreBudgetBytes / (query.size(1) * length * 4)));
  const int64_t tile_count = (count + tile_rows - 1) / tile_rows;
  const double neg_inf = -std::numeric_limits<double>::infinity();
  const int64_t blocks = options_.candidate_mode == Csa2CandidateMode::NONE
                             ? 0
                             : (length + options_.candidate_block_size - 1) /
                                   options_.candidate_block_size;
  const bool restrict_candidates =
      options_.candidate_mode == Csa2CandidateMode::CONSUME &&
      complete_candidate_blocks != blocks;
  // Decode keys end at the causal frontier. A full candidate pool also
  // excludes nothing; neither case needs a device mask or its index tensors.
  const bool causal_mask = length > (query_begin + 1) / options_.ratio;
  torch::Tensor columns;
  if (causal_mask || restrict_candidates) {
    columns = torch::arange(length, ints);
  }
  if (options_.candidate_mode == Csa2CandidateMode::PRODUCE &&
      blocks <= options_.candidate_topk_blocks) {
    result.complete_candidate_blocks = blocks;
  }
  const torch::Tensor key_transpose = keys.to(torch::kFloat32).t();
  std::vector<torch::Tensor> selections, pools;
  selections.reserve(tile_count);
  pools.reserve(tile_count);
  for (int64_t row = 0; row < count; row += tile_rows) {
    const int64_t rows = std::min(tile_rows, count - row);
    const bool mask_rows = length > (query_begin + row + 1) / options_.ratio;
    torch::Tensor lengths;
    if (mask_rows || options_.candidate_mode == Csa2CandidateMode::PRODUCE) {
      lengths = torch::floor_divide(
          torch::arange(
              query_begin + row + 1, query_begin + row + rows + 1, ints),
          options_.ratio);
    }
    torch::Tensor logits =
        (torch::matmul(query.slice(0, row, row + rows).to(torch::kFloat32),
                       key_transpose)
             .relu() *
         weights.slice(0, row, row + rows).to(torch::kFloat32).unsqueeze(-1))
            .sum(1);
    if (mask_rows) {
      logits.masked_fill_(columns.unsqueeze(0) >= lengths.unsqueeze(1),
                          neg_inf);
    }
    if (options_.candidate_mode != Csa2CandidateMode::NONE) {
      const int64_t width = options_.candidate_block_size;
      if (options_.candidate_mode == Csa2CandidateMode::PRODUCE) {
        torch::Tensor padded =
            torch::full({rows, blocks * width}, neg_inf, logits.options());
        padded.slice(1, 0, length).copy_(logits);
        torch::Tensor block_scores =
            std::get<0>(padded.view({rows, blocks, width}).max(-1));
        const torch::Tensor last =
            torch::floor_divide((lengths - 1).clamp_min(0), width).unsqueeze(1);
        block_scores.scatter_(1, last, std::numeric_limits<double>::infinity());
        pools.emplace_back(std::get<1>(block_scores.topk(
            std::min(blocks, options_.candidate_topk_blocks), -1)));
      } else if (restrict_candidates) {
        torch::Tensor allowed =
            torch::zeros({rows, blocks}, query.options().dtype(torch::kBool));
        allowed.scatter_(1, candidates.slice(0, row, row + rows), true);
        logits.masked_fill_(
            ~allowed.index_select(1, torch::floor_divide(columns, width)),
            neg_inf);
      }
    }
    auto [values, selected] = logits.topk(std::min(options_.topk, length), -1);
    selected.masked_fill_(values == neg_inf, -1);
    selections.emplace_back(std::move(selected));
  }
  result.indices = torch::cat(selections, 0);
  if (!pools.empty()) {
    result.candidates = torch::cat(pools, 0);
  }
  return result;
}

Csa2Selection Csa2Indexer::select_decode(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& table,
    int64_t query_begin,
    int64_t block_size,
    const torch::Tensor& candidates,
    int64_t complete_candidate_blocks) const {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(0), 1);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(weights.size(0), 1);
  CHECK_EQ(weights.size(1), query.size(1));
  CHECK_EQ(cache.dim(), 3);
  CHECK_EQ(cache.size(1), block_size);
  CHECK_EQ(cache.size(2), query.size(2));
  CHECK_EQ(table.dim(), 1);
  CHECK_GE(query_begin, 0);
  const int64_t length = (query_begin + 1) / options_.ratio;
  CHECK_GE(table.numel() * block_size, length);
  const auto slots = cache_slots(table.unsqueeze(0),
                                 /*seq=*/0,
                                 /*begin=*/0,
                                 length,
                                 block_size,
                                 /*ring=*/false);
  return select(query,
                weights,
                cache.view({-1, cache.size(2)}).index_select(0, slots),
                query_begin,
                candidates,
                complete_candidate_blocks);
}

void Csa2Indexer::select_decode_batch(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& cache,
    const torch::Tensor& table,
    const std::vector<int32_t>& query_begin,
    int64_t block_size,
    std::vector<Csa2Selection>& selections) const {
  const int64_t batch_size = query_begin.size();
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(query.size(0), batch_size);
  CHECK_EQ(weights.size(0), batch_size);
  CHECK_EQ(weights.size(1), query.size(1));
  CHECK_EQ(cache.dim(), 3);
  CHECK_EQ(cache.size(1), block_size);
  CHECK_EQ(cache.size(2), query.size(2));
  CHECK_EQ(table.dim(), 2);
  CHECK_GE(table.size(0), batch_size);
  CHECK_EQ(selections.size(), batch_size);
  for (int32_t begin : query_begin) {
    CHECK_GE(begin, 0);
    CHECK_GE(table.size(1) * block_size, (begin + 1) / options_.ratio);
  }
#if defined(USE_MLU)
  if (query.device().type() == torch::kPrivateUse1 && block_size == 128 &&
      cache.scalar_type() == torch::kBFloat16 && query.size(1) == 32 &&
      query.size(2) == 128 && options_.ratio == 2 && options_.topk == 512 &&
      options_.candidate_mode == Csa2CandidateMode::NONE) {
    const auto ints = query.options().dtype(torch::kInt32);
    std::vector<int64_t> active;
    active.reserve(batch_size);
    for (int64_t seq = 0; seq < batch_size; ++seq) {
      if ((query_begin[seq] + 1) / options_.ratio == 0) {
        selections[seq].indices =
            torch::empty({1, 0}, query.options().dtype(torch::kInt64));
        continue;
      }
      active.emplace_back(seq);
    }
    const int64_t active_size = active.size();
    if (active_size == 0) {
      return;
    }
    const auto host = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true);
    auto host_lengths = torch::empty({active_size}, host);
    auto* lengths_data = host_lengths.data_ptr<int32_t>();
    for (int64_t row = 0; row < active_size; ++row) {
      lengths_data[row] =
          static_cast<int32_t>((query_begin[active[row]] + 1) / options_.ratio);
    }
    const auto lengths = host_lengths.to(query.device(), /*non_blocking=*/true);
    auto selected_query = query;
    auto selected_weights = weights;
    auto physical_pages = table.slice(0, 0, batch_size);
    if (active_size != batch_size) {
      // Empty histories never enter the SDK. Keep the remaining rows batched.
      auto host_active = torch::empty({active_size}, host.dtype(torch::kInt64));
      std::copy(active.begin(), active.end(), host_active.data_ptr<int64_t>());
      const auto rows = host_active.to(query.device(), /*non_blocking=*/true);
      selected_query = query.index_select(0, rows);
      selected_weights = weights.index_select(0, rows);
      physical_pages = physical_pages.index_select(0, rows);
    }
    auto indices = torch::empty({active_size, 1, options_.topk}, ints);
    const auto selected_lengths = torch::empty_like(lengths);
    // Identity output pages preserve logical indices for cross-layer Reuse.
    const auto logical_pages =
        torch::arange(table.size(1), ints).repeat({active_size, 1});
    kernel::mlu::masked_indexer_select_paged_kv(
        selected_query.to(torch::kBFloat16).unsqueeze(1),
        cache.unsqueeze(1),
        selected_weights.unsqueeze(1),
        logical_pages,
        std::nullopt,
        std::nullopt,
        lengths,
        physical_pages,
        /*is_prefill=*/false,
        options_.topk,
        block_size,
        /*softmax_scale=*/1.0,
        std::nullopt,
        std::nullopt,
        indices,
        selected_lengths,
        /*is_score_float=*/true,
        /*compress_ratio=*/1);
    const auto logical_indices =
        indices.view({active_size, options_.topk}).to(torch::kInt64);
    for (int64_t row = 0; row < active_size; ++row) {
      const int64_t seq = active[row];
      const int64_t count = std::min<int64_t>(
          (query_begin[seq] + 1) / options_.ratio, options_.topk);
      selections[seq].indices =
          logical_indices.slice(0, row, row + 1).slice(1, 0, count);
    }
    return;
  }
#endif
  for (int64_t seq = 0; seq < batch_size; ++seq) {
    auto selected = select_decode(query.slice(0, seq, seq + 1),
                                  weights.slice(0, seq, seq + 1),
                                  cache,
                                  table[seq],
                                  query_begin[seq],
                                  block_size,
                                  selections[seq].candidates,
                                  selections[seq].complete_candidate_blocks);
    selections[seq].indices = std::move(selected.indices);
    if (selected.candidates.defined()) {
      selections[seq].candidates = std::move(selected.candidates);
      selections[seq].complete_candidate_blocks =
          selected.complete_candidate_blocks;
    }
  }
}

torch::Tensor csa2_sparse_attention(const torch::Tensor& query,
                                    const torch::Tensor& window,
                                    const torch::Tensor& compressed,
                                    const torch::Tensor& indices,
                                    const torch::Tensor& sink,
                                    int64_t query_begin,
                                    int64_t window_size) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(window.dim(), 2);
  CHECK_EQ(window.size(1), query.size(2));
  CHECK_GT(window_size, 0);
  CHECK_GE(query_begin, 0);
  CHECK_EQ(sink.numel(), query.size(1));
  const int64_t count = query.size(0);
  const int64_t dim = query.size(2);
  CHECK_GT(dim, 0);
  const int64_t old_begin = std::max<int64_t>(0, query_begin - window_size + 1);
  CHECK_EQ(window.size(0), query_begin - old_begin + count);
  const bool has_main = compressed.defined() && compressed.size(0) > 0;
  if (has_main) {
    CHECK_EQ(compressed.dim(), 2);
    CHECK_EQ(compressed.size(1), dim);
    CHECK_EQ(indices.dim(), 2);
    CHECK_EQ(indices.size(0), count);
  }
  torch::Tensor output = torch::empty_like(query);
  const auto ints = query.options().dtype(torch::kInt64);
  const double neg_inf = -std::numeric_limits<double>::infinity();
  for (int64_t row = 0; row < count; row += 16) {
    const int64_t rows = std::min<int64_t>(16, count - row);
    const torch::Tensor pos =
        torch::arange(query_begin + row, query_begin + row + rows, ints);
    const torch::Tensor win_pos = pos.unsqueeze(1) - window_size + 1 +
                                  torch::arange(window_size, ints).unsqueeze(0);
    torch::Tensor valid = win_pos.ge(0);
    torch::Tensor selected =
        window.index_select(0, (win_pos - old_begin).clamp_min(0).flatten())
            .view({rows, window_size, dim});
    if (has_main) {
      const torch::Tensor topk = indices.slice(0, row, row + rows);
      const torch::Tensor main_rows =
          compressed.index_select(0, topk.clamp_min(0).flatten())
              .view({rows, topk.size(1), dim});
      selected = torch::cat({selected, main_rows}, 1);
      valid = torch::cat({valid, topk.ge(0)}, 1);
    }
    torch::Tensor logits =
        torch::bmm(query.slice(0, row, row + rows).to(torch::kFloat32),
                   selected.to(torch::kFloat32).transpose(1, 2)) /
        std::sqrt(dim);
    logits.masked_fill_(~valid.unsqueeze(1), neg_inf);
    const torch::Tensor sinks = sink.to(torch::kFloat32)
                                    .view({1, query.size(1), 1})
                                    .expand({rows, query.size(1), 1});
    const torch::Tensor probability = torch::cat({logits, sinks}, -1)
                                          .softmax(-1)
                                          .slice(-1, 0, logits.size(-1));
    output.slice(0, row, row + rows)
        .copy_(torch::bmm(probability, selected.to(torch::kFloat32))
                   .to(query.scalar_type()));
  }
  return output;
}
Csa2Attention::Csa2Attention(Csa2CachePlan::Mode mode,
                             int64_t window_size,
                             int64_t block_size,
                             Csa2IndexOptions index_options)
    : mode_(mode),
      window_size_(window_size),
      block_size_(block_size),
      ratio_(index_options.ratio),
      topk_(index_options.topk) {
  CHECK_GT(window_size_, 0);
  CHECK_GT(block_size_, 0);
  if (mode_ != Csa2CachePlan::Mode::SWA) {
    CHECK(ratio_ == 1 || ratio_ == 2);
  }
  if (mode_ == Csa2CachePlan::Mode::FULL ||
      mode_ == Csa2CachePlan::Mode::REINDEX) {
    indexer_.emplace(index_options);
  }
}

torch::Tensor Csa2Attention::forward(
    const torch::Tensor& query,
    const torch::Tensor& kv,
    const torch::Tensor& index_query,
    const torch::Tensor& index_weights,
    const torch::Tensor& sink,
    const ExecutionBatchMetadata& batch,
    const Csa2LayerCacheView& cache,
    std::vector<Csa2Selection>& selections) const {
  CHECK_GE(batch.num_reqs, 0);
  if (batch.num_reqs == 0) {
    CHECK_EQ(batch.num_tokens, 0);
    return torch::zeros_like(query);
  }
  CHECK_EQ(batch.num_scheduled_tokens.size(), batch.num_reqs);
  CHECK_EQ(batch.num_computed_tokens.size(), batch.num_reqs);
  CHECK_EQ(batch.query_start_loc.size(), batch.num_reqs + 1);
  CHECK_EQ(batch.query_start_loc.front(), 0);
  CHECK_EQ(batch.query_start_loc.back(), batch.num_tokens);
  CHECK_EQ(selections.size(), batch.num_reqs);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(kv.dim(), 2);
  CHECK_EQ(query.size(0), kv.size(0));
  CHECK_EQ(query.size(2), kv.size(1));
  CHECK_LE(batch.num_tokens, query.size(0));
  CHECK_GE(batch.num_tokens, 0);
  // Validate the entire batch before publishing any layer state.
  for (int32_t seq = 0; seq < batch.num_reqs; ++seq) {
    CHECK_GE(batch.num_scheduled_tokens[seq], 0);
    CHECK_GE(batch.num_computed_tokens[seq], 0);
    CHECK_EQ(batch.query_start_loc[seq + 1] - batch.query_start_loc[seq],
             batch.num_scheduled_tokens[seq]);
  }
  if (batch.num_tokens == 0) {
    return torch::zeros_like(query);
  }
  CHECK_EQ(cache.swa_block_table.dim(), 2);
  CHECK_GE(cache.swa_block_table.size(0), batch.num_reqs);
  CHECK_GE(cache.swa_block_table.size(1) * block_size_, window_size_);
  CHECK_EQ(cache.swa.size(-1), kv.size(1));
  torch::Tensor swa = cache.swa.view({-1, kv.size(1)});
  bool decode = false;
#if defined(USE_MLU)
  // CPU and non-512 fixtures retain the projected reference path. The SDK
  // kernel is selected before mutation, only for unpadded single-token decode.
  decode = block_size_ == 128 && query.device().type() == torch::kPrivateUse1 &&
           query.scalar_type() == torch::kBFloat16 && query.size(2) == 512 &&
           query.size(0) == batch.num_reqs &&
           batch.is_prefilling.size() == batch.num_reqs &&
           std::all_of(batch.num_scheduled_tokens.begin(),
                       batch.num_scheduled_tokens.end(),
                       [](int32_t count) { return count == 1; }) &&
           std::none_of(batch.is_prefilling.begin(),
                        batch.is_prefilling.end(),
                        [](uint8_t prefill) { return prefill != 0; });
#endif
  torch::Tensor output =
      decode ? torch::empty_like(query) : torch::zeros_like(query);
  const int64_t capacity = window_size_ + topk_;
  torch::Tensor selected, lengths;
  if (decode) {
    selected = torch::empty({batch.num_tokens, capacity, query.size(2)},
                            query.options());
    lengths =
        torch::empty({batch.num_tokens}, query.options().dtype(torch::kInt32));
  }
  if (decode && indexer_) {
    indexer_->select_decode_batch(index_query,
                                  index_weights,
                                  cache.index,
                                  cache.main_block_table,
                                  batch.num_computed_tokens,
                                  block_size_,
                                  selections);
  }
  for (int32_t seq = 0; seq < batch.num_reqs; ++seq) {
    const int64_t count = batch.num_scheduled_tokens[seq];
    if (count == 0) {
      continue;
    }
    const int64_t offset = batch.query_start_loc[seq];
    const int64_t begin = batch.num_computed_tokens[seq];
    const int64_t end = begin + count;
    const int64_t old_begin = std::max<int64_t>(0, begin - window_size_ + 1);
    torch::Tensor window;
    if (!decode) {
      window = torch::cat({swa.index_select(0,
                                            cache_slots(cache.swa_block_table,
                                                        seq,
                                                        old_begin,
                                                        begin,
                                                        block_size_,
                                                        /*ring=*/true)),
                           kv.slice(0, offset, offset + count)},
                          0);
    }
    torch::Tensor main;
    auto& selection = selections[seq];
    if (mode_ != Csa2CachePlan::Mode::SWA) {
      torch::Tensor slots;
      if (!decode) {
        slots = cache_slots(cache.main_block_table,
                            seq,
                            0,
                            end / ratio_,
                            block_size_,
                            /*ring=*/false);
      }
      main = cache.main.view({-1, kv.size(1)});
      if (!decode) {
        main = main.index_select(0, slots);
      }
      if (indexer_ && !decode) {
        Csa2Selection selected = indexer_->select(
            index_query.slice(0, offset, offset + count),
            index_weights.slice(0, offset, offset + count),
            cache.index.view({-1, cache.index.size(-1)}).index_select(0, slots),
            begin,
            selection.candidates,
            selection.complete_candidate_blocks);
        selection.indices = std::move(selected.indices);
        if (selected.candidates.defined()) {
          selection.candidates = std::move(selected.candidates);
          selection.complete_candidate_blocks =
              selected.complete_candidate_blocks;
        }
      } else {
        CHECK(selection.indices.defined())
            << "CSA2 decode/Reuse requires its source selection";
        CHECK_EQ(selection.indices.size(0), count);
      }
    }
#if defined(USE_MLU)
    if (decode) {
      const bool compressed = mode_ != Csa2CachePlan::Mode::SWA;
      kernel::mlu::csa2_decode_gather(
          swa,
          compressed ? main : swa,
          kv[offset],
          cache.swa_block_table[seq],
          compressed ? cache.main_block_table[seq] : cache.swa_block_table[seq],
          compressed ? selection.indices[0]
                     : torch::empty({0}, query.options().dtype(torch::kInt64)),
          begin,
          window_size_,
          topk_,
          selected[offset],
          lengths.slice(0, offset, offset + 1));
      continue;
    }
#endif
    output.slice(0, offset, offset + count)
        .copy_(csa2_sparse_attention(query.slice(0, offset, offset + count),
                                     window,
                                     main,
                                     selection.indices,
                                     sink,
                                     begin,
                                     window_size_));
    // Read and consume old ring contents before a chunk can overwrite them.
    const int64_t tail =
        std::min(count, cache.swa_block_table.size(1) * block_size_);
    swa.index_copy_(0,
                    cache_slots(cache.swa_block_table,
                                seq,
                                end - tail,
                                end,
                                block_size_,
                                /*ring=*/true),
                    kv.slice(0, offset + count - tail, offset + count));
  }
#if defined(USE_MLU)
  if (decode) {
    auto table = torch::arange(batch.num_tokens * capacity,
                               query.options().dtype(torch::kInt32))
                     .view({batch.num_tokens, 1, capacity});
    auto projected = query.unsqueeze(1);
    auto flat_selected = selected.view({-1, 1, 1, query.size(2)});
    auto out = output.unsqueeze(1);
    std::optional<torch::Tensor> lse;
    kernel::mlu::batch_decode(projected,
                              flat_selected,
                              out,
                              table,
                              lengths,
                              std::nullopt,
                              lse,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              "half",
                              capacity,
                              -1,
                              -1,
                              1.0 / std::sqrt(query.size(2)),
                              false,
                              -1,
                              std::nullopt,
                              -1,
                              sink.to(torch::kFloat32));
  }
#endif
  return output;
}
}  // namespace xllm::layer
