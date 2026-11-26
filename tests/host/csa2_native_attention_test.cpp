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

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/layers/mlu/deepseek_v41/csa2_attention.h"

namespace xllm::layer {
namespace {
TEST(Csa2NativeIndexerTest, CausalTopkMasksUnwrittenCompressedTokens) {
  Csa2Indexer indexer({.ratio = 2, .topk = 3});
  const auto query = torch::ones({8, 1, 1});
  const auto weights = torch::ones({8, 1});
  const auto keys = torch::tensor({{1.0f}, {2.0f}, {3.0f}, {4.0f}});
  const auto selection =
      indexer.select(query, weights, keys, /*query_begin=*/0);
  const auto expected = torch::tensor({{-1, -1, -1},
                                       {0, -1, -1},
                                       {0, -1, -1},
                                       {1, 0, -1},
                                       {1, 0, -1},
                                       {2, 1, 0},
                                       {2, 1, 0},
                                       {3, 2, 1}},
                                      torch::kInt64);
  EXPECT_TRUE(torch::equal(selection.indices, expected));
}

TEST(Csa2NativeIndexerTest, ChunkAndSingleTokenSelectionsAgree) {
  Csa2Indexer indexer({.ratio = 1, .topk = 3});
  const auto keys = torch::arange(1, 71, torch::kFloat32).view({70, 1});
  const auto all =
      indexer.select(torch::ones({70, 1, 1}), torch::ones({70, 1}), keys, 0)
          .indices;
  const auto chunk =
      indexer.select(torch::ones({37, 1, 1}), torch::ones({37, 1}), keys, 33)
          .indices;
  EXPECT_TRUE(torch::equal(all.slice(0, 33), chunk));
  const auto last =
      indexer.select(torch::ones({1, 1, 1}), torch::ones({1, 1}), keys, 69)
          .indices;
  EXPECT_TRUE(torch::equal(last, all.slice(0, 69)));
}

TEST(Csa2NativeIndexerTest, DecodeReadsLogicalHistoryFromPermutedPages) {
  Csa2Indexer indexer({.ratio = 2, .topk = 3});
  const auto cache =
      torch::tensor({8.0f, 9.0f, 1.0f, 2.0f, 3.0f, 100.0f}).view({3, 2, 1});
  const auto table = torch::tensor({1, 2, 0}, torch::kInt32);
  const auto query = torch::ones({1, 1, 1});
  const auto weights = torch::ones({1, 1});
  const auto selected = indexer.select_decode(
      query, weights, cache, table, /*query_begin=*/5, /*block_size=*/2);
  EXPECT_TRUE(torch::equal(
      selected.indices, torch::tensor({2, 1, 0}, torch::kInt64).view({1, 3})));
  const auto empty = indexer.select_decode(
      query, weights, cache, table, /*query_begin=*/0, /*block_size=*/2);
  EXPECT_EQ(empty.indices.sizes(), torch::IntArrayRef({1, 0}));
}

TEST(Csa2NativeIndexerTest, DecodeBatchPreservesRaggedLogicalHistories) {
  Csa2Indexer indexer({.ratio = 2, .topk = 3});
  const auto cache =
      torch::tensor({8.0f, 9.0f, 1.0f, 2.0f, 3.0f, 100.0f}).view({3, 2, 1});
  const auto tables =
      torch::tensor({1, 2, 0, 2, 0, 1, 0, 1, 2}, torch::kInt32).view({3, 3});
  std::vector<Csa2Selection> selected(3);
  indexer.select_decode_batch(torch::ones({3, 1, 1}),
                              torch::ones({3, 1}),
                              cache,
                              tables,
                              {5, 0, 3},
                              /*block_size=*/2,
                              selected);
  EXPECT_TRUE(
      torch::equal(selected[0].indices,
                   torch::tensor({2, 1, 0}, torch::kInt64).view({1, 3})));
  EXPECT_EQ(selected[1].indices.sizes(), torch::IntArrayRef({1, 0}));
  EXPECT_TRUE(torch::equal(selected[2].indices,
                           torch::tensor({1, 0}, torch::kInt64).view({1, 2})));
}

TEST(Csa2NativeIndexerTest, CandidatePoolKeepsNewestBlockAndLimitsConsumer) {
  const auto query = torch::ones({4, 1, 1});
  const auto weights = torch::ones({4, 1});
  const auto keys = torch::tensor({{100.0f}, {99.0f}, {2.0f}, {1.0f}});
  Csa2Indexer producer({.ratio = 1,
                        .topk = 1,
                        .candidate_mode = Csa2CandidateMode::PRODUCE,
                        .candidate_block_size = 2,
                        .candidate_topk_blocks = 1});
  const auto first = producer.select(query, weights, keys, 0);
  EXPECT_EQ(first.indices[3][0].item<int64_t>(), 0);
  EXPECT_EQ(first.candidates[3][0].item<int64_t>(), 1);
  Csa2Indexer consumer({.ratio = 1,
                        .topk = 2,
                        .candidate_mode = Csa2CandidateMode::CONSUME,
                        .candidate_block_size = 2,
                        .candidate_topk_blocks = 1});
  const auto next = consumer.select(query, weights, keys, 0, first.candidates);
  EXPECT_TRUE(
      torch::equal(next.indices[3], torch::tensor({2, 3}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(next.indices[2], torch::tensor({2, -1}, torch::kInt64)));
}

TEST(Csa2NativeIndexerTest, CompleteCandidatePoolMatchesExplicitMask) {
  const Csa2Indexer producer({.ratio = 1,
                              .topk = 3,
                              .candidate_mode = Csa2CandidateMode::PRODUCE,
                              .candidate_block_size = 2,
                              .candidate_topk_blocks = 4});
  const Csa2Indexer consumer({.ratio = 1,
                              .topk = 3,
                              .candidate_mode = Csa2CandidateMode::CONSUME,
                              .candidate_block_size = 2,
                              .candidate_topk_blocks = 4});
  const auto keys = torch::arange(1, 9, torch::kFloat32).view({8, 1});
  const auto query = torch::ones({8, 1, 1});
  const auto weights = torch::ones({8, 1});
  const auto pool = producer.select(query, weights, keys, /*query_begin=*/0);
  ASSERT_EQ(pool.complete_candidate_blocks, 4);
  const auto masked = consumer.select(query, weights, keys, 0, pool.candidates);
  const auto complete = consumer.select(
      query, weights, keys, 0, pool.candidates, pool.complete_candidate_blocks);
  EXPECT_TRUE(torch::equal(masked.indices, complete.indices));
  EXPECT_TRUE(torch::equal(complete.indices[7],
                           torch::tensor({7, 6, 5}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(complete.indices[0],
                           torch::tensor({0, -1, -1}, torch::kInt64)));
  const auto cache = keys.view({4, 2, 1});
  const auto tables = torch::arange(4, torch::kInt32).repeat({2, 1});
  const auto decode_query = torch::ones({2, 1, 1});
  const auto decode_weights = torch::ones({2, 1});
  std::vector<Csa2Selection> selected(2);
  producer.select_decode_batch(decode_query,
                               decode_weights,
                               cache,
                               tables,
                               {3, 7},
                               /*block_size=*/2,
                               selected);
  ASSERT_EQ(selected[0].complete_candidate_blocks, 2);
  ASSERT_EQ(selected[1].complete_candidate_blocks, 4);
  auto explicit_mask = selected;
  for (auto& selection : explicit_mask) {
    selection.complete_candidate_blocks = 0;
  }
  consumer.select_decode_batch(decode_query,
                               decode_weights,
                               cache,
                               tables,
                               {3, 7},
                               /*block_size=*/2,
                               selected);
  consumer.select_decode_batch(decode_query,
                               decode_weights,
                               cache,
                               tables,
                               {3, 7},
                               /*block_size=*/2,
                               explicit_mask);
  EXPECT_TRUE(torch::equal(selected[0].indices, explicit_mask[0].indices));
  EXPECT_TRUE(torch::equal(selected[1].indices, explicit_mask[1].indices));
}

TEST(Csa2NativeIndexerTest, CompletePoolProofDoesNotCoverLargerKeySpan) {
  const Csa2Indexer producer({.ratio = 1,
                              .topk = 3,
                              .candidate_mode = Csa2CandidateMode::PRODUCE,
                              .candidate_block_size = 2,
                              .candidate_topk_blocks = 2});
  const Csa2Indexer consumer({.ratio = 1,
                              .topk = 3,
                              .candidate_mode = Csa2CandidateMode::CONSUME,
                              .candidate_block_size = 2,
                              .candidate_topk_blocks = 2});
  const auto query = torch::ones({1, 1, 1});
  const auto weights = torch::ones({1, 1});
  const auto keys = torch::arange(1, 7, torch::kFloat32).view({6, 1});
  const auto pool =
      producer.select(query, weights, keys.slice(0, 0, 4), /*query_begin=*/3);
  ASSERT_EQ(pool.complete_candidate_blocks, 2);
  const auto actual = consumer.select(
      query, weights, keys, 5, pool.candidates, pool.complete_candidate_blocks);
  EXPECT_TRUE(torch::equal(
      actual.indices, torch::tensor({3, 2, 1}, torch::kInt64).view({1, 3})));
  const auto partial = producer.select(query, weights, keys, /*query_begin=*/5);
  EXPECT_EQ(partial.complete_candidate_blocks, 0);
}

TEST(Csa2NativeIndexerTest, EmptyKeysAndEmptyRowsKeepRankTwoSelections) {
  Csa2Indexer indexer({.ratio = 2, .topk = 2});
  const auto empty_keys = indexer.select(
      torch::ones({1, 2, 3}), torch::ones({1, 2}), torch::empty({0, 3}), 0);
  EXPECT_EQ(empty_keys.indices.sizes(), torch::IntArrayRef({1, 0}));
  const auto empty_rows = indexer.select(
      torch::empty({0, 2, 3}), torch::empty({0, 2}), torch::ones({4, 3}), 0);
  EXPECT_EQ(empty_rows.indices.sizes(), torch::IntArrayRef({0, 2}));
}

TEST(Csa2NativeAttentionTest, SinkParticipatesOnlyInDenominator) {
  const auto actual = csa2_sparse_attention(torch::zeros({1, 1, 2}),
                                            torch::tensor({{2.0f, 4.0f}}),
                                            {},
                                            {},
                                            torch::tensor({0.0f}),
                                            /*query_begin=*/0,
                                            /*window_size=*/4);
  EXPECT_TRUE(torch::equal(actual, torch::tensor({{{1.0f, 2.0f}}})));
}

TEST(Csa2NativeAttentionTest, MasksInvalidSelectionsAndCombinesWindowAndMain) {
  const auto actual =
      csa2_sparse_attention(torch::zeros({1, 1, 2}),
                            torch::tensor({{2.0f, 4.0f}}),
                            torch::tensor({{8.0f, 12.0f}, {1000.0f, 2000.0f}}),
                            torch::tensor({{0, -1}}, torch::kInt64),
                            torch::tensor({0.0f}),
                            /*query_begin=*/0,
                            /*window_size=*/1);
  EXPECT_TRUE(
      torch::allclose(actual, torch::tensor({{{10.0f / 3, 16.0f / 3}}})));
}

// Independent scalar softmax oracle, including the learned sink.
TEST(Csa2NativeAttentionTest,
     MatchesScalarReferenceAcrossTileAndWindowBoundary) {
  constexpr int64_t kRows = 37, kHeads = 2, kDim = 3, kWindow = 5;
  const auto query =
      torch::sin(torch::arange(kRows * kHeads * kDim, torch::kFloat32))
          .view({kRows, kHeads, kDim});
  const auto window = torch::cos(torch::arange(kRows * kDim, torch::kFloat32))
                          .view({kRows, kDim});
  const auto sink = torch::tensor({0.2f, -0.3f});
  const auto actual =
      csa2_sparse_attention(query, window, {}, {}, sink, 0, kWindow);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t head = 0; head < kHeads; ++head) {
      double denominator = std::exp(sink[head].item<double>());
      std::vector<double> numerator(kDim, 0);
      for (int64_t key = std::max<int64_t>(0, row - kWindow + 1); key <= row;
           ++key) {
        double dot = 0;
        for (int64_t dim = 0; dim < kDim; ++dim) {
          dot += query[row][head][dim].item<double>() *
                 window[key][dim].item<double>();
        }
        const double score = std::exp(dot / std::sqrt(kDim));
        denominator += score;
        for (int64_t dim = 0; dim < kDim; ++dim) {
          numerator[dim] += score * window[key][dim].item<double>();
        }
      }
      for (int64_t dim = 0; dim < kDim; ++dim) {
        EXPECT_NEAR(actual[row][head][dim].item<double>(),
                    numerator[dim] / denominator,
                    1e-6);
      }
    }
  }
  const auto chunk = csa2_sparse_attention(query.slice(0, 19),
                                           window.slice(0, 15),
                                           {},
                                           {},
                                           sink,
                                           /*query_begin=*/19,
                                           kWindow);
  EXPECT_TRUE(torch::allclose(actual.slice(0, 19), chunk, 1e-6, 1e-6));
}

TEST(Csa2NativeAttentionTest, Bfloat16PreservesOutputDtype) {
  const auto query = torch::zeros({1, 2, 2}, torch::kBFloat16);
  const auto window = torch::tensor({{2.0f, 4.0f}}).to(torch::kBFloat16);
  const auto actual =
      csa2_sparse_attention(query, window, {}, {}, torch::zeros({2}), 0, 4);
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::equal(actual.to(torch::kFloat32),
                           torch::tensor({{{1.0f, 2.0f}, {1.0f, 2.0f}}})));
}

TEST(Csa2NativeCachedTest, LongChunkReadsOldRingBeforeOverwrite) {
  const Csa2Attention attention(Csa2CachePlan::Mode::SWA, 4, 2, {});
  const auto query =
      torch::sin(torch::arange(26, torch::kFloat32)).view({13, 1, 2});
  const auto kv = torch::arange(1, 27, torch::kFloat32).view({13, 2});
  const auto sink = torch::zeros({1});
  Csa2LayerCacheView cache;
  cache.swa = torch::zeros({2, 2, 2});
  cache.swa_block_table = torch::tensor({{0, 1}}, torch::kInt32);
  std::vector<Csa2Selection> selections(1);
  ExecutionBatchMetadata batch{.num_reqs = 1,
                               .num_tokens = 13,
                               .num_scheduled_tokens = {13},
                               .num_computed_tokens = {0},
                               .query_start_loc = {0, 13}};
  const auto full =
      attention.forward(query, kv, {}, {}, sink, batch, cache, selections);
  cache.swa.zero_();
  batch.num_tokens = 5;
  batch.num_scheduled_tokens = {5};
  batch.query_start_loc = {0, 5};
  const auto first = attention.forward(query.slice(0, 0, 5),
                                       kv.slice(0, 0, 5),
                                       {},
                                       {},
                                       sink,
                                       batch,
                                       cache,
                                       selections);
  batch.num_tokens = 8;
  batch.num_scheduled_tokens = {8};
  batch.num_computed_tokens = {5};
  batch.query_start_loc = {0, 8};
  const auto second = attention.forward(query.slice(0, 5),
                                        kv.slice(0, 5),
                                        {},
                                        {},
                                        sink,
                                        batch,
                                        cache,
                                        selections);
  EXPECT_TRUE(torch::allclose(full, torch::cat({first, second}), 1e-6, 1e-6));
  const auto physical =
      kv.index_select(0, torch::tensor({12, 9, 10, 11}, torch::kInt64));
  EXPECT_TRUE(torch::equal(cache.swa.view({4, 2}), physical));
}

TEST(Csa2NativeCachedTest, PackedRequestsUseTheirOwnPages) {
  const Csa2Attention attention(Csa2CachePlan::Mode::SWA, 2, 1, {});
  Csa2LayerCacheView cache;
  cache.swa = torch::zeros({4, 1, 1});
  cache.swa_block_table = torch::tensor({{0, 1}, {2, 3}}, torch::kInt32);
  std::vector<Csa2Selection> selections(2);
  const ExecutionBatchMetadata batch{.num_reqs = 2,
                                     .num_tokens = 5,
                                     .num_scheduled_tokens = {3, 2},
                                     .num_computed_tokens = {0, 0},
                                     .query_start_loc = {0, 3, 5}};
  const auto output = attention.forward(
      torch::zeros({5, 1, 1}),
      torch::tensor({{10.0f}, {20.0f}, {30.0f}, {100.0f}, {200.0f}}),
      {},
      {},
      torch::zeros({1}),
      batch,
      cache,
      selections);
  const auto expected = torch::tensor({5.0f, 10.0f, 50.0f / 3, 50.0f, 100.0f});
  EXPECT_TRUE(torch::allclose(output.flatten(), expected));
}

TEST(Csa2NativeCachedTest, FullReindexAndReuseShareOnlySelectionAndMainCache) {
  Csa2LayerCacheView cache;
  cache.swa = torch::zeros({1, 1, 1});
  cache.swa_block_table = torch::tensor({{0}}, torch::kInt32);
  cache.main = torch::tensor({2.0f, 4.0f, 6.0f, 8.0f}).view({4, 1, 1});
  cache.index = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}).view({4, 1, 1});
  cache.main_block_table = torch::tensor({{0, 1, 2, 3}}, torch::kInt32);
  const ExecutionBatchMetadata batch{.num_reqs = 1,
                                     .num_tokens = 4,
                                     .num_scheduled_tokens = {4},
                                     .num_computed_tokens = {0},
                                     .query_start_loc = {0, 4}};
  std::vector<Csa2Selection> selections(1);
  const Csa2IndexOptions options{.ratio = 1, .topk = 1};
  Csa2Attention full(Csa2CachePlan::Mode::FULL, 1, 1, options);
  const auto query = torch::zeros({4, 1, 1});
  const auto kv = torch::ones({4, 1});
  const auto output = full.forward(query,
                                   kv,
                                   torch::ones({4, 1, 1}),
                                   torch::ones({4, 1}),
                                   torch::zeros({1}),
                                   batch,
                                   cache,
                                   selections);
  const auto source_picks = selections[0].indices.clone();
  Csa2Attention reuse(Csa2CachePlan::Mode::REUSE, 1, 1, options);
  cache.swa = torch::zeros_like(cache.swa);  // next layer owns its own SWA
  const auto reused = reuse.forward(
      query, kv, {}, {}, torch::zeros({1}), batch, cache, selections);
  EXPECT_TRUE(torch::equal(reused, output));
  EXPECT_TRUE(torch::equal(selections[0].indices, source_picks));
  Csa2Attention reindex(Csa2CachePlan::Mode::REINDEX, 1, 1, options);
  reindex.forward(query,
                  kv,
                  torch::ones({4, 1, 1}),
                  -torch::ones({4, 1}),
                  torch::zeros({1}),
                  batch,
                  cache,
                  selections);
  EXPECT_EQ(selections[0].indices[3][0].item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(cache.main.flatten(),
                           torch::tensor({2.0f, 4.0f, 6.0f, 8.0f})));
}

TEST(Csa2NativeCachedTest, IdleDpRowsDoNotTouchCacheOrSelections) {
  const Csa2Attention attention(Csa2CachePlan::Mode::SWA, 4, 2, {});
  const auto query = torch::ones({1, 1, 2});
  std::vector<Csa2Selection> selections;
  const auto output = attention.forward(
      query, {}, {}, {}, {}, ExecutionBatchMetadata{}, {}, selections);
  EXPECT_TRUE(torch::equal(output, torch::zeros_like(query)));
}

}  // namespace
}  // namespace xllm::layer
