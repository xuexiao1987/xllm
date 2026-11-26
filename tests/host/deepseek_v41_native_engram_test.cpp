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
#include <fstream>
#include <nlohmann/json.hpp>

#include "core/layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_hash.h"
#include "core/layers/mlu/deepseek_v41/engram_table.h"

namespace xllm::layer {
namespace {
DeepseekV41EngramHash small_hash() {
  return DeepseekV41EngramHash(torch::arange(8, torch::kInt64),
                               torch::tensor({3, 5, 7}, torch::kInt64),
                               torch::tensor({5, 7}, torch::kInt64),
                               torch::tensor({0, 5}, torch::kInt64),
                               /*pad_token_id=*/2,
                               /*compressed_vocab_size=*/8,
                               /*table_rows=*/12);
}

TEST(DeepseekV41NativeEngramTest, MatchesFrozenInternalHashForBothLayers) {
  std::ifstream stream(std::string(XLLM_HOST_DATA_DIR) +
                       "/deepseek_v41_engram_hash.json");
  ASSERT_TRUE(stream.good());
  const auto fixture = nlohmann::json::parse(stream);
  ASSERT_EQ(fixture["reference_sha"],
            "5026beb058c166cdd108f7feddc7dbbec317485d");
  const auto ids = torch::tensor(fixture["inputs"].get<std::vector<int64_t>>(),
                                 torch::kInt64);
  const auto history = torch::tensor({{-1, -1, -1}, {2, 1, -1}}, torch::kInt64);
  const auto parameters = build_deepseek_v41_hash_parameters(
      {{1, 14}, {384006168, 384016682}, 4, 8, 16000000, 8});
  const DeepseekV41EngramInput input(ids, history);
  EXPECT_EQ(input.tokens_.data_ptr(), ids.data_ptr());
  EXPECT_EQ(input.history_.data_ptr(), history.data_ptr());
  const auto saved_ids = ids.clone();
  const auto saved_history = history.clone();
  for (int64_t layer = 0; layer < 2; ++layer) {
    const auto sizes =
        torch::tensor(fixture["head_sizes"][layer].get<std::vector<int64_t>>(),
                      torch::kInt64);
    const auto offsets = sizes.cumsum(0) - sizes;
    EXPECT_TRUE(torch::equal(parameters.primes[layer], sizes));
    EXPECT_TRUE(torch::equal(parameters.offsets[layer], offsets));
    EXPECT_TRUE(torch::equal(
        parameters.multipliers[layer],
        torch::tensor(fixture["multipliers"][layer].get<std::vector<int64_t>>(),
                      torch::kInt64)));
    DeepseekV41EngramHash hasher(torch::arange(8, torch::kInt64),
                                 parameters.multipliers[layer],
                                 parameters.primes[layer],
                                 parameters.offsets[layer],
                                 2,
                                 8,
                                 sizes.sum().item<int64_t>());
    const auto actual = hasher.forward(input.tokens_, input.history_, {5, 3});
    for (int64_t row = 0; row < ids.numel(); ++row) {
      const auto expected = torch::tensor(
          fixture["hashes"][row][layer].get<std::vector<int64_t>>(),
          torch::kInt64);
      EXPECT_TRUE(torch::equal(actual[row], expected));
    }
  }
  EXPECT_TRUE(torch::equal(ids, saved_ids));
  EXPECT_TRUE(torch::equal(history, saved_history));
}

TEST(DeepseekV41NativeEngramTest, NormalizesPackedInt32InputWithoutMutation) {
  const auto tokens = torch::tensor({1, 2, 3}, torch::kInt32);
  const auto history = torch::tensor({{-1, -1}, {4, 2}}, torch::kInt32);
  const DeepseekV41EngramInput input(tokens, history);
  EXPECT_EQ(input.tokens_.scalar_type(), torch::kInt64);
  EXPECT_EQ(input.history_.scalar_type(), torch::kInt64);
  const auto hashes =
      small_hash().forward(input.tokens_, input.history_, {2, 1});
  const auto first = small_hash().forward(
      input.tokens_.slice(0, 0, 2), input.history_.slice(0, 0, 1), {2});
  const auto second = small_hash().forward(
      input.tokens_.slice(0, 2, 3), input.history_.slice(0, 1, 2), {1});
  EXPECT_TRUE(torch::equal(hashes, torch::cat({first, second})));
  EXPECT_TRUE(torch::equal(tokens, input.tokens_));
  EXPECT_TRUE(torch::equal(history, input.history_));
}

TEST(DeepseekV41NativeEngramTest, BucketPrimesAreUniqueAcrossLayers) {
  const auto parameters =
      build_deepseek_v41_hash_parameters({{1, 14}, {36, 88}, 3, 2, 5, 8});
  EXPECT_TRUE(torch::equal(
      parameters.primes,
      torch::tensor({{5, 7, 11, 13}, {17, 19, 23, 29}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      parameters.offsets,
      torch::tensor({{0, 5, 12, 23}, {0, 17, 36, 59}}, torch::kInt64)));
  const auto reordered =
      build_deepseek_v41_hash_parameters({{14, 1}, {36, 88}, 3, 2, 5, 8});
  EXPECT_TRUE(
      torch::equal(parameters.multipliers.flip({0}), reordered.multipliers));
}

TEST(DeepseekV41NativeEngramTest,
     RejectsInsufficientTableAndInvalidHashGeometry) {
  EXPECT_DEATH(build_deepseek_v41_hash_parameters({{1}, {35}, 3, 2, 5, 8}),
               "exceed embedding");
  EXPECT_DEATH(build_deepseek_v41_hash_parameters({{1}, {36}, 1, 2, 5, 8}),
               "max_ngram_size");
  EXPECT_DEATH(build_deepseek_v41_hash_parameters({{-1}, {36}, 3, 2, 5, 8}),
               "id");
}

TEST(DeepseekV41NativeEngramTest,
     ChunkedRequestsAndEmptyRowsKeepHistoryIsolated) {
  const auto hasher = small_hash();
  const auto full =
      hasher.forward(torch::tensor({3, 4, 5, 6, 7, 1}, torch::kInt64),
                     torch::full({2, 2}, -1, torch::kInt64),
                     {4, 2});
  const auto first = hasher.forward(torch::tensor({3, 4, 7}, torch::kInt64),
                                    torch::full({2, 2}, -1, torch::kInt64),
                                    {2, 1});
  const auto second =
      hasher.forward(torch::tensor({5, 6, 1}, torch::kInt64),
                     torch::tensor({{4, 3}, {7, -1}}, torch::kInt64),
                     {2, 1});
  EXPECT_TRUE(torch::equal(
      first, full.index_select(0, torch::tensor({0, 1, 4}, torch::kInt64))));
  EXPECT_TRUE(torch::equal(
      second, full.index_select(0, torch::tensor({2, 3, 5}, torch::kInt64))));
  const auto decode =
      hasher.forward(torch::tensor({6}, torch::kInt64),
                     torch::tensor({{-1, -1}, {5, 4}, {-1, -1}}, torch::kInt64),
                     {0, 1, 0});
  EXPECT_TRUE(torch::equal(decode, full.slice(0, 3, 4)));
}

TEST(DeepseekV41NativeEngramTest, ImageAndNegativeHistoryStopOlderLookback) {
  const auto hasher = small_hash();
  const auto actual = hasher.forward(
      torch::tensor({5, 5, 5, 5}, torch::kInt64),
      torch::tensor({{-1, 7}, {129264, 3}, {129265, 4}, {2, 2}}, torch::kInt64),
      {1, 1, 1, 1});
  for (int64_t row = 1; row < 4; ++row) {
    EXPECT_TRUE(torch::equal(actual[0], actual[row]));
  }
  EXPECT_EQ(hasher
                .forward(torch::empty({0}, torch::kInt64),
                         torch::empty({0, 2}, torch::kInt64),
                         {})
                .sizes(),
            torch::IntArrayRef({0, 2}));
}

TEST(DeepseekV41NativeEngramTest,
     HashTableAndProjectionFormActualLocalForward) {
  const auto hasher = small_hash();
  const auto tokens = torch::tensor({3, 129264}, torch::kInt64);
  const auto hashes =
      hasher.forward(tokens, torch::full({1, 2}, -1, torch::kInt64), {2});
  DeepseekV41EngramTable table(torch::full({12, 2}, 0x38, torch::kUInt8),
                               torch::full({12, 1}, 127, torch::kUInt8),
                               0,
                               1);
  auto values = torch::empty({hashes.numel(), 2}, torch::kUInt8);
  auto scales = torch::empty({hashes.numel(), 1}, torch::kUInt8);
  table.lookup_rows(hashes, values, scales);
  const auto rows = dequantize_deepseek_v41_engram_rows(
                        values, scales, deepseek_v41_fp8_values(torch::kCPU))
                        .view({2, 2, 2});
  const auto hidden = torch::ones({2, 2, 2});
  const auto actual = deepseek_v41_engram_project(hidden,
                                                  rows,
                                                  tokens,
                                                  torch::ones({6, 4}),
                                                  torch::ones({2, 2}),
                                                  torch::ones({2, 2}),
                                                  1e-6);
  const double dot =
      8.0 / std::sqrt(1.0 + 1e-6) / std::sqrt(16.0 + 1e-6) / std::sqrt(2.0);
  const double expected = 1.0 + 4.0 / (1.0 + std::exp(-std::sqrt(dot)));
  EXPECT_TRUE(
      torch::allclose(actual[0], torch::full({2, 2}, expected), 1e-6, 1e-6));
  EXPECT_TRUE(torch::equal(actual[1], hidden[1]));
}

TEST(DeepseekV41NativeEngramTest, RejectsBadHashBucketsAndRequestPacking) {
  EXPECT_DEATH((DeepseekV41EngramHash(torch::arange(8, torch::kInt64),
                                      torch::tensor({3, 5}, torch::kInt64),
                                      torch::tensor({7}, torch::kInt64),
                                      torch::tensor({9}, torch::kInt64),
                                      2,
                                      8,
                                      12)),
               "buckets exceed");
  const auto hasher = small_hash();
  EXPECT_DEATH(hasher.forward(torch::tensor({3}, torch::kInt64),
                              torch::full({1, 2}, -1, torch::kInt64),
                              {2}),
               "rows");
}
}  // namespace
}  // namespace xllm::layer
