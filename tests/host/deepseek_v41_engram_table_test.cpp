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

#include "core/layers/mlu/deepseek_v41/engram_table.h"

namespace xllm::layer {
namespace {
torch::Tensor lookup(const DeepseekV41EngramTable& table,
                     const torch::Tensor& hashes) {
  torch::Tensor weights =
      torch::empty({hashes.numel(), table.head_dim()}, torch::kUInt8);
  torch::Tensor scales =
      torch::empty({hashes.numel(), table.scale_dim()}, torch::kUInt8);
  table.lookup_rows(hashes, weights, scales);
  return dequantize_deepseek_v41_engram_rows(
      weights, scales, deepseek_v41_fp8_values(torch::kCPU));
}

TEST(DeepseekV41EngramTableTest, CopiesAcrossBoundedLoadChunks) {
  // Exceed the 64 MiB copy bound and probe both sides of the boundary.
  constexpr int64_t kWidth = 128;
  constexpr int64_t kBoundary = 64 * 1024 * 1024 / (kWidth + 1);
  auto weight = torch::full({kBoundary + 2, kWidth}, 0x38, torch::kUInt8);
  auto scale = torch::full({kBoundary + 2, 1}, 127, torch::kUInt8);
  weight[kBoundary - 1].fill_(0xc0);
  scale[kBoundary].fill_(128);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  const auto actual = lookup(
      table,
      torch::tensor({kBoundary - 1, kBoundary, kBoundary + 1}, torch::kInt64));
  EXPECT_TRUE(actual[0].eq(-2).all().item<bool>());
  EXPECT_TRUE(actual[1].eq(2).all().item<bool>());
  EXPECT_TRUE(actual[2].eq(1).all().item<bool>());
}

TEST(DeepseekV41EngramTableTest, DecodesSignedFp8AndGroupedExponentScales) {
  // E4M3FN bytes for [1, -2, 0.5, 4]; exponents [0, 1].
  const auto weight = torch::tensor({{0x38, 0xc0, 0x30, 0x48}}, torch::kUInt8);
  const auto scale = torch::tensor({{127, 128}}, torch::kUInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  const auto actual = lookup(table, torch::tensor({0}, torch::kInt64));
  const auto expected = torch::tensor({{1.0f, -2.0f, 1.0f, 8.0f}});
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(DeepseekV41EngramTableTest, UnevenRankShardsReassembleInRequestOrder) {
  const auto weight = torch::tensor(
      {{0x38, 0xc0}, {0x30, 0x48}, {0x00, 0x40}, {0x3c, 0xb0}, {0x48, 0x38}},
      torch::kUInt8);
  const auto scale =
      torch::tensor({{127}, {128}, {126}, {127}, {129}}, torch::kUInt8);
  const auto hashes = torch::tensor({4, 1, 4, 0, 2, 3}, torch::kInt64);
  torch::Tensor actual = torch::zeros({6, 2});
  for (int64_t rank = 0; rank < 3; ++rank) {
    DeepseekV41EngramTable shard(weight, scale, rank, /*world_size=*/3);
    actual += lookup(shard, hashes);
  }
  const auto expected = torch::tensor({{16.0f, 4.0f},
                                       {1.0f, 8.0f},
                                       {16.0f, 4.0f},
                                       {1.0f, -2.0f},
                                       {0.0f, 1.0f},
                                       {1.5f, -0.5f}});
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(DeepseekV41EngramTableTest, OwnsDataAfterCheckpointStorageIsReleased) {
  auto weight = torch::full({2, 4}, 0x38, torch::kUInt8);
  auto scale = torch::full({2, 1}, 127, torch::kUInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  weight.zero_();
  scale.zero_();
  weight = torch::Tensor();
  scale = torch::Tensor();
  EXPECT_TRUE(torch::equal(lookup(table, torch::tensor({1}, torch::kInt64)),
                           torch::ones({1, 4})));
}

TEST(DeepseekV41EngramTableTest, ReusedStagingClearsRowsOwnedByOtherRanks) {
  const auto weight = torch::full({4, 2}, 0x38, torch::kUInt8);
  const auto scale = torch::full({4, 1}, 127, torch::kUInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/1, /*world_size=*/2);
  torch::Tensor weights = torch::empty({2, 2}, torch::kUInt8);
  torch::Tensor scales = torch::empty({2, 1}, torch::kUInt8);
  table.lookup_rows(torch::tensor({2, 3}, torch::kInt64), weights, scales);
  EXPECT_EQ(weights.sum().item<int64_t>(), 4 * 0x38);
  table.lookup_rows(torch::tensor({0, 1}, torch::kInt64), weights, scales);
  EXPECT_EQ(weights.sum().item<int64_t>(), 0);
  EXPECT_EQ(scales.sum().item<int64_t>(), 0);
}

TEST(DeepseekV41EngramTableTest, EmptyShardAndEmptyBatchDoNotAccessStorage) {
  const auto weight = torch::full({1, 2}, 0x38, torch::kUInt8);
  const auto scale = torch::full({1, 1}, 127, torch::kUInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/4);
  EXPECT_EQ(table.begin(), table.end());
  EXPECT_TRUE(torch::equal(lookup(table, torch::tensor({0}, torch::kInt64)),
                           torch::zeros({1, 2})));
  EXPECT_EQ(lookup(table, torch::empty({0}, torch::kInt64)).numel(), 0);
}

TEST(DeepseekV41EngramTableTest, ReadsNoncontiguousCheckpointViews) {
  const auto weight = torch::full({4, 3}, 0x38, torch::kUInt8).t();
  const auto scale = torch::full({2, 3}, 128, torch::kUInt8).t();
  ASSERT_FALSE(weight.is_contiguous());
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  EXPECT_TRUE(torch::equal(lookup(table, torch::tensor({2, 0}, torch::kInt64)),
                           torch::full({2, 4}, 2.0f)));
}

TEST(DeepseekV41EngramTableTest,
     TypedFp8StorageIsReinterpretedNotNumericallyCast) {
  const auto weight =
      torch::tensor({{0x38, 0xc0}}, torch::kUInt8).view(torch::kFloat8_e4m3fn);
  const auto scale =
      torch::tensor({{128}}, torch::kUInt8).view(torch::kFloat8_e8m0fnu);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  EXPECT_TRUE(torch::equal(lookup(table, torch::tensor({0}, torch::kInt64)),
                           torch::tensor({{2.0f, -4.0f}})));
}

TEST(DeepseekV41EngramTableTest, SignedStoragePreservesRawBytes) {
  const auto weight =
      torch::tensor({{0x38, 0xc0}}, torch::kUInt8).view(torch::kInt8);
  const auto scale = torch::tensor({{128}}, torch::kUInt8).view(torch::kInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  EXPECT_TRUE(torch::equal(lookup(table, torch::tensor({0}, torch::kInt64)),
                           torch::tensor({{2.0f, -4.0f}})));
}

TEST(DeepseekV41EngramTableTest, ScaleEndpointsAndInvalidNanByte) {
  const auto weight = torch::full({2, 1}, 0x38, torch::kUInt8);
  const auto scale = torch::tensor({{0}, {254}}, torch::kUInt8);
  DeepseekV41EngramTable table(weight, scale, /*rank=*/0, /*world_size=*/1);
  const auto actual = lookup(table, torch::tensor({0, 1}, torch::kInt64));
  EXPECT_EQ(actual[0][0].item<float>(), 0x1p-127f);
  EXPECT_EQ(actual[1][0].item<float>(), 0x1p127f);
  EXPECT_DEATH((DeepseekV41EngramTable(
                   weight, torch::full({2, 1}, 255, torch::kUInt8), 0, 1)),
               "invalid NaN byte");
}

TEST(DeepseekV41EngramTableTest, RejectsInvalidCheckpointLayoutAndRank) {
  const auto weight = torch::zeros({4, 4}, torch::kUInt8);
  const auto scale = torch::zeros({4, 1}, torch::kUInt8);
  EXPECT_DEATH((DeepseekV41EngramTable(weight, scale, 2, 2)), "rank");
  EXPECT_DEATH(
      (DeepseekV41EngramTable(weight.to(torch::kFloat32), scale, 0, 1)),
      "E4M3FN bytes");
  EXPECT_DEATH(
      (DeepseekV41EngramTable(weight, scale.to(torch::kFloat32), 0, 1)),
      "UE8M0 bytes");
  EXPECT_DEATH((DeepseekV41EngramTable(
                   weight, torch::zeros({4, 3}, torch::kUInt8), 0, 1)),
               "scale groups");
  EXPECT_DEATH((DeepseekV41EngramTable(
                   weight, torch::zeros({3, 1}, torch::kUInt8), 0, 1)),
               "size");
}
}  // namespace
}  // namespace xllm::layer
