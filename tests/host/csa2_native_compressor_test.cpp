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
#include <torch/torch.h>

#include <cmath>
#include <vector>

#include "core/layers/mlu/deepseek_v41/csa2_attention.h"
#include "core/layers/mlu/deepseek_v41/csa2_compressor.h"

namespace xllm::layer {
namespace {
ExecutionBatchMetadata batch_of(std::vector<int32_t> lengths,
                                std::vector<int32_t> begins) {
  ExecutionBatchMetadata batch;
  batch.num_reqs = static_cast<int32_t>(lengths.size());
  batch.num_scheduled_tokens = std::move(lengths);
  batch.num_computed_tokens = std::move(begins);
  batch.query_start_loc.emplace_back(0);
  batch.query_start_loc.reserve(batch.num_reqs + 1);
  for (int32_t length : batch.num_scheduled_tokens) {
    batch.query_start_loc.emplace_back(batch.query_start_loc.back() + length);
  }
  batch.num_tokens = batch.query_start_loc.back();
  return batch;
}
Csa2CompressionCacheView make_cache() {
  return {torch::full({8, 32, 4}, -19, torch::kBFloat16),
          torch::full({8, 32, 8}, -23, torch::kFloat32),
          torch::tensor({{4, 1, 5, 7}, {2, 3, 6, 0}}, torch::kInt64),
          torch::tensor({{3, 0}, {1, 2}}, torch::kInt64)};
}
torch::Tensor rope_table() {
  const auto angle = torch::arange(256, torch::kFloat32).unsqueeze(1) * 0.01;
  return torch::cat({angle.cos().repeat({1, 2}), angle.sin().repeat({1, 2})},
                    1);
}

TEST(Csa2NativeCompressorTest, ScalarOraclePreservesRoundingAndStartPosition) {
  auto cache = make_cache();
  const auto raw = torch::tensor({{0.13f, -1.04f, 2.01f, 0.9f},
                                  {1.71f, 0.23f, -0.61f, 1.03f},
                                  {2.22f, 0.01f, 0.05f, -3.2f},
                                  {0.98f, 4.37f, 2.49f, 0.99f}});
  const auto scores = torch::tensor({{30.f, -20.f, 0.01f, 0.99f},
                                     {29.f, -19.f, 0.05f, -0.11f},
                                     {-100.f, 1.f, 0.f, 5.f},
                                     {100.f, 3.f, 0.f, 6.f}});
  const auto gamma = torch::tensor({0.9f, 1.2f, 1.03f, 0.7f});
  const auto rope = rope_table();
  Csa2Compressor compressor(2, 32, 2, 1e-6);
  const auto result =
      compressor.forward(raw, scores, gamma, rope, batch_of({4}, {0}), cache);
  EXPECT_TRUE(
      torch::equal(result.positions, torch::tensor({0, 2}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(result.slots, torch::tensor({128, 129}, torch::kInt64)));
  auto expected = torch::empty({2, 4}, torch::kFloat32);
  for (int64_t group = 0; group < 2; ++group) {
    double values[4];
    double square_sum = 0;
    for (int64_t col = 0; col < 4; ++col) {
      const double a = scores[group * 2][col].item<double>();
      const double b = scores[group * 2 + 1][col].item<double>();
      const double p = 1.0 / (1.0 + std::exp(b - a));
      const double pooled = p * raw[group * 2][col].item<double>() +
                            (1 - p) * raw[group * 2 + 1][col].item<double>();
      values[col] = torch::tensor(pooled).to(torch::kBFloat16).item<double>();
      square_sum += values[col] * values[col];
    }
    for (int64_t col = 0; col < 4; ++col) {
      expected[group][col] = values[col] / std::sqrt(square_sum / 4 + 1e-6) *
                             gamma[col].item<double>();
    }
  }
  expected = expected.to(torch::kBFloat16);
  EXPECT_TRUE(torch::equal(result.latent, expected));
  auto rotated = expected.to(torch::kFloat32);
  for (int64_t row = 0; row < 2; ++row) {
    const double a = expected[row][2].item<double>();
    const double b = expected[row][3].item<double>();
    rotated[row][2] = a * std::cos(row * 0.02) - b * std::sin(row * 0.02);
    rotated[row][3] = a * std::sin(row * 0.02) + b * std::cos(row * 0.02);
  }
  EXPECT_TRUE(
      torch::equal(cache.main[4].slice(0, 0, 2), rotated.to(torch::kBFloat16)));
  EXPECT_EQ(cache.main[1][0][0].item<float>(), -19);
  EXPECT_TRUE(torch::all(cache.state == -23).item<bool>());
}

TEST(Csa2NativeCompressorTest, ChunkDecodeMatchesWholeAcrossPagesAndStateRing) {
  const auto raw =
      (torch::arange(300, torch::kFloat32).sin() * 3).view({75, 4});
  const auto scores = torch::arange(300, torch::kFloat32).cos().view({75, 4});
  const auto norm = torch::ones({4});
  const auto rope = rope_table();
  for (bool full_tail : {false, true}) {
    Csa2Compressor compressor(2, 32, 2, 1e-6, full_tail);
    auto whole_cache = make_cache(), chunk_cache = make_cache();
    const auto whole = compressor.forward(
        raw, scores, norm, rope, batch_of({75}, {0}), whole_cache);
    std::vector<torch::Tensor> rows, pos;
    rows.reserve(6);
    pos.reserve(6);
    int32_t offset = 0;
    for (int32_t length : {31, 1, 1, 32, 9, 1}) {
      const auto part =
          compressor.forward(raw.slice(0, offset, offset + length),
                             scores.slice(0, offset, offset + length),
                             norm,
                             rope,
                             batch_of({length}, {offset}),
                             chunk_cache);
      rows.emplace_back(part.latent);
      pos.emplace_back(part.positions);
      offset += length;
    }
    EXPECT_TRUE(torch::equal(torch::cat(rows), whole.latent));
    EXPECT_TRUE(torch::equal(torch::cat(pos), whole.positions));
    EXPECT_TRUE(torch::equal(chunk_cache.main, whole_cache.main));
    // The pending odd token must occupy the state page selected by SWA.
    EXPECT_TRUE(torch::equal(chunk_cache.state[3][10].slice(0, 0, 4), raw[74]));
    EXPECT_TRUE(torch::equal(chunk_cache.state[3][10].slice(0, 4), scores[74]));
    if (full_tail) {
      for (int64_t token = 43; token < 75; ++token) {
        const int64_t page = (token / 32) % 2 == 0 ? 3 : 0;
        EXPECT_TRUE(torch::equal(chunk_cache.state[page][token % 32],
                                 whole_cache.state[page][token % 32]));
      }
    }
  }
}

TEST(Csa2NativeCompressorTest, PackedSequencesKeepIndependentOddTails) {
  Csa2Compressor compressor(2, 32, 0, 1e-6);
  auto cache = make_cache();
  const auto norm = torch::ones({4});
  const auto first =
      torch::tensor({{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}});
  const auto zero = torch::zeros_like(first);
  const auto pending = compressor.forward(
      first, zero, norm, {}, batch_of({1, 1}, {0, 0}), cache);
  EXPECT_EQ(pending.latent.size(0), 0);
  const auto second = first + 4;
  const auto completed = compressor.forward(
      second, zero, norm, {}, batch_of({1, 1}, {1, 1}), cache);
  auto reference = make_cache();
  const auto all = torch::stack({first[0], second[0], first[1], second[1]});
  const auto expected = compressor.forward(all,
                                           torch::zeros_like(all),
                                           norm,
                                           {},
                                           batch_of({2, 2}, {0, 0}),
                                           reference);
  EXPECT_TRUE(torch::equal(expected.latent, completed.latent));
  EXPECT_TRUE(torch::equal(reference.main, cache.main));
  EXPECT_TRUE(
      torch::equal(completed.slots, torch::tensor({128, 64}, torch::kInt64)));
}

TEST(Csa2NativeCompressorTest, RatioOneAndIdleDoNotRequireState) {
  Csa2Compressor compressor(1, 32, 0, 1e-6);
  auto cache = make_cache();
  cache.state = torch::Tensor();
  cache.state_block_table = torch::Tensor();
  const auto raw = torch::tensor({{1.f, -1.f, 1.f, -1.f}}, torch::kBFloat16);
  const auto result = compressor.forward(
      raw, {}, torch::ones({4}), {}, batch_of({1, 0}, {33, 0}), cache);
  EXPECT_TRUE(torch::equal(result.latent, raw));
  EXPECT_EQ(result.slots[0].item<int64_t>(), 33);
  EXPECT_EQ(result.positions[0].item<int64_t>(), 33);
  const auto untouched = cache.main.clone();
  const auto idle = compressor.forward(raw, {}, torch::ones({4}), {}, {}, {});
  EXPECT_EQ(idle.latent.size(0), 0);
  EXPECT_TRUE(torch::equal(untouched, cache.main));
}

TEST(Csa2NativeCompressorTest, CompressionAndAttentionComposeAcrossOddChunk) {
  const auto raw = torch::arange(28, torch::kFloat32).sin().view({7, 4});
  const auto scores = torch::zeros_like(raw);
  const auto query = raw.unsqueeze(1).to(torch::kBFloat16);
  const auto kv = (raw * 2).to(torch::kBFloat16);
  Csa2Compressor compressor(2, 32, 0, 1e-6);
  Csa2Attention attention(
      Csa2CachePlan::Mode::FULL, 4, 32, {.ratio = 2, .topk = 4});
  auto run = [&](const std::vector<int32_t>& chunks) {
    auto cache = make_cache();
    Csa2LayerCacheView view{torch::zeros({8, 32, 4}, torch::kBFloat16),
                            cache.main,
                            torch::zeros_like(cache.main),
                            cache.state_block_table,
                            cache.main_block_table};
    std::vector<torch::Tensor> outputs;
    outputs.reserve(chunks.size());
    int32_t offset = 0;
    for (int32_t count : chunks) {
      const auto batch = batch_of({count}, {offset});
      const auto compressed =
          compressor.forward(raw.slice(0, offset, offset + count),
                             scores.slice(0, offset, offset + count),
                             torch::ones({4}),
                             {},
                             batch,
                             cache);
      // Index projection is identity for this integrated cache-boundary test.
      view.index.view({-1, 4}).index_copy_(
          0, compressed.slots, compressed.latent);
      std::vector<Csa2Selection> selections(1);
      outputs.emplace_back(
          attention.forward(query.slice(0, offset, offset + count),
                            kv.slice(0, offset, offset + count),
                            query.slice(0, offset, offset + count),
                            torch::ones({count, 1}),
                            torch::zeros({1}),
                            batch,
                            view,
                            selections));
      offset += count;
    }
    return torch::cat(outputs);
  };
  EXPECT_TRUE(torch::equal(run({7}), run({3, 1, 2, 1})));
}

TEST(Csa2NativeCompressorTest, RejectsMalformedPackingBeforeMutation) {
  Csa2Compressor compressor(2, 32, 0, 1e-6);
  auto cache = make_cache();
  auto batch = batch_of({2}, {0});
  batch.query_start_loc.back() = 1;
  EXPECT_DEATH(compressor.forward(torch::ones({2, 4}),
                                  torch::ones({2, 4}),
                                  torch::ones({4}),
                                  {},
                                  batch,
                                  cache),
               "num_tokens");
}
}  // namespace
}  // namespace xllm::layer
