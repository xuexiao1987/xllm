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

#include "core/layers/mlu/deepseek_v41/attention_projection.h"

namespace xllm::layer::deepseek_v41 {
namespace {
TEST(DeepseekV41ProjectionTest, NormPreservesFloat32GammaBeforeBfloatRounding) {
  const auto x =
      torch::tensor({{1.25f, -3.5f, 0.0625f, 8.f}}, torch::kBFloat16);
  const auto gamma = torch::tensor({1.003f, 0.997f, 1.007f, 1.004f});
  const auto actual = norm(x, gamma, 1e-6);
  const auto xf = x.to(torch::kFloat64);
  double sum = 0;
  for (int64_t col = 0; col < 4; ++col) {
    sum += std::pow(xf[0][col].item<double>(), 2);
  }
  auto expected = torch::empty({1, 4}, torch::kFloat64);
  for (int64_t col = 0; col < 4; ++col) {
    expected[0][col] = xf[0][col].item<double>() * gamma[col].item<double>() /
                       std::sqrt(sum / 4 + 1e-6);
  }
  EXPECT_TRUE(torch::equal(actual, expected.to(torch::kBFloat16)));
}

TEST(DeepseekV41ProjectionTest,
     TailRotationUsesAbsolutePositionAndInterleavedPairs) {
  const auto x = torch::arange(1, 17, torch::kFloat32).view({2, 2, 4});
  const auto positions = torch::tensor({1, 0}, torch::kInt32);
  const auto rope = torch::tensor({{1.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}});
  const auto actual = rotate(x.clone(), positions, rope, /*dim=*/2);
  const auto expected =
      torch::tensor({{{1.f, 2.f, -4.f, 3.f}, {5.f, 6.f, -8.f, 7.f}},
                     {{9.f, 10.f, 11.f, 12.f}, {13.f, 14.f, 15.f, 16.f}}});
  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_TRUE(torch::equal(rotate(actual.clone(),
                                  positions,
                                  rope,
                                  /*dim=*/2,
                                  /*inverse=*/true),
                           x));
}

TEST(DeepseekV41ProjectionTest,
     GroupedCheckpointAndTensorParallelShardsMatchScalarOracle) {
  const auto heads = torch::arange(48, torch::kFloat32)
                         .view({3, 8, 2})
                         .div(4)
                         .to(torch::kBFloat16);
  // Released wo_a checkpoints store [output groups, rank, group input].
  const auto weight = torch::arange(32, torch::kFloat32)
                          .view({4, 2, 4})
                          .sub(16)
                          .div(4)
                          .to(torch::kBFloat16);
  auto expected = torch::zeros({3, 8});
  const auto input = heads.to(torch::kFloat32).reshape({3, 16});
  const auto matrix = weight.to(torch::kFloat32);
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t group = 0; group < 4; ++group) {
      for (int64_t out = 0; out < 2; ++out) {
        double value = 0;
        for (int64_t col = 0; col < 4; ++col) {
          value += input[row][group * 4 + col].item<double>() *
                   matrix[group][out][col].item<double>();
        }
        expected[row][group * 2 + out] = value;
      }
    }
  }
  const auto full = grouped_output(heads, weight, /*groups=*/4, /*rank_dim=*/2);
  EXPECT_TRUE(torch::equal(full, expected.to(torch::kBFloat16)));
  const auto left =
      grouped_output(heads.slice(1, 0, 4), weight.slice(0, 0, 2), 2, 2);
  const auto right =
      grouped_output(heads.slice(1, 4, 8), weight.slice(0, 2, 4), 2, 2);
  EXPECT_TRUE(torch::equal(torch::cat({left, right}, 1), full));
}
TEST(DeepseekV41ProjectionTest, FlattenedCheckpointPreservesGroupsAndTpShards) {
  const auto heads = torch::arange(48, torch::kFloat32).reshape({3, 8, 2});
  const auto grouped = torch::arange(32, torch::kFloat32).reshape({4, 2, 4});
  const auto flat = grouped.reshape({8, 4});
  const auto normalized = grouped_output_weight(flat, 4, 2, 4);
  EXPECT_TRUE(torch::equal(normalized, grouped));
  EXPECT_TRUE(torch::equal(grouped_output_weight(grouped, 4, 2, 4), grouped));
  const auto full = grouped_output(heads, grouped, 4, 2);
  const auto left =
      grouped_output(heads.slice(1, 0, 4), normalized.slice(0, 0, 2), 2, 2);
  const auto right =
      grouped_output(heads.slice(1, 4, 8), normalized.slice(0, 2, 4), 2, 2);
  EXPECT_TRUE(torch::equal(torch::cat({left, right}, 1), full));
}

TEST(DeepseekV41ProjectionTest, RejectsSameSizeButWrongCheckpointLayout) {
  const auto weight = torch::zeros({4, 8});
  EXPECT_DEATH(grouped_output_weight(weight, 4, 2, 4), "Invalid shape");
}
}  // namespace
}  // namespace xllm::layer::deepseek_v41
