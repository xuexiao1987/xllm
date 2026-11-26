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

#include "core/kv_cache/layout/csa2_cache_plan.h"
#include "core/kv_cache/layout/deepseek_v4_cache_policy.h"

namespace xllm {
TEST(Csa2MluLayoutTest, MixedC1C2BudgetUsesBfloat16IndexWithoutScale) {
  const auto policy = get_dsv4_cache_policy(torch::kBFloat16);
  EXPECT_EQ(policy.index_dtype, torch::kBFloat16);
  EXPECT_FALSE(policy.has_indexer_cache_scale);
  const Csa2CachePlan plan({0, 2, 2, 1, 1}, {1, 3});
  EXPECT_EQ(
      plan.token_unit_bytes(
          128, 512, 128, 2, policy.index_dtype_size, policy.scale_dtype_size),
      491520);
  // The NPU path retains INT8 indices and one FP16 scale per token.
  EXPECT_EQ(plan.token_unit_bytes(128, 512, 128, 2), 443136);
  EXPECT_EQ(plan.swa_block_bytes(128, 512, 2), 786432);
}
TEST(Csa2MluLayoutTest, C1OnlyDoesNotAllocateCompressorState) {
  const auto policy = get_dsv4_cache_policy(torch::kBFloat16);
  const Csa2CachePlan plan({1, 1}, {0});
  EXPECT_EQ(
      plan.token_unit_bytes(
          128, 512, 128, 2, policy.index_dtype_size, policy.scale_dtype_size),
      327680);
  EXPECT_EQ(plan.swa_block_bytes(128, 512, 2), 262144);
}
}  // namespace xllm
