/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kv_cache/layout/csa2_cache_plan.h"

#include <gtest/gtest.h>
namespace xllm {
TEST(Csa2CachePlanTest, SharedLayersDoNotMultiplyMainCacheOrState) {
  const Csa2CachePlan plan({0, 2, 2, 1, 1, 1}, {1, 3});
  EXPECT_EQ(plan.source_count(2), 1);
  EXPECT_EQ(plan.source_count(1), 1);
  EXPECT_FALSE(plan.owns_main(2));
  EXPECT_TRUE(plan.owns_main(3));
  EXPECT_EQ(plan.swa_block_bytes(4, 16, 2), 6 * 4 * 16 * 2 + 32 * 16 * 8);
  EXPECT_EQ(plan.token_unit_bytes(4, 16, 8, 2), 3 * 4 * (32 + 8 + 2));
}
TEST(Csa2CachePlanTest, C1OnlyDoesNotReserveCompressorState) {
  const Csa2CachePlan plan({1, 1}, {0});
  EXPECT_EQ(plan.swa_block_bytes(128, 512, 2), 2 * 128 * 512 * 2);
  EXPECT_EQ(plan.token_unit_bytes(128, 512, 128, 2),
            2 * 128 * (1024 + 128 + 2));
}
TEST(Csa2CachePlanTest, RejectsInvalidOwnership) {
  EXPECT_DEATH((Csa2CachePlan({0, 2}, {0})), "ratio");
  EXPECT_DEATH((Csa2CachePlan({2, 2}, {1, 0})), "source");
  EXPECT_DEATH((Csa2CachePlan({2, 2}, {2})), "source");
}
}  // namespace xllm
