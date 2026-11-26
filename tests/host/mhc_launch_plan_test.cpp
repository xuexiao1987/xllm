/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <vector>

#include "core/kernels/mlu/bangc/src/fused_mhc_common/launch_plan.h"

namespace xllm::kernel::mlu::bangc::fused_mhc_detail {
namespace {
TEST(MhcLaunchPlanTest, CoversEveryTokenExactlyOnceAcrossGroupsAndRounds) {
  for (int32_t clusters : {1, 2, 4, 8, 16}) {
    for (int64_t tokens : {1,
                           3,
                           8,
                           9,
                           17,
                           33,
                           63,
                           65,
                           88,
                           89,
                           128,
                           129,
                           257,
                           300,
                           640,
                           1024,
                           2049,
                           4097}) {
      SCOPED_TRACE(clusters);
      SCOPED_TRACE(tokens);
      const Plan plan = make_plan(tokens, Topology{clusters, 4});
      EXPECT_EQ(plan.groups * plan.p, clusters * 4);
      EXPECT_LE(plan.round_tokens, round_cap(plan.p));
      EXPECT_GT(plan.round_tokens, 0);
      std::vector<int32_t> visits(tokens, 0);
      for (int32_t group = 0; group < plan.groups; ++group) {
        for (int32_t round = 0; round < plan.rounds; ++round) {
          const int64_t begin =
              group * plan.group_tokens + round * plan.round_tokens;
          const int64_t end =
              std::min({begin + plan.round_tokens,
                        static_cast<int64_t>((group + 1) * plan.group_tokens),
                        tokens});
          for (int64_t token = begin; token < end; ++token) {
            ++visits[token];
          }
        }
      }
      EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](int32_t visits) {
        return visits == 1;
      }));
    }
  }
}

TEST(MhcLaunchPlanTest, KeepsLargeValidCountsWithinIntegerBounds) {
  const int64_t tokens = int64_t{1} << 30;
  for (int32_t clusters : {1, 2, 4, 8, 16}) {
    const Plan plan = make_plan(tokens, Topology{clusters, 4});
    EXPECT_GE(static_cast<int64_t>(plan.groups) * plan.group_tokens, tokens);
    EXPECT_GE(static_cast<int64_t>(plan.rounds) * plan.round_tokens,
              plan.group_tokens);
    EXPECT_LT(static_cast<int64_t>(plan.rounds - 1) * plan.round_tokens,
              plan.group_tokens);
  }
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc::fused_mhc_detail
