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

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/stopping_checker.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {
class ExecutionBatchBuilderTest : public ::testing::TestWithParam<bool> {};

TEST_P(ExecutionBatchBuilderTest, NativeRequirementBuildsRealPrefillMetadata) {
  auto& config = ModelConfig::get_instance();
  const std::string old_impl = config.model_impl();
  ScopeGuard restore([&] { config.model_impl(old_impl); });
  config.model_impl("native");
  RequestSamplingParam sampling;
  StoppingChecker stopping;
  auto make_sequence = [&](std::vector<int32_t> tokens) {
    SequenceParams params;
    params.seq_capacity = 32;
    params.stopping_checker = &stopping;
    params.sampling_param = &sampling;
    params.enable_schedule_overlap = false;
    IncrementalDecoder decoder(/*prompt=*/"",
                               /*num_prompt_tokens=*/tokens.size(),
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    return Sequence(
        /*index=*/0, tokens, torch::Tensor(), MMData{}, decoder, params);
  };
  BlockManager::Options options;
  options.num_blocks(8).block_size(4);
  BlockManagerImpl manager(options);
  auto first = make_sequence({1, 2, 3, 4});
  auto second = make_sequence({5, 6, 7, 8, 9, 10, 11, 12});
  first.add_blocks(BlockType::KV, manager.allocate(1));
  second.add_blocks(BlockType::KV, manager.allocate(2));
  second.kv_state().incr_kv_cache_tokens_num(/*size=*/4);
  ModelArgs args;
  args.requires_execution_batch(GetParam());
  args.engram_layer_ids({1});
  args.engram_max_ngram_size(4);
  Batch batch({&first, &second});
  auto input = batch.prepare_forward_input(args, nullptr);
  EXPECT_TRUE(
      torch::equal(input.token_ids_host,
                   torch::tensor({1, 2, 3, 4, 9, 10, 11, 12}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.positions_host,
                   torch::tensor({0, 1, 2, 3, 4, 5, 6, 7}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.input_params.embedding.engram_token_history,
                   torch::tensor({{-1, -1, -1}, {8, 7, 6}}, torch::kInt32)));
  const auto& metadata = input.input_params.meta.execution_batch;
  if (GetParam()) {
    EXPECT_EQ(metadata.num_reqs, 2);
    EXPECT_EQ(metadata.num_tokens, 8);
    EXPECT_EQ(metadata.num_computed_tokens, std::vector<int32_t>({0, 4}));
    EXPECT_EQ(metadata.num_scheduled_tokens, std::vector<int32_t>({4, 4}));
    EXPECT_EQ(metadata.query_start_loc, std::vector<int32_t>({0, 4, 8}));
    EXPECT_EQ(metadata.is_prefilling, std::vector<uint8_t>({1, 1}));
  } else {
    EXPECT_EQ(metadata.num_reqs, 0);
    EXPECT_EQ(metadata.num_tokens, 0);
    EXPECT_TRUE(metadata.num_scheduled_tokens.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(NativeCapability,
                         ExecutionBatchBuilderTest,
                         ::testing::Bool());
}  // namespace
}  // namespace xllm
