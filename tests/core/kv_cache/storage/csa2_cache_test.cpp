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

#include <gtest/gtest.h>

#include "core/framework/model/model_args.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"
#include "worker.pb.h"
namespace xllm {
TEST(Csa2CacheTest, EstimatedBudgetMatchesActualOwnedCpuTensors) {
  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(6)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(16)
      .index_head_dim(8)
      .index_n_heads(2)
      .window_size(8)
      .compress_ratios({0, 2, 2, 1, 1, 1})
      .kv_source_layer_ids({1, 3});
  KVCacheEstimateOptions estimate;
  estimate.dtype = torch::kBFloat16;
  estimate.block_size = 4;
  estimate.world_size = 1;
  estimate.n_local_kv_heads = 1;
  estimate.max_seqs_per_batch = 2;
  estimate.max_tokens_per_batch = 16;
  estimate.cache_size_in_bytes = 1024 * 1024;
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, estimate);
  KVCacheShape shape(capacity, args, 1);
  proto::KVCacheShape serialized;
  shape.to_proto(&serialized);
  shape = KVCacheShape::from_proto(serialized);
  EXPECT_EQ(shape.pool_count(BlockType::C2), capacity.c2_count());
  EXPECT_EQ(shape.pool_count(BlockType::C1), capacity.c1_count());
  EXPECT_FALSE(shape.has_pool(BlockType::C4));
  KVCacheCreateOptions options;
  options.model_type("deepseek_v41")
      .device(torch::Device(torch::kCPU))
      .dtype(estimate.dtype)
      .num_layers(6)
      .window_size(args.window_size())
      .block_size(4)
      .head_dim(16)
      .index_head_dim(8)
      .compress_ratios(args.compress_ratios())
      .kv_source_layer_ids(args.kv_source_layer_ids());
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  ASSERT_EQ(caches.size(), 6);
  int64_t allocated_bytes = 0;
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(caches[i].get_swa_cache().sizes().vec(),
              (std::vector<int64_t>{capacity.swa_count(), 4, 16}));
    EXPECT_EQ(caches[i].get_k_cache().defined(), i == 1 || i == 3);
    EXPECT_EQ(caches[i].get_compress_kv_state().defined(), i == 1);
    for (const auto& tensor : caches[i].get_cache_tensors()) {
      allocated_bytes += tensor.tensor.nbytes();
    }
  }
  EXPECT_EQ(caches[1].get_index_cache().scalar_type(), torch::kInt8);
  EXPECT_EQ(caches[1].get_indexer_cache_scale()->scalar_type(),
            torch::kFloat16);
  EXPECT_EQ(caches[1].get_compress_kv_state().size(1), 32);
  EXPECT_EQ(caches[1].get_k_cache().size(0) * 2,
            caches[3].get_k_cache().size(0));
  EXPECT_LE(allocated_bytes, estimate.cache_size_in_bytes);
  const int64_t next_unit_bytes = 3 * 4 * (16 * 2 + 8 + 2);
  EXPECT_LT(estimate.cache_size_in_bytes - allocated_bytes, next_unit_bytes);
}

TEST(Csa2CacheTest, BaseV4PoolIdentitySurvivesNewAndLegacyMessages) {
  ModelArgs args;
  args.model_type("deepseek_v4");
  KVCacheCapacity capacity;
  capacity.block_size(4).swa_count(10).c4_count(20).c128_count(30);
  KVCacheShape shape(capacity, args, 1);
  proto::KVCacheShape message;
  shape.to_proto(&message);
  for (const bool legacy : {false, true}) {
    if (legacy) {
      message.clear_pool_types();
    }
    const KVCacheShape restored = KVCacheShape::from_proto(message);
    EXPECT_EQ(restored.pool_count(BlockType::SWA), 10);
    EXPECT_EQ(restored.pool_count(BlockType::C4), 20);
    EXPECT_EQ(restored.pool_count(BlockType::C128), 30);
    EXPECT_FALSE(restored.has_pool(BlockType::C2));
  }
}

#if defined(USE_NPU)
TEST(Csa2ModelArgsTest, NestedTextConfigRetainsC1AndC2Semantics) {
  JsonReader json;
  ASSERT_TRUE(
      json.parse_text(R"({"model_type":"deepseek_v41", "dtype":"bfloat16",
    "text_config":{"model_type":"deepseek_v41_text", "num_hidden_layers":6,
    "hidden_size":16, "num_attention_heads":2, "head_dim":8, "intermediate_size":32,
    "compress_ratios":[0,2,2,1,1,1,0], "kv_source_layer_ids":[1,3], "index_source_layer_ids":[1,3,4],
    "sliding_window":8,"engram_layer_ids":[1],"engram_max_ngram_size":4,
    "max_position_embeddings":64}})"));
  const auto loader = ModelRegistry::get_model_args_loader("deepseek_v41");
  ASSERT_TRUE(loader);
  ModelArgs args;
  ASSERT_TRUE(loader(json, &args));
  EXPECT_EQ(args.model_type(), "deepseek_v41");
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_EQ(args.n_layers(), 6);
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 2, 2, 1, 1, 1}));
  EXPECT_EQ(args.kv_source_layer_ids(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(args.csa2_manager_ratios(), (std::vector<int32_t>{0, 2, 1}));
  EXPECT_EQ(args.window_size(), 8);
  EXPECT_EQ(args.moe_intermediate_size(), 32);
  EXPECT_EQ(args.n_hash_layers(), 0);
  EXPECT_EQ(args.engram_max_ngram_size(), 4);
}
TEST(Csa2ModelArgsTest, InvalidLayoutsFailBeforeEngineConstruction) {
  const auto loader = ModelRegistry::get_model_args_loader("deepseek_v41");
  ASSERT_TRUE(loader);
  for (const std::string patch_text : {R"({"compress_ratios":[0,2]})",
                                       R"({"compress_ratios":[0,4,4,1,1,1]})",
                                       R"({"kv_source_layer_ids":[3,1]})",
                                       R"({"index_source_layer_ids":[1,3,3]})",
                                       R"({"candidate_source_layer_id":1})"}) {
    JsonReader json;
    ASSERT_TRUE(json.parse_text(R"({"num_hidden_layers":6,
      "hidden_size":16,"num_attention_heads":2,"head_dim":8,
      "compress_ratios":[0,2,2,1,1,1],"kv_source_layer_ids":[1,3],
      "index_source_layer_ids":[1,3,4],"candidate_source_layer_id":3})"));
    JsonReader patch;
    ASSERT_TRUE(patch.parse_text(patch_text));
    json.mutable_data().update(patch.data());
    ModelArgs args;
    EXPECT_DEATH(loader(json, &args), "Check failed");
  }
}

#endif
}  // namespace xllm
