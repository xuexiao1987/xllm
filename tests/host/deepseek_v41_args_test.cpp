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

#include "models/llm/deepseek_v41_args.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {
JsonReader config() {
  JsonReader json;
  json.mutable_data() = {{"model_type", "deepseek_v41"},
                         {"dtype", "bfloat16"},
                         {"bos_token_id", 0},
                         {"eos_token_id", 1},
                         {"text_config",
                          {{"hidden_size", 5120},
                           {"num_hidden_layers", 6},
                           {"num_attention_heads", 64},
                           {"head_dim", 512},
                           {"q_lora_rank", 1536},
                           {"compress_ratios", {0, 2, 2, 2, 1, 1}},
                           {"kv_source_layer_ids", {1, 4}},
                           {"index_source_layer_ids", {1, 2, 4}},
                           {"engram_layer_ids", {0, 3}},
                           {"max_position_embeddings", 1048576}}}};
  return json;
}

TEST(DeepseekV41ArgsTest, NestedConfigPreservesDistinctCacheModes) {
  const auto json = config();
  ModelArgs args;
  load_deepseek_v41_model_args(json, &args);
  const Csa2CachePlan plan(args.compress_ratios(),
                           args.kv_source_layer_ids(),
                           args.index_source_layer_ids());
  EXPECT_EQ(args.hidden_size(), 5120);
  EXPECT_TRUE(args.requires_execution_batch());
  EXPECT_EQ(args.n_heads(), 64);
  EXPECT_EQ(args.head_dim(), 512);
  EXPECT_EQ(args.q_lora_rank(), 1536);
  EXPECT_EQ(args.max_seq_len(), 1048576);
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_EQ(args.n_hash_layers(), 0);
  EXPECT_EQ(args.engram_layer_ids(), (std::vector<int32_t>{0, 3}));
  EXPECT_EQ(args.csa2_manager_ratios(), (std::vector<int32_t>{0, 2, 1}));
  EXPECT_EQ(plan.layer(0).mode, Csa2CachePlan::Mode::SWA);
  EXPECT_EQ(plan.layer(1).mode, Csa2CachePlan::Mode::FULL);
  EXPECT_EQ(plan.layer(2).mode, Csa2CachePlan::Mode::REINDEX);
  EXPECT_EQ(plan.layer(3).mode, Csa2CachePlan::Mode::REUSE);
  EXPECT_EQ(plan.layer(5).source, 4);
}

TEST(DeepseekV41ArgsTest, TextOverridesRootWithoutMutatingSource) {
  auto json = config();
  json.mutable_data()["hidden_size"] = 64;
  json.mutable_data()["dtype"] = "float32";
  json.mutable_data()["text_config"]["torch_dtype"] = "bfloat16";
  const auto before = json.data();
  ModelArgs args;
  load_deepseek_v41_model_args(json, &args);
  EXPECT_EQ(args.hidden_size(), 5120);
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_EQ(json.data(), before);
}

TEST(DeepseekV41ArgsTest,
     NativeGeometryUsesTextOverridesAndInt64EmbeddingRows) {
  auto json = config();
  json.mutable_data()["engram_head_dim"] = 16;
  auto& text = json.mutable_data()["text_config"];
  text["engram_num_embeddings"] = {5000000000LL, 5000000021LL};
  text["engram_vocab_size"] = 16000003;
  text["engram_compressed_vocab_size"] = 120000;
  text["engram_n_heads"] = 8;
  text["engram_head_dim"] = 128;
  text["engram_pad_token_id"] = 7;
  text["candidate_source_layer_id"] = 4;
  text["candidate_topk_blocks"] = 8;
  text["candidate_block_size"] = 64;
  ModelArgs args;
  load_deepseek_v41_model_args(json, &args);
  EXPECT_EQ(args.engram_num_embeddings(),
            (std::vector<int64_t>{5000000000LL, 5000000021LL}));
  EXPECT_EQ(args.engram_vocab_size(), 16000003);
  EXPECT_EQ(args.engram_compressed_vocab_size(), 120000);
  EXPECT_EQ(args.engram_n_heads(), 8);
  EXPECT_EQ(args.engram_head_dim(), 128);
  EXPECT_EQ(args.engram_pad_token_id(), 7);
  EXPECT_EQ(args.candidate_source_layer_id(), 4);
  EXPECT_EQ(args.candidate_topk_blocks(), 8);
  EXPECT_EQ(args.candidate_block_size(), 64);
}

TEST(DeepseekV41ArgsTest, FlatTextExportHasSameSemantics) {
  const auto nested = config();
  JsonReader flat;
  flat.mutable_data() = nested.data();
  flat.mutable_data().update(nested.data().at("text_config"));
  flat.mutable_data().erase("text_config");
  ModelArgs args;
  load_deepseek_v41_model_args(flat, &args);
  EXPECT_EQ(args.model_type(), "deepseek_v41");
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{0, 2, 2, 2, 1, 1}));
  EXPECT_EQ(args.kv_source_layer_ids(), (std::vector<int32_t>{1, 4}));
}

TEST(DeepseekV41ArgsTest, MoeAndWindowAliasesPreserveV11Defaults) {
  auto json = config();
  auto& text = json.mutable_data()["text_config"];
  text["sliding_window"] = 256;
  text["num_experts_per_tok"] = 8;
  text["intermediate_size"] = 3072;
  ModelArgs args;
  load_deepseek_v41_model_args(json, &args);
  EXPECT_EQ(args.window_size(), 256);
  EXPECT_EQ(args.moe_intermediate_size(), 3072);
  EXPECT_EQ(args.n_activated_experts(), 8);
  EXPECT_TRUE(args.norm_topk_prob());
  EXPECT_FLOAT_EQ(args.factor(), 16.0f);
  EXPECT_FLOAT_EQ(args.hc_eps(), 1e-6f);
  EXPECT_EQ(args.engram_max_ngram_size(), 4);
}

TEST(DeepseekV41ArgsTest, DraftTailDoesNotAllocateBackboneCache) {
  auto json = config();
  json.mutable_data()["text_config"]["compress_ratios"] =
      std::vector<int32_t>{0, 2, 2, 2, 1, 1, 1, 1, 1};
  ModelArgs args;
  load_deepseek_v41_model_args(json, &args);
  EXPECT_EQ(args.compress_ratios().size(), 6);
}

TEST(DeepseekV41ArgsTest, RejectsShortRatioList) {
  auto json = config();
  json.mutable_data()["text_config"]["compress_ratios"] =
      std::vector<int32_t>{0, 2};
  ModelArgs args;
  EXPECT_DEATH(load_deepseek_v41_model_args(json, &args), "compress_ratios");
}

TEST(DeepseekV41ArgsTest, RejectsV4CompressionInsteadOfNormalizingIt) {
  auto json = config();
  json.mutable_data()["text_config"]["compress_ratios"][1] = 4;
  ModelArgs args;
  EXPECT_DEATH(load_deepseek_v41_model_args(json, &args), "C1/C2");
}

TEST(DeepseekV41ArgsTest, RejectsCandidateReuseAcrossKvGroups) {
  auto json = config();
  json.mutable_data()["text_config"]["candidate_source_layer_id"] = 2;
  ModelArgs args;
  EXPECT_DEATH(load_deepseek_v41_model_args(json, &args), "cross KV groups");
}

TEST(DeepseekV41ArgsTest, RejectsNonpositiveWindowAndEngramDepth) {
  auto json = config();
  json.mutable_data()["text_config"]["window_size"] = 0;
  ModelArgs args;
  EXPECT_DEATH(load_deepseek_v41_model_args(json, &args), "window_size");
  json.mutable_data()["text_config"]["window_size"] = 128;
  json.mutable_data()["text_config"]["engram_max_ngram_size"] = 0;
  EXPECT_DEATH(load_deepseek_v41_model_args(json, &args),
               "engram_max_ngram_size");
}

TEST(DeepseekV41ArgsTest, V4SchemaExtractionPreservesV4SpecificDefaults) {
  JsonReader json;
  json.mutable_data() = {{"dim", 128},
                         {"n_heads", 4},
                         {"num_hidden_layers", 2},
                         {"compress_ratios", {4, 128}},
                         {"dspark_target_layer_ids", {1, 2, 3}},
                         {"dspark_block_size", 8}};
  ModelArgs args;
  load_deepseek_v4_model_args(json, &args);
  EXPECT_EQ(args.model_type(), "deepseek_v4");
  EXPECT_FALSE(args.requires_execution_batch());
  EXPECT_EQ(args.head_dim(), 32);
  EXPECT_EQ(args.intermediate_size(), 512);
  EXPECT_EQ(args.n_hash_layers(), 3);
  EXPECT_EQ(args.dspark_num_layers(), 3);
  EXPECT_EQ(args.block_size(), 8);
  EXPECT_EQ(args.compress_ratios(), (std::vector<int32_t>{4, 128}));
}
}  // namespace
}  // namespace xllm
