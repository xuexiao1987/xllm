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
#include "models/llm/deepseek_v41_token_map.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "core/framework/tokenizer/tokenizers/tokenizers.h"

namespace xllm {
namespace {
class DeepseekV41TokenMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "xllm_token_map_XXXXXX")
            .string();
    const char* path = mkdtemp(pattern.data());
    ASSERT_NE(path, nullptr);
    directory_ = path;
    json_ = R"json({"version":"1.0","truncation":null,"padding":null,
      "added_tokens":[],"normalizer":{"type":"Lowercase"},"pre_tokenizer":null,
      "post_processor":null,"decoder":null,"model":{"type":"WordLevel",
      "vocab":{"É":0,"ｅ":1,"E":2," ":3,"\t":4," a ":5,"a":6,"unk":7,"�":8},
      "unk_token":"unk"}})json";
    std::ofstream file(directory_ / "tokenizer.json");
    file << json_;
    ASSERT_TRUE(file.good());
    args_.vocab_size(9).engram_compressed_vocab_size(5);
  }
  void TearDown() override { std::filesystem::remove_all(directory_); }
  std::filesystem::path directory_;
  std::string json_;
  ModelArgs args_;
};

TEST_F(DeepseekV41TokenMapTest,
       RealRustNormalizationMatchesTrainingEquivalenceClasses) {
  const auto actual = build_deepseek_v41_token_map(
      args_, (directory_ / "tokenizer.json").string());
  EXPECT_TRUE(torch::equal(
      actual, torch::tensor({0, 0, 0, 1, 1, 2, 2, 3, 4}, torch::kInt32)));
}

TEST_F(DeepseekV41TokenMapTest, RejectsMismatchedCompressedVocabulary) {
  args_.engram_compressed_vocab_size(6);
  EXPECT_DEATH(build_deepseek_v41_token_map(
                   args_, (directory_ / "tokenizer.json").string()),
               "compressed.size");
}

TEST_F(DeepseekV41TokenMapTest,
       InvalidUtf8DoesNotExposePreviousNormalizationOutput) {
  std::unique_ptr<void, decltype(&tokenizers_free)> tokenizer(
      tokenizers_new_from_str(json_.data(), json_.size()), tokenizers_free);
  const char* data = nullptr;
  size_t size = 0;
  ASSERT_EQ(tokenizers_normalize(tokenizer.get(), "ABC", 3, &data, &size), 0);
  EXPECT_EQ(std::string(data, size), "abc");
  const char invalid[] = {static_cast<char>(0xff)};
  EXPECT_EQ(tokenizers_normalize(tokenizer.get(), invalid, 1, &data, &size), 1);
  EXPECT_EQ(data, nullptr);
  EXPECT_EQ(size, 0);
  ASSERT_EQ(tokenizers_normalize(tokenizer.get(), "XYZ", 3, &data, &size), 0);
  EXPECT_EQ(std::string(data, size), "xyz");
}
}  // namespace
}  // namespace xllm
