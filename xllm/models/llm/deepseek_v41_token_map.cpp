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

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "core/framework/tokenizer/tokenizers/tokenizers.h"

namespace xllm {
torch::Tensor build_deepseek_v41_token_map(const ModelArgs& args,
                                           const std::string& path) {
  std::ifstream input(path);
  CHECK(input.good()) << "Cannot read Engram tokenizer: " << path;
  auto config =
      nlohmann::json::parse(input, nullptr, /*allow_exceptions=*/false);
  CHECK(!config.is_discarded())
      << "Cannot parse Engram tokenizer JSON: " << path;
  CHECK(config.is_object())
      << "Engram tokenizer JSON must be an object: " << path;
  CHECK(config.contains("model") && config["model"].is_object())
      << "Engram tokenizer JSON requires a model object: " << path;
  const auto& model = config["model"];
  CHECK(model.contains("type") && model["type"].is_string() &&
        model.contains("vocab") &&
        (model["vocab"].is_object() || model["vocab"].is_array()))
      << "Engram tokenizer JSON requires model type and vocabulary: " << path;
  config["normalizer"] = nlohmann::json::parse(R"json({
    "type":"Sequence", "normalizers":[
      {"type":"NFKC"}, {"type":"NFD"}, {"type":"StripAccents"},
      {"type":"Lowercase"},
      {"type":"Replace", "pattern":{"Regex":"[ \t\r\n]+"}, "content":" "},
      {"type":"Replace", "pattern":{"Regex":"^ $"}, "content":"\ue000"},
      {"type":"Strip", "strip_left":true, "strip_right":true},
      {"type":"Replace", "pattern":{"String":"\ue000"}, "content":" "}
    ]})json");
  const std::string json = config.dump();
  std::unique_ptr<void, decltype(&tokenizers_free)> tokenizer(
      tokenizers_new_from_str(json.data(), json.size()), tokenizers_free);
  size_t vocab_size = 0;
  tokenizers_get_vocab_size(tokenizer.get(), &vocab_size);
  CHECK_EQ(vocab_size, args.vocab_size());
  auto result = torch::empty({static_cast<int64_t>(vocab_size)}, torch::kInt32);
  std::unordered_map<std::string, int32_t> compressed;
  compressed.reserve(vocab_size);
  for (uint32_t id = 0; id < vocab_size; ++id) {
    const char* data = nullptr;
    size_t length = 0;
    tokenizers_decode(
        tokenizer.get(), &id, 1, /*skip_special_tokens=*/0, &data, &length);
    const std::string decoded(data, length);
    std::string key;
    if (decoded.find("\xef\xbf\xbd") != std::string::npos) {
      tokenizers_id_to_token(tokenizer.get(), id, &data, &length);
      key.assign(data, length);
    } else {
      const int32_t status = tokenizers_normalize(
          tokenizer.get(), decoded.data(), decoded.size(), &data, &length);
      CHECK_EQ(status, 0) << "Engram tokenizer normalization failed for token "
                          << id << " in " << path << " (status " << status
                          << ": 1=invalid UTF-8, 2=normalizer error)";
      key = length == 0 ? decoded : std::string(data, length);
    }
    const auto [entry, inserted] = compressed.try_emplace(
        std::move(key), static_cast<int32_t>(compressed.size()));
    result.data_ptr<int32_t>()[id] = entry->second;
  }
  CHECK_EQ(compressed.size(), args.engram_compressed_vocab_size());
  return result;
}
}  // namespace xllm
