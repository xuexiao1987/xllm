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

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

namespace xllm::deepseek_template {

// V4 assigns unknown IDs rank zero (tied with the first call) and does not
// promise stable ties. V4.1 retains unknown IDs after known calls, stably.
enum class ToolResultOrder : int8_t {
  UNKNOWN_AS_FIRST_CALL,
  STABLE_UNKNOWN_LAST
};

// Input is normalized message JSON: role/content/tool IDs are strings.
// Content blocks become authoritative; the returned user content is empty.
// Task boundaries and the positions of non-tool blocks are retained.
nlohmann::ordered_json merge_tool_messages(nlohmann::ordered_json messages);
nlohmann::ordered_json sort_tool_results_by_call_order(
    nlohmann::ordered_json messages,
    ToolResultOrder ordering);
void append_content_blocks(std::string& prompt,
                           const nlohmann::ordered_json& blocks);

}  // namespace xllm::deepseek_template
