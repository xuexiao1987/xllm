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

#include "core/framework/chat_template/deepseek_message_utils.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm::deepseek_template {
namespace {
constexpr char kRoleUser[] = "user";
constexpr char kRoleTool[] = "tool";
constexpr char kRoleAssistant[] = "assistant";
}  // namespace

nlohmann::ordered_json merge_tool_messages(nlohmann::ordered_json messages) {
  nlohmann::ordered_json merged = nlohmann::json::array();
  merged.get_ref<nlohmann::ordered_json::array_t&>().reserve(messages.size());

  for (auto& msg : messages) {
    std::string role = msg.value("role", "");

    if (role == kRoleTool) {
      nlohmann::ordered_json tool_block;
      tool_block["type"] = "tool_result";
      const auto tool_call_id = msg.find("tool_call_id");
      tool_block["tool_use_id"] = tool_call_id == msg.end()
                                      ? nlohmann::ordered_json("")
                                      : std::move(*tool_call_id);
      tool_block["content"] = std::move(msg["content"]);

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks")) {
        merged.back()["content"] = "";
        merged.back()["content_blocks"].emplace_back(std::move(tool_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content"] = "";
        new_msg["content_blocks"] = nlohmann::ordered_json::array();
        new_msg["content_blocks"].emplace_back(std::move(tool_block));
        merged.emplace_back(std::move(new_msg));
      }
    } else if (role == kRoleUser) {
      nlohmann::ordered_json text_block;
      text_block["type"] = "text";
      text_block["text"] = std::move(msg["content"]);

      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks") &&
          !merged.back().contains("task")) {
        merged.back()["content"] = "";
        merged.back()["content_blocks"].emplace_back(std::move(text_block));
      } else {
        nlohmann::ordered_json new_msg;
        new_msg["role"] = kRoleUser;
        new_msg["content"] = "";
        new_msg["content_blocks"] = nlohmann::ordered_json::array();
        new_msg["content_blocks"].emplace_back(std::move(text_block));
        for (const auto& key : {"task", "wo_eos", "mask"}) {
          if (msg.contains(key)) {
            new_msg[key] = std::move(msg[key]);
          }
        }
        merged.emplace_back(std::move(new_msg));
      }
    } else {
      merged.emplace_back(std::move(msg));
    }
  }
  return merged;
}

nlohmann::ordered_json sort_tool_results_by_call_order(
    nlohmann::ordered_json messages,
    ToolResultOrder ordering) {
  const bool stable = ordering == ToolResultOrder::STABLE_UNKNOWN_LAST;
  const int32_t unknown_order =
      stable ? std::numeric_limits<int32_t>::max() : 0;
  std::unordered_map<std::string, int32_t> last_tc_order;

  for (auto& msg : messages) {
    std::string role = msg.value("role", "");
    if (role == kRoleAssistant && msg.contains("tool_calls") &&
        msg["tool_calls"].is_array()) {
      last_tc_order.clear();
      last_tc_order.reserve(msg["tool_calls"].size());
      int32_t idx = 0;
      for (const auto& tc : msg["tool_calls"]) {
        std::string tc_id;
        if (tc.contains("id")) {
          tc_id = tc.value("id", "");
        } else if (tc.contains("function") && tc["function"].contains("id")) {
          tc_id = tc["function"].value("id", "");
        }
        if (!tc_id.empty()) {
          last_tc_order[std::move(tc_id)] = idx;
        }
        ++idx;
      }
    } else if (role == kRoleUser && msg.contains("content_blocks")) {
      if (last_tc_order.empty()) {
        continue;
      }
      auto& blocks = msg["content_blocks"];
      int32_t tool_count = 0;
      int32_t previous_order = -1;
      bool strictly_ordered = true;
      for (const auto& b : blocks) {
        if (b.value("type", "") == "tool_result") {
          ++tool_count;
          const auto order_it = last_tc_order.find(b.value("tool_use_id", ""));
          const int32_t order = order_it != last_tc_order.end()
                                    ? order_it->second
                                    : unknown_order;
          strictly_ordered = strictly_ordered && previous_order < order;
          previous_order = order;
        }
      }
      // Equal ranks still use the selected sort: stable for V4.1,
      // unspecified ties for V4.
      if (tool_count > 1 && !strictly_ordered) {
        std::vector<nlohmann::ordered_json> tool_blocks;
        tool_blocks.reserve(tool_count);
        std::vector<size_t> tool_positions;
        tool_positions.reserve(tool_count);
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
          auto& b = blocks[block_idx];
          if (b.value("type", "") == "tool_result") {
            tool_positions.emplace_back(block_idx);
            tool_blocks.emplace_back(std::move(b));
          }
        }
        const auto precedes = [&](const nlohmann::ordered_json& a,
                                  const nlohmann::ordered_json& b) {
          const auto ia = last_tc_order.find(a.value("tool_use_id", ""));
          const auto ib = last_tc_order.find(b.value("tool_use_id", ""));
          const int32_t oa =
              ia != last_tc_order.end() ? ia->second : unknown_order;
          const int32_t ob =
              ib != last_tc_order.end() ? ib->second : unknown_order;
          return oa < ob;
        };
        if (stable) {
          std::stable_sort(tool_blocks.begin(), tool_blocks.end(), precedes);
        } else {
          std::sort(tool_blocks.begin(), tool_blocks.end(), precedes);
        }
        for (size_t idx = 0; idx < tool_positions.size(); ++idx) {
          blocks[tool_positions[idx]] = std::move(tool_blocks[idx]);
        }
      }
    }
  }
  return messages;
}

void append_content_blocks(std::string& prompt,
                           const nlohmann::ordered_json& blocks) {
  const std::string empty_content;
  bool first_part = true;
  for (const auto& block : blocks) {
    std::string btype = block.value("type", "");
    if (btype != "text" && btype != "tool_result") {
      continue;
    }
    if (!first_part) {
      prompt += "\n\n";
    }
    first_part = false;
    if (btype == "text") {
      const auto text = block.find("text");
      if (text != block.end()) {
        prompt += text->get_ref<const std::string&>();
      }
    } else {
      const auto tool_content = block.find("content");
      const std::string& tc = tool_content == block.end()
                                  ? empty_content
                                  : tool_content->get_ref<const std::string&>();
      prompt += "<tool_result>";
      prompt += tc;
      prompt += "</tool_result>";
    }
  }
}

}  // namespace xllm::deepseek_template
