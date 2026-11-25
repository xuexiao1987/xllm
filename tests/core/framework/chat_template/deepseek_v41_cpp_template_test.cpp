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

#include "core/framework/chat_template/deepseek_v41_cpp_template.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "core/framework/chat_template/deepseek_message_utils.h"
#include "core/framework/chat_template/deepseek_v4_cpp_template.h"

namespace xllm {
namespace {

xllm::ChatMessages read_messages(const nlohmann::ordered_json& rows) {
  xllm::ChatMessages messages;
  messages.reserve(rows.size());
  for (const auto& row : rows) {
    xllm::Message message(row.at("role").get<std::string>(),
                          row.at("content").get<std::string>());
    if (row.contains("reasoning")) {
      message.reasoning_content = row.at("reasoning").get<std::string>();
    }
    if (row.contains("tool_call_id")) {
      message.tool_call_id = row.at("tool_call_id").get<std::string>();
    }
    if (row.contains("tool_calls")) {
      xllm::Message::ToolCallVec calls;
      calls.reserve(row.at("tool_calls").size());
      for (const auto& call : row.at("tool_calls")) {
        calls.emplace_back(
            xllm::Message::ToolCall{call.at("id").get<std::string>(),
                                    "function",
                                    {call.at("name").get<std::string>(),
                                     call.at("arguments").get<std::string>()}});
      }
      message.tool_calls = std::move(calls);
    }
    messages.emplace_back(std::move(message));
  }
  return messages;
}

TEST(DeepseekMessageUtils, BothVersionsPreserveFrozenPromptBytes) {
  std::ifstream input(std::filesystem::path(__FILE__).parent_path() /
                      "data/deepseek_v4_v41_prompts.json");
  ASSERT_TRUE(input.good());
  nlohmann::ordered_json fixture;
  input >> fixture;
  DeepseekV4CppTemplate v4{TokenizerArgs{}};
  DeepseekV41CppTemplate v41{TokenizerArgs{}};
  for (const auto& row : fixture.at("cases")) {
    SCOPED_TRACE(row.at("name").get<std::string>());
    const auto messages = read_messages(row.at("messages"));
    const auto p4 = v4.apply(messages, {}, row.at("kwargs"));
    const auto p41 = v41.apply(messages, {}, row.at("kwargs"));
    ASSERT_TRUE(p4.has_value());
    ASSERT_TRUE(p41.has_value());
    EXPECT_EQ(*p4, row.at("v4").get<std::string>());
    EXPECT_EQ(*p41, row.at("v41").get<std::string>());
  }
}

TEST(DeepseekMessageUtils, TaskBoundaryAndMixedBlockPositionsArePreserved) {
  using Json = nlohmann::ordered_json;
  const Json messages = Json::array(
      {{{"role", "user"},
        {"content", "first"},
        {"task", "prefix"},
        {"mask", true}},
       {{"role", "user"}, {"content", "second"}},
       {{"role", "tool"}, {"content", "result"}, {"tool_call_id", "x"}},
       {{"role", "user"}, {"content", "third"}}});
  const Json merged = deepseek_template::merge_tool_messages(messages);
  ASSERT_EQ(merged.size(), 2);
  EXPECT_EQ(merged[0]["task"], "prefix");
  EXPECT_EQ(merged[0]["mask"], true);
  EXPECT_EQ(merged[1]["content"], "");
  std::string rendered;
  deepseek_template::append_content_blocks(rendered,
                                           merged[1]["content_blocks"]);
  EXPECT_EQ(rendered, "second\n\n<tool_result>result</tool_result>\n\nthird");
  EXPECT_EQ(messages[0]["content"], "first");
}

TEST(DeepseekMessageUtils, OrderingPoliciesKeepTextSlotsAndStableUnknownTies) {
  using Json = nlohmann::ordered_json;
  const Json messages = Json::array(
      {{{"role", "assistant"},
        {"tool_calls",
         Json::array(
             {{{"id", "a"}}, {{"function", {{"id", "b"}}}}, {{"id", "c"}}})}},
       {{"role", "user"},
        {"content_blocks",
         Json::array(
             {{{"type", "tool_result"}, {"tool_use_id", "c"}},
              {{"type", "text"}, {"text", "stay"}},
              {{"type", "tool_result"}, {"tool_use_id", "unknown1"}},
              {{"type", "tool_result"}, {"tool_use_id", "b"}},
              {{"type", "tool_result"}, {"tool_use_id", "unknown2"}}})}}});
  const auto ordered = deepseek_template::sort_tool_results_by_call_order(
      messages, deepseek_template::ToolResultOrder::STABLE_UNKNOWN_LAST);
  EXPECT_EQ(ordered[1]["content_blocks"][0]["tool_use_id"], "b");
  EXPECT_EQ(ordered[1]["content_blocks"][1]["text"], "stay");
  EXPECT_EQ(ordered[1]["content_blocks"][2]["tool_use_id"], "c");
  EXPECT_EQ(ordered[1]["content_blocks"][3]["tool_use_id"], "unknown1");
  EXPECT_EQ(ordered[1]["content_blocks"][4]["tool_use_id"], "unknown2");
  const auto legacy = deepseek_template::sort_tool_results_by_call_order(
      messages, deepseek_template::ToolResultOrder::UNKNOWN_AS_FIRST_CALL);
  EXPECT_EQ(legacy[1]["content_blocks"][1]["text"], "stay");
  EXPECT_EQ(legacy[1]["content_blocks"][3]["tool_use_id"], "b");
  EXPECT_EQ(legacy[1]["content_blocks"][4]["tool_use_id"], "c");
}

TEST(DeepseekMessageUtils, TemplatesPreserveInvalidUtf8Replacement) {
  DeepseekV4CppTemplate v4{TokenizerArgs{}};
  DeepseekV41CppTemplate v41{TokenizerArgs{}};
  ChatMessages messages{Message("user", "test")};
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "test";
  tool.function.parameters = {
      {"description", std::string(1, static_cast<char>(0xff))}};
  std::vector<JsonTool> tools;
  tools.reserve(1);
  tools.emplace_back(std::move(tool));
  const auto p4 = v4.apply(messages, tools, {});
  const auto p41 = v41.apply(messages, tools, {});
  ASSERT_TRUE(p4.has_value());
  ASSERT_TRUE(p41.has_value());
  EXPECT_NE(p4->find("\xef\xbf\xbd"), std::string::npos);
  EXPECT_NE(p41->find("\xef\xbf\xbd"), std::string::npos);
}

TEST(DeepseekMessageUtils, TemplatesPreserveFatalUnknownRole) {
  DeepseekV4CppTemplate v4{TokenizerArgs{}};
  DeepseekV41CppTemplate v41{TokenizerArgs{}};
  ChatMessages messages{Message("unknown_role", "test")};
  EXPECT_DEATH(v4.apply(messages), "Unknown role");
  EXPECT_DEATH(v41.apply(messages), "Unknown role");
}

TEST(DeepseekV41CppTemplate, DefaultThinkingHasNumericSystemBudget) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  const auto result = encoder.apply_with_generation_mode(
      {Message("user", "hello")}, {}, nlohmann::ordered_json::object());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(
      result->prompt,
      "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 "
      "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
      "<｜User｜>hello<｜Assistant｜><think>");
  EXPECT_EQ(result->generation_mode, ChatTemplateGenerationMode::REASONING);
}

TEST(DeepseekV41CppTemplate, ThinkingPrecedenceControlsPromptAndParsing) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  const std::vector<nlohmann::ordered_json> chat_modes = {
      {{"reasoning_effort", "none"}, {"thinking", true}},
      {{"thinking", false}, {"enable_thinking", true}},
      {{"enable_thinking", false}},
      {{"thinking_mode", "chat"}, {"thinking", true}},
  };
  for (const auto& kwargs : chat_modes) {
    const auto result = encoder.apply_with_generation_mode(
        {Message("user", "hello")}, {}, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->prompt,
              "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>");
    EXPECT_EQ(result->generation_mode, ChatTemplateGenerationMode::CHAT);
  }
  const auto explicit_mode =
      encoder.apply_with_generation_mode({Message("user", "hello")},
                                         {},
                                         {{"thinking_mode", "thinking"},
                                          {"reasoning_effort", "none"},
                                          {"thinking", false}});
  ASSERT_TRUE(explicit_mode.has_value());
  EXPECT_EQ(explicit_mode->prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜><think>");
  EXPECT_EQ(explicit_mode->generation_mode,
            ChatTemplateGenerationMode::REASONING);
}

TEST(DeepseekV41CppTemplate, EffortTiersAndNumericBounds) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  const std::vector<std::pair<std::string, std::string>> budgets = {
      {"low", "25"},
      {"high", "50"},
      {"xhigh", "75"},
      {"max", "100"},
      {"1", "1"},
      {"100", "100"},
      {"0007", "7"},
      {"0", "50"},
      {"101", "50"},
      {"bad", "50"},
      {"999999999999999999999", "50"}};
  for (const auto& [effort, budget] : budgets) {
    SCOPED_TRACE(effort);
    const auto prompt = encoder.apply(
        {Message("user", "hi")}, {}, {{"reasoning_effort", effort}});
    ASSERT_TRUE(prompt.has_value());
    EXPECT_NE(prompt->find("Reasoning Effort: " + budget + " (range 1-100"),
              std::string::npos);
  }
}

TEST(DeepseekV41CppTemplate, SystemHistoryDropsOldReasoning) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  Message assistant("assistant", "old answer");
  assistant.reasoning_content = "private old reasoning";
  const auto prompt = encoder.apply({Message("system", "first"),
                                     Message("user", "old question"),
                                     assistant,
                                     Message("system", "updated"),
                                     Message("user", "new question")});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(prompt->find("private old reasoning"), std::string::npos);
  EXPECT_NE(prompt->find("old answer<｜end▁of▁sentence｜><｜System｜>updated"),
            std::string::npos);
  EXPECT_TRUE(
      prompt->ends_with("<｜User｜>new question<｜Assistant｜><think>"));
}

TEST(DeepseekV41CppTemplate, ToolCallsUseSpacedDsmlAndResultsFollowCallOrder) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  Message assistant("assistant", "");
  assistant.tool_calls = Message::ToolCallVec{
      {"a", "function", {"lookup", R"({"city":"beijing","count":2})"}},
      {"b", "function", {"lookup", R"({"city":"shanghai"})"}}};
  Message second("tool", "second result");
  second.tool_call_id = "b";
  Message first("tool", "first result");
  first.tool_call_id = "a";
  const auto prompt =
      encoder.apply({Message("user", "weather"), assistant, second, first},
                    {},
                    {{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">"),
            std::string::npos);
  EXPECT_NE(prompt->find("<｜DSML｜ parameter name=\"count\" string=\"false\">2"
                         "</｜DSML｜ parameter>"),
            std::string::npos);
  EXPECT_NE(prompt->find("<tool_result>first result</tool_result>\n\n"
                         "<tool_result>second result</tool_result>"),
            std::string::npos);
}

TEST(DeepseekV41CppTemplate, TextPartsAndEscapedLiteralMatchStringContent) {
  DeepseekV41CppTemplate encoder{TokenizerArgs{}};
  const auto plain =
      encoder.apply({Message("user", "literal \\<image> hello")});
  const auto parts = encoder.apply(
      {Message("user",
               MMContentVec{MMContent("text", "literal \\<image> "),
                            MMContent("text", "hello")})});
  ASSERT_TRUE(plain.has_value());
  EXPECT_EQ(parts, plain);
  EXPECT_NE(plain->find("literal <image> hello"), std::string::npos);
}

}  // namespace
}  // namespace xllm
