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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/kv_cache/layout/csa2_cache_plan.h"

namespace {
std::vector<int32_t> parse(const std::string& input) {
  std::vector<int32_t> values;
  values.reserve(input.size());
  std::istringstream stream(input);
  std::string value;
  while (std::getline(stream, value, ',')) {
    values.emplace_back(std::stoi(value));
  }
  return values;
}
}  // namespace

int main(int argc, char** argv) {
  CHECK_EQ(argc, 5);
  const auto ratios = parse(argv[1]);
  const xllm::Csa2CachePlan plan(ratios, parse(argv[2]), parse(argv[3]));
  plan.validate_candidate_source(std::stoi(argv[4]));
  for (const int32_t ratio : plan.manager_ratios()) {
    std::cout << ratio << ' ';
  }
  std::cout << '\n';
  const char* modes[] = {"swa", "full", "reindex", "reuse"};
  for (int64_t i = 0; i < static_cast<int64_t>(ratios.size()); ++i) {
    const auto& layer = plan.layer(i);
    std::cout << modes[static_cast<int32_t>(layer.mode)] << ' ' << layer.ratio
              << ' ' << layer.source << ' ' << layer.owns_main << ' '
              << layer.owns_state << '\n';
  }
}
