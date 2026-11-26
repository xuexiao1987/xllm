/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <vector>

namespace xllm {

struct ExecutionBatchMetadata {
  int32_t num_reqs = 0;
  int64_t num_tokens = 0;
  std::vector<int32_t> num_scheduled_tokens;
  std::vector<int32_t> num_computed_tokens;
  std::vector<int32_t> query_start_loc;
  std::vector<uint8_t> is_prefilling;
};

}  // namespace xllm
