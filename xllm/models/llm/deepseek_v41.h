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

#pragma once

#include "models/llm/deepseek_v41_args.h"

#if defined(USE_NPU) || defined(USE_MLU)
#include "models/model_registry.h"

namespace xllm {
REGISTER_MODEL_ARGS(deepseek_v41,
                    [&] { load_deepseek_v41_model_args(json, args); });
REGISTER_MODEL_ARGS(deepseek_v41_text,
                    [&] { load_deepseek_v41_model_args(json, args); });
}  // namespace xllm
#endif
