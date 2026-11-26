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

#include "core/util/json_reader.h"
#include "core/util/type_traits.h"

#define LOAD_ARG(arg_name, json_name)                          \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    }                                                          \
  }()

#define LOAD_ARG_OR(arg_name, json_name, default_value)                     \
  [&] {                                                                     \
    auto value = args->arg_name();                                          \
    using value_type = remove_optional_t<decltype(value)>;                  \
    args->arg_name() = json.value_or<value_type>(json_name, default_value); \
  }()

#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)             \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    } else {                                                   \
      args->arg_name() = __VA_ARGS__();                        \
    }                                                          \
  }()
