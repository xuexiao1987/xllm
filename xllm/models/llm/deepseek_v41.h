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

#if defined(USE_NPU)
#include "core/kv_cache/layout/csa2_cache_plan.h"
#include "models/llm/deepseek_v4.h"

namespace xllm {
inline void load_deepseek_v41_model_args(const JsonReader& source,
                                         ModelArgs* args) {
  JsonReader json;
  json.mutable_data() = source.data();
  if (const auto* text = source.resolve("text_config")) {
    json.mutable_data().update(*text);
  }
  load_deepseek_v4_model_args(json, args);
  args->model_type("deepseek_v41");
  LOAD_ARG_OR_FUNC(dtype, "torch_dtype", [&] {
    return json.value_or<std::string>("dtype", "bfloat16");
  });
  LOAD_ARG_OR_FUNC(window_size, "window_size", [&] {
    return json.value_or<int32_t>("sliding_window", 128);
  });
  LOAD_ARG(kv_source_layer_ids, "kv_source_layer_ids");
  LOAD_ARG(index_source_layer_ids, "index_source_layer_ids");
  LOAD_ARG(engram_layer_ids, "engram_layer_ids");
  LOAD_ARG_OR(engram_max_ngram_size, "engram_max_ngram_size", 4);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 0);
  LOAD_ARG_OR_FUNC(moe_intermediate_size, "moe_intermediate_size", [&] {
    return static_cast<int32_t>(args->intermediate_size());
  });
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  args->n_activated_experts(args->num_experts_per_tok());
  args->max_seq_len(args->max_position_embeddings());
  CHECK_GT(args->n_layers(), 0);
  CHECK_GE(args->compress_ratios().size(),
           static_cast<size_t>(args->n_layers()));
  for (const int32_t ratio : args->compress_ratios()) {
    CHECK(ratio == 0 || ratio == 1 || ratio == 2)
        << "CSA2 text cache supports C1/C2";
  }
  args->compress_ratios().resize(args->n_layers());
  CHECK_GT(args->window_size(), 0);
  CHECK_GE(args->engram_max_ngram_size(), 1);
  const Csa2CachePlan plan(args->compress_ratios(),
                           args->kv_source_layer_ids(),
                           args->index_source_layer_ids());
  plan.validate_candidate_source(
      json.value_or<int32_t>("candidate_source_layer_id", -1));
  args->csa2_manager_ratios(plan.manager_ratios());
}
REGISTER_MODEL_ARGS(deepseek_v41,
                    [&] { load_deepseek_v41_model_args(json, args); });
REGISTER_MODEL_ARGS(deepseek_v41_text,
                    [&] { load_deepseek_v41_model_args(json, args); });
}  // namespace xllm
#endif
