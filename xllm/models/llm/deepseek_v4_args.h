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

#include "core/framework/model/model_args.h"
#include "models/model_args_loader.h"

namespace xllm {

inline void load_deepseek_v4_model_args(const JsonReader& json,
                                        ModelArgs* args) {
  // --------------------------------------------------------------------------
  // Base args (shared/common HF CausalLM args).
  // NOTE: These are intentionally kept here even if they are not DSv4-specific.
  // --------------------------------------------------------------------------
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("norm_eps", 1e-6f);
  });

  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR(rope_head_dim, "qk_rope_head_dim", 64);

  // LoRA / groups
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1024);
  LOAD_ARG_OR(o_lora_rank, "o_lora_rank", 1024);
  LOAD_ARG_OR(o_groups, "o_groups", 8);

  // --------------------------------------------------------------------------
  // DeepSeek V4 args.
  // --------------------------------------------------------------------------
  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR(compress_rope_theta, "compress_rope_theta", 160000.0f);
  LOAD_ARG_OR(window_size, "window_size", 128);

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_activated_experts, "n_activated_experts", 6);
  LOAD_ARG_OR_FUNC(num_experts_per_tok, "num_experts_per_tok", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(n_hash_layers, "num_hash_layers", 3);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sqrtsoftplus");
  LOAD_ARG_OR(swiglu_limit, "swiglu_limit", 10.0f);

  // Indexer
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 512);

  // HC / DSA helpers
  LOAD_ARG_OR(hc_mult, "hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "hc_eps", 1e-6f);
  LOAD_ARG_OR(factor, "rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);

  // DSpark draft metadata. The 0731 checkpoint stores three draft layers
  // under mtp.0/1/2 and uses DeepSeek-specific top-level key names.
  LOAD_ARG_OR(markov_rank, "dspark_markov_rank", 0);
  args->dspark_num_layers() = static_cast<int32_t>(
      json.value_or<std::vector<int32_t>>("dspark_target_layer_ids",
                                          std::vector<int32_t>{})
          .size());
  LOAD_ARG_OR_FUNC(block_size, "block_size", [&] {
    return json.value_or<int64_t>("dspark_block_size", 0);
  });

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
}

}  // namespace xllm
