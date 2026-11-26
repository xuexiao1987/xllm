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

#include <algorithm>

#include "core/framework/config/kv_cache_config.h"
#include "core/kernels/mlu/mlu_ops_api.h"
#include "core/layers/common/deepseek_v4_rotary_embedding.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_decoder_layer.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_hash.h"
#include "models/llm/deepseek_v41.h"
#include "models/llm/deepseek_v41_token_map.h"
#include "models/llm/llm_model_base.h"

namespace xllm::mlu::model {
class DeepseekV41ModelImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV41ModelImpl(const ModelContext& context)
      : args_(context.get_model_args()) {
    const auto& parallel = context.get_parallel_args();
    CHECK_EQ(parallel.cp_size(), 1) << "V4.1 Native CP is not enabled in V1.2";
    CHECK_EQ(parallel.kv_split_size_effective(), 1);
    CHECK_EQ(parallel.layerwise_split_size(), 1);
    CHECK_EQ(context.num_speculative_tokens(), 0)
        << "V4.1 Native speculative decoding belongs to V2";
    CHECK(args_.layers_to_capture().empty());
    CHECK_EQ(args_.n_hash_layers(), 0);
    CHECK_GT(args_.hc_mult(), 0);
    CHECK_GT(args_.hc_sinkhorn_iters(), 0);
    CHECK_EQ(KVCacheConfig::get_instance().block_size(), 128);
    CHECK(args_.requires_execution_batch());
    const auto& options = context.get_tensor_options();
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    const auto rope_options = options.dtype(torch::kFloat32);
    layer::DeepseekV4RotaryEmbedding plain(
        args_.qk_rope_head_dim(),
        args_.max_position_embeddings(),
        /*interleaved=*/true,
        args_.rope_theta(),
        args_.compress_rope_theta(),
        /*scaling_factor=*/1,
        /*extrapolation_factor=*/1,
        args_.beta_fast(),
        args_.beta_slow(),
        /*attn_factor=*/1,
        /*mscale=*/0,
        /*mscale_all_dim=*/0,
        /*original_max_position_embeddings=*/0,
        rope_options);
    layer::DeepseekV4RotaryEmbedding compressed(
        args_.qk_rope_head_dim(),
        args_.max_position_embeddings(),
        /*interleaved=*/true,
        args_.rope_theta(),
        args_.compress_rope_theta(),
        args_.factor(),
        /*extrapolation_factor=*/1,
        args_.beta_fast(),
        args_.beta_slow(),
        /*attn_factor=*/1,
        /*mscale=*/0,
        /*mscale_all_dim=*/0,
        args_.rope_scaling_original_max_position_embeddings(),
        rope_options);
    plain_rope_ = plain.get_cos_sin_cache("default");
    compressed_rope_ = compressed.get_cos_sin_cache("c4");
    layers_.reserve(args_.n_layers());
    for (int32_t id = 0; id < args_.n_layers(); ++id) {
      layers_.emplace_back(
          register_module("layer_" + std::to_string(id),
                          layer::DeepseekV41DecoderLayer(context, id)));
    }
  }

  torch::Tensor get_input_embeddings(torch::Tensor tokens) {
    return embed_tokens_(tokens);
  }
  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }
  void set_word_embedding(layer::WordEmbedding& embedding) {
    embed_tokens_ = embedding;
  }

  void load_hash_state(const StateDict& state) {
    for (auto& layer : layers_) {
      layer->load_hash_state(state);
    }
  }
  void load_state_dict(const StateDict& state) {
    embed_tokens_->load_state_dict(state.get_dict_with_prefix(
        std::vector<std::string>{"embed.", "embed_tokens."}));
    norm_->load_state_dict(state.get_dict_with_prefix("norm."));
    for (size_t id = 0; id < layers_.size(); ++id) {
      layers_[id]->load_state_dict(
          state.get_dict_with_prefix("layers." + std::to_string(id) + "."));
    }
  }
  void verify_loaded_weights() const {
    CHECK(embed_tokens_->is_weight_loaded()) << "Missing embed.weight";
    norm_->verify_loaded_weights("norm.");
    for (size_t id = 0; id < layers_.size(); ++id) {
      layers_[id]->verify_loaded_weights("layers." + std::to_string(id) + ".");
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& caches,
                      const ModelParamsView& raw_input) {
    torch::NoGradGuard guard;
    CHECK(!raw_input.enable_graph)
        << "V4.1 Native graph execution belongs to V9";
    CHECK(!raw_input.is_spec_verify);
    CHECK(!raw_input.prefill_without_cache);
    CHECK_EQ(caches.size(), layers_.size());
    const auto& counts = raw_input.parallel.dp_global_token_nums;
    std::optional<ModelInputSnapshot> snapshot;
    std::optional<ModelParamsView> padded;
    const bool host_block_tables =
        std::any_of(raw_input.multi_block_tables.begin(),
                    raw_input.multi_block_tables.end(),
                    [&tokens](const torch::Tensor& table) {
                      return table.device() != tokens.device();
                    });
    if (tokens.numel() == 0 || host_block_tables ||
        std::find(counts.begin(), counts.end(), 0) != counts.end()) {
      snapshot.emplace(raw_input.clone());
      padded.emplace(snapshot->view());
      auto& local_counts = padded->parallel.dp_global_token_nums;
      std::replace(local_counts.begin(), local_counts.end(), 0, 1);
      // Engine block tables stay on CPU for other backends. CSA2 indexes device
      // caches, so materialize one model-local copy before traversing layers.
      for (auto& table : padded->multi_block_tables) {
        table = table.to(tokens.device(), /*non_blocking=*/true);
      }
    }
    if (tokens.numel() == 0) {
      tokens = torch::zeros({1}, tokens.options());
      positions = torch::zeros({1}, positions.options());
    }
    const auto& input = padded ? *padded : raw_input;
    const auto& batch = input.meta.execution_batch;
    CHECK(batch.num_reqs > 0 || batch.num_tokens == 0);
    CHECK_EQ(batch.num_reqs, input.meta.num_sequences);
    CHECK(batch.num_reqs == 0 || batch.num_tokens == tokens.numel());
    const layer::DeepseekV41EngramInput engram_input =
        batch.num_reqs > 0 && !args_.engram_layer_ids().empty()
            ? layer::DeepseekV41EngramInput(
                  tokens, input.embedding.engram_token_history)
            : layer::DeepseekV41EngramInput();
    auto embeddings = input.embedding.input_embedding;
    if (!embeddings.defined()) {
      embeddings = embed_tokens_(tokens);
    }
    auto hidden = embeddings.unsqueeze(1).repeat({1, args_.hc_mult(), 1});
    auto pre = torch::zeros({tokens.numel(), args_.hc_mult()},
                            hidden.options().dtype(torch::kFloat32));
    pre.select(1, 0).fill_(1);
    std::vector<layer::Csa2Selection> selections(
        static_cast<size_t>(batch.num_reqs));
    std::optional<layer::PendingMHC> pending;
    for (size_t id = 0; id < layers_.size(); ++id) {
      if (!input.synchronize_layer(static_cast<uint32_t>(id))) {
        return ModelOutput();
      }
      hidden = layers_[id]->forward(
          hidden,
          pre,
          tokens,
          positions,
          args_.compress_ratios().at(id) == 0 ? plain_rope_ : compressed_rope_,
          caches,
          input,
          engram_input,
          selections,
          pending);
      if (!input.record_layer(static_cast<uint32_t>(id), hidden.device())) {
        return ModelOutput();
      }
    }
    hidden = kernel::mlu::fused_mul_reduce_sum(hidden, pre);
    const auto [output, residual] = norm_(hidden, std::nullopt);
    return ModelOutput(output, residual);
  }

 private:
  ModelArgs args_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  std::vector<layer::DeepseekV41DecoderLayer> layers_;
  torch::Tensor plain_rope_, compressed_rope_;
};
TORCH_MODULE(DeepseekV41Model);

class DeepseekV41ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DeepseekV41Model> {
 public:
  explicit DeepseekV41ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV41Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    const auto& args = loader->model_args();
    if (!args.engram_layer_ids().empty()) {
      const layer::DeepseekV41HashOptions options{
          args.engram_layer_ids(),
          args.engram_num_embeddings(),
          args.engram_max_ngram_size(),
          args.engram_n_heads(),
          args.engram_vocab_size(),
          args.engram_compressed_vocab_size()};
      const auto hash = layer::build_deepseek_v41_hash_parameters(options);
      model_->load_hash_state(StateDict(
          {{"token_map",
            build_deepseek_v41_token_map(
                args, loader->model_weights_path() + "/tokenizer.json")},
           {"multipliers", hash.multipliers},
           {"primes", hash.primes},
           {"offsets", hash.offsets}}));
    }
    for (const auto& state : loader->get_state_dicts()) {
      std::unordered_map<std::string, torch::Tensor> tensors;
      tensors.reserve(state->size());
      for (const auto& [key, value] : *state) {
        const std::string name = !prefix.empty() && key.starts_with(prefix)
                                     ? key.substr(prefix.size())
                                     : key;
        CHECK(tensors.emplace(name, value).second)
            << "Duplicate normalized weight " << name;
      }
      const StateDict normalized(std::move(tensors));
      model_->load_state_dict(normalized);
      if (!embedding_mode_) {
        lm_head_->load_state_dict(normalized.get_dict_with_prefix(
            tie_word_embeddings
                ? std::vector<std::string>{"embed.", "embed_tokens."}
                : std::vector<std::string>{"head.", "lm_head."}));
      }
    }
    model_->verify_loaded_weights();
    if (!embedding_mode_) {
      CHECK(lm_head_->is_weight_loaded()) << "Missing head.weight";
    }
  }
};
TORCH_MODULE(DeepseekV41ForCausalLM);
REGISTER_CAUSAL_MODEL(deepseek_v41, DeepseekV41ForCausalLM);
}  // namespace xllm::mlu::model
