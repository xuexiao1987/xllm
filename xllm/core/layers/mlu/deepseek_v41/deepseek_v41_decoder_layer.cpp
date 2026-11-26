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
#include "core/layers/mlu/deepseek_v41/deepseek_v41_decoder_layer.h"

#include <algorithm>
#include <utility>

namespace xllm::layer {
DeepseekV41DecoderLayerImpl::DeepseekV41DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();
  attention_ = register_module("attn", DeepseekV41Attention(context, layer_id));
  const auto& ids = args.engram_layer_ids();
  const auto it = std::find(ids.begin(), ids.end(), layer_id);
  if (it != ids.end()) {
    engram_ = register_module(
        "engram",
        DeepseekV41NativeEngram(
            context, static_cast<int32_t>(std::distance(ids.begin(), it))));
  }
  attn_norm_ = register_module("attn_norm", RMSNorm(context));
  ffn_norm_ = register_module("ffn_norm", RMSNorm(context));
  attn_hc_ = register_module("hc_attn",
                             MHCPre(args.hc_mult(),
                                    args.hidden_size(),
                                    args.hc_sinkhorn_iters(),
                                    args.hc_eps(),
                                    args.rms_norm_eps(),
                                    options));
  ffn_hc_ = register_module("hc_ffn",
                            MHCPre(args.hc_mult(),
                                   args.hidden_size(),
                                   args.hc_sinkhorn_iters(),
                                   args.hc_eps(),
                                   args.rms_norm_eps(),
                                   options));
  post_ = register_module("hc_post", MHCPost(args.rms_norm_eps()));
  use_fused_mhc_ = attn_hc_->supports_fused_delayed_pre();
  // Engram and the model head consume materialized multi-stream residuals.
  defer_mhc_post_ =
      use_fused_mhc_ && layer_id + 1 < args.n_layers() &&
      std::find(ids.begin(), ids.end(), layer_id + 1) == ids.end();
  QuantArgs quant = context.get_quant_args();
  const std::string prefix =
      "layers." + std::to_string(layer_id) + ".ffn.shared_experts.";
  for (const auto& [source, target] :
       std::vector<std::pair<std::string, std::string>>{
           {"w13", "gate_proj"}, {"w13", "up_proj"}, {"w2", "down_proj"}}) {
    auto scheme = context.get_quant_args().module_quant_scheme(prefix + source);
    if (scheme != nullptr) {
      auto mapped = *scheme;
      mapped.targets = {prefix + target};
      quant.compressed_groups().emplace_back(std::move(mapped));
    }
  }
  moe_ =
      register_module("ffn",
                      DeepseekV4SparseMoEBlock(context.with_quant_args(quant),
                                               /*use_hash=*/false));
}

void DeepseekV41DecoderLayerImpl::load_state_dict(const StateDict& state) {
  attention_->load_state_dict(state.get_dict_with_prefix("attn."));
  if (engram_) {
    engram_->load_state_dict(state.get_dict_with_prefix("engram."));
  }
  attn_norm_->load_state_dict(state.get_dict_with_prefix("attn_norm."));
  ffn_norm_->load_state_dict(state.get_dict_with_prefix("ffn_norm."));
  const StateDict ffn = state.get_dict_with_prefix("ffn.");
  std::unordered_map<std::string, torch::Tensor> tensors;
  tensors.reserve(ffn.size());
  for (const auto& [key, value] : ffn) {
    std::string name = key;
    for (const auto& [source, target] :
         std::vector<std::pair<std::string, std::string>>{
             {".w1.", ".gate_proj."},
             {".w3.", ".up_proj."},
             {".w2.", ".down_proj."}}) {
      const size_t pos = name.rfind(source);
      if (pos != std::string::npos) {
        name.replace(pos, source.size(), target);
        break;
      }
    }
    tensors.emplace(std::move(name), value);
  }
  moe_->load_state_dict(
      StateDict(std::move(tensors), std::string(ffn.prefix())));
  attn_hc_->load_state_dict(get_hc_state(state, "hc_attn_pre.", "hc_attn_"));
  ffn_hc_->load_state_dict(get_hc_state(state, "hc_ffn_pre.", "hc_ffn_"));
}

void DeepseekV41DecoderLayerImpl::load_hash_state(const StateDict& state) {
  if (engram_) {
    engram_->load_hash_state(state);
  }
}

void DeepseekV41DecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  attention_->verify_loaded_weights();
  attn_norm_->verify_loaded_weights(prefix + "attn_norm.");
  ffn_norm_->verify_loaded_weights(prefix + "ffn_norm.");
  attn_hc_->verify_loaded_weights(prefix + "hc_attn_pre.");
  ffn_hc_->verify_loaded_weights(prefix + "hc_ffn_pre.");
  moe_->verify_loaded_weights();
  if (engram_) {
    engram_->verify_loaded_weights();
  }
}

torch::Tensor DeepseekV41DecoderLayerImpl::forward(
    torch::Tensor hidden,
    torch::Tensor& pre_mix,
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    std::vector<KVCache>& caches,
    const ModelParamsView& input,
    const DeepseekV41EngramInput& engram_input,
    std::vector<Csa2Selection>& selections,
    std::optional<PendingMHC>& pending) {
  if (input.meta.execution_batch.num_reqs == 0) {
    // Idle DP ranks still join MoE collectives; no dummy row owns KV/history.
    auto dummy = hidden.select(/*dim=*/1, /*index=*/0).contiguous();
    const auto route = moe_->prep_route(dummy);
    dummy = moe_->forward_selected(
        dummy, route.reduce_weight, route.expert_id, input);
    return dummy.unsqueeze(1).expand_as(hidden).contiguous();
  }
  if (engram_) {
    hidden = engram_->forward(hidden,
                              tokens,
                              engram_input,
                              input.meta.execution_batch.num_scheduled_tokens);
  }
  MHCPreOutput attn;
  torch::Tensor x;
  if (pending.has_value()) {
    std::tie(x, hidden, attn.pre, attn.post, attn.comb) =
        attn_hc_->fused_delayed_pre(pending->x,
                                    pending->residual,
                                    pending->post,
                                    pending->comb,
                                    pre_mix,
                                    attn_norm_->weight());
    pending.reset();
  } else {
    attn = attn_hc_->forward(hidden, std::nullopt, pre_mix);
    x = std::get<0>(attn_norm_(attn.output, std::nullopt));
  }
  x = attention_->forward(x, positions, rope, caches, input, selections);
  MHCPreOutput ffn;
  if (use_fused_mhc_) {
    std::tie(x, hidden, ffn.pre, ffn.post, ffn.comb) =
        ffn_hc_->fused_delayed_pre(
            x, hidden, attn.post, attn.comb, attn.pre, ffn_norm_->weight());
  } else {
    hidden = std::get<0>(post_->forward(x, hidden, attn.post, attn.comb));
    ffn = ffn_hc_->forward(hidden, std::nullopt, attn.pre);
    x = std::get<0>(ffn_norm_(ffn.output, std::nullopt));
  }
  const auto route = moe_->prep_route(x);
  x = moe_->forward_selected(x, route.reduce_weight, route.expert_id, input);
  pre_mix = ffn.pre;
  if (defer_mhc_post_) {
    pending.emplace(PendingMHC{x, hidden, ffn.post, ffn.comb});
    return x;
  }
  return std::get<0>(post_->forward(x, hidden, ffn.post, ffn.comb));
}
}  // namespace xllm::layer
