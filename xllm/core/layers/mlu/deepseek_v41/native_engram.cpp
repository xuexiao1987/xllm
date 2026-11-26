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
#include "core/layers/mlu/deepseek_v41/native_engram.h"

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm::layer {
DeepseekV41NativeEngramImpl::DeepseekV41NativeEngramImpl(
    const ModelContext& context,
    int32_t engram_id)
    : args_(context.get_model_args()),
      engram_id_(engram_id),
      lookup_group_(context.get_parallel_args().tp_group_),
      options_(context.get_tensor_options()) {
  CHECK_EQ(args_.engram_layer_ids().size(),
           args_.engram_num_embeddings().size());
  CHECK_GE(engram_id_, 0);
  CHECK_LT(engram_id_, args_.engram_layer_ids().size());
  CHECK_GT(args_.engram_num_embeddings().at(engram_id_), 0);
  CHECK_GE(args_.engram_max_ngram_size(), 2);
  CHECK_GT(args_.engram_n_heads(), 0);
  CHECK_GT(args_.engram_head_dim(), 0);
  CHECK_GT(args_.engram_compressed_vocab_size(), 0);
  CHECK_GE(args_.engram_pad_token_id(), 0);
  CHECK_LT(args_.engram_pad_token_id(), args_.vocab_size());
  fp8_values_ = deepseek_v41_fp8_values(options_.device());
}

void DeepseekV41NativeEngramImpl::load_hash_state(const StateDict& state) {
  const auto token_map = state.get_tensor("token_map");
  CHECK_EQ(token_map.numel(), args_.vocab_size());
  const auto multipliers = state.get_tensor("multipliers");
  const auto primes = state.get_tensor("primes");
  const auto offsets = state.get_tensor("offsets");
  for (const auto& tensor : {multipliers, primes, offsets}) {
    CHECK(tensor.defined());
    CHECK_EQ(tensor.size(0), args_.engram_layer_ids().size());
  }
  hash_.emplace(token_map,
                multipliers[engram_id_],
                primes[engram_id_],
                offsets[engram_id_],
                args_.engram_pad_token_id(),
                args_.engram_compressed_vocab_size(),
                args_.engram_num_embeddings().at(engram_id_));
}

void DeepseekV41NativeEngramImpl::load_state_dict(const StateDict& state) {
  const int64_t cols =
      (args_.engram_max_ngram_size() - 1) * args_.engram_n_heads();
  for (const std::string name : {"wkv.weight", "q_weight", "k_weight"}) {
    const auto value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    const std::vector<int64_t> shape =
        name == "wkv.weight"
            ? std::vector<int64_t>{(args_.hc_mult() + 1) * args_.hidden_size(),
                                   cols * args_.engram_head_dim()}
            : std::vector<int64_t>{args_.hc_mult(), args_.hidden_size()};
    CHECK(value.sizes().vec() == shape) << "Invalid Engram " << name;
    CHECK(value.scalar_type() == torch::kBFloat16 ||
          (name != "wkv.weight" && value.scalar_type() == torch::kFloat32));
    weights_[name] = value.to(options_.device()).contiguous();
  }
  // The ModelLoader owns all shard mappings until load_model returns. Retain
  // views until both tensors arrive, then own only this TP rank's partition.
  const auto weight = state.get_tensor("embed.weight");
  const auto scale = state.get_tensor("embed.scale");
  if (weight.defined()) {
    CHECK(!table_ && !pending_weight_.defined());
    pending_weight_ = weight;
  }
  if (scale.defined()) {
    CHECK(!table_ && !pending_scale_.defined());
    pending_scale_ = scale;
  }
  if (pending_weight_.defined() && pending_scale_.defined()) {
    CHECK_EQ(pending_weight_.size(0),
             args_.engram_num_embeddings().at(engram_id_));
    CHECK_EQ(pending_weight_.size(1), args_.engram_head_dim());
    table_.emplace(pending_weight_,
                   pending_scale_,
                   lookup_group_->rank(),
                   lookup_group_->world_size());
    pending_weight_ = torch::Tensor();
    pending_scale_ = torch::Tensor();
  }
}

void DeepseekV41NativeEngramImpl::verify_loaded_weights() const {
  CHECK(hash_.has_value());
  CHECK(table_.has_value()) << "Missing Engram embedding table or scales";
  for (const std::string name : {"wkv.weight", "q_weight", "k_weight"}) {
    CHECK(weights_.contains(name)) << "Missing Engram " << name;
  }
}

torch::Tensor DeepseekV41NativeEngramImpl::forward(
    const torch::Tensor& hidden,
    const torch::Tensor& tokens,
    const DeepseekV41EngramInput& host_input,
    const std::vector<int32_t>& query_lengths) {
  const auto hashes =
      hash_->forward(host_input.tokens_, host_input.history_, query_lengths);
  const auto host =
      torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
  auto weight_rows = torch::empty({hashes.numel(), table_->head_dim()}, host);
  auto scale_rows = torch::empty({hashes.numel(), table_->scale_dim()}, host);
  table_->lookup_rows(hashes, weight_rows, scale_rows);
  auto rows = dequantize_deepseek_v41_engram_rows(
                  weight_rows.to(hidden.device(), /*non_blocking=*/true),
                  scale_rows.to(hidden.device(), /*non_blocking=*/true),
                  fp8_values_)
                  .to(hidden.scalar_type());
  rows = parallel_state::reduce(rows, lookup_group_);
  rows = rows.view({tokens.numel(), hashes.size(1), table_->head_dim()});
  return deepseek_v41_engram_project(hidden,
                                     rows,
                                     tokens,
                                     weights_.at("wkv.weight"),
                                     weights_.at("q_weight"),
                                     weights_.at("k_weight"),
                                     args_.rms_norm_eps());
}
}  // namespace xllm::layer
