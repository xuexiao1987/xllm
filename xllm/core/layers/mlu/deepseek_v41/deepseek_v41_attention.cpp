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
#include "core/layers/mlu/deepseek_v41/deepseek_v41_attention.h"

#include <algorithm>
#include <cmath>

#include "core/layers/mlu/deepseek_v41/attention_projection.h"

namespace xllm::layer {
namespace {
constexpr int64_t kBlockSize = 128;
}  // namespace

DeepseekV41AttentionImpl::DeepseekV41AttentionImpl(const ModelContext& context,
                                                   int32_t layer_id)
    : args_(context.get_model_args()),
      layer_id_(layer_id),
      rank_(context.get_parallel_args().tp_group_->rank()),
      tp_size_(context.get_parallel_args().tp_group_->world_size()),
      options_(context.get_tensor_options()),
      layout_(Csa2CachePlan(args_.compress_ratios(),
                            args_.kv_source_layer_ids(),
                            args_.index_source_layer_ids())
                  .layer(layer_id)) {
  CHECK_EQ(options_.dtype().toScalarType(), torch::kBFloat16);
  CHECK_GT(args_.o_groups(), 0);
  CHECK_EQ(args_.o_groups() % tp_size_, 0);
  CHECK_EQ(args_.n_heads() % args_.o_groups(), 0);
  CHECK_GT(args_.qk_rope_head_dim(), 0);
  CHECK_EQ(args_.qk_rope_head_dim() % 2, 0);
  CHECK_LE(args_.qk_rope_head_dim(), args_.head_dim());
  const Csa2CachePlan plan(args_.compress_ratios(),
                           args_.kv_source_layer_ids(),
                           args_.index_source_layer_ids());
  Csa2IndexOptions index;
  index.ratio = std::max(1, layout_.ratio);
  index.topk = args_.index_topk();
  const int32_t candidate = args_.candidate_source_layer_id();
  if (candidate >= 0) {
    index.candidate_mode = layer_id == candidate  ? Csa2CandidateMode::PRODUCE
                           : layer_id > candidate ? Csa2CandidateMode::CONSUME
                                                  : Csa2CandidateMode::NONE;
    index.candidate_block_size = args_.candidate_block_size();
    index.candidate_topk_blocks = args_.candidate_topk_blocks();
  }
  attention_ = std::make_unique<Csa2Attention>(
      layout_.mode, args_.window_size(), kBlockSize, index);
  if (layout_.source >= 0) {
    const auto& ratios = plan.manager_ratios();
    main_manager_ = static_cast<int32_t>(
        std::distance(ratios.begin(),
                      std::find(ratios.begin(), ratios.end(), layout_.ratio)));
    CHECK_LT(main_manager_, ratios.size());
  }
  if (layout_.owns_main) {
    compressor_.emplace(layout_.ratio,
                        kBlockSize,
                        args_.qk_rope_head_dim(),
                        args_.rms_norm_eps());
  }
  const std::string prefix = "layers." + std::to_string(layer_id) + ".attn.";
  const auto& quant = context.get_quant_args();
  auto* group = context.get_parallel_args().tp_group_;
  wq_b_ =
      register_module("wq_b",
                      ColumnParallelLinear(args_.q_lora_rank(),
                                           args_.n_heads() * args_.head_dim(),
                                           /*bias=*/false,
                                           /*gather_output=*/false,
                                           quant.for_module(prefix + "wq_b"),
                                           group,
                                           options_));
  wo_b_ =
      register_module("wo_b",
                      RowParallelLinear(args_.o_groups() * args_.o_lora_rank(),
                                        args_.hidden_size(),
                                        /*bias=*/false,
                                        /*input_is_parallelized=*/true,
                                        /*reduce_results=*/true,
                                        quant.for_module(prefix + "wo_b"),
                                        group,
                                        options_));
  weight_shapes_ = {{"wq_a.weight", {args_.q_lora_rank(), args_.hidden_size()}},
                    {"wkv.weight", {args_.head_dim(), args_.hidden_size()}},
                    {"q_norm.weight", {args_.q_lora_rank()}},
                    {"kv_norm.weight", {args_.head_dim()}},
                    {"wo_a.weight",
                     {args_.o_groups(),
                      args_.o_lora_rank(),
                      args_.n_heads() * args_.head_dim() / args_.o_groups()}},
                    {"attn_sink", {args_.n_heads()}}};
  if (layout_.owns_main) {
    weight_shapes_.emplace(
        "compressor.wkv.weight",
        std::vector<int64_t>{args_.head_dim(), args_.hidden_size()});
    weight_shapes_.emplace("compressor.norm.weight",
                           std::vector<int64_t>{args_.head_dim()});
    weight_shapes_.emplace(
        "indexer.wk.weight",
        std::vector<int64_t>{args_.index_head_dim(), args_.head_dim()});
    weight_shapes_.emplace("indexer.k_norm.weight",
                           std::vector<int64_t>{args_.index_head_dim()});
    if (layout_.ratio == 2) {
      weight_shapes_.emplace(
          "compressor.wgate.weight",
          std::vector<int64_t>{args_.head_dim(), args_.hidden_size()});
    }
  }
  if (layout_.mode == Csa2CachePlan::Mode::FULL ||
      layout_.mode == Csa2CachePlan::Mode::REINDEX) {
    CHECK_GT(args_.index_n_heads(), 0);
    CHECK_GE(args_.index_head_dim(), args_.qk_rope_head_dim());
    weight_shapes_.emplace(
        "indexer.wq_b.weight",
        std::vector<int64_t>{args_.index_n_heads() * args_.index_head_dim(),
                             args_.q_lora_rank()});
    weight_shapes_.emplace(
        "indexer.weights_proj.weight",
        std::vector<int64_t>{args_.index_n_heads(), args_.hidden_size()});
  }
}

void DeepseekV41AttentionImpl::load_state_dict(const StateDict& state) {
  wq_b_->load_state_dict(state.get_dict_with_prefix("wq_b."));
  wo_b_->load_state_dict(state.get_dict_with_prefix("wo_b."));
  for (const auto& [name, shape] : weight_shapes_) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    if (name == "wo_a.weight") {
      value = deepseek_v41::grouped_output_weight(
          value, shape[0], shape[1], shape[2]);
    } else {
      CHECK(value.sizes().vec() == shape) << "Invalid shape for " << name;
    }
    const bool norm_weight = name.find("norm.weight") != std::string::npos;
    CHECK(value.scalar_type() == torch::kBFloat16 ||
          ((norm_weight || name == "attn_sink") &&
           value.scalar_type() == torch::kFloat32))
        << "Expected BF16 projection or BF16/FP32 normalization for " << name;
    if (name == "wo_a.weight" || name == "attn_sink") {
      value = value.chunk(tp_size_, 0).at(rank_);
    }
    const torch::ScalarType dtype =
        name == "attn_sink" ? torch::kFloat32 : value.scalar_type();
    weights_[name] = value.to(options_.device(), dtype).contiguous();
  }
}

void DeepseekV41AttentionImpl::verify_loaded_weights() const {
  for (const auto& [name, shape] : weight_shapes_) {
    CHECK(weights_.contains(name))
        << "Missing layers." << layer_id_ << ".attn." << name;
  }
  CHECK(wq_b_->is_weight_loaded());
  CHECK(wo_b_->is_weight_loaded());
}

torch::Tensor DeepseekV41AttentionImpl::forward(
    const torch::Tensor& hidden,
    const torch::Tensor& positions,
    const torch::Tensor& rope,
    std::vector<KVCache>& caches,
    const ModelParamsView& input,
    std::vector<Csa2Selection>& selections) {
  using deepseek_v41::dense_projection;
  using deepseek_v41::norm;
  using deepseek_v41::rotate;
  const auto& batch = input.meta.execution_batch;
  if (batch.num_reqs == 0) {
    return torch::zeros_like(hidden);
  }
  const int64_t rope_dim = args_.qk_rope_head_dim();
  const int64_t dim = args_.head_dim();
  const double eps = args_.rms_norm_eps();
  const auto qr = norm(dense_projection(hidden, weights_.at("wq_a.weight")),
                       weights_.at("q_norm.weight"),
                       eps);
  const auto query =
      rotate(wq_b_->forward(qr).view({-1, args_.n_heads() / tp_size_, dim}),
             positions,
             rope,
             rope_dim);
  const auto kv =
      rotate(norm(dense_projection(hidden, weights_.at("wkv.weight")),
                  weights_.at("kv_norm.weight"),
                  eps),
             positions,
             rope,
             rope_dim);
  Csa2LayerCacheView cache;
  cache.swa = caches.at(layer_id_).get_swa_cache();
  cache.swa_block_table = input.multi_block_tables.at(0);
  if (layout_.source >= 0) {
    cache.main = caches.at(layout_.source).get_k_cache();
    cache.index = caches.at(layout_.source).get_index_cache();
    cache.main_block_table = input.multi_block_tables.at(main_manager_);
  }
  if (compressor_) {
    torch::Tensor raw, scores;
    if (layout_.ratio == 2) {
      raw = torch::mm(
          hidden, weights_.at("compressor.wkv.weight").t(), torch::kFloat32);
      scores = torch::mm(
          hidden, weights_.at("compressor.wgate.weight").t(), torch::kFloat32);
    } else {
      raw = dense_projection(hidden, weights_.at("compressor.wkv.weight"));
    }
    const Csa2CompressionCacheView compression{
        cache.main,
        caches.at(layer_id_).get_compress_state(),
        cache.main_block_table,
        cache.swa_block_table};
    const auto rows =
        compressor_->forward(raw,
                             scores,
                             weights_.at("compressor.norm.weight"),
                             rope,
                             batch,
                             compression);
    if (rows.positions.numel() > 0) {
      const auto keys = rotate(
          norm(dense_projection(rows.latent, weights_.at("indexer.wk.weight")),
               weights_.at("indexer.k_norm.weight"),
               eps),
          rows.positions,
          rope,
          rope_dim);
      cache.index.view({-1, args_.index_head_dim()})
          .index_copy_(0, rows.slots, keys);
    }
  }
  torch::Tensor index_query, index_weights;
  if (layout_.mode == Csa2CachePlan::Mode::FULL ||
      layout_.mode == Csa2CachePlan::Mode::REINDEX) {
    index_query =
        rotate(dense_projection(qr, weights_.at("indexer.wq_b.weight"))
                   .view({-1, args_.index_n_heads(), args_.index_head_dim()}),
               positions,
               rope,
               rope_dim)
            .to(torch::kFloat32);
    index_weights =
        dense_projection(hidden, weights_.at("indexer.weights_proj.weight"))
            .to(torch::kFloat32) /
        std::sqrt(static_cast<double>(args_.index_n_heads()) *
                  args_.index_head_dim());
  }
  auto output = attention_->forward(query,
                                    kv,
                                    index_query,
                                    index_weights,
                                    weights_.at("attn_sink"),
                                    batch,
                                    cache,
                                    selections);
  output =
      rotate(std::move(output), positions, rope, rope_dim, /*inverse=*/true);
  const int64_t groups = args_.o_groups() / tp_size_;
  return wo_b_->forward(deepseek_v41::grouped_output(
      output, weights_.at("wo_a.weight"), groups, args_.o_lora_rank()));
}
}  // namespace xllm::layer
