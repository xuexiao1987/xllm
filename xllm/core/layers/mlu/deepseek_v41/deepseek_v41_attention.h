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

#include <unordered_map>

#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/common/linear.h"
#include "core/layers/mlu/deepseek_v41/csa2_attention.h"
#include "core/layers/mlu/deepseek_v41/csa2_compressor.h"

namespace xllm::layer {

// Owns projections only. KV pages and per-forward index selections remain
// owned by the engine and model respectively.
class DeepseekV41AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV41AttentionImpl(const ModelContext& context, int32_t layer_id);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& positions,
                        const torch::Tensor& rope,
                        std::vector<KVCache>& caches,
                        const ModelParamsView& input,
                        std::vector<Csa2Selection>& selections);

 private:
  ModelArgs args_;
  int32_t layer_id_;
  int32_t rank_;
  int32_t tp_size_;
  torch::TensorOptions options_;
  Csa2CachePlan::Layer layout_;
  int32_t main_manager_ = -1;
  std::unique_ptr<Csa2Attention> attention_;
  std::optional<Csa2Compressor> compressor_;
  ColumnParallelLinear wq_b_{nullptr};
  RowParallelLinear wo_b_{nullptr};
  std::unordered_map<std::string, torch::Tensor> weights_;
  std::unordered_map<std::string, std::vector<int64_t>> weight_shapes_;
};
TORCH_MODULE(DeepseekV41Attention);
}  // namespace xllm::layer
