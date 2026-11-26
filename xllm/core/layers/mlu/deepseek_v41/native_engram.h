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

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>

#include <optional>
#include <unordered_map>

#include "core/framework/model_context/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "core/layers/mlu/deepseek_v41/engram_table.h"

namespace xllm::layer {
class DeepseekV41NativeEngramImpl final : public torch::nn::Module {
 public:
  DeepseekV41NativeEngramImpl(const ModelContext& context, int32_t engram_id);
  void load_hash_state(const StateDict& state);
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& tokens,
                        const DeepseekV41EngramInput& host_input,
                        const std::vector<int32_t>& query_lengths);

 private:
  ModelArgs args_;
  int32_t engram_id_;
  ProcessGroup* lookup_group_;
  torch::TensorOptions options_;
  std::optional<DeepseekV41EngramHash> hash_;
  std::optional<DeepseekV41EngramTable> table_;
  torch::Tensor pending_weight_, pending_scale_;
  torch::Tensor fp8_values_;
  std::unordered_map<std::string, torch::Tensor> weights_;
};
TORCH_MODULE(DeepseekV41NativeEngram);
}  // namespace xllm::layer
