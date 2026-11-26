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

#include "core/layers/mlu/deepseek_v41/deepseek_v41_engram.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace xllm::layer {
namespace {
constexpr int64_t kImageSentinel = 129264;
constexpr int64_t kImagePad = 129265;

torch::Tensor host_vector(const torch::Tensor& tensor) {
  CHECK(tensor.defined() && tensor.device().is_cpu());
  CHECK_EQ(tensor.dim(), 1);
  CHECK(tensor.scalar_type() == torch::kInt32 ||
        tensor.scalar_type() == torch::kInt64);
  return tensor.to(torch::kCPU,
                   torch::kInt64,
                   /*non_blocking=*/false,
                   /*copy=*/true,
                   torch::MemoryFormat::Contiguous);
}
}  // namespace

DeepseekV41EngramInput::DeepseekV41EngramInput(const torch::Tensor& tokens,
                                               const torch::Tensor& history)
    : tokens_(tokens.to(torch::kCPU, torch::kInt64)),
      history_(history.to(torch::kCPU, torch::kInt64)) {}

DeepseekV41EngramHash::DeepseekV41EngramHash(const torch::Tensor& token_map,
                                             const torch::Tensor& multipliers,
                                             const torch::Tensor& primes,
                                             const torch::Tensor& offsets,
                                             int64_t pad_token_id,
                                             int64_t compressed_vocab_size,
                                             int64_t table_rows)
    : token_map_(host_vector(token_map)),
      multipliers_(host_vector(multipliers)),
      primes_(host_vector(primes)),
      offsets_(host_vector(offsets)),
      pad_token_id_(pad_token_id),
      depth_(multipliers_.numel() - 1) {
  CHECK_GT(compressed_vocab_size, 0);
  CHECK_GT(table_rows, 0);
  CHECK_GE(pad_token_id_, 0);
  CHECK_LT(pad_token_id_, token_map_.numel());
  CHECK_GE(depth_, 1);
  CHECK_GT(primes_.numel(), 0);
  CHECK_EQ(primes_.numel() % depth_, 0);
  CHECK_EQ(primes_.numel(), offsets_.numel());
  heads_ = primes_.numel() / depth_;
  CHECK_GE(token_map_.min().item<int64_t>(), 0);
  CHECK_LT(token_map_.max().item<int64_t>(), compressed_vocab_size);
  CHECK_GT(multipliers_.min().item<int64_t>(), 0);
  CHECK_LE(multipliers_.max().item<int64_t>(),
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(1, compressed_vocab_size - 1));
  CHECK_GT(primes_.min().item<int64_t>(), 0);
  CHECK_LE(primes_.max().item<int64_t>(), table_rows);
  CHECK_GE(offsets_.min().item<int64_t>(), 0);
  CHECK((offsets_ <= table_rows - primes_).all().item<bool>())
      << "Engram hash buckets exceed embedding rows";
}

torch::Tensor DeepseekV41EngramHash::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& history,
    const std::vector<int32_t>& query_lengths) const {
  CHECK(tokens.device().is_cpu() && history.device().is_cpu());
  CHECK_EQ(tokens.dim(), 1);
  CHECK_EQ(history.dim(), 2);
  CHECK(tokens.scalar_type() == torch::kInt32 ||
        tokens.scalar_type() == torch::kInt64);
  CHECK(history.scalar_type() == torch::kInt32 ||
        history.scalar_type() == torch::kInt64);
  CHECK_EQ(history.size(0), query_lengths.size());
  CHECK_EQ(history.size(1), depth_);
  int64_t rows = 0;
  for (const int32_t count : query_lengths) {
    CHECK_GE(count, 0);
    rows += count;
  }
  CHECK_EQ(rows, tokens.numel());
  const torch::Tensor ids = tokens.to(torch::kInt64).contiguous();
  const torch::Tensor old = history.to(torch::kInt64).contiguous();
  torch::Tensor result = torch::empty({rows, primes_.numel()},
                                      tokens.options().dtype(torch::kInt64));
  const int64_t* current = ids.const_data_ptr<int64_t>();
  const int64_t* previous = old.const_data_ptr<int64_t>();
  const int64_t* map = token_map_.const_data_ptr<int64_t>();
  const int64_t* mul = multipliers_.const_data_ptr<int64_t>();
  const int64_t* prime = primes_.const_data_ptr<int64_t>();
  const int64_t* offset = offsets_.const_data_ptr<int64_t>();
  int64_t* output = result.mutable_data_ptr<int64_t>();
  int64_t start = 0;
  for (int64_t seq = 0; seq < static_cast<int64_t>(query_lengths.size());
       ++seq) {
    for (int64_t row = 0; row < query_lengths[seq]; ++row) {
      int64_t rolling = 0;
      bool blocked = false;
      for (int64_t shift = 0; shift <= depth_; ++shift) {
        const int64_t relative = row - shift;
        const int64_t token = relative >= 0
                                  ? current[start + relative]
                                  : previous[seq * depth_ - relative - 1];
        blocked = blocked || token < 0 || token == kImageSentinel ||
                  token == kImagePad;
        const int64_t value =
            blocked ? map[pad_token_id_]
                    : map[std::min(token, token_map_.numel() - 1)];
        rolling ^= value * mul[shift];
        if (shift == 0) {
          continue;
        }
        for (int64_t head = 0; head < heads_; ++head) {
          const int64_t col = (shift - 1) * heads_ + head;
          output[(start + row) * primes_.numel() + col] =
              rolling % prime[col] + offset[col];
        }
      }
    }
    start += query_lengths[seq];
  }
  return result;
}

torch::Tensor deepseek_v41_engram_project(const torch::Tensor& hidden,
                                          const torch::Tensor& rows,
                                          const torch::Tensor& tokens,
                                          const torch::Tensor& wkv,
                                          const torch::Tensor& q_weight,
                                          const torch::Tensor& k_weight,
                                          double eps) {
  CHECK_EQ(hidden.dim(), 3);
  CHECK_EQ(rows.dim(), 3);
  CHECK_EQ(rows.size(0), hidden.size(0));
  CHECK_EQ(tokens.dim(), 1);
  CHECK_EQ(tokens.size(0), hidden.size(0));
  const int64_t streams = hidden.size(1);
  const int64_t width = hidden.size(2);
  CHECK_GT(width, 0);
  CHECK_GT(eps, 0);
  CHECK(wkv.sizes() == torch::IntArrayRef({(streams + 1) * width,
                                           rows.size(1) * rows.size(2)}));
  CHECK(q_weight.sizes() == torch::IntArrayRef({streams, width}));
  CHECK(k_weight.sizes() == q_weight.sizes());
  const torch::Tensor kv =
      torch::matmul(rows.flatten(1).to(wkv.scalar_type()), wkv.t());
  const torch::Tensor keys =
      kv.slice(1, 0, streams * width).view_as(hidden).to(torch::kFloat32);
  const torch::Tensor value = kv.slice(1, streams * width).to(torch::kFloat32);
  const torch::Tensor h = hidden.to(torch::kFloat32);
  torch::Tensor dot =
      (h * keys * q_weight.to(torch::kFloat32) * k_weight.to(torch::kFloat32))
          .sum(-1);
  dot *= torch::rsqrt(h.square().mean(-1) + eps) *
         torch::rsqrt(keys.square().mean(-1) + eps) / std::sqrt(width);
  torch::Tensor gate = torch::sigmoid(torch::where(dot < 0, -1.0, 1.0) *
                                      dot.abs().clamp_min(1e-6).sqrt());
  const torch::Tensor text_mask =
      tokens.ne(kImageSentinel) & tokens.ne(kImagePad);
  gate = torch::where(text_mask.unsqueeze(-1), gate, torch::zeros_like(gate));
  return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(hidden.scalar_type());
}
}  // namespace xllm::layer
