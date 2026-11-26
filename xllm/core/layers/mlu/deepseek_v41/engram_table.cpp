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

#include "core/layers/mlu/deepseek_v41/engram_table.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>

namespace xllm::layer {
namespace {
void check_bytes(const torch::Tensor& tensor, const char* name) {
  CHECK(tensor.defined()) << "Missing Engram " << name;
  CHECK_EQ(tensor.dim(), 2) << "Engram " << name << " must be a matrix";
  CHECK_GT(tensor.size(1), 0) << "Empty Engram " << name << " row";
}
}  // namespace

DeepseekV41EngramTable::DeepseekV41EngramTable(const torch::Tensor& weight,
                                               const torch::Tensor& scale,
                                               int64_t rank,
                                               int64_t world_size) {
  CHECK_GT(world_size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size);
  check_bytes(weight, "weight");
  check_bytes(scale, "scale");
  CHECK(weight.scalar_type() == torch::kUInt8 ||
        weight.scalar_type() == torch::kInt8 ||
        weight.scalar_type() == torch::kFloat8_e4m3fn)
      << "Engram weight must contain E4M3FN bytes";
  CHECK(scale.scalar_type() == torch::kUInt8 ||
        scale.scalar_type() == torch::kInt8 ||
        scale.scalar_type() == torch::kFloat8_e8m0fnu)
      << "Engram scale must contain UE8M0 bytes";
  CHECK_EQ(weight.size(0), scale.size(0));
  CHECK_GT(weight.size(0), 0);
  CHECK_EQ(weight.size(1) % scale.size(1), 0)
      << "Engram scale groups must divide the embedding width";
  begin_ = weight.size(0) * rank / world_size;
  end_ = weight.size(0) * (rank + 1) / world_size;
  const auto bytes =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8);
  weight_ = torch::empty({end_ - begin_, weight.size(1)}, bytes);
  scale_ = torch::empty({end_ - begin_, scale.size(1)}, bytes);
  // Bound checkpoint copies and scale validation temporaries for large tables.
  constexpr int64_t kCopyBytes = 64 * 1024 * 1024;
  const int64_t rows =
      std::max<int64_t>(1, kCopyBytes / (weight.size(1) + scale.size(1)));
  for (int64_t begin = begin_; begin < end_; begin += rows) {
    const int64_t end = std::min(end_, begin + rows);
    weight_.slice(0, begin - begin_, end - begin_)
        .copy_(weight.slice(0, begin, end).view(torch::kUInt8));
    auto scales = scale_.slice(0, begin - begin_, end - begin_);
    scales.copy_(scale.slice(0, begin, end).view(torch::kUInt8));
    CHECK(!scales.eq(255).any().item<bool>())
        << "Engram UE8M0 scale contains invalid NaN byte 255";
  }
}

void DeepseekV41EngramTable::lookup_rows(const torch::Tensor& hashes,
                                         torch::Tensor& weight_rows,
                                         torch::Tensor& scale_rows) const {
  CHECK(hashes.device().is_cpu() && hashes.is_contiguous());
  CHECK_EQ(hashes.scalar_type(), torch::kInt64);
  for (const auto& tensor : {weight_rows, scale_rows}) {
    CHECK(tensor.device().is_cpu() && tensor.is_contiguous());
    CHECK_EQ(tensor.scalar_type(), torch::kUInt8);
    CHECK_EQ(tensor.dim(), 2);
    CHECK_EQ(tensor.size(0), hashes.numel());
  }
  CHECK_EQ(weight_rows.size(1), head_dim());
  CHECK_EQ(scale_rows.size(1), scale_dim());
  const int64_t* ids = hashes.const_data_ptr<int64_t>();
  const uint8_t* table = weight_.const_data_ptr<uint8_t>();
  const uint8_t* scales = scale_.const_data_ptr<uint8_t>();
  uint8_t* output = weight_rows.mutable_data_ptr<uint8_t>();
  uint8_t* output_scales = scale_rows.mutable_data_ptr<uint8_t>();
  for (int64_t index = 0; index < hashes.numel(); ++index) {
    const int64_t id = ids[index];
    if (id < begin_ || id >= end_) {
      std::memset(output + index * head_dim(), 0, head_dim());
      std::memset(output_scales + index * scale_dim(), 0, scale_dim());
      continue;
    }
    const int64_t row = id - begin_;
    std::memcpy(
        output + index * head_dim(), table + row * head_dim(), head_dim());
    std::memcpy(output_scales + index * scale_dim(),
                scales + row * scale_dim(),
                scale_dim());
  }
}

torch::Tensor deepseek_v41_fp8_values(const torch::Device& device) {
  return torch::arange(256, torch::kInt16)
      .to(torch::kUInt8)
      .view(torch::kFloat8_e4m3fn)
      .to(device, torch::kFloat32);
}

torch::Tensor dequantize_deepseek_v41_engram_rows(
    const torch::Tensor& weight_rows,
    const torch::Tensor& scale_rows,
    const torch::Tensor& fp8_values) {
  // Shapes and byte encoding are established by the table loader. Keep this
  // device path free of host synchronization and repeated value validation.
  const torch::Tensor scales =
      torch::exp2(scale_rows.to(torch::kFloat32) - 127)
          .repeat_interleave(weight_rows.size(1) / scale_rows.size(1), -1);
  return fp8_values.index_select(0, weight_rows.to(torch::kInt64).flatten())
             .view_as(scales) *
         scales;
}

}  // namespace xllm::layer
