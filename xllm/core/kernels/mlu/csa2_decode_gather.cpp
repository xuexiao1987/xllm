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

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <limits>

#include "core/kernels/mlu/bangc/src/csa2_decode_gather/launcher.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
void csa2_decode_gather(const torch::Tensor& swa,
                        const torch::Tensor& compressed,
                        const torch::Tensor& current,
                        const torch::Tensor& swa_table,
                        const torch::Tensor& compressed_table,
                        const torch::Tensor& topk,
                        int64_t position,
                        int64_t window_size,
                        int64_t topk_capacity,
                        torch::Tensor output,
                        torch::Tensor length) {
  CHECK_EQ(swa.device().type(), torch::kPrivateUse1);
  CHECK_EQ(swa.dim(), 2);
  CHECK_EQ(swa.size(1), 512);
  CHECK_EQ(compressed.dim(), 2);
  CHECK_EQ(compressed.size(1), 512);
  for (const auto& tensor : {swa, compressed, current, output}) {
    CHECK_EQ(tensor.device(), swa.device());
    CHECK_EQ(tensor.scalar_type(), torch::kBFloat16);
    CHECK(tensor.is_contiguous());
  }
  for (const auto& tensor : {swa_table, compressed_table, length}) {
    CHECK_EQ(tensor.device(), swa.device());
    CHECK_EQ(tensor.scalar_type(), torch::kInt32);
    CHECK(tensor.is_contiguous());
  }
  CHECK_EQ(topk.device(), swa.device());
  CHECK_EQ(topk.scalar_type(), torch::kInt64);
  CHECK(topk.is_contiguous());
  CHECK_EQ(swa_table.dim(), 1);
  CHECK_EQ(compressed_table.dim(), 1);
  CHECK_EQ(topk.dim(), 1);
  CHECK_EQ(current.numel(), 512);
  CHECK_EQ(length.numel(), 1);
  CHECK_GE(position, 0);
  CHECK_GT(window_size, 0);
  CHECK_GE(swa_table.numel() * 128, window_size);
  CHECK_GE(topk_capacity, topk.numel());
  CHECK_GE(output.numel(), (window_size + topk_capacity) * 512);
  CHECK_LE(window_size, std::numeric_limits<int32_t>::max());
  CHECK_LE(topk.numel(), std::numeric_limits<int32_t>::max());
  CHECK_LE(swa_table.numel(), std::numeric_limits<int32_t>::max());
  bangc::csa2_decode_gather_launch(torch_mlu::getCurMLUStream(),
                                   swa.data_ptr(),
                                   compressed.data_ptr(),
                                   current.data_ptr(),
                                   swa_table.data_ptr<int32_t>(),
                                   compressed_table.data_ptr<int32_t>(),
                                   topk.data_ptr<int64_t>(),
                                   static_cast<int32_t>(topk.numel()),
                                   position,
                                   static_cast<int32_t>(window_size),
                                   static_cast<int32_t>(swa_table.numel()),
                                   output.data_ptr(),
                                   length.data_ptr<int32_t>());
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
}
}  // namespace xllm::kernel::mlu
