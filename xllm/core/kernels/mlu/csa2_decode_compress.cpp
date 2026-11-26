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

#include "compiler/triton/include/jit_kernel.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
void csa2_decode_compress(const torch::Tensor& raw,
                          const torch::Tensor& scores,
                          const torch::Tensor& norm_weight,
                          const torch::Tensor& rope,
                          torch::Tensor state,
                          torch::Tensor main,
                          const torch::Tensor& state_table,
                          const torch::Tensor& main_table,
                          int64_t position,
                          int64_t ratio,
                          int64_t rope_dim,
                          double norm_eps,
                          bool save_full_tail,
                          torch::Tensor latent,
                          torch::Tensor positions,
                          torch::Tensor slots) {
  CHECK_EQ(raw.device().type(), torch::kPrivateUse1);
  CHECK_EQ(raw.numel(), 512);
  CHECK(ratio == 1 || ratio == 2);
  CHECK_GE(position, 0);
  CHECK_GE(rope_dim, 0);
  CHECK_LE(rope_dim, 512);
  CHECK_EQ(rope_dim % 2, 0);
  CHECK_EQ(main.scalar_type(), torch::kBFloat16);
  CHECK_EQ(raw.scalar_type(), ratio == 2 ? torch::kFloat32 : torch::kBFloat16);
  CHECK_EQ(norm_weight.numel(), 512);
  CHECK_GT(norm_eps, 0);
  for (const auto& tensor :
       {raw, norm_weight, main, main_table, latent, positions, slots}) {
    CHECK_EQ(tensor.device(), raw.device());
    CHECK(tensor.is_contiguous());
  }
  CHECK_EQ(main_table.scalar_type(), torch::kInt32);
  CHECK_EQ(positions.scalar_type(), torch::kInt64);
  CHECK_EQ(slots.scalar_type(), torch::kInt64);
  CHECK_EQ(latent.scalar_type(), torch::kBFloat16);
  const bool emit = ratio == 1 || position % 2 == 1;
  CHECK_EQ(latent.numel(), emit ? 512 : 0);
  CHECK_EQ(positions.numel(), emit ? 1 : 0);
  CHECK_EQ(slots.numel(), emit ? 1 : 0);
  if (rope_dim > 0) {
    CHECK_EQ(rope.device(), raw.device());
    CHECK_EQ(rope.scalar_type(), torch::kFloat32);
    CHECK(rope.is_contiguous());
    CHECK_EQ(rope.size(1), 2 * rope_dim);
  }
  if (ratio == 2) {
    for (const auto& tensor : {scores, state, state_table}) {
      CHECK_EQ(tensor.device(), raw.device());
      CHECK(tensor.is_contiguous());
    }
    CHECK_EQ(scores.scalar_type(), torch::kFloat32);
    CHECK_EQ(state.scalar_type(), torch::kFloat32);
    CHECK_EQ(state_table.scalar_type(), torch::kInt32);
    CHECK_EQ(scores.numel(), 512);
    CHECK_GT(state_table.numel(), 0);
  }
  auto& kernel = triton_jit::JITKernel::get(
      "xllm.core.kernels.mlu.triton_kernel.csa2_decode_compress",
      "csa2_decode_compress_kernel");
  kernel.launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
                {1, 1, 1},
                {/*num_warps=*/4, /*num_stages=*/1},
                raw,
                scores,
                norm_weight,
                rope,
                state,
                main,
                state_table,
                main_table,
                latent,
                positions,
                slots,
                position,
                static_cast<int32_t>(state_table.numel()),
                static_cast<float>(norm_eps),
                static_cast<int32_t>(ratio),
                static_cast<int32_t>(rope_dim),
                save_full_tail,
                /*HEAD_DIM=*/512);
}
}  // namespace xllm::kernel::mlu
