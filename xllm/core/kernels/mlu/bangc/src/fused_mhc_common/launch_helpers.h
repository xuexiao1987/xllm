/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cnrt.h>

#include "core/kernels/mlu/bangc/src/fused_mhc_common/launch_plan.h"

namespace xllm::kernel::mlu::bangc::fused_mhc_detail {

// Topology of the current device, cached per thread for the last device.
inline Topology query_topology() {
  static thread_local int32_t cached_dev = -1;
  static thread_local Topology cached;
  int32_t dev = 0;
  if (cnrtGetDevice(&dev) != cnrtSuccess) {
    return Topology{};
  }
  if (dev != cached_dev) {
    Topology t;
    if (cnrtDeviceGetAttribute(&t.clusters, cnrtAttrClusterCount, dev) !=
            cnrtSuccess ||
        cnrtDeviceGetAttribute(&t.cores_per_cluster,
                               cnrtAttrMcorePerCluster,
                               dev) != cnrtSuccess) {
      return Topology{};
    }
    cached = t;
    cached_dev = dev;
  }
  return cached;
}

inline cnrtFunctionType_t union_type(int32_t clusters) {
  switch (clusters) {
    case 1:
      return cnrtFuncTypeUnion1;
    case 2:
      return cnrtFuncTypeUnion2;
    case 4:
      return cnrtFuncTypeUnion4;
    case 8:
      return cnrtFuncTypeUnion8;
    case 16:
      return cnrtFuncTypeUnion16;
    default:
      return cnrtFuncTypeUnion1;
  }
}

}  // namespace xllm::kernel::mlu::bangc::fused_mhc_detail
