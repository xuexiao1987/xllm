include(cc_library)

add_subdirectory(bangc)

file(GLOB_RECURSE MLU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE MLU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

set(MLU_KERNEL_DEPS
  torch
  :platform
  :triton_jit
  xllm_mlu_mhc_decode
  xllm_mlu_mhc_prefill
  xllm_mlu_csa2_gather
  cnclep
  torch_mlu
  -Wl,--no-as-needed
  torch_mlu_ops
  ${MLU_CNRT_LIBRARY}
  -Wl,--as-needed
  Python::Python
)

cc_library(
  NAME
    mlu_kernels
  HDRS
    ${MLU_HEADER_FILES}
  SRCS
    ${MLU_SOURCE_FILES}
  DEPS
    ${MLU_KERNEL_DEPS}
)
