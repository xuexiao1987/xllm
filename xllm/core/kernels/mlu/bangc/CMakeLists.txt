# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

if(NOT NEUWARE_HOME)
  set(NEUWARE_HOME "$ENV{NEUWARE_HOME}")
  if(NOT NEUWARE_HOME)
    set(NEUWARE_HOME "/usr/local/neuware")
  endif()
endif()
list(APPEND CMAKE_MODULE_PATH
  "${NEUWARE_HOME}/cmake" "${NEUWARE_HOME}/cmake/modules")
set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
find_package(BANG REQUIRED)
# These kernels are tuned for MLU590. Other device architectures need their
# own verified implementation before extending this target.
string(APPEND BANG_CNCC_FLAGS
  " -std=c++17 -fPIC --bang-mlu-arch=mtp_592")
set(BANG_configuration_types Debug Release RelWithDebInfo MinSizeRel)
list(APPEND BANG_CNCC_FLAGS_DEBUG -O0 -g)
list(APPEND BANG_CNCC_FLAGS_RELEASE -O3 -DNDEBUG)
list(APPEND BANG_CNCC_FLAGS_RELWITHDEBINFO -O3 -g -DNDEBUG)
list(APPEND BANG_CNCC_FLAGS_MINSIZEREL -Os -DNDEBUG)
# Preserve the decode kernel's measured scheduling; do not change the flags
# of existing kernels or the token-split prefill kernel.
bang_add_library(xllm_mlu_mhc_decode STATIC
  src/fused_mhc_decode/fused_mhc_decode.mlu
  OPTIONS -Xbang-cnas -fno-soft-pipeline)
bang_add_library(xllm_mlu_mhc_prefill STATIC
  src/fused_mhc_prefill/fused_mhc_prefill.mlu)
bang_add_library(xllm_mlu_csa2_gather STATIC
  src/csa2_decode_gather/gather.mlu)
foreach(target xllm_mlu_mhc_decode xllm_mlu_mhc_prefill xllm_mlu_csa2_gather)
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/xllm" "${NEUWARE_HOME}/include")
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()
