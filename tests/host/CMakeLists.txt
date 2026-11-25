# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.26)
project(xllm_host_tests LANGUAGES CXX)

# This is a test-only entry point, not a CPU backend for the engine.
get_filename_component(XLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(XLLM_HOST_SANITIZER "" CACHE STRING "Host sanitizer: empty, address,undefined, or thread")
if(XLLM_HOST_SANITIZER)
  if(NOT XLLM_HOST_SANITIZER MATCHES "^(address,undefined|thread)$")
    message(FATAL_ERROR "Unsupported XLLM_HOST_SANITIZER: ${XLLM_HOST_SANITIZER}")
  endif()
  add_compile_options("-fsanitize=${XLLM_HOST_SANITIZER}" -fno-omit-frame-pointer)
  add_link_options("-fsanitize=${XLLM_HOST_SANITIZER}")
endif()

include(FetchContent)
# Host-test dependencies only. Production dependency pins are unchanged.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59)
FetchContent_Declare(glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG 7b134a5c82c0c0b5698bb6bf7a835b230c5638e4)
FetchContent_MakeAvailable(googletest glog)

set(BUILD_TESTING ON CACHE BOOL "" FORCE)
enable_testing()
include(GoogleTest)
list(APPEND CMAKE_MODULE_PATH "${XLLM_ROOT}/cmake")
include(cc_test)
add_custom_target(all_tests)
set(COMMON_INCLUDE_DIRS
  "${XLLM_ROOT}/xllm" "${XLLM_ROOT}/xllm/core" "${XLLM_ROOT}/tests")

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
add_test(NAME host_python_model_test
  COMMAND "${Python3_EXECUTABLE}" -m pytest -q tests/host
    --junitxml=${CMAKE_CURRENT_BINARY_DIR}/python-model.xml)
set_tests_properties(host_python_model_test PROPERTIES
  WORKING_DIRECTORY "${XLLM_ROOT}" LABELS "cpu;python")
get_property(host_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${host_tests} PROPERTIES TIMEOUT 120)

cc_test(NAME csa2_cache_plan_test CPU_ONLY DISCOVER_TESTS
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/csa2_cache_plan_test.cpp"
  DEPS GTest::gtest_main glog::glog)

add_executable(csa2_layout_probe csa2_layout_probe.cpp)
target_include_directories(csa2_layout_probe PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(csa2_layout_probe PRIVATE glog::glog)
add_test(NAME csa2_layout_cross_language
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_csa2_layout_parity.py"
          $<TARGET_FILE:csa2_layout_probe>)
add_dependencies(all_tests csa2_layout_probe)
