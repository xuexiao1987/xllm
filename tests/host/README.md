<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Host CPU validation

This standalone test build runs device-independent production code on macOS
and Linux. It is not a CPU inference backend and does not build the engine,
initialize accelerator SDKs, or supply fake native operators.

Run from the repository root with Python 3.12, a C++20 compiler and Ninja:

```sh
python3.12 -m venv .deps/mac-cpu-venv
.deps/mac-cpu-venv/bin/python -m pip install -r tests/host/requirements.txt
export PATH="$PWD/.deps/mac-cpu-venv/bin:$PATH"
cmake -S tests/host -B build-host-mac -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython3_EXECUTABLE="$PWD/.deps/mac-cpu-venv/bin/python"
cmake --build build-host-mac --target all_tests --parallel 6
ctest --test-dir build-host-mac --output-on-failure --no-tests=error \
  --timeout 120 --output-junit results.xml
python -m pytest -q -rs tests/scripts \
  --junitxml=build-host-mac/python-scripts.xml
```

CMake fetches fixed GoogleTest, Abseil and glog revisions for this test build
only. An existing checkout can be supplied with the standard
`FETCHCONTENT_SOURCE_DIR_GOOGLETEST`, `FETCHCONTENT_SOURCE_DIR_ABSEIL` and
`FETCHCONTENT_SOURCE_DIR_GLOG` options; record its actual SHA in the run record.
The production submodule pins remain unchanged. `requirements.txt` fixes the
host tools; save `python -m pip freeze` alongside results for transitive versions.

The C++ targets reuse existing tests and real implementation sources for
scheduler notifications, PD cancellation identities, heartbeat callback lifetime,
KV transfer topology, worker startup arguments, virtual KV page geometry/state,
and stopping conditions. GoogleTest discovers cases from the compiled binaries.

`host_python_cache_test` explicitly runs three runtime-neutral cache test modules
with `--noconftest`. Their production imports do not need `xllm_export` or kernel
initialization. Do not extend this selection to model/runtime tests that require
`tests/python/conftest.py`. `host_python_model_test` runs device-independent
model preparation tests in this directory; it does not prove a device Forward.

The V4.1 checkpoint tests execute the base `ModelWeightLoader` on CPU tensors.
They cover nested text configuration, CSA2 weight ownership, dynamic INT8 versus
FLOAT projections, TP row/column splits, CP Q replication, and missing or invalid
weights. Backbone tests also cover hidden-sharded embedding, vocabulary-sharded
head aliases and replicated layer/final norms.

Engram tests execute hashing, packed request windows, INT8 head-bucket loading,
group-32 dequantization, TP shard reassembly, sidecar validation and the gated
residual projection. mHC tests cover the previous-sublayer pre coefficients,
Sinkhorn normalization, residual matrix orientation and final learned merge.
The fixtures in `data/` contain numerical outputs from the pinned internal
reference, not from the implementation under test. TP tests assemble real CPU
tensors; they do not test a distributed collective.

The V4.1 model is registered through the existing model factory and executor.
MoE tests exercise the shared base W8/W4 checkpoint loader, orthogonal
MoE-TP/EP shards, BF16/FP32 expert Forward against pinned internal outputs,
unbiased routing weights and explicit backend selection. EP/TP numerical tests
sum actual local partials; they do not substitute for distributed device tests.

CSA2 tests cover cache history, compression layouts, chunk boundaries and the
attention implementation against independent CPU reference calculations. The
signal tests exercise the production entry-point adapter and worker load
cancellation context, including restoration of host signal handlers.

These host tests do not validate NPU kernels, distributed collectives, real
checkpoint inference, device peak memory or device shutdown. Those require
separate accelerator evidence. FP8 loading, Engram host staging and DP shared
storage, and graph execution are outside the initial V4.1 execution path.
Passing these tests alone is not model acceptance.

The build-tool suite uses controlled subprocess boundaries, including fixtures
that test registration without compiling a binary. It is separate evidence from
the C++ tests that actually execute. On macOS, Linux ELF loader tests report
skips. Keep them in the report as pending Linux-host checks, not passed cases.
No other skip or missing dependency should be counted as CPU validation success.

For native memory and undefined-behavior checks, configure a separate directory:

```sh
cmake -S tests/host -B build-host-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython3_EXECUTABLE="$PWD/.deps/mac-cpu-venv/bin/python" \
  -DXLLM_HOST_SANITIZER=address,undefined
cmake --build build-host-asan --target all_tests --parallel 6
ctest --test-dir build-host-asan --output-on-failure --no-tests=error \
  --timeout 120 -E '^host_python_' --output-junit results.xml
```

The sanitizer applies to compiled C++ targets and dependencies, not to the
prebuilt Python/Torch wheels. ThreadSanitizer is separately selectable with
`-DXLLM_HOST_SANITIZER=thread`. Device allocation, copies/events, collectives,
Graph capture/replay, real model precision and performance require the target
accelerator and remain outside this host gate.
