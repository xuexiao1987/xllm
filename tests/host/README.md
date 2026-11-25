<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# DeepSeek V4.1 host tests

These tests run device-independent model implementation code on macOS or Linux.
They do not provide a CPU engine backend or emulate accelerator kernels.

From the repository root, use Python 3.12, a C++20 compiler and Ninja:

```sh
python3.12 -m venv .deps/model-test-venv
.deps/model-test-venv/bin/python -m pip install -r tests/host/requirements.txt
export PATH="$PWD/.deps/model-test-venv/bin:$PATH"
cmake -S tests/host -B build-host-model -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython3_EXECUTABLE="$PWD/.deps/model-test-venv/bin/python"
cmake --build build-host-model --target all_tests --parallel 6
ctest --test-dir build-host-model --output-on-failure --no-tests=error \
  --timeout 120 --output-junit results.xml
```

The entry point builds the CSA2 cache plan tests and the C++/Python layout
parity probe, and runs the Python model tests in this directory. It uses the
existing `cc_test` compiled-test discovery path. GoogleTest and glog revisions
are fixed for this standalone test build; production dependency pins do not
change. Existing sources can be supplied through
`FETCHCONTENT_SOURCE_DIR_GOOGLETEST` and `FETCHCONTENT_SOURCE_DIR_GLOG`.

Python tests cover checkpoint aliases and shapes, dense and expert TP shards,
Engram hashing and chunked INT8 table loading, mHC mixing, MoE routing, CSA2
history and compression, and model/executor wiring. V4 regression tests cover
shared checkpoint code. Numerical fixtures and their provenance are documented
in [data/README.md](data/README.md). CPU summation of rank-local partials does
not test distributed device communication.

For native memory and undefined-behavior checks, configure a separate build
with `-DXLLM_HOST_SANITIZER=address,undefined`; run CTest with
`-E '^host_python_'`. The sanitizer applies to compiled C++ targets and their
dependencies, not prebuilt Python/Torch wheels. ThreadSanitizer can be selected
with `-DXLLM_HOST_SANITIZER=thread`.

Accelerator kernels, distributed communication, real-checkpoint inference,
peak device memory and performance require separate device tests. The model
uses the base engine's startup and shutdown behavior; this change does not add
loading cancellation or a new process signal policy. Passing host tests alone
is not device acceptance.
