/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#pragma once

#include <atomic>

namespace xllm {

// The caller owns the stop state and must keep it alive until its workers stop.
// An empty context leaves process signals entirely under the host's control.
class ShutdownContext final {
 public:
  ShutdownContext() = default;
  explicit ShutdownContext(const std::atomic<bool>& requested)
      : requested_(&requested) {}

  bool requested() const {
    return requested_ != nullptr && requested_->load(std::memory_order_relaxed);
  }

 private:
  const std::atomic<bool>* requested_ = nullptr;
};

}  // namespace xllm
