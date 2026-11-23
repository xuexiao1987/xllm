/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#pragma once

#include <atomic>
#include <exception>
#include <memory>

namespace xllm {

// Unwinds only a cancelled load to the worker initialization boundary.
// Execution errors remain errors and must not be treated as cancellation.
class ModelLoadCancelled final : public std::exception {
 public:
  const char* what() const noexcept override {
    return "Model loading cancelled";
  }
};

// Copies share cancellation state, including the Python state-dict views.
// This context belongs to one worker lifetime, not to a global model registry.
class ModelLoadContext final {
 public:
  void cancel() { cancelled_->store(true, std::memory_order_relaxed); }
  bool cancelled() const { return cancelled_->load(std::memory_order_relaxed); }
  void check_cancelled() const {
    if (cancelled()) {
      throw ModelLoadCancelled();
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> cancelled_ =
      std::make_shared<std::atomic<bool>>(false);
};

}  // namespace xllm
