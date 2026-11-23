/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#pragma once

#include <csignal>

#include "core/common/shutdown_context.h"

namespace xllm {

// Only executable entry points own this scope, before starting model loading.
// Library constructors must never install process signal policy.
class ShutdownSignals final {
 public:
  explicit ShutdownSignals(bool handle_hangup = false);
  ~ShutdownSignals();
  ShutdownSignals(const ShutdownSignals&) = delete;
  ShutdownSignals& operator=(const ShutdownSignals&) = delete;

  ShutdownContext context() const;

 private:
  struct sigaction previous_int_{};
  struct sigaction previous_term_{};
  struct sigaction previous_hup_{};
  bool handle_hangup_ = false;
};

}  // namespace xllm
