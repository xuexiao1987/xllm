/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "server/shutdown_signals.h"

#include <glog/logging.h>

#include <atomic>

namespace xllm {
namespace {
// Process-lifetime storage also keeps contexts valid during worker teardown.
std::atomic<bool> g_shutdown_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void shutdown_signal_handler(int /*signum*/) {
  g_shutdown_requested.store(true, std::memory_order_relaxed);
}
}  // namespace

ShutdownSignals::ShutdownSignals(bool handle_hangup)
    : handle_hangup_(handle_hangup) {
  g_shutdown_requested.store(false, std::memory_order_relaxed);
  struct sigaction action = {};
  action.sa_handler = shutdown_signal_handler;
  sigemptyset(&action.sa_mask);
  CHECK_EQ(sigaction(SIGINT, &action, &previous_int_), 0);
  CHECK_EQ(sigaction(SIGTERM, &action, &previous_term_), 0);
  if (handle_hangup_) {
    CHECK_EQ(sigaction(SIGHUP, &action, &previous_hup_), 0);
  }
}

ShutdownSignals::~ShutdownSignals() {
  CHECK_EQ(sigaction(SIGINT, &previous_int_, nullptr), 0);
  CHECK_EQ(sigaction(SIGTERM, &previous_term_, nullptr), 0);
  if (handle_hangup_) {
    CHECK_EQ(sigaction(SIGHUP, &previous_hup_, nullptr), 0);
  }
}

ShutdownContext ShutdownSignals::context() const {
  return ShutdownContext(g_shutdown_requested);
}

}  // namespace xllm
