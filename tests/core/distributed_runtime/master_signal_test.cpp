/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <stdexcept>

#include "core/distributed_runtime/master.h"

namespace xllm {
namespace {
volatile std::sig_atomic_t received_signal = 0;
void host_handler(int signum) { received_signal = signum; }

// Exercise the production Master constructor without loading another model.
class HostMaster final : public Master {
 public:
  explicit HostMaster(const Options& options, bool fail_initialization = false)
      : Master(options) {
    if (fail_initialization) {
      throw std::runtime_error("host initialization failed");
    }
  }
  const ModelArgs* model_args() const override { return nullptr; }
};

class MasterSignalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    received_signal = 0;
    struct sigaction action = {};
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaction(SIGINT, &action, &previous_int_), 0);
    ASSERT_EQ(sigaction(SIGTERM, &action, &previous_term_), 0);
    options_.backend("llm").model_path("host-constructor-probe");
  }
  void TearDown() override {
    EXPECT_EQ(sigaction(SIGINT, &previous_int_, nullptr), 0);
    EXPECT_EQ(sigaction(SIGTERM, &previous_term_, nullptr), 0);
  }
  void check_host_signals() {
    for (int signum : {SIGINT, SIGTERM}) {
      struct sigaction action = {};
      ASSERT_EQ(sigaction(signum, nullptr, &action), 0);
      EXPECT_EQ(action.sa_handler, host_handler);
      ASSERT_EQ(raise(signum), 0);
      EXPECT_EQ(received_signal, signum);
    }
  }
  Options options_;

 private:
  struct sigaction previous_int_{};
  struct sigaction previous_term_{};
};

TEST_F(MasterSignalTest, ConstructorAndDestructionPreserveHostSignals) {
  {
    HostMaster master(options_);
    check_host_signals();
  }
  check_host_signals();
}

TEST_F(MasterSignalTest, InitializationFailurePreservesHostSignals) {
  EXPECT_THROW(HostMaster(options_, /*fail_initialization=*/true),
               std::runtime_error);
  check_host_signals();
}

TEST_F(MasterSignalTest,
       ExplicitStopContextEndsIdleLoopWithoutSignalOwnership) {
  std::atomic<bool> requested{false};
  options_.shutdown_context(ShutdownContext(requested));
  HostMaster master(options_);
  master.run();
  check_host_signals();
  requested.store(true);
  master.wait();
  check_host_signals();
}
}  // namespace
}  // namespace xllm
