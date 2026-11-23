/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "server/shutdown_signals.h"

#include <gtest/gtest.h>

#include <atomic>
#include <csignal>
#include <future>
#include <stdexcept>
#include <thread>

#include "core/framework/model_loader/model_load_context.h"

namespace xllm {
namespace {
volatile std::sig_atomic_t host_signal = 0;

void host_handler(int signum) { host_signal = signum; }

class ShutdownSignalsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    host_signal = 0;
    struct sigaction action = {};
    action.sa_handler = host_handler;
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaction(SIGINT, &action, &original_int_), 0);
    ASSERT_EQ(sigaction(SIGTERM, &action, &original_term_), 0);
    ASSERT_EQ(sigaction(SIGHUP, &action, &original_hup_), 0);
  }

  void TearDown() override {
    EXPECT_EQ(sigaction(SIGINT, &original_int_, nullptr), 0);
    EXPECT_EQ(sigaction(SIGTERM, &original_term_, nullptr), 0);
    EXPECT_EQ(sigaction(SIGHUP, &original_hup_, nullptr), 0);
  }

 private:
  struct sigaction original_int_{};
  struct sigaction original_term_{};
  struct sigaction original_hup_{};
};

TEST_F(ShutdownSignalsTest, LibraryContextPreservesHostHandlers) {
  ShutdownContext library;
  std::atomic<bool> requested{false};
  ShutdownContext explicit_context(requested);
  for (int signum : {SIGINT, SIGTERM}) {
    ASSERT_EQ(raise(signum), 0);
    EXPECT_EQ(host_signal, signum);
    EXPECT_FALSE(library.requested());
    EXPECT_FALSE(explicit_context.requested());
  }
  requested.store(true);
  EXPECT_TRUE(explicit_context.requested());
  EXPECT_FALSE(library.requested());
}

TEST_F(ShutdownSignalsTest, EntrySignalsNotifyWorkersAndRestoreHost) {
  for (int signum : {SIGINT, SIGTERM}) {
    host_signal = 0;
    {
      ShutdownSignals signals;
      const ShutdownContext context = signals.context();
      ModelLoadContext loading;
      std::promise<void> ready;
      auto worker = std::async(std::launch::async, [&]() {
        ready.set_value();
        while (!context.requested()) {
          std::this_thread::yield();
        }
        loading.cancel();
        loading.check_cancelled();
      });
      ready.get_future().wait();
      ASSERT_EQ(raise(signum), 0);
      EXPECT_THROW(worker.get(), ModelLoadCancelled);
      EXPECT_EQ(host_signal, 0);
    }
    ASSERT_EQ(raise(signum), 0);
    EXPECT_EQ(host_signal, signum);
  }
}

TEST_F(ShutdownSignalsTest, InitializationFailureRestoresHostHandlers) {
  EXPECT_THROW(
      {
        ShutdownSignals signals;
        throw std::runtime_error("startup failure");
      },
      std::runtime_error);
  for (int signum : {SIGINT, SIGTERM}) {
    ASSERT_EQ(raise(signum), 0);
    EXPECT_EQ(host_signal, signum);
  }
}

TEST_F(ShutdownSignalsTest, StandaloneWorkerAlsoHandlesHangup) {
  {
    ShutdownSignals signals(/*handle_hangup=*/true);
    ASSERT_EQ(raise(SIGHUP), 0);
    EXPECT_TRUE(signals.context().requested());
    EXPECT_EQ(host_signal, 0);
  }
  ASSERT_EQ(raise(SIGHUP), 0);
  EXPECT_EQ(host_signal, SIGHUP);
}

}  // namespace
}  // namespace xllm
