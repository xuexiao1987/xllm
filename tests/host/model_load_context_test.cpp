/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "framework/model_loader/model_load_context.h"

#include <gtest/gtest.h>

#include <future>
#include <thread>

namespace xllm {
namespace {

TEST(ModelLoadContextTest, CancellationReachesAnInFlightLoadOnAnotherThread) {
  ModelLoadContext worker;
  ModelLoadContext other_worker;
  std::promise<void> ready;
  std::promise<void> resume;
  auto resumed = resume.get_future();
  auto load =
      std::async(std::launch::async, [context = worker, &ready, &resumed]() {
        context.check_cancelled();
        ready.set_value();
        resumed.wait();
        context.check_cancelled();
      });
  ready.get_future().wait();
  worker.cancel();
  resume.set_value();
  EXPECT_THROW(load.get(), ModelLoadCancelled);
  EXPECT_NO_THROW(other_worker.check_cancelled());
}

TEST(ModelLoadContextTest, StateDictViewRetainsCancellationAfterOwnerRelease) {
  ModelLoadContext view;
  {
    ModelLoadContext worker;
    view = worker;
    worker.cancel();
  }
  EXPECT_THROW(view.check_cancelled(), ModelLoadCancelled);
  view.cancel();
  EXPECT_THROW(view.check_cancelled(), ModelLoadCancelled);
}

}  // namespace
}  // namespace xllm
