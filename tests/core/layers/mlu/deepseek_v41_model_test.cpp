/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "core/framework/model/llm_model_params.h"
#include "core/platform/device.h"
#include "models/llm/mlu/deepseek_v41.h"

namespace xllm::mlu::model {
namespace {

void compare_csa2_decode(Csa2CachePlan::Mode mode,
                         const std::vector<int32_t>& positions) {
  const torch::Device device(torch::kPrivateUse1, 0);
  Device(device).set_seed();
  torch::manual_seed(1701);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto q = torch::randn({2, 4, 512}, bf16) * 0.25;
  const auto kv = torch::randn({2, 512}, bf16);
  const auto iq = torch::randn({2, 32, 128}, bf16).to(torch::kFloat32);
  const auto iw = torch::rand({2, 32});
  const auto sink = torch::tensor({-0.5, 0.25, 1.5, -1.0}, bf16);
  const int64_t pages =
      *std::max_element(positions.begin(), positions.end()) / 256 + 1;
  layer::Csa2LayerCacheView cpu{
      torch::randn({8, 128, 512}, bf16),
      torch::randn({2 * pages, 128, 512}, bf16),
      torch::randn({2 * pages, 128, 128}, bf16),
      torch::tensor({3, 1, 6, 4}, torch::kInt32).view({2, 2}),
      torch::randperm(2 * pages, torch::kInt32).view({2, pages})};
  layer::Csa2LayerCacheView mlu{cpu.swa.to(device),
                                cpu.main.to(device),
                                cpu.index.to(device),
                                cpu.swa_block_table.to(device),
                                cpu.main_block_table.to(device)};
  ExecutionBatchMetadata batch{2, 2, {1, 1}, positions, {0, 1, 2}, {0, 0}};
  layer::Csa2IndexOptions options;
  options.ratio = 2;
  options.topk = 512;
  layer::Csa2Attention attention(mode,
                                 /*window_size=*/128,
                                 /*block_size=*/128,
                                 options);
  std::vector<layer::Csa2Selection> cpu_selection(2), mlu_selection(2);
  // Include an interior masked selection: gathering must compact stably,
  // rather than assuming all negative indices form a suffix.
  cpu_selection[0].indices =
      torch::tensor({2, -1, 0, 1}, torch::kInt64).view({1, 4});
  cpu_selection[1].indices = torch::full({1, 4}, -1, torch::kInt64);
  for (int32_t seq = 0; seq < 2; ++seq) {
    mlu_selection[seq].indices = cpu_selection[seq].indices.to(device);
  }
  const auto expected =
      attention.forward(q, kv, iq, iw, sink, batch, cpu, cpu_selection);
  const auto actual = attention
                          .forward(q.to(device),
                                   kv.to(device),
                                   iq.to(device),
                                   iw.to(device),
                                   sink.to(device),
                                   batch,
                                   mlu,
                                   mlu_selection)
                          .cpu();
  EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/0.03, /*atol=*/0.015));
  EXPECT_TRUE(torch::equal(mlu.swa.cpu(), cpu.swa));
}

TEST(Csa2DecodeTest, NativeGatherPreservesStableRowsAndRingWritesBitwise) {
  const torch::Device device(torch::kPrivateUse1, 0);
  Device(device).set_seed();
  torch::manual_seed(1704);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto original = torch::randn({8 * 128, 512}, bf16);
  const auto compressed = torch::randn({6 * 128, 512}, bf16);
  const auto current = torch::randn({512}, bf16);
  const std::vector<int32_t> swa_pages{5, 1, 7, 3};
  const std::vector<int32_t> compressed_pages{3, 5, 0, 4, 1, 2};
  const auto swa_table = torch::tensor(swa_pages, torch::kInt32).to(device);
  const auto compressed_table =
      torch::tensor(compressed_pages, torch::kInt32).to(device);
  const auto device_compressed = compressed.to(device);
  const auto device_current = current.to(device);
  for (int64_t position : {0, 127, 128, 255, 1025}) {
    for (int64_t count : {0, 1, 3, 16, 17, 127, 128, 129, 511, 512, 513}) {
      for (int64_t window : {1, 128, 256}) {
        SCOPED_TRACE(position);
        SCOPED_TRACE(count);
        SCOPED_TRACE(window);
        const int64_t win = std::min(position + 1, window);
        // Include holes, duplicates and reversed page order. Output padding
        // must remain untouched; only the current ring slot may change.
        std::vector<int64_t> indices(count);
        for (int64_t id = 0; id < count; ++id) {
          indices[id] = id % 3 == 1 || (position == 127 && count == 128)
                            ? -1
                            : (count - id) % 257;
        }
        std::vector<torch::Tensor> rows;
        rows.reserve(win + count);
        for (int64_t logical = position - win + 1; logical <= position;
             ++logical) {
          const int64_t physical =
              swa_pages[(logical / 128) % 4] * 128 + logical % 128;
          rows.emplace_back(logical == position ? current : original[physical]);
        }
        for (int64_t logical : indices) {
          if (logical < 0) {
            continue;
          }
          rows.emplace_back(compressed[compressed_pages[logical / 128] * 128 +
                                       logical % 128]);
        }
        auto swa = original.to(device);
        auto output = torch::full({window + 768, 512}, -9, bf16).to(device);
        auto length =
            torch::zeros({1}, torch::dtype(torch::kInt32).device(device));
        kernel::mlu::csa2_decode_gather(
            swa,
            device_compressed,
            device_current,
            swa_table,
            compressed_table,
            torch::tensor(indices, torch::kInt64).to(device),
            position,
            window,
            /*topk_capacity=*/768,
            output,
            length);
        auto expected_cache = original.clone();
        expected_cache[swa_pages[(position / 128) % 4] * 128 + position % 128]
            .copy_(current);
        const auto host_output = output.cpu();
        EXPECT_EQ(length.cpu().item<int32_t>(), rows.size());
        EXPECT_TRUE(torch::equal(host_output.slice(0, 0, rows.size()),
                                 torch::stack(rows)));
        EXPECT_TRUE(
            host_output.slice(0, rows.size()).eq(-9).all().item<bool>());
        EXPECT_TRUE(torch::equal(swa.cpu(), expected_cache));
      }
    }
  }
}

TEST(Csa2DecodeTest, CompleteCandidatePoolMatchesMaskedSelectionBitwise) {
  const torch::Device device(torch::kPrivateUse1, 0);
  Device(device).set_seed();
  torch::manual_seed(1705);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (int64_t ratio : {1, 2}) {
    const layer::Csa2Indexer producer(
        {.ratio = ratio,
         .topk = 512,
         .candidate_mode = layer::Csa2CandidateMode::PRODUCE,
         .candidate_block_size = 8,
         .candidate_topk_blocks = 2048});
    const layer::Csa2Indexer consumer(
        {.ratio = ratio,
         .topk = 512,
         .candidate_mode = layer::Csa2CandidateMode::CONSUME,
         .candidate_block_size = 8,
         .candidate_topk_blocks = 2048});
    for (int64_t length : {1, 8, 9, 513}) {
      for (int64_t count : {1, 33}) {
        const auto keys = torch::randn({length, 16}, options);
        const auto query = torch::randn({count, 4, 16}, options);
        const auto weights = torch::rand({count, 4}, options);
        for (int64_t begin : {int64_t{0}, ratio * length - 1}) {
          const auto pool = producer.select(query.slice(0, 0, 1),
                                            weights.slice(0, 0, 1),
                                            keys,
                                            ratio * length - 1);
          const auto candidates = pool.candidates.repeat({count, 1});
          ASSERT_EQ(pool.complete_candidate_blocks, (length + 7) / 8);
          const auto masked =
              consumer.select(query, weights, keys, begin, candidates);
          const auto complete = consumer.select(query,
                                                weights,
                                                keys,
                                                begin,
                                                candidates,
                                                pool.complete_candidate_blocks);
          EXPECT_TRUE(torch::equal(masked.indices, complete.indices))
              << "ratio=" << ratio << " length=" << length << " count=" << count
              << " begin=" << begin;
        }
      }
    }
  }
}

TEST(Csa2DecodeTest, MatchesCpuSwaAcrossRingWrap) {
  compare_csa2_decode(Csa2CachePlan::Mode::SWA, {127, 256});
}

TEST(Csa2DecodeTest, MatchesCpuReuseWithMaskedSelections) {
  compare_csa2_decode(Csa2CachePlan::Mode::REUSE, {127, 256});
}

TEST(Csa2DecodeTest, MatchesCpuFullWithEmptyAndPopulatedHistory) {
  compare_csa2_decode(Csa2CachePlan::Mode::FULL, {0, 256});
}

TEST(Csa2DecodeTest, MatchesCpuPagedTopkPastSelectionCapacity) {
  compare_csa2_decode(Csa2CachePlan::Mode::FULL, {1025, 1537});
}

TEST(Csa2DecodeTest, CompressionMatchesCpuAtOddTailsAndPageBoundaries) {
  const torch::Device device(torch::kPrivateUse1, 0);
  Device(device).set_seed();
  torch::manual_seed(1702);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  const std::vector<int32_t> starts{0, 1, 127, 128, 255, 256, 1025};
  const int64_t batch_size = starts.size();
  std::vector<int32_t> prefix(batch_size + 1);
  std::iota(prefix.begin(), prefix.end(), 0);
  ExecutionBatchMetadata batch{static_cast<int32_t>(batch_size),
                               static_cast<int32_t>(batch_size),
                               std::vector<int32_t>(batch_size, 1),
                               starts,
                               prefix,
                               std::vector<uint8_t>(batch_size, 0)};
  const auto phase = torch::randn({1026, 32}, fp32);
  const auto rope = torch::cat({phase.cos().repeat_interleave(2, 1),
                                phase.sin().repeat_interleave(2, 1)},
                               1);
  const auto weight = torch::rand({512}, bf16) + 0.5;
  for (int64_t ratio : {1, 2}) {
    for (bool save_tail : {false, true}) {
      const auto raw =
          torch::randn({batch_size, 512}, ratio == 2 ? fp32 : bf16);
      const auto scores = torch::randn({batch_size, 512}, fp32);
      layer::Csa2CompressionCacheView cpu{
          torch::randn({64, 128, 512}, bf16),
          torch::randn({14, 32, 1024}, fp32),
          torch::randperm(63, torch::kInt32).view({batch_size, 9}),
          torch::randperm(14, torch::kInt32).view({batch_size, 2})};
      layer::Csa2CompressionCacheView mlu{cpu.main.to(device),
                                          cpu.state.to(device),
                                          cpu.main_block_table.to(device),
                                          cpu.state_block_table.to(device)};
      layer::Csa2Compressor compressor(ratio,
                                       /*block_size=*/128,
                                       /*rope_dim=*/64,
                                       /*norm_eps=*/1e-6,
                                       save_tail);
      const auto expected =
          compressor.forward(raw, scores, weight, rope, batch, cpu);
      const auto actual = compressor.forward(raw.to(device),
                                             scores.to(device),
                                             weight.to(device),
                                             rope.to(device),
                                             batch,
                                             mlu);
      EXPECT_TRUE(torch::allclose(actual.latent.cpu(),
                                  expected.latent,
                                  /*rtol=*/0.01,
                                  /*atol=*/0.015));
      EXPECT_TRUE(torch::equal(actual.positions.cpu(), expected.positions));
      EXPECT_TRUE(torch::equal(actual.slots.cpu(), expected.slots));
      EXPECT_TRUE(torch::allclose(mlu.main.cpu(),
                                  cpu.main,
                                  /*rtol=*/0.01,
                                  /*atol=*/0.015));
      EXPECT_TRUE(torch::equal(mlu.state.cpu(), cpu.state));
    }
  }
}

class DeepseekV41ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Device(device_).set_seed();
    group_ = create_process_group(/*rank=*/0,
                                  /*world_size=*/1,
                                  /*rank_size=*/1,
                                  /*port=*/0,
                                  /*trans=*/false,
                                  "127.0.0.1",
                                  "v41_native_test",
                                  device_);
    args_.model_type() = "deepseek_v41";
    args_.requires_execution_batch() = true;
    args_.hidden_size() = 256;
    args_.vocab_size() = 128;
    args_.n_layers() = 6;
    args_.n_heads() = 4;
    args_.n_kv_heads() = 1;
    args_.head_dim() = 64;
    args_.q_lora_rank() = 128;
    args_.qk_rope_head_dim() = 32;
    args_.o_groups() = 2;
    args_.o_lora_rank() = 64;
    args_.index_n_heads() = 2;
    args_.index_head_dim() = 32;
    args_.index_topk() = 2;
    args_.compress_ratios() = {0, 2, 2, 2, 1, 1};
    args_.kv_source_layer_ids() = {1, 4};
    args_.index_source_layer_ids() = {1, 2, 4};
    args_.window_size() = 128;
    args_.max_position_embeddings() = 256;
    args_.rope_theta() = 10000;
    args_.compress_rope_theta() = 10000;
    args_.factor() = 1;
    args_.hc_mult() = 4;
    args_.hc_sinkhorn_iters() = 20;
    args_.hc_eps() = 1e-6;
    args_.rms_norm_eps() = 1e-6;
    args_.n_routed_experts() = 4;
    args_.n_shared_experts() = 1;
    args_.num_experts_per_tok() = 2;
    args_.n_group() = 2;
    args_.topk_group() = 2;
    args_.norm_topk_prob() = true;
    args_.routed_scaling_factor() = 1;
    args_.moe_intermediate_size() = 128;
    args_.hidden_act() = "silu";
    args_.scoring_func() = "sqrtsoftplus";
    args_.topk_method() = "noaux_tc";
    args_.swiglu_limit() = 10;
    args_.engram_layer_ids() = {1, 4};
    args_.engram_num_embeddings() = {64, 96};
    args_.engram_max_ngram_size() = 3;
    args_.engram_n_heads() = 2;
    args_.engram_head_dim() = 32;
    args_.engram_vocab_size() = 5;
    args_.engram_compressed_vocab_size() = 128;
    args_.engram_pad_token_id() = 0;
  }

  DeepseekV41Model loaded_model() {
    const int64_t dim = args_.hidden_size();
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, group_.get());
    parallel.tp_group_ = group_.get();
    parallel.dp_local_process_group_ = group_.get();
    parallel.single_rank_group_ = group_.get();
    parallel.moe_ep_group_ = group_.get();
    parallel.cp_group_ = group_.get();
    const ModelContext context(parallel, args_, QuantArgs(), options_);
    DeepseekV41Model model(context);
    const auto hash = layer::build_deepseek_v41_hash_parameters(
        {{1, 4}, {64, 96}, 3, 2, 5, 128});
    model->load_hash_state(
        StateDict({{"token_map", torch::arange(128, torch::kInt64)},
                   {"multipliers", hash.multipliers},
                   {"primes", hash.primes},
                   {"offsets", hash.offsets}}));
    std::unordered_map<std::string, torch::Tensor> weights;
    const auto dense = [&](const std::string& name, torch::IntArrayRef shape) {
      weights[name] = (torch::randn(shape) * 0.025).to(torch::kBFloat16);
    };
    const auto norm = [&](const std::string& name, int64_t dim) {
      weights[name] = torch::ones({dim});
    };
    dense("embed.weight", {128, dim});
    norm("norm.weight", dim);
    const Csa2CachePlan plan(args_.compress_ratios(),
                             args_.kv_source_layer_ids(),
                             args_.index_source_layer_ids());
    for (int32_t id = 0; id < args_.n_layers(); ++id) {
      const std::string root = "layers." + std::to_string(id) + ".";
      const std::string attn = root + "attn.";
      dense(attn + "wq_a.weight", {128, dim});
      dense(attn + "wq_b.weight", {256, 128});
      dense(attn + "wkv.weight", {64, dim});
      dense(attn + "wo_a.weight", {128, 128});
      dense(attn + "wo_b.weight", {dim, 128});
      norm(attn + "q_norm.weight", 128);
      norm(attn + "kv_norm.weight", 64);
      weights[attn + "attn_sink"] = torch::zeros({4});
      const auto& layout = plan.layer(id);
      if (layout.owns_main) {
        dense(attn + "compressor.wkv.weight", {64, dim});
        norm(attn + "compressor.norm.weight", 64);
        dense(attn + "indexer.wk.weight", {32, 64});
        norm(attn + "indexer.k_norm.weight", 32);
        if (layout.ratio == 2) {
          dense(attn + "compressor.wgate.weight", {64, dim});
        }
      }
      if (layout.mode == Csa2CachePlan::Mode::FULL ||
          layout.mode == Csa2CachePlan::Mode::REINDEX) {
        dense(attn + "indexer.wq_b.weight", {64, 128});
        dense(attn + "indexer.weights_proj.weight", {2, dim});
      }
      norm(root + "attn_norm.weight", dim);
      norm(root + "ffn_norm.weight", dim);
      for (const std::string phase : {"attn", "ffn"}) {
        const std::string hc = root + "hc_" + phase + "_pre.";
        dense(hc + "hc_fn", {24, 4 * dim});
        weights[hc + "hc_base"] = torch::randn({24}) * 0.05;
        weights[hc + "hc_scale"] = torch::full({3}, 0.1);
      }
      dense(root + "ffn.gate.weight", {4, dim});
      weights[root + "ffn.gate.e_score_correction_bias"] = torch::zeros({4});
      for (int32_t expert = 0; expert < 5; ++expert) {
        const std::string prefix =
            root + "ffn." +
            (expert == 4 ? "shared_experts."
                         : "experts." + std::to_string(expert) + ".");
        dense(prefix + "w1.weight", {128, dim});
        dense(prefix + "w3.weight", {128, dim});
        dense(prefix + "w2.weight", {dim, 128});
      }
    }
    std::vector<torch::Tensor> checkpoint_bytes;
    checkpoint_bytes.reserve(args_.engram_layer_ids().size());
    for (size_t id = 0; id < args_.engram_layer_ids().size(); ++id) {
      const std::string root =
          "layers." + std::to_string(args_.engram_layer_ids()[id]) + ".engram.";
      const int64_t rows = args_.engram_num_embeddings()[id];
      dense(root + "wkv.weight", {5 * dim, 128});
      weights[root + "q_weight"] = torch::ones({4, dim});
      weights[root + "k_weight"] = torch::ones({4, dim});
      checkpoint_bytes.emplace_back(
          torch::randint(32, 56, {rows, 32}, torch::kUInt8));
      model->load_state_dict(
          StateDict({{root + "embed.weight", checkpoint_bytes.back()}}));
      model->load_state_dict(
          StateDict({{root + "embed.scale",
                      torch::full({rows, 1}, 124, torch::kUInt8)}}));
    }
    model->load_state_dict(StateDict(std::move(weights)));
    model->verify_loaded_weights();
    // The table must own its partition after loading, not alias checkpoint
    // mappings. Invalid bytes below must not influence subsequent forwards.
    for (auto& bytes : checkpoint_bytes) {
      bytes.fill_(255);
    }
    model->eval();
    return model;
  }

  std::vector<KVCache> caches() const {
    const Csa2CachePlan plan(args_.compress_ratios(),
                             args_.kv_source_layer_ids(),
                             args_.index_source_layer_ids());
    std::vector<KVCache> result;
    result.reserve(args_.n_layers());
    for (int32_t id = 0; id < args_.n_layers(); ++id) {
      DeepSeekV4KVCacheTensors tensors;
      tensors.swa_cache = torch::zeros({4, 128, 64}, options_);
      const auto& layout = plan.layer(id);
      if (layout.owns_main) {
        tensors.key_cache = torch::zeros({4, 128, 64}, options_);
        tensors.index_cache = torch::zeros({4, 128, 32}, options_);
        if (layout.owns_state) {
          tensors.compress_state =
              torch::zeros({4, 32, 128}, options_.dtype(torch::kFloat32));
        }
      }
      result.emplace_back(tensors);
    }
    return result;
  }

  torch::Tensor forward(DeepseekV41Model& model,
                        std::vector<KVCache>& cache,
                        const std::vector<int32_t>& starts,
                        const std::vector<int32_t>& lengths,
                        bool prefill,
                        bool host_history = false) const {
    LlmModelParams params;
    auto& batch = params.meta.execution_batch;
    batch.num_reqs = static_cast<int32_t>(starts.size());
    batch.num_scheduled_tokens = lengths;
    batch.num_computed_tokens = starts;
    batch.query_start_loc = {0};
    batch.query_start_loc.reserve(starts.size() + 1);
    batch.is_prefilling.assign(starts.size(), prefill);
    params.meta.num_sequences = batch.num_reqs;
    params.meta.batch_forward_type =
        prefill ? BatchForwardType::CHUNKED_PREFILL : BatchForwardType::DECODE;
    std::vector<int64_t> ids, positions, history;
    ids.reserve(256);
    positions.reserve(256);
    history.reserve(starts.size() * 2);
    for (size_t row = 0; row < starts.size(); ++row) {
      batch.query_start_loc.emplace_back(batch.query_start_loc.back() +
                                         lengths[row]);
      for (int32_t token = starts[row]; token < starts[row] + lengths[row];
           ++token) {
        ids.emplace_back(1 + token % 126);
        positions.emplace_back(token);
      }
      for (int32_t offset = 1; offset <= 2; ++offset) {
        history.emplace_back(
            starts[row] >= offset ? 1 + (starts[row] - offset) % 126 : -1);
      }
    }
    batch.num_tokens = static_cast<int64_t>(ids.size());
    params.parallel.dp_global_token_nums = {static_cast<int32_t>(ids.size())};
    params.embedding.engram_token_history =
        torch::tensor(history, torch::kInt64).view({batch.num_reqs, 2});
    if (!host_history && batch.num_reqs > 0) {
      params.embedding.engram_token_history =
          params.embedding.engram_token_history.to(device_);
    }
    // Two independent request page ranges; compare mixed decode against
    // separate requests to detect cache/history crossing request boundaries.
    const auto tables = torch::tensor({{0, 1}, {2, 3}}, torch::kInt32)
                            .slice(0, 0, batch.num_reqs);
    params.multi_block_tables = {tables, tables, tables};
    const ModelParamsView view(params);
    const auto output =
        model->forward(torch::tensor(ids, torch::kInt64).to(device_),
                       torch::tensor(positions, torch::kInt64).to(device_),
                       cache,
                       view);
    EXPECT_EQ(params.parallel.dp_global_token_nums.front(), ids.size());
    return output.hidden_states.cpu();
  }

  torch::Device device_{torch::kPrivateUse1, 0};
  torch::TensorOptions options_ =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  std::unique_ptr<ProcessGroup> group_;
  ModelArgs args_;
};

TEST_F(DeepseekV41ModelTest, OddChunksAndDecodeMatchWholePrefill) {
  auto model = loaded_model();
  auto whole_cache = caches();
  auto split_cache = caches();
  const auto whole = forward(model, whole_cache, {0}, {9}, true);
  const auto first = forward(model, split_cache, {0}, {3}, true);
  const auto second = forward(model, split_cache, {3}, {5}, true);
  const auto last = forward(model, split_cache, {8}, {1}, false);
  const auto split = torch::cat({first, second, last});
  ASSERT_TRUE(torch::isfinite(split).all().item<bool>());
  EXPECT_GT(split.abs().max().item<float>(), 0.1);
  EXPECT_TRUE(torch::allclose(split, whole, 0.03, 0.03))
      << "max error: " << (split - whole).abs().max().item<float>();
}

TEST_F(DeepseekV41ModelTest, RaggedBatchPreservesRequestCachesAndHistory) {
  auto model = loaded_model();
  auto batched = caches();
  auto separate_a = caches();
  auto separate_b = caches();
  forward(model, batched, {0, 0}, {3, 6}, true);
  const auto actual = forward(model, batched, {3, 6}, {1, 1}, false);
  forward(model, separate_a, {0}, {3}, true);
  forward(model, separate_b, {0}, {6}, true);
  const auto expected =
      torch::cat({forward(model, separate_a, {3}, {1}, false),
                  forward(model, separate_b, {6}, {1}, false)});
  ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_TRUE(torch::allclose(actual, expected, 0.03, 0.03));
}

TEST_F(DeepseekV41ModelTest, HostAndDeviceHistoryMatchAcrossEngramLayers) {
  auto model = loaded_model();
  auto host_cache = caches();
  auto device_cache = caches();
  const auto host = forward(model,
                            host_cache,
                            {0, 0},
                            {3, 6},
                            true,
                            /*host_history=*/true);
  const auto device = forward(model, device_cache, {0, 0}, {3, 6}, true);
  EXPECT_TRUE(torch::equal(host, device));
  const auto host_decode = forward(model,
                                   host_cache,
                                   {3, 6},
                                   {1, 1},
                                   false,
                                   /*host_history=*/true);
  const auto device_decode =
      forward(model, device_cache, {3, 6}, {1, 1}, false);
  EXPECT_TRUE(torch::equal(host_decode, device_decode));
}

TEST_F(DeepseekV41ModelTest, IdleBatchLeavesEveryCacheUntouched) {
  auto model = loaded_model();
  auto cache = caches();
  const auto output = forward(model, cache, {}, {}, false);
  EXPECT_EQ(output.size(0), 1);
  for (auto& layer : cache) {
    EXPECT_EQ(torch::count_nonzero(layer.get_swa_cache()).item<int64_t>(), 0);
  }
  for (const int32_t id : {1, 4}) {
    EXPECT_EQ(torch::count_nonzero(cache[id].get_k_cache()).item<int64_t>(), 0);
    EXPECT_EQ(torch::count_nonzero(cache[id].get_index_cache()).item<int64_t>(),
              0);
  }
  EXPECT_EQ(torch::count_nonzero(cache[1].get_compress_state()).item<int64_t>(),
            0);
}

TEST_F(DeepseekV41ModelTest,
       DelayedMhc5120PreservesEngramAndForwardBoundaries) {
  args_.hidden_size() = 5120;
  torch::manual_seed(1703);
  auto model = loaded_model();
  auto batched = caches();
  auto separate_a = caches();
  auto separate_b = caches();
  forward(model, batched, {0, 0}, {3, 6}, true);
  const auto actual = forward(model, batched, {3, 6}, {1, 1}, false);
  forward(model, separate_a, {0}, {3}, true, /*host_history=*/true);
  forward(model, separate_b, {0}, {6}, true, /*host_history=*/true);
  const auto expected =
      torch::cat({forward(model, separate_a, {3}, {1}, false),
                  forward(model, separate_b, {6}, {1}, false)});
  ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/0.03, /*atol=*/0.03))
      << "max error: " << (actual - expected).abs().max().item<float>();

  // No pending state may escape a forward or belong to an idle rank.
  auto idle_cache = caches();
  const auto idle = forward(model, idle_cache, {}, {}, false);
  EXPECT_EQ(idle.size(0), 1);
  for (auto& layer : idle_cache) {
    EXPECT_EQ(torch::count_nonzero(layer.get_swa_cache()).item<int64_t>(), 0);
  }
  auto fresh = caches();
  const auto replay = forward(model, fresh, {0, 0}, {3, 6}, true);
  auto again = caches();
  EXPECT_TRUE(
      torch::equal(replay, forward(model, again, {0, 0}, {3, 6}, true)));
}

}  // namespace
}  // namespace xllm::mlu::model
