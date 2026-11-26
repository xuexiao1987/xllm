/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {

// Cache geometry is shared by estimation and allocation. Reindex/Reuse own
// SWA only; their main KV and index K belong to the preceding Full layer.
class Csa2CachePlan final {
 public:
  static constexpr int64_t kStateRows = 32;
  enum class Mode : int8_t { SWA, FULL, REINDEX, REUSE };
  struct Layer {
    Mode mode;
    int32_t ratio;
    int32_t source;
    bool owns_main;
    bool owns_state;
  };

  Csa2CachePlan(std::vector<int32_t> ratios,
                const std::vector<int32_t>& sources)
      : Csa2CachePlan(std::move(ratios), sources, sources) {}

  Csa2CachePlan(std::vector<int32_t> ratios,
                std::vector<int32_t> sources,
                const std::vector<int32_t>& index_sources)
      : ratios_(std::move(ratios)), sources_(std::move(sources)) {
    for (const int32_t ratio : ratios_) {
      CHECK(ratio == 0 || ratio == 1 || ratio == 2)
          << "CSA2 text cache supports C1/C2";
    }
    validate_sources(sources_);
    validate_sources(index_sources);
    for (const int32_t source : sources_) {
      CHECK(ratios_[source] == 1 || ratios_[source] == 2);
      CHECK(std::binary_search(
          index_sources.begin(), index_sources.end(), source))
          << "KV sources must also be index sources";
    }
    layers_.reserve(ratios_.size());
    int32_t source = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(ratios_.size()); ++i) {
      const bool full = owns_main(i);
      source = full ? i : source;
      const bool indexed =
          std::binary_search(index_sources.begin(), index_sources.end(), i);
      CHECK(!indexed || (source >= 0 && ratios_[i] > 0));
      const bool swa = ratios_[i] == 0 || source < 0;
      CHECK(swa || ratios_[i] == ratios_[source])
          << "CSA2 consumers must use their source compression ratio";
      const Mode mode = swa       ? Mode::SWA
                        : full    ? Mode::FULL
                        : indexed ? Mode::REINDEX
                                  : Mode::REUSE;
      layers_.push_back(
          {mode, ratios_[i], swa ? -1 : source, full, full && ratios_[i] == 2});
    }
    manager_ratios_.reserve(3);
    manager_ratios_.emplace_back(0);
    for (const int32_t ratio : {2, 1}) {
      if (source_count(ratio) > 0) {
        manager_ratios_.emplace_back(ratio);
      }
    }
  }

  const Layer& layer(int64_t index) const { return layers_.at(index); }
  const std::vector<int32_t>& manager_ratios() const { return manager_ratios_; }

  void validate_candidate_source(int32_t source) const {
    CHECK_GE(source, -1);
    if (source == -1) {
      return;
    }
    CHECK_LT(source, static_cast<int32_t>(layers_.size()));
    CHECK(layer(source).mode == Mode::FULL ||
          layer(source).mode == Mode::REINDEX)
        << "CSA2 candidate source must be an index-producing layer";
    for (int32_t i = source + 1; i < static_cast<int32_t>(layers_.size());
         ++i) {
      CHECK(layer(i).mode == Mode::SWA ||
            layer(i).source == layer(source).source)
          << "CSA2 candidate source cannot cross KV groups";
    }
  }
  bool owns_main(int64_t layer) const {
    return std::binary_search(sources_.begin(), sources_.end(), layer);
  }
  int64_t source_count(int32_t ratio) const {
    return std::count_if(sources_.begin(), sources_.end(), [&](int32_t layer) {
      return ratios_[layer] == ratio;
    });
  }
  int64_t swa_block_bytes(int64_t block_size,
                          int64_t head_dim,
                          int64_t dtype_bytes) const {
    return static_cast<int64_t>(ratios_.size()) * block_size * head_dim *
               dtype_bytes +
           source_count(2) * kStateRows * head_dim * 2 * sizeof(float);
  }
  int64_t main_block_bytes(int64_t block_size,
                           int64_t head_dim,
                           int64_t index_dim,
                           int64_t dtype_bytes,
                           int64_t index_dtype_bytes = 1,
                           int64_t scale_bytes = 2) const {
    return block_size * (head_dim * dtype_bytes +
                         index_dim * index_dtype_bytes + scale_bytes);
  }
  // Two logical blocks cover one C2 block and two C1 blocks.
  int64_t token_unit_bytes(int64_t block_size,
                           int64_t head_dim,
                           int64_t index_dim,
                           int64_t dtype_bytes,
                           int64_t index_dtype_bytes = 1,
                           int64_t scale_bytes = 2) const {
    return (source_count(2) + 2 * source_count(1)) *
           main_block_bytes(block_size,
                            head_dim,
                            index_dim,
                            dtype_bytes,
                            index_dtype_bytes,
                            scale_bytes);
  }

 private:
  void validate_sources(const std::vector<int32_t>& sources) const {
    int32_t previous = -1;
    for (const int32_t source : sources) {
      CHECK_GT(source, previous);
      CHECK_LT(source, static_cast<int32_t>(ratios_.size()));
      previous = source;
    }
  }
  std::vector<Layer> layers_;
  std::vector<int32_t> manager_ratios_;
  std::vector<int32_t> ratios_;
  std::vector<int32_t> sources_;
};
}  // namespace xllm
