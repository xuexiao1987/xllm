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

/* SeedSequence algorithm adapted from NumPy (MIT):
Copyright (c) 2015 Melissa E. O'Neill
Copyright (c) 2019 NumPy Developers
https://github.com/numpy/numpy/blob/v2.3.0/numpy/random/bit_generator.pyx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "core/layers/mlu/deepseek_v41/deepseek_v41_hash.h"

#include <glog/logging.h>

#include <array>
#include <bit>
#include <limits>

namespace xllm::layer {
namespace {
// NumPy SeedSequence constants from numpy/random/bit_generator.pyx.
// Keep the upstream INIT_A/B, MULT_A/B and MIX_MULT_L/R names recognizable:
// A mixes input entropy; B expands the mixed pool into output seed words.
constexpr uint32_t kSeedInitA = 0x43b0d7e5U;
constexpr uint32_t kSeedMultA = 0x931e8875U;
constexpr uint32_t kSeedInitB = 0x8b51f9ddU;
constexpr uint32_t kSeedMultB = 0x58f38dedU;
constexpr uint32_t kSeedMixMultL = 0xca01f9ddU;
constexpr uint32_t kSeedMixMultR = 0x4973f715U;
constexpr int32_t kSeedWordBits = std::numeric_limits<uint32_t>::digits;
constexpr int32_t kSeedXorShift = kSeedWordBits / 2;
// NumPy's default entropy pool is four uint32 words (128 bits).
constexpr int32_t kSeedPoolWords = 4;
// PCG64 needs four uint64 words: a 128-bit state and a 128-bit stream seed.
constexpr int32_t kPcgSeedWords = 4;
constexpr int32_t kSeedWordsPerOutput =
    std::numeric_limits<uint64_t>::digits / kSeedWordBits;
constexpr int32_t kSeedOutputWords = kPcgSeedWords * kSeedWordsPerOutput;

// NumPy SeedSequence's single-integer seed protocol, followed by PCG64
// XSL-RR and Lemire bounded sampling. These exact draws are part of the
// trained Engram hash, not interchangeable random initialization.
std::array<uint64_t, kPcgSeedWords> seed_words(uint64_t seed) {
  uint32_t hash_constant = kSeedInitA;
  auto mix_hash = [&hash_constant](uint32_t value) {
    value ^= hash_constant;
    hash_constant *= kSeedMultA;
    value *= hash_constant;
    return value ^ (value >> kSeedXorShift);
  };
  std::array<uint32_t, kSeedPoolWords> pool;
  for (int32_t i = 0; i < kSeedPoolWords; ++i) {
    pool[i] = mix_hash(static_cast<uint32_t>(seed));
    seed >>= kSeedWordBits;
  }
  for (int32_t source = 0; source < kSeedPoolWords; ++source) {
    for (int32_t target = 0; target < kSeedPoolWords; ++target) {
      if (source == target) {
        continue;
      }
      uint32_t value =
          kSeedMixMultL * pool[target] - kSeedMixMultR * mix_hash(pool[source]);
      pool[target] = value ^ (value >> kSeedXorShift);
    }
  }
  std::array<uint64_t, kPcgSeedWords> words{};
  hash_constant = kSeedInitB;
  // Pack consecutive uint32 draws low word first, matching NumPy's uint64
  // generate_state output independently of the host's byte order.
  for (int32_t i = 0; i < kSeedOutputWords; ++i) {
    uint32_t value = pool[i % kSeedPoolWords] ^ hash_constant;
    hash_constant *= kSeedMultB;
    value *= hash_constant;
    value ^= value >> kSeedXorShift;
    words[i / kSeedWordsPerOutput] |=
        static_cast<uint64_t>(value)
        << (kSeedWordBits * (i % kSeedWordsPerOutput));
  }
  return words;
}

torch::Tensor hash_multipliers(const DeepseekV41HashOptions& options) {
  CHECK_GT(options.compressed_vocab_size, 0)
      << "engram_compressed_vocab_size must be positive";
  // NumPy PCG64 XSL-RR 128/64 constants (not the PCG64DXSM variant):
  // numpy/random/src/pcg64/pcg64.h, PCG_DEFAULT_MULTIPLIER_{HIGH,LOW}.
  constexpr uint64_t kPcgMultiplierHigh = 2549297995355413924ULL;
  constexpr uint64_t kPcgMultiplierLow = 4865540595714422341ULL;
  constexpr int32_t kPcgWordBits = std::numeric_limits<uint64_t>::digits;
  // The top six state bits select a rotation of the 64-bit output word.
  constexpr int32_t kPcgRotationBits = 6;
  constexpr int32_t kPcgRotationShift = 2 * kPcgWordBits - kPcgRotationBits;
  constexpr __uint128_t kPcgMultiplier =
      (static_cast<__uint128_t>(kPcgMultiplierHigh) << kPcgWordBits) |
      kPcgMultiplierLow;
  // Training seeds each layer with 10007 * layer_id, then maps bounded
  // draws to odd hash multipliers via 2 * draw + 1. Preserve both conventions
  // to address the same trained Engram embeddings.
  constexpr uint64_t kEngramLayerSeedMultiplier = 10007ULL;
  constexpr uint64_t kOddMultiplierScale = 2;
  const int64_t layers = options.layer_ids.size();
  const int64_t shifts = options.max_ngram_size;
  auto result = torch::empty({layers, shifts}, torch::kInt64);
  const uint64_t bound = std::numeric_limits<int64_t>::max() /
                         options.compressed_vocab_size / kOddMultiplierScale;
  // The model's vocabulary uses NumPy's 64-bit bounded-integer path.
  CHECK_GT(bound, std::numeric_limits<uint32_t>::max());
  // Unsigned wraparound computes 2^64 % bound for Lemire rejection sampling.
  const uint64_t rejection_threshold = -bound % bound;
  for (int64_t layer = 0; layer < layers; ++layer) {
    const auto [state_high, state_low, sequence_high, sequence_low] =
        seed_words(kEngramLayerSeedMultiplier * options.layer_ids[layer]);
    const __uint128_t initial =
        (static_cast<__uint128_t>(state_high) << kPcgWordBits) | state_low;
    // PCG's stream increment must be odd: (sequence << 1) | 1.
    const __uint128_t increment =
        (((static_cast<__uint128_t>(sequence_high) << kPcgWordBits) |
          sequence_low)
         << 1) |
        1;
    __uint128_t state = (increment + initial) * kPcgMultiplier + increment;
    for (int64_t shift = 0; shift < shifts; ++shift) {
      __uint128_t product;
      do {
        state = state * kPcgMultiplier + increment;
        const uint64_t value = static_cast<uint64_t>(state >> kPcgWordBits) ^
                               static_cast<uint64_t>(state);
        const int32_t rotation =
            static_cast<int32_t>(state >> kPcgRotationShift);
        const uint64_t random = std::rotr(value, rotation);
        product = static_cast<__uint128_t>(random) * bound;
      } while (static_cast<uint64_t>(product) < rejection_threshold);
      result.data_ptr<int64_t>()[layer * shifts + shift] = static_cast<int64_t>(
          (product >> kPcgWordBits) * kOddMultiplierScale + 1);
    }
  }
  return result;
}

bool is_prime(int64_t value) {
  if (value < 2) {
    return false;
  }
  if (value % 2 == 0) {
    return value == 2;
  }
  for (int64_t divisor = 3; divisor <= value / divisor; divisor += 2) {
    if (value % divisor == 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

DeepseekV41HashParameters build_deepseek_v41_hash_parameters(
    const DeepseekV41HashOptions& options) {
  CHECK_GE(options.max_ngram_size, 2);
  CHECK_GT(options.n_heads, 0);
  CHECK_GT(options.bucket_vocab_size, 1);
  CHECK_GT(options.compressed_vocab_size, 0);
  CHECK_EQ(options.layer_ids.size(), options.num_embeddings.size());
  for (int32_t id : options.layer_ids) {
    CHECK_GE(id, 0);
  }
  const int64_t layers = static_cast<int64_t>(options.layer_ids.size());
  const int64_t cols = (options.max_ngram_size - 1) * options.n_heads;
  auto primes = torch::empty({layers, cols}, torch::kInt64);
  auto offsets = torch::empty_like(primes);
  int64_t prime = options.bucket_vocab_size - 1;
  for (int64_t layer = 0; layer < layers; ++layer) {
    int64_t offset = 0;
    for (int64_t col = 0; col < cols; ++col) {
      do {
        CHECK_LT(prime, options.num_embeddings[layer] - offset)
            << "Engram prime buckets exceed embedding rows";
        ++prime;
      } while (!is_prime(prime));
      primes.data_ptr<int64_t>()[layer * cols + col] = prime;
      offsets.data_ptr<int64_t>()[layer * cols + col] = offset;
      offset += prime;
    }
  }
  return {hash_multipliers(options), primes, offsets};
}

}  // namespace xllm::layer
