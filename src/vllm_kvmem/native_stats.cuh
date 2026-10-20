// SPDX-License-Identifier: Apache-2.0
// KVMem's optional side output. No Q/K loads, dot products or attention-output changes.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

struct KVMemStats {
  int32_t const* rows;
  int32_t const* requests;
  int32_t const* pieces;
  float* lse;
  int32_t context, width, heads;

  __device__ void add(float* dst, float value) const {
    if (!isfinite(value)) return;
    auto ptr = reinterpret_cast<unsigned int*>(dst);
    unsigned int old = atomicCAS(ptr, 0U, 0U), assumed;
    do {
      assumed = old;
      float previous = __uint_as_float(assumed);
      float maximum = fmaxf(previous, value);
      float combined = maximum + log1pf(expf(fminf(previous, value) - maximum));
      old = atomicCAS(ptr, assumed, __float_as_uint(combined));
    } while (old != assumed);
  }

  // Four adjacent lanes own sixteen softmax weights from native MMA fragments.
  // Pieces split at BOTH logical-segment and sixteen-token boundaries. A partial
  // segment or split-K edge may produce several contributors: CAS logaddexp
  // merges them without an ordering assumption or an unstable exp(logit).
  __device__ void write(int sample, int head, int piece, float value, bool merge) const {
    size_t offset = (size_t(sample) * heads + head) * width + piece;
    if (merge) add(lse + offset, value);
    else lse[offset] = value;
  }

  __device__ void store16_weight(float a, float b, float c, float d, float center,
                                 int base, int limit, int sample, int head, bool merge) const {
    if (base >= limit) return;
    int request = requests[sample];
    int lane = threadIdx.x % 32;
    int token = base + 2 * (lane % 4);
    a = token < limit ? a : 0.f;
    b = token + 1 < limit ? b : 0.f;
    c = token + 8 < limit ? c : 0.f;
    d = token + 9 < limit ? d : 0.f;
    int first = pieces[request * context + base];
    int last = pieces[request * context + min(base + 15, limit - 1)];
    // Different quads can have different causal tails/piece counts. Only the
    // four lanes exchanging data may participate in each shuffle's barrier.
    unsigned int active = __activemask() & (0xfU << (lane & ~3));
    if (first == last) {
      // Sixteen native FP32 softmax weights, one log and one store per quad.
      float sum = a + b + c + d;
      sum += __shfl_xor_sync(active, sum, 1);
      sum += __shfl_xor_sync(active, sum, 2);
      if ((lane & 3) == 0 && sum > 0.f)
        write(sample, head, first, center + logf(sum), merge);
      return;
    }
    int ids[4] = {
      token < limit ? pieces[request * context + token] : -1,
      token + 1 < limit ? pieces[request * context + token + 1] : -1,
      token + 8 < limit ? pieces[request * context + token + 8] : -1,
      token + 9 < limit ? pieces[request * context + token + 9] : -1};
    // Piece IDs are monotone and contiguous across this dense fragment. This
    // also handles several tiny logical segments within one MMA fragment.
    for (int piece = first; piece <= last; ++piece) {
      float sum = (ids[0] == piece ? a : 0.f) + (ids[1] == piece ? b : 0.f)
                + (ids[2] == piece ? c : 0.f) + (ids[3] == piece ? d : 0.f);
      sum += __shfl_xor_sync(active, sum, 1);
      sum += __shfl_xor_sync(active, sum, 2);
      if ((lane & 3) == 0 && sum > 0.f)
        write(sample, head, piece, center + logf(sum), merge);
    }
  }
};
