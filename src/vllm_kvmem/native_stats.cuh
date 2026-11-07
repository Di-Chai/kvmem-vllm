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
                                 int base, int limit, int sample, int head, bool merge,
                                 bool aligned = false) const {
    if (base >= limit) return;
    int lane = threadIdx.x % 32;
    int token = base + 2 * (lane % 4);
    a = token < limit ? a : 0.f;
    b = token + 1 < limit ? b : 0.f;
    c = token + 8 < limit ? c : 0.f;
    d = token + 9 < limit ? d : 0.f;
    // Only the opt-in FA2 variant supplies this per-request qualification.
    // Keep the exact causal masking/reduction order. A final partial tile is
    // valid; an unaligned split-K still needs the original map/CAS path.
    if (aligned && !merge && (base & 15) == 0) {
      unsigned int active = __activemask() & (0xfU << (lane & ~3));
      float sum = a + b + c + d;
      sum += __shfl_xor_sync(active, sum, 1);
      sum += __shfl_xor_sync(active, sum, 2);
      if ((lane & 3) == 0 && sum > 0.f)
        write(sample, head, base >> 4, center + logf(sum), false);
      return;
    }
    int request = requests[sample];
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

  // Host-qualified, non-graph FA2 only. The arithmetic below is deliberately
  // identical to the aligned branch above; this separate entry has no piece
  // lookup or CAS path for the compiler to retain in its instantiated kernel.
  __device__ void store16_weight_static_aligned(float a, float b, float c, float d,
                                 float center, int base, int limit, int sample,
                                 int head) const {
    if (base >= limit) return;
    int lane = threadIdx.x % 32;
    int token = base + 2 * (lane % 4);
    a = token < limit ? a : 0.f;
    b = token + 1 < limit ? b : 0.f;
    c = token + 8 < limit ? c : 0.f;
    d = token + 9 < limit ? d : 0.f;
    unsigned int active = __activemask() & (0xfU << (lane & ~3));
    float sum = a + b + c + d;
    sum += __shfl_xor_sync(active, sum, 1);
    sum += __shfl_xor_sync(active, sum, 2);
    if ((lane & 3) == 0 && sum > 0.f)
      write(sample, head, base >> 4, center + logf(sum), false);
  }

  // Four consecutive sixteen-token fragments (n = 0..3) for one Q row. Same
  // values as four store16_weight_static_aligned calls: each lane forms
  // a + b + c + d in the same order and every fragment is reduced as
  // (lane0 + lane1) + (lane2 + lane3), only commuted where IEEE addition is
  // exact. A reduce-scatter leaves fragment n on quad lane n, so the quad
  // issues three shuffles, one log and one contiguous store instead of eight
  // shuffles, four logs and four predicated stores.
  __device__ void store64_weight_static_aligned(float const (&v)[4][4], float center,
                                 int base, int limit, int sample, int head) const {
    if (base >= limit) return;
    int lane = threadIdx.x % 32, quad = lane & 3;
    float s[4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      int token = base + 16 * n + 2 * quad;
      float a = token < limit ? v[n][0] : 0.f;
      float b = token + 1 < limit ? v[n][1] : 0.f;
      float c = token + 8 < limit ? v[n][2] : 0.f;
      float d = token + 9 < limit ? v[n][3] : 0.f;
      s[n] = a + b + c + d;
    }
    unsigned int active = __activemask() & (0xfU << (lane & ~3));
    bool odd = quad & 1, high = quad & 2;
    // Pair lanes (0,1) and (2,3): keep fragments whose parity matches this lane.
    float r0 = __shfl_xor_sync(active, odd ? s[0] : s[1], 1);
    float r1 = __shfl_xor_sync(active, odd ? s[2] : s[3], 1);
    float low_pair = (odd ? s[1] : s[0]) + r0;
    float high_pair = (odd ? s[3] : s[2]) + r1;
    // Pair lanes (0,2) and (1,3): lane quad keeps fragment quad.
    float other = __shfl_xor_sync(active, high ? low_pair : high_pair, 2);
    float sum = (high ? high_pair : low_pair) + other;
    int token = base + 16 * quad;
    if (token < limit && sum > 0.f)
      write(sample, head, token >> 4, center + logf(sum), false);
  }
};
