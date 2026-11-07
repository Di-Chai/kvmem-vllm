// SPDX-License-Identifier: Apache-2.0
#include "native_stats.cuh"
#include <flashinfer/attention/variants.cuh>

struct KVMemAttention : flashinfer::DefaultAttention<false, false, false, false> {
  using Base = flashinfer::DefaultAttention<false, false, false, false>;
  template <typename Params>
  __host__ __device__ KVMemAttention(Params const& p, uint32_t b, uint8_t* s) : Base(p, b, s) {}

  template <typename Traits, typename Params, typename Acc>
  __device__ void CaptureSoftmax(Params const& p, uint32_t batch, uint32_t packed,
                          uint32_t kv_base, uint32_t qo_len, uint32_t kv_len,
                          uint32_t chunk_end, flashinfer::uint_fastdiv group,
                          Acc (*frag)[Traits::NUM_MMA_KV][8], Acc (*maxima)[2], dim3 tid,
                          uint32_t kv_head) const {
    static_assert(!Traits::USE_SOFTMAX_VO_SPLIT, "Stats currently qualify head dimensions <= 256");
    KVMemStats stats{p.stat_rows + p.stat_offset, p.stat_requests, p.stat_pieces,
                    p.stat_lse, p.stat_context, p.stat_width, int(p.num_qo_heads)};
    bool merge = p.partition_kv && (p.kv_chunk_size_ptr[0] % 16 != 0);
    #pragma unroll
    for (uint32_t m = 0; m < Traits::NUM_MMA_Q; ++m) {
      #pragma unroll
      for (uint32_t r = 0; r < 2; ++r) {
        uint32_t q, h;
        group.divmod(packed + m * 16 + tid.x / 4 + 8 * r, q, h);
        if (q >= qo_len) continue;  // Uniform within each four-lane group.
        int sample = stats.rows[p.q_indptr[batch] + q];
        if (sample < 0) continue;
        #pragma unroll
        for (uint32_t n = 0; n < Traits::NUM_MMA_KV; ++n) {
          uint32_t reg = 2 * r;
          stats.store16_weight(float(frag[m][n][reg]), float(frag[m][n][reg+1]),
              float(frag[m][n][reg+4]), float(frag[m][n][reg+5]),
              float(maxima[m][r]) * p.sm_scale, kv_base + n * 16,
              min(chunk_end, kv_len - qo_len + q + 1), sample,
              kv_head * uint32_t(group) + h, merge);
        }
      }
    }
  }
};

// Separate generated ABI only when the service option is enabled. The runtime
// flag changes across graph replays; this class/module choice does not.
struct KVMemAlignedAttention : flashinfer::DefaultAttention<false, false, false, false> {
  using Base = flashinfer::DefaultAttention<false, false, false, false>;
  template <typename Params>
  __host__ __device__ KVMemAlignedAttention(Params const& p, uint32_t b, uint8_t* s) : Base(p, b, s) {}

  template <typename Traits, typename Params, typename Acc>
  __device__ void CaptureSoftmax(Params const& p, uint32_t batch, uint32_t packed,
                          uint32_t kv_base, uint32_t qo_len, uint32_t kv_len,
                          uint32_t chunk_end, flashinfer::uint_fastdiv group,
                          Acc (*frag)[Traits::NUM_MMA_KV][8], Acc (*maxima)[2], dim3 tid,
                          uint32_t kv_head) const {
    static_assert(!Traits::USE_SOFTMAX_VO_SPLIT, "Stats currently qualify head dimensions <= 256");
    KVMemStats stats{p.stat_rows + p.stat_offset, p.stat_requests, p.stat_pieces,
                    p.stat_lse, p.stat_context, p.stat_width, int(p.num_qo_heads)};
    bool merge = p.partition_kv && (p.kv_chunk_size_ptr[0] % 16 != 0);
    #pragma unroll
    for (uint32_t m = 0; m < Traits::NUM_MMA_Q; ++m) {
      #pragma unroll
      for (uint32_t r = 0; r < 2; ++r) {
        uint32_t q, h;
        group.divmod(packed + m * 16 + tid.x / 4 + 8 * r, q, h);
        if (q >= qo_len) continue;  // Uniform within each four-lane group.
        int sample = stats.rows[p.q_indptr[batch] + q];
        if (sample < 0) continue;
        // Sample requests carry the runner's global owner index. FA2 batch
        // indices alone are insufficient when a decode prefix precedes prefill.
        bool aligned = p.stat_aligned[stats.requests[sample]] != 0;
        #pragma unroll
        for (uint32_t n = 0; n < Traits::NUM_MMA_KV; ++n) {
          uint32_t reg = 2 * r;
          stats.store16_weight(float(frag[m][n][reg]), float(frag[m][n][reg+1]),
              float(frag[m][n][reg+4]), float(frag[m][n][reg+5]),
              float(maxima[m][r]) * p.sm_scale, kv_base + n * 16,
              min(chunk_end, kv_len - qo_len + q + 1), sample,
              kv_head * uint32_t(group) + h, merge, aligned);
        }
      }
    }
  }
};

// This class is never selected while capturing or by a graph-planned wrapper.
// The host has proved a complete dense-16 piece domain and page-aligned KV
// partition starts. No runtime general-path branch is instantiated here.
struct KVMemStaticAlignedAttention : flashinfer::DefaultAttention<false, false, false, false> {
  using Base = flashinfer::DefaultAttention<false, false, false, false>;
  template <typename Params>
  __host__ __device__ KVMemStaticAlignedAttention(Params const& p, uint32_t b, uint8_t* s) : Base(p, b, s) {}

  template <typename Traits, typename Params, typename Acc>
  __device__ void CaptureSoftmax(Params const& p, uint32_t batch, uint32_t packed,
                          uint32_t kv_base, uint32_t qo_len, uint32_t kv_len,
                          uint32_t chunk_end, flashinfer::uint_fastdiv group,
                          Acc (*frag)[Traits::NUM_MMA_KV][8], Acc (*maxima)[2], dim3 tid,
                          uint32_t kv_head) const {
    static_assert(!Traits::USE_SOFTMAX_VO_SPLIT, "Stats currently qualify head dimensions <= 256");
    KVMemStats stats{p.stat_rows + p.stat_offset, p.stat_requests, p.stat_pieces,
                    p.stat_lse, p.stat_context, p.stat_width, int(p.num_qo_heads)};
    #pragma unroll
    for (uint32_t m = 0; m < Traits::NUM_MMA_Q; ++m) {
      #pragma unroll
      for (uint32_t r = 0; r < 2; ++r) {
        uint32_t q, h;
        group.divmod(packed + m * 16 + tid.x / 4 + 8 * r, q, h);
        if (q >= qo_len) continue;
        int sample = stats.rows[p.q_indptr[batch] + q];
        if (sample < 0) continue;
        #pragma unroll
        for (uint32_t n = 0; n < Traits::NUM_MMA_KV; ++n) {
          uint32_t reg = 2 * r;
          stats.store16_weight_static_aligned(float(frag[m][n][reg]), float(frag[m][n][reg+1]),
              float(frag[m][n][reg+4]), float(frag[m][n][reg+5]),
              float(maxima[m][r]) * p.sm_scale, kv_base + n * 16,
              min(chunk_end, kv_len - qo_len + q + 1), sample,
              kv_head * uint32_t(group) + h);
        }
      }
    }
  }
};

// Same host qualification and statistics as KVMemStaticAlignedAttention.
// Groups of four MMA-KV fragments use one quad reduce-scatter per Q row.
struct KVMemStaticQuadAttention : flashinfer::DefaultAttention<false, false, false, false> {
  using Base = flashinfer::DefaultAttention<false, false, false, false>;
  template <typename Params>
  __host__ __device__ KVMemStaticQuadAttention(Params const& p, uint32_t b, uint8_t* s) : Base(p, b, s) {}

  template <typename Traits, typename Params, typename Acc>
  __device__ void CaptureSoftmax(Params const& p, uint32_t batch, uint32_t packed,
                          uint32_t kv_base, uint32_t qo_len, uint32_t kv_len,
                          uint32_t chunk_end, flashinfer::uint_fastdiv group,
                          Acc (*frag)[Traits::NUM_MMA_KV][8], Acc (*maxima)[2], dim3 tid,
                          uint32_t kv_head) const {
    static_assert(!Traits::USE_SOFTMAX_VO_SPLIT, "Stats currently qualify head dimensions <= 256");
    KVMemStats stats{p.stat_rows + p.stat_offset, p.stat_requests, p.stat_pieces,
                    p.stat_lse, p.stat_context, p.stat_width, int(p.num_qo_heads)};
    #pragma unroll
    for (uint32_t m = 0; m < Traits::NUM_MMA_Q; ++m) {
      // Each lane owns Q rows r=0 and r=1 (packed +0 and +8). Different quads
      // of one warp usually sample different r; one warp-uniform pass over a
      // per-quad selected row replaces two divergent passes. A quad sampling
      // both rows takes the second pass for r=1. Statistics are row-local.
      uint32_t q[2], h[2];
      int sample[2];
      #pragma unroll
      for (uint32_t r = 0; r < 2; ++r) {
        group.divmod(packed + m * 16 + tid.x / 4 + 8 * r, q[r], h[r]);
        sample[r] = q[r] < qo_len ? stats.rows[p.q_indptr[batch] + q[r]] : -1;
      }
      const bool want0 = sample[0] >= 0, want1 = sample[1] >= 0;
      const unsigned int warp = __activemask();
      #pragma unroll
      for (uint32_t pass = 0; pass < 2; ++pass) {
        const bool act = pass == 0 ? (want0 || want1) : (want0 && want1);
        if (!__any_sync(warp, act)) break;
        if (!act) continue;
        const bool one = pass == 1 || !want0;
        const uint32_t row_q = one ? q[1] : q[0];
        const float center = float(one ? maxima[m][1] : maxima[m][0]) * p.sm_scale;
        const int limit = min(chunk_end, kv_len - qo_len + row_q + 1);
        const int row_sample = one ? sample[1] : sample[0];
        const int head = kv_head * uint32_t(group) + (one ? h[1] : h[0]);
        if constexpr (Traits::NUM_MMA_KV % 4 == 0) {
          #pragma unroll
          for (uint32_t n0 = 0; n0 < Traits::NUM_MMA_KV; n0 += 4) {
            float v[4][4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              v[i][0] = float(one ? frag[m][n0 + i][2] : frag[m][n0 + i][0]);
              v[i][1] = float(one ? frag[m][n0 + i][3] : frag[m][n0 + i][1]);
              v[i][2] = float(one ? frag[m][n0 + i][6] : frag[m][n0 + i][4]);
              v[i][3] = float(one ? frag[m][n0 + i][7] : frag[m][n0 + i][5]);
            }
            stats.store64_weight_static_aligned(v, center, kv_base + n0 * 16, limit,
                                                row_sample, head);
          }
        } else {
          #pragma unroll
          for (uint32_t n = 0; n < Traits::NUM_MMA_KV; ++n) {
            stats.store16_weight_static_aligned(
                float(one ? frag[m][n][2] : frag[m][n][0]), float(one ? frag[m][n][3] : frag[m][n][1]),
                float(one ? frag[m][n][6] : frag[m][n][4]), float(one ? frag[m][n][7] : frag[m][n][5]),
                center, kv_base + n * 16, limit, row_sample, head);
          }
        }
      }
    }
  }
};
