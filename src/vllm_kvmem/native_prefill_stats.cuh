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
