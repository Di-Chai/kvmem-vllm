// Unmodified qw3 scalable page-softmax kernels; see provenance and LICENSE.
constexpr uint32_t kSoftmaxPagesTile = 512;
constexpr uint32_t kSoftmaxPagesThreads = 128;

template <typename QueryT, typename KbarT>
__global__ void block_attn_softmax_pages_logits_kernel(
        float *logits,
        const QueryT *q_multi,
        const KbarT *kbar_multi,
        uint32_t distribution_base,
        uint32_t distribution_count,
        uint32_t n_tiles,
        uint32_t n_tokens,
        uint32_t q_layer_stride,
        uint32_t n_blocks,
        uint32_t kbar_layer_stride,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        float scale,
        uint32_t excl_lo_end,
        uint32_t excl_hi_begin,
        uint32_t n_subblocks) {
    const uint64_t job = blockIdx.x;
    const uint32_t local_dist =
        static_cast<uint32_t>(job / n_tiles);
    const uint32_t tile = static_cast<uint32_t>(
        job - static_cast<uint64_t>(local_dist) * n_tiles);
    if (local_dist >= distribution_count) return;

    const uint32_t dist = distribution_base + local_dist;
    const uint32_t qh = dist % n_heads;
    const uint32_t lt = dist / n_heads;
    const uint32_t t = lt % n_tokens;
    const uint32_t l = lt / n_tokens;
    const uint32_t group = n_heads / n_kv_heads;
    const uint32_t kvh = qh / group;
    const uint32_t total = n_blocks * n_subblocks;
    const uint32_t p0 = tile * kSoftmaxPagesTile;
    const uint32_t p1 = min(p0 + kSoftmaxPagesTile, total);
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    constexpr uint32_t kWarps =
        kSoftmaxPagesThreads / 32;

    const uint64_t qrow =
        static_cast<uint64_t>(n_heads) * head_dim;
    const QueryT *q_l =
        q_multi +
        static_cast<uint64_t>(l) * q_layer_stride * qrow;
    const QueryT *qr =
        q_l +
        (static_cast<uint64_t>(t) * n_heads + qh) * head_dim;
    const KbarT *kbar_l =
        kbar_multi +
        static_cast<uint64_t>(l) * kbar_layer_stride *
            n_subblocks * n_kv_heads * head_dim;
    float *out =
        logits + static_cast<uint64_t>(local_dist) * total;

    for (uint32_t p = p0 + warp; p < p1; p += kWarps) {
        const uint32_t w = p / n_subblocks;
        float dot = 0.0f;
        if (w >= excl_lo_end && w < excl_hi_begin) {
            const KbarT *kbar_row =
                kbar_l +
                (static_cast<uint64_t>(p) * n_kv_heads + kvh) *
                    head_dim;
            for (uint32_t d = lane; d < head_dim; d += 32) {
                dot += static_cast<float>(qr[d]) *
                       static_cast<float>(kbar_row[d]);
            }
#pragma unroll
            for (uint32_t offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
        }
        if (lane == 0) {
            out[p] =
                (w >= excl_lo_end && w < excl_hi_begin)
                    ? dot * scale : -INFINITY;
        }
    }
}

__global__ void block_attn_softmax_pages_norm_kernel(
        float *row_max,
        float *row_inv_sum,
        const float *logits,
        uint32_t distribution_count,
        uint32_t total) {
    const uint32_t dist = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (dist >= distribution_count) return;
    __shared__ float reduce[kSoftmaxPagesThreads];
    const float *row =
        logits + static_cast<uint64_t>(dist) * total;

    float local_max = -INFINITY;
    for (uint32_t p = tid; p < total; p += blockDim.x) {
        local_max = fmaxf(local_max, row[p]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            reduce[tid] =
                fmaxf(reduce[tid], reduce[tid + s]);
        }
        __syncthreads();
    }
    const float m = reduce[0];
    float local_sum = 0.0f;
    if (isfinite(m)) {
        for (uint32_t p = tid; p < total; p += blockDim.x) {
            local_sum += __expf(row[p] - m);
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        row_max[dist] = m;
        row_inv_sum[dist] =
            reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    }
}

__global__ void block_attn_softmax_pages_accumulate_logits_kernel(
        float *score,
        const float *logits,
        const float *row_max,
        const float *row_inv_sum,
        uint32_t distribution_count,
        uint32_t n_blocks,
        uint32_t n_subblocks,
        uint32_t reduce_max,
        float distribution_weight,
        uint32_t excl_lo_end,
        uint32_t excl_hi_begin) {
    const uint32_t w =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_blocks || w < excl_lo_end ||
        w >= excl_hi_begin) {
        return;
    }
    const uint32_t total = n_blocks * n_subblocks;
    float accum = 0.0f;
    for (uint32_t dist = 0; dist < distribution_count; ++dist) {
        const float inv = row_inv_sum[dist];
        if (!(inv > 0.0f)) continue;
        const float m = row_max[dist];
        const float *row =
            logits + static_cast<uint64_t>(dist) * total +
            static_cast<uint64_t>(w) * n_subblocks;
        float mass = 0.0f;
        if (reduce_max && n_subblocks > 1) {
            for (uint32_t sb = 0; sb < n_subblocks; ++sb) {
                mass = fmaxf(
                    mass, __expf(row[sb] - m) * inv);
            }
        } else {
            for (uint32_t sb = 0; sb < n_subblocks; ++sb) {
                mass += __expf(row[sb] - m) * inv;
            }
        }
        accum += mass;
    }
    score[w] += accum * distribution_weight;
}
