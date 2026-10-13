// block_attn_score_softmax_pages_kernel: one CUDA block per (layer,token). For
// each query head it computes logit[w] = scale·(q[l,t,h]·k̄[l,w,h/group]) for every
// page w, softmaxes those logits OVER PAGES, then atomicAdds (1/(L·H))·softmax_w
// into the global score[w]. Summed across the (L·n_tokens) grid this gives
//   score[w] = Σ_t mean_l mean_h softmax_w(scale·(q·k̄)),
// the accumulated attention mass each page receives. score must be pre-zeroed.
// Page logits live in dynamic shared memory s_logit[n_blocks]; the per-head
// max/sum reductions are serialized on thread 0 (n_blocks is small, ~hundreds).
// reduce_max (default 0): with n_subblocks>1, score each block by its best
// sub-block's mass (max over sub-blocks) instead of the sum, so score[w] =
// Σ_t mean_l mean_h max_sb softmax(scale·(q·k̄_{w,sb})) — MaxSim late interaction.
// reduce_max is a no-op at n_subblocks==1 (single sub-block => max == sum).
template <typename QueryT, typename KbarT>
__global__ void block_attn_score_softmax_pages_kernel(
                                                      float *score,
                                                      const QueryT *q_multi,
                                                      const KbarT *kbar_multi,
                                                      const int32_t *group_begin,
                                                      const int32_t *group_end,
                                                      uint32_t n_groups,
                                                      uint32_t n_layers,
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
                                                      uint32_t n_subblocks,
                                                      uint32_t reduce_max) {
    const uint32_t lt = blockIdx.x;
    const uint32_t l = lt / n_tokens;
    const uint32_t t = lt % n_tokens;
    if (l >= n_layers) return;
    extern __shared__ float s_logit[];   // [n_blocks * n_subblocks]
    __shared__ float s_max;
    __shared__ float s_sum;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t group = n_heads / n_kv_heads;
    const uint64_t qrow = static_cast<uint64_t>(n_heads) * head_dim;
    // Softmax runs at SUB-BLOCK granularity: total = n_blocks*n_subblocks logits.
    // logit index p maps to block w = p/n_subblocks, sub-block sb = p%n_subblocks.
    // A block's sub-block masses are summed back into score[w] at the end, so a
    // concentrated relevant sub-region keeps its mass instead of being diluted by
    // the block mean. n_subblocks==1 (plain mean-k) => total==n_blocks, p==w,
    // byte-identical to the pre-sub-block path.
    const uint32_t total = n_blocks * n_subblocks;
    // Per-layer base uses kbar_layer_stride (in blocks) × n_subblocks. The stride
    // may exceed n_blocks when g_kbar_multi_ is allocated at a fixed session stride
    // (ctx_blocks) so preserved [0,D) slices stay valid as the block count grows.
    // n_blocks remains the loop/count bound. Callers that don't fix-stride pass
    // kbar_layer_stride == n_blocks (byte-identical).
    const KbarT *kbar_l =
        kbar_multi +
        static_cast<uint64_t>(l) * kbar_layer_stride * n_subblocks * n_kv_heads *
            head_dim;
    const QueryT *q_l =
        q_multi + static_cast<uint64_t>(l) * q_layer_stride * qrow;
    // Per-head weight: each (layer,token) contributes mean over heads, and the
    // grid sum over layers must end up as a mean over L. n_tokens is summed (not
    // averaged) so the final score accumulates one unit of mass per query token.
    const float head_w = 1.0f / (static_cast<float>(n_layers) *
                                 static_cast<float>(n_heads));
    for (uint32_t qh = 0; qh < n_heads; ++qh) {
        const uint32_t kvh = qh / group;
        const QueryT *qr =
            q_l + (static_cast<uint64_t>(t) * n_heads + qh) * head_dim;
        // logit[p] = scale * dot(q[l,t,qh], k̄[l,w,sb,kvh]) for every sub-block.
        // Sub-blocks of a block in the always-kept sink/recent bands
        // [0,excl_lo_end) / [excl_hi_begin,n_blocks) are masked to -inf so their
        // softmax mass redistributes onto the retrievable middle (they are pinned).
        for (uint32_t p = tid; p < total; p += nthreads) {
            const uint32_t w = p / n_subblocks;
            if (w < excl_lo_end || w >= excl_hi_begin) {
                s_logit[p] = -INFINITY;
                continue;
            }
            const KbarT *kbar_row =
                kbar_l + (static_cast<uint64_t>(p) * n_kv_heads + kvh) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += static_cast<float>(qr[d]) *
                       static_cast<float>(kbar_row[d]);
            }
            s_logit[p] = dot * scale;
        }
        __syncthreads();
        if (tid == 0) {
            float m = -INFINITY;
            for (uint32_t p = 0; p < total; ++p) m = fmaxf(m, s_logit[p]);
            s_max = m;
        }
        __syncthreads();
        const float m = s_max;
        for (uint32_t p = tid; p < total; p += nthreads) {
            s_logit[p] = __expf(s_logit[p] - m);
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (uint32_t p = 0; p < total; ++p) sum += s_logit[p];
            s_sum = sum > 0.0f ? sum : 1.0f;
        }
        __syncthreads();
        const float inv = head_w / s_sum;
        if (n_groups > 0 && group_begin && group_end) {
            // Semantic retrieval: reduce the globally normalized attention
            // probabilities over each variable-length group. reduce_max keeps
            // the existing MaxSim behavior; zero produces true group mass.
            for (uint32_t g = tid; g < n_groups; g += nthreads) {
                const uint32_t p0 = static_cast<uint32_t>(
                    max(group_begin[g], 0));
                const uint32_t p1 = min(
                    static_cast<uint32_t>(max(group_end[g], 0)), total);
                float mass = 0.0f;
                for (uint32_t p = p0; p < p1; ++p) {
                    mass = reduce_max
                        ? fmaxf(mass, s_logit[p])
                        : mass + s_logit[p];
                }
                atomicAdd(&score[g], mass * inv);
            }
        } else if (reduce_max && n_subblocks > 1) {
            // MaxSim-style reduction: score each block by its single best
            // sub-block's softmax mass (max over the document side), still
            // summed across query tokens / heads / layers by the grid atomicAdd.
            // Counters the block mean's dilution of a concentrated sub-region.
            // Masked bands have s_logit==0 (post-exp of -inf) so a fully-masked
            // block contributes 0, matching the sum path. n_subblocks==1 falls
            // through to the sum branch (byte-identical, single sub-block/block).
            for (uint32_t w = tid; w < n_blocks; w += nthreads) {
                float mx = 0.0f;
                for (uint32_t sb = 0; sb < n_subblocks; ++sb)
                    mx = fmaxf(mx, s_logit[w * n_subblocks + sb]);
                atomicAdd(&score[w], mx * inv);
            }
        } else {
            for (uint32_t p = tid; p < total; p += nthreads) {
                atomicAdd(&score[p / n_subblocks], s_logit[p] * inv);
            }
        }
        __syncthreads();  // s_logit reused by the next head
    }
}
