#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "qw3/cuda/mean_k_reference.cuh"

// Test-only bridge; retrieval in the serving worker remains batched PyTorch.
extern "C" int meank_reference(float *out, const float *query, const float *means,
    uint32_t layers, uint32_t tokens, uint32_t blocks, uint32_t heads,
    uint32_t kv_heads, uint32_t dim, uint32_t sink, uint32_t recent,
    void *stream_ptr) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto error = cudaMemsetAsync(out, 0, blocks * sizeof(float), stream);
    if (error != cudaSuccess) return error;
    block_attn_score_softmax_pages_kernel<float, float>
        <<<layers * tokens, 128, blocks * sizeof(float), stream>>>(
            out, query, means, nullptr, nullptr, 0, layers, tokens, tokens,
            blocks, blocks, heads, kv_heads, dim, 1.0f / sqrtf(dim),
            sink, blocks - recent, 1, 0);
    return cudaGetLastError();
}
