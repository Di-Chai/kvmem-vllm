#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include "qw3_mean_k_kernel.cuh"

torch::Tensor score(torch::Tensor q, torch::Tensor k, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.device() == k.device(), "CUDA Q/index must share a device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf, "qw3 MeanK requires FP16 Q/index");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.is_contiguous() && k.is_contiguous(), "Expected contiguous LTHD/LBHD");
    const auto L=q.size(0), T=q.size(1), H=q.size(2), D=q.size(3), B=k.size(1), KH=k.size(2);
    TORCH_CHECK(L>0 && T>0 && H>0 && D>0 && B>0 && B<=8192 && KH>0 && H%KH==0 &&
        k.size(0)==L && k.size(3)==D && std::isfinite(scale) && scale>0, "Invalid bounded MeanK geometry/scale");
    c10::cuda::CUDAGuard guard(q.device());
    auto result=torch::zeros({B},q.options().dtype(at::kFloat));
    block_attn_score_softmax_pages_kernel<at::Half,at::Half><<<L*T,128,B*sizeof(float),at::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<float>(),q.data_ptr<at::Half>(),k.data_ptr<at::Half>(),nullptr,nullptr,0,
        L,T,T,B,B,H,KH,D,static_cast<float>(scale),0,B,1,0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("score", &score); }
