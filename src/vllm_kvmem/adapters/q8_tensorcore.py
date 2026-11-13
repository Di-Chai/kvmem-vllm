"""Experimental Q8_0 x Q8_1 grouped integer Tensor Core linear.

Not registered in serving. Keeps GGUF bytes and FP16 group scales unchanged;
activation quantization follows the pinned GGUF CUDA roundf/FP16-scale path.
Each group has an exact INT32 dot, scaled in FP32. Reduction ordering differs
from the upstream DP4A kernel, so model qualification is still required.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, D, K: tl.constexpr, GROUPS: tl.constexpr):
    group = tl.program_id(0)
    row = tl.program_id(1)
    lane = tl.arange(0, 32)
    x = tl.load(X + row * K + group * 32 + lane).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    # The pinned CUDA extension is built with --use_fast_math: division by
    # constant 127 becomes multiplication; variable division uses div.approx.
    d = maximum * (1.0 / 127.0)
    ratio = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[x, tl.where(maximum == 0, 1.0, d)], dtype=tl.float32,
        is_pure=True, pack=1)
    # CUDA roundf is ties-away-from-zero, unlike tl.extra.cuda.libdevice.rint.
    q = tl.floor(tl.abs(ratio) + 0.5) * tl.where(ratio < 0, -1.0, 1.0)
    tl.store(Q + row * K + group * 32 + lane, q.to(tl.int8))
    tl.store(D + row * GROUPS + group, d.to(tl.float16))


@triton.jit(do_not_specialize=['M'])
def _matmul(Q, D, W, P, M, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BG: tl.constexpr,
            SPLIT: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    groups: tl.constexpr = K // 32
    lanes = tl.arange(0, 32)
    acc = tl.full((BG, BM, BN), 0, tl.float32)
    for base in range(split * BG, groups, BG * SPLIT):
        gi = base + tl.arange(0, BG)
        aq = tl.load(Q + mi[None, :, None] * K + gi[:, None, None] * 32
                     + lanes[None, None, :],
                     (mi[None, :, None] < M) & (gi[:, None, None] < groups), 0)
        wp = W + ni[None, None, :] * (groups * 34) + gi[:, None, None] * 34
        bq = tl.load(wp + 2 + lanes[None, :, None],
                     (ni[None, None, :] < N) & (gi[:, None, None] < groups), 0)
        dots = tl.dot(aq, bq.to(tl.int8, bitcast=True)).to(tl.float32)
        ad = tl.load(D + mi[None, :] * groups + gi[:, None],
                     (mi[None, :] < M) & (gi[:, None] < groups), 0).to(tl.float32)
        sp = W + ni[None, :] * (groups * 34) + gi[:, None] * 34
        mask = (ni[None, :] < N) & (gi[:, None] < groups)
        lo = tl.load(sp, mask, 0).to(tl.uint16)
        hi = tl.load(sp + 1, mask, 0).to(tl.uint16)
        bd = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        acc += dots * (ad[:, :, None] * bd[:, None, :])
    total = tl.sum(acc, 0)
    tl.store(P + split * M * N + mi[:, None] * N + ni[None, :], total,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit(do_not_specialize=['SIZE'])
def _reduce(P, Y, SIZE, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT)
    values = tl.load(P + splits[:, None] * SIZE + offset[None, :],
                     offset[None, :] < SIZE, 0)
    tl.store(Y + offset, tl.sum(values, 0), offset < SIZE)


def quantize_activations(x):
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous() or x.shape[1] % 32:
        raise ValueError("Q8 activation input requires contiguous CUDA [M,K], K divisible by 32")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("unsupported activation dtype")
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    d = torch.empty((m, k // 32), dtype=torch.float16, device=x.device)
    with torch.cuda.device(x.device):
        _quantize[(k // 32, m)](x, q, d, k, k // 32, num_warps=1, enable_fp_fusion=False)
    return q, d


def q8_tensorcore_linear(x, weight, *, split_k=4, block_n=64, block_groups=4):
    """Diagnostic only; no weight conversion/repacking or serving mutation."""
    if (weight.ndim != 2 or weight.dtype != torch.uint8 or not weight.is_contiguous()
            or weight.device != x.device or x.ndim != 2
            or weight.shape[1] * 32 != x.shape[1] * 34):
        raise ValueError("expected GGUF Q8_0 byte matrix [N,K/32*34] on input device")
    if split_k not in (1, 2, 4, 8, 16) or block_n not in (32, 64, 128) or block_groups not in (1, 2, 4, 8):
        raise ValueError("unsupported diagnostic tile")
    q, d = quantize_activations(x)
    m, k = x.shape
    n = weight.shape[0]
    partial = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _matmul[(triton.cdiv(m, 16), triton.cdiv(n, block_n), split_k)](
            q, d, weight, partial, m, n, k, 16, block_n, block_groups, split_k,
            num_warps=4, enable_fp_fusion=False)
        _reduce[(triton.cdiv(m * n, 256),)](partial, y, m * n, split_k, 256)
    return y
