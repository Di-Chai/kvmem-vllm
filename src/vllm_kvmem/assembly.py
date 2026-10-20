"""Fused immutable-K RoPE/FP8 assembly into engine-owned, strided page views.

All tokens across requests can share a launch. The scheduler validates page
leases and CPU position bounds before dispatch. Negative slots are padding.
V is already quantized with this layer's unchanged V scale; copy its bytes once.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _assemble(raw, values, positions, slots, trig, kcache, vcache, kscale,
              r0: tl.constexpr, r1: tl.constexpr,
              v0: tl.constexpr, v1: tl.constexpr,
              k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr,
              c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
              trig_stride: tl.constexpr, PAGE: tl.constexpr,
              DIM: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
              COPY_V: tl.constexpr, RAW_TYPE: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + token)
    if slot >= 0:
        position = tl.load(positions + token).to(tl.int64)
        d = tl.arange(0, BLOCK)
        mask = d < DIM
        x = tl.load(raw + token*r0 + head*r1 + d, mask=mask, other=0.0).to(tl.float32)
        other_d = tl.where(d < HALF, d + HALF, d - HALF)
        mate = tl.load(raw + token*r0 + head*r1 + other_d,
                       mask=d < 2*HALF, other=0.0).to(tl.float32)
        freq = d % HALF
        cos = tl.load(trig + position*trig_stride + freq).to(tl.float32)
        sin = tl.load(trig + position*trig_stride + HALF + freq).to(tl.float32)
        rotated = tl.where(d < HALF, x*cos - mate*sin, x*cos + mate*sin)
        rotated = tl.where(d < 2*HALF, rotated, x).to(RAW_TYPE).to(tl.float32)
        if RAW_TYPE == tl.float32:
            scaled = tl.div_rn(rotated, tl.load(kscale))
        else:
            scaled = rotated / tl.load(kscale)
        scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        page, offset = slot // PAGE, slot % PAGE
        tl.store(kcache + page*k0 + offset*k1 + head*k2 + d, scaled, mask=mask)
        if COPY_V:
            source_bytes = values.to(tl.pointer_type(tl.uint8))
            destination_bytes = vcache.to(tl.pointer_type(tl.uint8))
            v = tl.load(source_bytes + token*v0 + head*v1 + d, mask=mask, other=0)
            tl.store(destination_bytes + page*c0 + offset*c1 + head*c2 + d, v, mask=mask)


def assemble_fp8(raw_k, values, positions, slots, cos_sin_cache, key_cache, value_cache, k_scale):
    """Caches [pages,tokens,heads,dim], arbitrary outer strides, contiguous dim.

    `values=None` retains resident V and only rebuilds moved K. This is essential
    for incremental assembly. Staged V must not alias the destination cache.
    """
    if raw_k.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("Expected NHD raw K and PTHD page views")
    tokens, heads, dim = raw_k.shape
    if raw_k.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("Unsupported immutable raw K dtype")
    if key_cache.dtype != torch.float8_e4m3fn or value_cache.dtype != key_cache.dtype:
        raise ValueError("This assembly kernel is qualified for FP8 e4m3 only")
    if tuple(key_cache.shape[2:]) != (heads, dim):
        raise ValueError("Cache head geometry mismatch")
    if positions.shape != (tokens,) or slots.shape != (tokens,):
        raise ValueError("Position/slot count must match raw K")
    if positions.dtype not in (torch.int32, torch.int64) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("Positions and slots must be integer tensors")
    if cos_sin_cache.ndim != 2 or not 0 < cos_sin_cache.shape[1] <= dim or cos_sin_cache.shape[1] % 2:
        raise ValueError("Expected model NeoX cos/sin cache with even rotary dimension")
    tensors = [raw_k, positions, slots, cos_sin_cache, key_cache, value_cache, k_scale]
    if values is not None:
        if values.shape != raw_k.shape or values.dtype != value_cache.dtype:
            raise ValueError("Staged V shape/dtype mismatch")
        if values.untyped_storage().data_ptr() == value_cache.untyped_storage().data_ptr():
            raise ValueError("V source must be staged or omitted; in-place permutation races")
        tensors.append(values)
    if raw_k.device.type != "cuda" or any(t.device != raw_k.device for t in tensors):
        raise ValueError("Assembly requires all tensors on one CUDA device")
    if any(t.ndim and t.stride(-1) != 1 for t in tensors) or k_scale.numel() != 1:
        raise ValueError("Require contiguous inner dimensions and a validated per-tensor K scale")
    if not tokens:
        return
    _assemble[(tokens, heads)](
        raw_k, values if values is not None else value_cache, positions, slots,
        cos_sin_cache, key_cache, value_cache, k_scale,
        *raw_k.stride()[:2], *(values.stride()[:2] if values is not None else (0, 0)),
        *key_cache.stride()[:3], *value_cache.stride()[:3], cos_sin_cache.stride(0),
        key_cache.shape[1], dim, cos_sin_cache.shape[1] // 2,
        triton.next_power_of_2(dim), values is not None,
        (tl.float32 if raw_k.dtype == torch.float8_e4m3fn else
         tl.bfloat16 if raw_k.dtype == torch.bfloat16 else tl.float16),
        num_warps=4, enable_fp_fusion=raw_k.dtype != torch.float8_e4m3fn,
    )
