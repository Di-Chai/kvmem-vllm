"""Byte-preserving E4M3FN access on CUDA devices without native FP8 casts."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def software_fp8(device):
    return device.type == 'cuda' and torch.cuda.get_device_capability(device) < (8, 9)


def fp8_storage(tensor):
    """Keep the same address, shape and strides while avoiding an FP8 pointer."""
    packed = tensor.dtype == torch.float8_e4m3fn and software_fp8(tensor.device)
    return (tensor.view(torch.uint8) if packed else tensor), packed


@triton.jit
def fp8_to_float(value, PACKED: tl.constexpr):
    if PACKED:
        byte = value.to(tl.uint32)
        magnitude = byte & 127
        normal = (((magnitude >> 3) + 120) << 23) | ((magnitude & 7) << 20)
        subnormal = (magnitude.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)
        bits = tl.where(magnitude < 8, subnormal, normal)
        bits = tl.where(magnitude == 127, 0x7fc00000, bits).to(tl.uint32)
        return (bits | ((byte & 128) << 24)).to(tl.float32, bitcast=True)
    return value.to(tl.float32)


@triton.jit
def float_to_fp8(value):
    """Saturating FP32 -> E4M3FN bytes, round to nearest even, signed zero."""
    bits = value.to(tl.float32).to(tl.uint32, bitcast=True)
    absolute = bits & 0x7fffffff
    small = tl.where(absolute < 0x3c800000, absolute, 0).to(tl.float32, bitcast=True) * 512.
    lower = small.to(tl.uint32)
    remainder = small - lower.to(tl.float32)
    subnormal = lower + ((remainder > 0.5) | ((remainder == 0.5) & ((lower & 1) != 0)))
    normal = ((absolute + 0x7ffff + ((absolute >> 20) & 1)) >> 20) - 960
    normal = tl.minimum(normal, 126)
    normal = tl.where(absolute > 0x7f800000, 127, normal)
    magnitude = tl.where(absolute < 0x3c800000, subnormal, normal)
    return (magnitude | ((bits >> 24) & 128)).to(tl.uint8)
