"""Reference GPU materialization from immutable pre-RoPE K.

Codec operations use logical NHD views; physical attention backend strides are
preserved by the adapter. No inverse rotation of a quantized cache is permitted.
The reference implementation is a correctness oracle for fused implementations.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVFormat:
    dtype: torch.dtype
    head_size: int
    kv_heads: int
    page_tokens: int
    scale_mode: str = "tensor"

    def __post_init__(self):
        if self.dtype not in (torch.float8_e4m3fn, torch.float16, torch.bfloat16):
            raise ValueError("Unsupported KV dtype")
        if min(self.head_size, self.kv_heads, self.page_tokens) <= 0:
            raise ValueError("Invalid KV dimensions")
        if self.scale_mode != "tensor":
            raise ValueError("Per-head/block scale codecs not yet qualified")


def neox_rope(raw_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Model-provided cos/sin [tokens, rotary_dim/2], including its RoPE scaling.

    Raw K is post-normalization, pre-RoPE [tokens, heads, head_size]. Computation
    is float32 followed by the model activation cast, matching the staged oracle.
    """
    if raw_k.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("Expected NHD keys and matching two-dimensional trig tables")
    half = cos.shape[-1]
    if cos.shape[0] != raw_k.shape[0] or half * 2 > raw_k.shape[-1] or half == 0:
        raise ValueError("Rotary extent does not match K")
    if raw_k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Raw K must be captured before KV quantization")
    source = raw_k.float()
    left, right = source[..., :half], source[..., half:2*half]
    c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    return torch.cat((left*c - right*s, right*c + left*s, source[..., 2*half:]), dim=-1).to(raw_k.dtype)


def quantize(tensor: torch.Tensor, fmt: KVFormat, scale: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != (fmt.kv_heads, fmt.head_size):
        raise ValueError("KV shape mismatch")
    if scale.numel() != 1 or scale.device != tensor.device:
        raise ValueError("Codec requires a same-device per-tensor scale")
    # Validate scales once at adapter creation, outside the decode hot path.
    if not torch.isfinite(scale).all().item() or scale.item() <= 0:
        raise ValueError("Scale must be finite and positive")
    if fmt.dtype == torch.float8_e4m3fn:
        limit = torch.finfo(fmt.dtype).max
        return (tensor.float() / scale).clamp(-limit, limit).to(fmt.dtype)
    if scale.item() != 1:
        raise ValueError("Unquantized KV must have unit scale")
    return tensor.to(fmt.dtype)


def materialize_k(raw_k, cos, sin, fmt, scale):
    return quantize(neox_rope(raw_k, cos, sin), fmt, scale)


def paged_nhd_view(cache: torch.Tensor, fmt: KVFormat, layout: str):
    """Return K/V views, preserving backend ownership and strides.

    Accepted explicit layouts: [2, pages, tokens, heads, dim] (2PNHD),
    [pages, 2, tokens, heads, dim] (P2NHD), and head-major P2HND.
    Shape coincidence is never used to guess a backend layout.
    """
    if cache.dtype != fmt.dtype or cache.ndim != 5:
        raise ValueError("KV cache dtype/rank mismatch")
    if layout == "2PNHD" and cache.shape[0] == 2:
        k, v = cache[0], cache[1]
    elif layout == "P2NHD" and cache.shape[1] == 2:
        k, v = cache[:, 0], cache[:, 1]
    elif layout == "P2HND" and cache.shape[1] == 2:
        k, v = cache[:, 0].transpose(1, 2), cache[:, 1].transpose(1, 2)
    else:
        raise ValueError("Unsupported physical cache layout")
    if tuple(k.shape[1:]) != (fmt.page_tokens, fmt.kv_heads, fmt.head_size):
        raise ValueError("Physical cache shape does not match codec")
    return k, v
