"""Archive-layout assembly without per-record CPU tensor repacking.

Immutable records remain in their file layout:
[record][all raw-K layers][all FP8-V layers].
"""
import torch
from vllm.triton_utils import tl, triton


def read_packed_records(authority, handles, *, destination=None, region=None):
    """One bounded read; optionally fill an exclusively owned CPU staging tensor."""
    from .authority import record_key
    if not handles:
        raise ValueError("No immutable records requested")
    authority.poll()
    records = [authority.records[record_key(handle)] for handle in handles]
    if any(not record.archived or record.count != authority.shape[1] for record in records):
        raise ValueError("Packed assembly requires fully sealed, published records")
    leases = [authority.memory.repository.pin(handle) for handle in handles]
    try:
        if not all(leases):
            raise ValueError("Stale archive generation")
        kwargs = {}
        if destination is not None:
            if (destination.device.type != 'cpu' or destination.dtype != torch.uint8 or
                    destination.ndim != 1 or not destination.is_contiguous()):
                raise ValueError("Packed read destination must be contiguous CPU bytes")
            kwargs['destination'] = destination.numpy()
        job = (authority.archive.read(authority.memory.request_id, leases, **kwargs) if region is None else
               authority.archive.read_region(authority.memory.request_id, leases, *region, **kwargs))
        if job is None:
            raise BlockingIOError("Archive read queue is full")
        data, stats = job.result()
        size = authority.record_bytes if region is None else region[1]
        if len(data) != len(handles) * size:
            raise ValueError("Archive returned an incomplete packed record batch")
        result = torch.frombuffer(data, dtype=torch.uint8)
        if destination is not None and result.data_ptr() != destination.data_ptr():
            raise ValueError("Archive did not fill the requested staging buffer")
        return result, stats
    finally:
        for lease in leases:
            lease.reset()


@triton.jit
def _assemble_packed(payload, bases, page_table, trig, kcache, vcache, scale,
                     k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr,
                     v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr,
                     TRIG_STRIDE: tl.constexpr, RECORD_TOKENS: tl.constexpr,
                     LAYERS: tl.constexpr, LAYER: tl.constexpr, HEADS: tl.constexpr,
                     DIM: tl.constexpr, HALF: tl.constexpr, PAGE: tl.constexpr,
                     BLOCK: tl.constexpr, RAW_TYPE: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    record, offset = token // RECORD_TOKENS, token % RECORD_TOKENS
    position = tl.load(bases + record).to(tl.int64) + offset
    page = tl.load(page_table + position // PAGE).to(tl.int64)
    cache_offset = position % PAGE
    layer_elements: tl.constexpr = RECORD_TOKENS * HEADS * DIM
    raw_elements: tl.constexpr = LAYERS * layer_elements
    d = tl.arange(0, BLOCK)
    mask = d < DIM
    raw = payload.to(tl.pointer_type(RAW_TYPE))
    raw_base = record * (raw_elements * 3 // 2) + LAYER * layer_elements + offset * HEADS * DIM + head * DIM
    x = tl.load(raw + raw_base + d, mask=mask, other=0).to(tl.float32)
    other_d = tl.where(d < HALF, d + HALF, d - HALF)
    mate = tl.load(raw + raw_base + other_d, mask=d < 2*HALF, other=0).to(tl.float32)
    freq = d % HALF
    cos = tl.load(trig + position * TRIG_STRIDE + freq).to(tl.float32)
    sin = tl.load(trig + position * TRIG_STRIDE + HALF + freq).to(tl.float32)
    rotated = tl.where(d < HALF, x*cos - mate*sin, x*cos + mate*sin)
    rotated = tl.where(d < 2*HALF, rotated, x).to(RAW_TYPE).to(tl.float32)
    quantized = tl.minimum(tl.maximum(rotated / tl.load(scale), -448.0), 448.0)
    tl.store(kcache + page*k0 + cache_offset*k1 + head*k2 + d, quantized, mask=mask)
    value_base = record * raw_elements * 3 + raw_elements * 2 + LAYER * layer_elements + offset * HEADS * DIM + head * DIM
    value = tl.load(payload + value_base + d, mask=mask, other=0)
    dst = vcache.to(tl.pointer_type(tl.uint8))
    tl.store(dst + page*v0 + cache_offset*v1 + head*v2 + d, value, mask=mask)


def assemble_packed_layer(payload, bases, layer, *, layer_index, layers, block_tokens, raw_dtype):
    """Caller validates destination page ownership, uniqueness and position bounds."""
    key, value = layer.key, layer.value
    if key.ndim != 4 or value.shape != key.shape or key.dtype != torch.float8_e4m3fn or value.dtype != key.dtype:
        raise ValueError("Packed assembly needs matching native FP8 cache views")
    heads, dim = key.shape[-2:]
    if (min(layers, block_tokens) <= 0 or (layers * block_tokens * heads * dim) % 2 or
            not 0 <= layer_index < layers or
            raw_dtype not in (torch.bfloat16, torch.float16)):
        raise ValueError("Invalid packed source geometry")
    if (bases.ndim != 1 or bases.dtype not in (torch.int32, torch.int64) or not bases.is_contiguous() or
            payload.ndim != 1 or payload.dtype != torch.uint8 or not payload.is_contiguous() or
            payload.numel() != bases.numel() * layers * block_tokens * heads * dim * 3):
        raise ValueError("Packed record extent does not match destination bases")
    tensors = (payload, bases, key, value, layer.kernel_page_table, layer.cos_sin, layer.key_scale)
    if payload.device.type != 'cuda' or any(t.device != payload.device for t in tensors):
        raise ValueError("Packed assembly requires one CUDA device")
    if (layer.cos_sin.ndim != 2 or not 0 < layer.cos_sin.shape[1] <= dim or
            layer.cos_sin.shape[1] % 2 or layer.key_scale.numel() != 1 or
            any(t.ndim and t.stride(-1) != 1 for t in tensors)):
        raise ValueError("Invalid native scale, rotary geometry or inner stride")
    if not bases.numel():
        return
    _assemble_packed[(bases.numel() * block_tokens, heads)](
        payload, bases, layer.kernel_page_table, layer.cos_sin, key, value, layer.key_scale,
        *key.stride()[:3], *value.stride()[:3], layer.cos_sin.stride(0), block_tokens,
        layers, layer_index, heads, dim, layer.cos_sin.shape[1] // 2, key.shape[1],
        triton.next_power_of_2(dim), tl.bfloat16 if raw_dtype == torch.bfloat16 else tl.float16,
        num_warps=4)
