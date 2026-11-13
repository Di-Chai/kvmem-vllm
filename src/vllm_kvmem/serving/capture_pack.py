"""Bounded full-batch FP8 capture packing, outside compiled model graphs.

Only CPU tensor metadata is inspected. Unsupported layouts retain the original
Torch path; errors after a supported kernel starts are never silently retried.
One pointer table and one event retain opaque-pointer sources until completion.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _header(Starts, Global, Local, Rejected, Out, N, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    begin = tl.load(Starts + i, i < N + 1, 0).to(tl.int64)
    tl.store(Out + i, begin, i < N + 1)
    glob = tl.load(Global + begin, i < N, 0)
    local = tl.load(Local + begin, i < N, 0)
    rejected = tl.load(Rejected + i, i < N, 0).to(tl.int64)
    tl.store(Out + N + 1 + i, glob, i < N)
    tl.store(Out + 2 * N + 1 + i, local, i < N)
    tl.store(Out + 3 * N + 1 + i, rejected, i < N)


@triton.jit(do_not_specialize=['TOKENS'])
def _payload(Pointers, Scales, Out, TOKENS, WIDTH: tl.constexpr,
             LAYERS: tl.constexpr, BLOCK: tl.constexpr):
    layer, kind = tl.program_id(1), tl.program_id(2)
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    elements = TOKENS.to(tl.int64) * WIDTH
    pointer = tl.load(Pointers + layer * 4 + kind)
    stride = tl.load(Pointers + layer * 4 + 2 + kind)
    source = pointer.to(tl.pointer_type(tl.bfloat16))
    value = tl.load(source + offset // WIDTH * stride + offset % WIDTH,
                    offset < elements, 0).to(tl.float32)
    if kind == 1:
        # Match Torch's FP32 division, not approximate reciprocal multiplication.
        value = tl.div_rn(value, tl.load(Scales + layer).to(tl.float32))
    value = tl.minimum(tl.maximum(value, -448., propagate_nan=tl.PropagateNan.ALL),
                       448., propagate_nan=tl.PropagateNan.ALL)
    byte = value.to(tl.float8e4nv, fp_downcast_rounding='rtne').to(tl.uint8, bitcast=True)
    tl.store(Out + (kind * LAYERS + layer).to(tl.int64) * elements + offset,
             byte, offset < elements)


@dataclass(frozen=True)
class PackLimits:
    layers: int
    heads: int
    dim: int
    max_tokens: int
    max_requests: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.layers, self.heads, self.dim, self.max_tokens, self.max_requests)):
            raise ValueError('Capture pack limits must be positive integers')

    @property
    def payload_bytes(self):
        return 2*self.layers*self.heads*self.dim*self.max_tokens + (4*self.max_requests+1)*8

    @property
    def metadata_bytes(self):
        # One int64 [layers, 4] table on each of CPU pinned memory and GPU.
        return 2*self.layers*4*8


def eligibility(captured, batch, rejected, indices, scales, raw_dtype, limits):
    """Metadata-only eligibility, with no .item(), value checks or GPU fences."""
    n, tokens = batch.num_reqs, batch.num_tokens
    if tuple(indices) != tuple(range(n)):
        return 'split_or_reordered_owners'
    if (type(n) is not int or type(tokens) is not int or
            not 0 < n <= limits.max_requests or not n <= tokens <= limits.max_tokens):
        return 'batch_capacity'
    if raw_dtype != torch.float8_e4m3fn:
        return 'raw_dtype'
    if len(captured) != limits.layers:
        return 'layer_geometry'
    device = scales.device
    if (device.type != 'cuda' or scales.dtype != torch.float32 or scales.ndim != 1 or
            scales.numel() != limits.layers or scales.stride(0) != 1):
        return 'scale_layout'
    header = ((batch.query_start_loc, n+1, (torch.int32, torch.int64)),
              (batch.positions, tokens, (torch.int64,)),
              (captured[0].positions, tokens, (torch.int64,)),
              (rejected, n, (torch.int32, torch.int64)))
    if any(value.device != device or value.ndim != 1 or value.numel() < needed or
           value.stride(0) != 1 or value.dtype not in dtypes for value, needed, dtypes in header):
        return 'header_layout'
    width = limits.heads * limits.dim
    for item in captured:
        for value in (item.key, item.value):
            if (value.device != device or value.dtype != torch.bfloat16 or value.ndim != 3 or
                    value.shape[0] < tokens or value.shape[1:] != (limits.heads, limits.dim) or
                    value.stride(2) != 1 or value.stride(1) != limits.dim or value.stride(0) < width):
                return 'source_layout'
    return None


def source_signature(captured, scales):
    return (tuple((value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                  for item in captured for value in (item.key, item.value)),
            (scales.data_ptr(), tuple(scales.shape), tuple(scales.stride()), scales.dtype, scales.device))


class SourceBinding:
    """One source generation; replacement may query but never wait on its event."""
    def __init__(self):
        self.signature = self.stream = self.event = None
        self.owners = ()

    def can_replace(self, signature, stream):
        return ((signature == self.signature and stream == self.stream) or
                self.event is None or self.event.query())


class FusedCapturePacker:
    def __init__(self, limits, device):
        self.limits, self.device = limits, device
        self.host = torch.empty((limits.layers, 4), dtype=torch.int64, pin_memory=True)
        self.pointers = torch.empty_like(self.host, device=device)
        self.binding = SourceBinding()
        self.event = torch.cuda.Event()
        self.poisoned = False

    def try_pack(self, captured, batch, rejected, scales):
        if self.poisoned:
            raise RuntimeError('Capture pack cache has an unresolved CUDA failure')
        signature = source_signature(captured, scales)
        stream = torch.cuda.current_stream(self.device)
        identity = stream.cuda_stream
        if not self.binding.can_replace(signature, identity):
            return None, 'source_generation_inflight'
        try:
            if signature != self.binding.signature or identity != self.binding.stream:
                pointers = [(item.key.data_ptr(), item.value.data_ptr(), item.key.stride(0), item.value.stride(0))
                            for item in captured]
                # Host storage is overwritten only after the previous pointer
                # upload and opaque-pointer reads have actually completed.
                self.host.copy_(torch.tensor(pointers, dtype=torch.int64))
                owners = tuple(value for item in captured for value in (item.key, item.value)) + (scales,)
                self.binding.owners = owners
                for value in owners:
                    value.record_stream(stream)
                self.pointers.copy_(self.host, non_blocking=True)
                self.binding.signature, self.binding.stream = signature, identity
            n, tokens = batch.num_reqs, batch.num_tokens
            header_bytes = (4*n+1)*8
            output = torch.empty(header_bytes + 2*self.limits.layers*tokens*self.limits.heads*self.limits.dim,
                                 dtype=torch.uint8, device=self.device)
            # Triton does not tell the allocator about reads from another
            # stream. This also protects short-lived batch headers and the
            # pointer table if the runtime is released before kernels finish.
            for value in (self.pointers, batch.query_start_loc, batch.positions,
                          captured[0].positions, rejected):
                value.record_stream(stream)
            _header[(1,)](batch.query_start_loc, batch.positions, captured[0].positions, rejected,
                           output[:header_bytes].view(torch.int64), n,
                           triton.next_power_of_2(self.limits.max_requests+1))
            width = self.limits.heads*self.limits.dim
            _payload[(triton.cdiv(tokens*width, 1024), self.limits.layers, 2)](
                self.pointers, scales, output[header_bytes:], tokens, width, self.limits.layers, 1024,
                enable_fp_fusion=False)
            self.event.record(stream)
            self.binding.event = self.event
            return output, None
        except BaseException:
            # The single cache retains source owners and pinned pointer storage.
            # A present kernel/launch error must not become a silent fallback or
            # let a later call replace storage while work may still use it.
            self.poisoned = True
            raise


def try_pack(runtime, captured, batch, rejected, indices):
    limits = getattr(runtime, '_capture_pack_limits', None)
    if limits is None:
        return None, 'missing_capacity'
    reason = eligibility(captured, batch, rejected, indices, runtime.scales, runtime.raw_dtype, limits)
    if reason is not None:
        return None, reason
    if torch.cuda.is_current_stream_capturing():
        return None, 'graph_capture'
    cache = getattr(runtime, '_capture_pack_cache', None)
    if cache is None:
        if torch.cuda.get_device_capability(runtime.scales.device) < (8, 9):
            return None, 'device_capability'
        cache = FusedCapturePacker(limits, runtime.scales.device)
        runtime._capture_pack_cache = cache
    elif cache.limits != limits or cache.device != runtime.scales.device:
        # Runtime capacity/device is immutable. Do not accumulate specializations
        # or allocate a second table when an unexpected configuration is seen.
        return None, 'cache_geometry'
    return cache.try_pack(captured, batch, rejected, runtime.scales)
