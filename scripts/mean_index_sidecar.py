"""Experimental sealed-page MeanK sum; no serving path imports this module.

Read the already packed archive FP8 bytes, not pre-quantization K or the rotated
attention cache. Keep one FP32 accumulator per channel and visit tokens 0..31.
Only finite pages use the result. Nonfinite input takes the original CPU batch
finalizer, preserving its NaN payload/sign behavior. CPU division and FP16
conversion remain the original authority operations.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl


BLOCK_TOKENS = 32


@dataclass(frozen=True)
class Geometry:
    layers: int = 17
    scoring_layers: int = 16
    heads: int = 4
    dim: int = 256
    capacity_tokens: int = 16424

    def __post_init__(self):
        for field in ('layers', 'scoring_layers', 'heads', 'dim', 'capacity_tokens'):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError('Positive bounded integer geometry required')
        if (not self.scoring_layers < self.layers <= 32 or
                self.heads * self.dim > 2048 or self.capacity_tokens > 16424 or
                self.capacity_tokens < BLOCK_TOKENS):
            raise ValueError('Experimental target prefix / capacity is unsupported')

    @property
    def pages(self):
        return self.capacity_tokens // BLOCK_TOKENS

    def sizes(self, tokens):
        if type(tokens) is not int or tokens <= 0 or tokens % BLOCK_TOKENS or tokens > self.capacity_tokens:
            raise ValueError('Only bounded complete 32-token pages are supported')
        pages = tokens // BLOCK_TOKENS
        sums = pages * self.scoring_layers * self.heads * self.dim * 4
        flags = pages * self.scoring_layers * 4
        return dict(pages=pages, sum_bytes=sums, flag_bytes=flags,
                    d2h_bytes=sums+flags, gpu_bytes=sums+flags, pinned_bytes=sums+flags)


def qualify(raw, geometry, *, history_begin=0):
    if type(history_begin) is not int or history_begin < 0 or history_begin % BLOCK_TOKENS:
        raise ValueError('A partial historical page must use the original CPU path')
    if (raw.dtype != torch.float8_e4m3fn or raw.ndim != 4 or
            (raw.shape[0], raw.shape[2], raw.shape[3]) !=
            (geometry.layers, geometry.heads, geometry.dim) or not raw.is_contiguous()):
        raise ValueError('Only exact contiguous packed FP8 archive bytes are supported')
    return geometry.sizes(raw.shape[1])


def qualify_target_prefix(archive_names, target_names, geometry):
    if (not isinstance(archive_names, tuple) or not isinstance(target_names, tuple) or
            len(archive_names) != geometry.layers or len(target_names) != geometry.scoring_layers or
            any(not isinstance(name, str) or not name for name in archive_names) or
            len(set(archive_names)) != len(archive_names) or
            archive_names[:len(target_names)] != target_names):
        raise ValueError('Target layer order must prove the exact archive prefix, excluding MTP')
    return target_names


@triton.jit
def _ordered_sum(Raw, Sums, Invalid, TOKENS: tl.constexpr,
                 SCORE_LAYERS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    page, layer = tl.program_id(0), tl.program_id(1)
    channel = tl.arange(0, BLOCK)
    valid = channel < WIDTH
    total = tl.full((BLOCK,), 0., tl.float32)
    bad = tl.full((BLOCK,), False, tl.int1)
    # Explicit ordered additions; no tree reduction of token values and no
    # division, fast-math, or FP16 conversion in the sidecar.
    for token in range(32):
        code = tl.load(Raw + (layer*TOKENS+page*32+token)*WIDTH+channel,
                       valid, 0).to(tl.uint32)
        magnitude = code & 127
        exponent, mantissa = magnitude >> 3, magnitude & 7
        sign = (code & 128) << 24
        normal_bits = sign | ((exponent + 120) << 23) | (mantissa << 20)
        normal = normal_bits.to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.001953125
        subnormal_bits = subnormal.to(tl.uint32, bitcast=True) | sign
        value = tl.where(exponent == 0, subnormal_bits.to(tl.float32, bitcast=True), normal)
        bad = bad | (magnitude == 127)
        # A NaN-containing batch is recomputed by CPU before any index write.
        total = total + tl.where(magnitude == 127, 0., value)
    tl.store(Sums + (page*SCORE_LAYERS+layer)*WIDTH+channel, total, valid)
    tl.store(Invalid + page*SCORE_LAYERS+layer,
             tl.sum((bad & valid).to(tl.int32), 0) != 0)


class Ticket:
    """One explicit experimental consumer; release only after all CPU uses."""
    def __init__(self, owner, raw, sizes):
        self.owner, self.raw, self.sizes = owner, raw, sizes
        self.ready = self.released = False

    def resolve(self):
        if self.released or self.owner.poisoned:
            raise RuntimeError('Sidecar ticket is released or quarantined')
        if not self.ready:
            try:
                self.owner.finished.synchronize()
            except BaseException:
                self.owner.poisoned = True
                raise
            self.ready = True
        pages = self.sizes['pages']
        return self.owner.host_sums[:pages], self.owner.host_invalid[:pages]

    def release(self):
        if not self.ready or self.released or self.owner.poisoned:
            raise RuntimeError('Unresolved/released/quarantined sidecar cannot be reused')
        self.raw = None
        self.released = True
        self.owner.pending = None


class OrderedMeanSidecar:
    """One fixed GPU/pinned slot. Full or failed slots never allocate another.

    This component has an explicit single consumer contract, not a serving
    lease pool. Callers must finish using returned views before ticket.release.
    """
    def __init__(self, geometry=Geometry(), *, device='cuda', archive_names, target_names):
        self.geometry, self.device = geometry, torch.device(device)
        self.target_names = qualify_target_prefix(archive_names, target_names, geometry)
        if self.device.type != 'cuda':
            raise ValueError('The experimental sidecar requires CUDA')
        shape = (geometry.pages, geometry.scoring_layers, geometry.heads, geometry.dim)
        flags = (geometry.pages, geometry.scoring_layers)
        self.sums = torch.empty(shape, device=device, dtype=torch.float32)
        self.invalid = torch.empty(flags, device=device, dtype=torch.int32)
        self.host_sums = torch.empty(shape, pin_memory=True, dtype=torch.float32)
        self.host_invalid = torch.empty(flags, pin_memory=True, dtype=torch.int32)
        self.started = torch.cuda.Event(enable_timing=True)
        self.computed = torch.cuda.Event(enable_timing=True)
        self.finished = torch.cuda.Event(enable_timing=True)
        self.pending, self.poisoned = None, False
        self.producer_stream = torch.cuda.current_stream(device).cuda_stream

    @torch.inference_mode()
    def submit(self, raw, *, history_begin=0):
        sizes = qualify(raw, self.geometry, history_begin=history_begin)
        if raw.device.type != 'cuda' or raw.device != self.sums.device:
            raise ValueError('Packed input and sidecar must use the same CUDA device')
        if raw.device.index != torch.cuda.current_device():
            raise ValueError('The component requires its CUDA device to be current')
        if self.pending is not None or self.poisoned:
            raise RuntimeError('Bounded sidecar slot is busy or quarantined')
        if torch.cuda.is_current_stream_capturing():
            raise ValueError('This experimental interface does not capture CUDA graphs')
        if torch.cuda.current_stream(raw.device).cuda_stream != self.producer_stream:
            raise ValueError('The component requires one ordered producer stream')
        ticket = Ticket(self, raw, sizes)
        self.pending = ticket
        try:
            self.started.record()
            width = self.geometry.heads*self.geometry.dim
            _ordered_sum[(sizes['pages'], self.geometry.scoring_layers)](
                raw.view(torch.uint8), self.sums, self.invalid, raw.shape[1],
                self.geometry.scoring_layers, width, triton.next_power_of_2(width),
                num_warps=4, enable_fp_fusion=False)
            self.computed.record()
            n = sizes['pages']
            self.host_sums[:n].copy_(self.sums[:n], non_blocking=True)
            self.host_invalid[:n].copy_(self.invalid[:n], non_blocking=True)
            self.finished.record()
        except BaseException:
            # Unknown submitted work retains all sources and buffers. This
            # diagnostic process stops rather than retrying a partial launch.
            self.poisoned = True
            raise
        return ticket

    def timings(self):
        if self.pending is None or not self.pending.ready:
            raise RuntimeError('Timing requires the completed current ticket')
        return dict(kernel_ms=self.started.elapsed_time(self.computed),
                    d2h_ms=self.computed.elapsed_time(self.finished),
                    gpu_interval_ms=self.started.elapsed_time(self.finished))


def cpu_jobs(raw, sums):
    """Original record views including exact per-layer source strides."""
    pages = raw.shape[1] // BLOCK_TOKENS
    return [(raw[:, i*32:(i+1)*32].view(torch.uint8).numpy(), sums[i].numpy(), 32, True)
            for i in range(pages)]


class SumConsumer:
    """Component adapter at the original finalize call, before append commit.

    It changes only where a fully validated FP32 sum comes from. Nonfinite
    batches call the unchanged scalar native routine. CPU mean conversion,
    accepted-count commit, archive publication, and retirement stay original.
    """
    def __init__(self, sums, invalid, original):
        if (sums.device.type != 'cpu' or sums.dtype != torch.float32 or sums.ndim != 4 or
                not sums.is_contiguous() or invalid.device.type != 'cpu' or
                invalid.shape != sums.shape[:2] or invalid.dtype != torch.int32):
            raise ValueError('Malformed completed sidecar output')
        self.sums, self.invalid, self.original = sums, invalid, original
        self.consumed = self.fallback_records = 0

    def __call__(self, jobs, vectorized=False):
        if vectorized:
            raise ValueError('Component adapter compares the original scalar finalizer')
        end = self.consumed+len(jobs)
        if end > len(self.sums):
            raise ValueError('Sidecar accepted-record cursor exceeded the physical packet')
        shape = tuple(self.sums.shape[1:])
        outputs = []
        for job in jobs:
            if len(job) != 4:
                raise ValueError('Malformed sealed index job')
            raw, dest, count, fp8 = job
            if (count != 32 or fp8 is not True or raw.dtype != np.uint8 or raw.ndim != 4 or
                    raw.shape[1] != 32 or raw.shape[0] <= shape[0] or raw.shape[2:] != shape[1:] or
                    dest.dtype != np.float32 or dest.shape != shape or not dest.flags.c_contiguous or
                    not dest.flags.writeable):
                raise ValueError('Original finalize job differs from qualified physical pages')
            outputs.append(dest)
        if bool(self.invalid[self.consumed:end].any()):
            self.original(jobs)
            self.fallback_records += len(jobs)
        else:
            # Include these CPU writes in complete component timings. Serving
            # might later attach a bulk native copier; this experiment does not.
            for index, dest in enumerate(outputs, self.consumed):
                np.copyto(dest, self.sums[index].numpy())
        self.consumed = end
        return None

    @contextmanager
    def installed(self):
        from vllm_kvmem import _native
        if _native.finalize_mean_batch is not self.original:
            raise RuntimeError('Experimental adapter is not exclusive')
        _native.finalize_mean_batch = self
        try:
            yield self
            if self.consumed != len(self.sums):
                raise RuntimeError('Not all sidecar pages were consumed')
        finally:
            _native.finalize_mean_batch = self.original
