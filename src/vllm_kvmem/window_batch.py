"""Translate working positions without changing native request/SSM history.

One per-request offset maps the *next dense window position* to logical history.
Historical selected blocks may be noncontiguous; assembly owns that mapping.
This adapter only maps newly appended tokens after an acknowledged refresh.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .metadata import WindowLengths, WorkingSetMetadata

_active_lengths = ContextVar("kvmem_attention_window_lengths", default=None)


@contextmanager
def attention_windows(groups):
    token = _active_lengths.set(groups)
    try:
        yield
    finally:
        _active_lengths.reset(token)


def contextual_metadata(native):
    groups = _active_lengths.get()
    return WorkingSetMetadata(native, groups) if groups else native


@triton.jit
def _translate(Positions, QueryStart, Offsets, SeqLens, WindowPos, WindowLens,
               NUM_REQS: tl.constexpr, NUM_TOKENS: tl.constexpr,
               PAD_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    begin = tl.load(QueryStart + req)
    end = tl.load(QueryStart + req + 1)
    offset = tl.load(Offsets + req)
    length = tl.load(SeqLens + req)
    tl.store(WindowLens + req, length - offset)
    for start in range(begin, end, BLOCK):
        ids = start + tl.arange(0, BLOCK)
        pos = tl.load(Positions + ids, ids < end, other=0)
        tl.store(WindowPos + ids, pos - offset, ids < end)
    if req == 0:
        for start in range(NUM_TOKENS, PAD_TOKENS, BLOCK):
            ids = start + tl.arange(0, BLOCK)
            tl.store(WindowPos + ids, 0, ids < PAD_TOKENS)


@dataclass(frozen=True)
class WindowBatch:
    positions: torch.Tensor
    groups: dict[int, WindowLengths]


class WindowBatchAdapter:
    def __init__(self, attention_groups: tuple[int, ...], capacity: int):
        if not attention_groups or len(set(attention_groups)) != len(attention_groups) or capacity <= 0:
            raise ValueError("Invalid full-attention window configuration")
        self.attention_groups = attention_groups
        self.capacity = capacity
        self.offsets: dict[str, int] = {}
        self.history_ends: dict[str, int] = {}

    def commit(self, request_id: str, history_end: int, window_tokens: int):
        if not 0 < window_tokens <= min(history_end, self.capacity):
            raise ValueError("Invalid committed working window")
        offset = history_end - window_tokens
        if history_end < self.history_ends.get(request_id, 0):
            raise ValueError("Working window cannot precede the committed history")
        self.offsets[request_id] = offset
        self.history_ends[request_id] = history_end

    def free(self, request_id):
        self.offsets.pop(request_id, None)
        self.history_ends.pop(request_id, None)

    def prepare(self, batch) -> WindowBatch | None:
        offsets = [self.offsets.get(req, 0) for req in batch.req_ids]
        if not any(offsets):
            return None
        if batch.positions.device.type != "cuda" or batch.positions.ndim != 1:
            raise ValueError("Working positions require native CUDA 1D input positions")
        if len(offsets) != batch.num_reqs:
            raise ValueError("Request order does not match native batch")
        if any(offset < 0 or offset > computed for offset, computed in
               zip(offsets, batch.num_computed_tokens_np)):
            raise ValueError("Native logical position precedes working-set commit")
        cpu_offsets = torch.zeros(batch.num_reqs_after_padding, dtype=torch.int32)
        cpu_offsets[:batch.num_reqs] = torch.tensor(offsets, dtype=torch.int32)
        bounds = batch.seq_lens_cpu_upper_bound - cpu_offsets
        if torch.any(bounds < 0).item() or torch.any(bounds > self.capacity).item():
            raise ValueError("Native batch exceeds the admitted working-set extent")
        gpu_offsets = cpu_offsets.to(batch.positions.device)
        positions = torch.empty_like(batch.positions)
        seq_lens = torch.zeros_like(batch.seq_lens)
        _translate[(batch.num_reqs,)](
            batch.positions, batch.query_start_loc, gpu_offsets, batch.seq_lens,
            positions, seq_lens, batch.num_reqs, batch.num_tokens,
            batch.num_tokens_after_padding, BLOCK=128)
        lengths = WindowLengths(seq_lens, bounds, int(bounds.max().item()), positions)
        return WindowBatch(positions, {group: lengths for group in self.attention_groups})
