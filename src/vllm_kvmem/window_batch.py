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
        self.buffers = None

    def reserve_graph_buffers(self, max_tokens, max_requests, device):
        """Capture-time attention lengths and RoPE positions keep stable addresses."""
        self.buffers = (
            torch.empty(max_tokens, device=device, dtype=torch.int64),
            torch.zeros(max_requests, device=device, dtype=torch.int32),
            torch.zeros(max_requests, device=device, dtype=torch.int32),
        )
        self.offsets_cpu = torch.empty(max_requests, dtype=torch.int32, pin_memory=True)
        self.offsets_ready = None
        self.uploaded_offsets = None

    def capture_window(self, batch):
        if self.buffers is None:
            raise RuntimeError('Graph window buffers were not reserved')
        positions, seq_lens, _ = self.buffers
        n, t = batch.num_reqs_after_padding, batch.num_tokens_after_padding
        positions[:t].copy_(batch.positions[:t])
        seq_lens[:n].copy_(batch.seq_lens[:n])
        bounds = batch.seq_lens_cpu_upper_bound[:n]
        # Native FULL capture uses a worst-case extent. Do not bake a short
        # bootstrap window into the attention kernel's graph configuration.
        lens = WindowLengths(seq_lens[:n], bounds, max(self.capacity, int(bounds.max())), positions[:t])
        return WindowBatch(positions[:t], {g: lens for g in self.attention_groups})

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
        if not any(offsets) and self.buffers is None:
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
        if self.buffers is None:
            gpu_offsets = cpu_offsets.to(batch.positions.device)
            positions = torch.empty_like(batch.positions)
            seq_lens = torch.zeros_like(batch.seq_lens)
        else:
            positions, seq_lens, gpu_offsets = self.buffers
            positions = positions[:batch.num_tokens_after_padding]
            seq_lens = seq_lens[:batch.num_reqs_after_padding]
            gpu_offsets = gpu_offsets[:batch.num_reqs_after_padding]
            offset_key = tuple(cpu_offsets.tolist())
            if self.uploaded_offsets != offset_key:
                # A synchronous CPU copy stalls behind the preceding native
                # draft kernels. Offsets usually stay fixed throughout decode.
                # Retain pinned source storage until its DMA has completed.
                if self.offsets_ready is not None and not self.offsets_ready.query():
                    self.offsets_ready.synchronize()
                source = self.offsets_cpu[:batch.num_reqs_after_padding]
                source.copy_(cpu_offsets)
                gpu_offsets.copy_(source, non_blocking=True)
                self.offsets_ready = torch.cuda.Event()
                self.offsets_ready.record(torch.cuda.current_stream(batch.positions.device))
                self.uploaded_offsets = offset_key
            seq_lens.zero_()
        _translate[(batch.num_reqs,)](
            batch.positions, batch.query_start_loc, gpu_offsets, batch.seq_lens,
            positions, seq_lens, batch.num_reqs, batch.num_tokens,
            batch.num_tokens_after_padding, BLOCK=128)
        lengths = WindowLengths(seq_lens, bounds, int(bounds.max().item()), positions)
        return WindowBatch(positions, {group: lengths for group in self.attention_groups})
