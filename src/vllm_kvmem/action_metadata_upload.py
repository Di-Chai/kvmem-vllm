"""Bounded, engine-thread-owned asynchronous Action metadata staging.

One pinned arena and one device arena hold typed, disjoint owner slices. Their
combined tensor payload is at most 8 MiB, including free cached space. Only a
fence after ALL metadata consumers permits slice reuse. Request cancellation or
packet publication does not own these leases. No path synchronizes the GPU.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import get_ident
import time

import torch


CAPACITY_BYTES = 8 * 1024 * 1024
MAX_LEASES = 128


class TorchMetadataBackend:
    @staticmethod
    def capturing():
        return torch.cuda.is_current_stream_capturing()

    @staticmethod
    def stream(device):
        return torch.cuda.current_stream(device)

    @staticmethod
    def allocate(size, device):
        return (torch.empty(size, dtype=torch.uint8, pin_memory=True),
                torch.empty(size, dtype=torch.uint8, device=device))

    @staticmethod
    def upload(device, host):
        device.copy_(host, non_blocking=True)

    @staticmethod
    def fence(stream):
        event = torch.cuda.Event()
        event.record(stream)
        return event


def typed_views(buffer, segments, rows):
    boundary = 4 * segments
    row_begin = 2 * boundary  # Always aligned for int64, including odd segment counts.
    row_bytes = 8 * rows
    return (buffer[:boundary].view(torch.int32), buffer[boundary:row_begin].view(torch.int32),
            buffer[row_begin:row_begin+row_bytes].view(torch.int64),
            buffer[row_begin+row_bytes:row_begin+2*row_bytes].view(torch.int64))


@dataclass
class MetadataLease:
    offset: int
    size: int
    host: object
    device: object
    views: tuple
    stream: object
    event: object = None
    enqueued: bool = False
    released: bool = False

    def complete(self, event):
        """Adopt the ORIGINAL mass-D2H event, already recorded after all layers."""
        if self.event is not None or not self.enqueued or self.released:
            raise RuntimeError('Action metadata lease cannot accept this completion')
        self.event = event


class PackedActionMetadata:
    def __init__(self, *, max_bytes=CAPACITY_BYTES, max_leases=MAX_LEASES, backend=None):
        if (type(max_bytes) is not int or max_bytes < 16 or max_bytes % 16 or
                type(max_leases) is not int or max_leases <= 0):
            raise ValueError('Action metadata arena requires aligned positive byte/lease bounds')
        self.capacity = max_bytes // 2
        self.max_leases = max_leases
        self.backend = backend or TorchMetadataBackend()
        self.thread = get_ident()
        self.host = self.device = None
        self.device_id = None
        self.free = [(0, self.capacity)]
        self.pending = {}
        self.poisoned = None
        self.counters = dict(submissions=0, completed=0, packing_host_seconds=0.,
            copy_submit_host_seconds=0., uploaded_bytes=0, peak_pending_payload_bytes=0,
            peak_pending_leases=0, failed_submissions=0, fence_failures=0, query_failures=0)
        self.fallbacks = {}

    def _check(self):
        if get_ident() != self.thread:
            raise RuntimeError('Action metadata staging is engine-thread owned')
        if self.poisoned is not None:
            raise RuntimeError('Action metadata arena quarantined after CUDA event failure') from self.poisoned

    def _release(self, lease):
        if self.pending.get(lease.offset) is not lease or lease.released:
            raise RuntimeError('Action metadata credit released twice or by another owner')
        del self.pending[lease.offset]
        lease.released = True
        intervals = sorted([*self.free, (lease.offset, lease.size)])
        self.free = []
        for begin, size in intervals:
            if self.free and self.free[-1][0]+self.free[-1][1] == begin:
                start, previous = self.free[-1]
                self.free[-1] = (start, previous+size)
            else:
                self.free.append((begin, size))

    def poll(self):
        self._check()
        if not self.pending or self.backend.capturing():
            return
        # Bound the scan by max_leases; do not create cross-owner FIFO waits.
        for lease in tuple(self.pending.values()):
            try:
                ready = lease.event is not None and lease.event.query()
            except BaseException as error:
                self.poisoned = error
                self.counters['query_failures'] += 1
                raise
            if ready:
                self._release(lease)
                self.counters['completed'] += 1

    def _fallback(self, reason, profile):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0)+1
        with profile.stage('metadata.packed_fallback', reason=reason) if profile else nullcontext():
            pass

    @contextmanager
    def lease(self, starts, counts, rows, begin, local_begin, device, *, profile=None):
        self._check()
        device = torch.device(device)
        if device.type != 'cuda':
            self._fallback('non_cuda', profile)
            yield None
            return
        if self.backend.capturing():
            self._fallback('graph_capture', profile)
            yield None
            return
        self.poll()
        segments, samples = len(starts), len(rows)
        if len(counts) != segments or not segments or not samples:
            raise ValueError('Action metadata requires paired nonempty segments and sampled rows')
        size = 8*segments + 16*samples
        stream = self.backend.stream(device)
        selected = next(((i, offset, count) for i, (offset, count) in enumerate(self.free)
                         if count >= size), None)
        reason = ('oversize' if size > self.capacity else
                  'lease_limit' if len(self.pending) >= self.max_leases else
                  'byte_credit' if selected is None else
                  'different_device' if self.device_id is not None and self.device_id != stream.device else None)
        if reason is not None:
            self._fallback(reason, profile)
            yield None
            return
        if self.host is None:
            with profile.stage('metadata.packed_arena_allocate', allocation_bytes=2*self.capacity) if profile else nullcontext():
                host, gpu = self.backend.allocate(self.capacity, stream.device)
            self.host, self.device, self.device_id = host, gpu, stream.device
        index, offset, available = selected
        self.free[index:index+1] = [(offset+size, available-size)] if size < available else []
        host = self.host[offset:offset+size]
        gpu = self.device[offset:offset+size]
        lease = MetadataLease(offset, size, host, gpu, typed_views(gpu, segments, samples), stream)
        # Register ownership BEFORE packing/enqueue. It survives a later owner,
        # payload packing, download submission, or request cancellation failure.
        self.pending[offset] = lease
        self.counters['peak_pending_leases'] = max(self.counters['peak_pending_leases'], len(self.pending))
        used = 2*sum(item.size for item in self.pending.values())
        self.counters['peak_pending_payload_bytes'] = max(self.counters['peak_pending_payload_bytes'], used)
        try:
            started = time.perf_counter()
            with profile.stage('metadata.packed_cpu', shape=[segments, samples], dtype='int32+int64') if profile else nullcontext():
                views = typed_views(host, segments, samples)
                for target, values in zip(views, (starts, counts, [begin+r for r in rows], [local_begin+r for r in rows])):
                    # CPU conversion preserves Torch's integer overflow checks;
                    # never stage positions through float32 or an int32 cast.
                    target.copy_(torch.as_tensor(values, dtype=target.dtype, device='cpu'))
            self.counters['packing_host_seconds'] += time.perf_counter()-started
            started = time.perf_counter()
            with profile.stage('metadata.packed_h2d', gpu=True, transfer_bytes=size,
                               direction='H2D', shape=[size], dtype='uint8/int32/int64') if profile else nullcontext():
                lease.enqueued = True  # A failed copy may already have queued work.
                self.backend.upload(gpu, host)
            self.counters['copy_submit_host_seconds'] += time.perf_counter()-started
            self.counters['submissions'] += 1
            self.counters['uploaded_bytes'] += size
            yield lease
        except BaseException:
            self.counters['failed_submissions'] += 1
            raise
        finally:
            if not lease.enqueued:
                self._release(lease)
            elif lease.event is None:
                try:
                    # On normal integration complete() adopts the original
                    # recorded mass event. Exceptions fence every already
                    # enqueued consumer on the SAME stream, without waiting.
                    lease.event = self.backend.fence(stream)
                except BaseException as error:
                    # Never query an unrecorded Event: CUDA may report it ready.
                    # Keep this interval/arena owned and make the error fatal.
                    self.poisoned = error
                    self.counters['fence_failures'] += 1
                    raise

    def snapshot(self):
        # Called for diagnostics only; do not poll on publication worker paths.
        return dict(**self.counters, fallbacks=dict(self.fallbacks),
            arena_host_bytes=self.capacity if self.host is not None else 0,
            arena_gpu_bytes=self.capacity if self.device is not None else 0,
            max_combined_arena_bytes=2*self.capacity, max_pending_leases=self.max_leases,
            pending_leases=len(self.pending), pending_payload_bytes=2*sum(x.size for x in self.pending.values()),
            quarantined=self.poisoned is not None,
            scope='Combined owned pinned+GPU tensor payload, including free arena. Torch allocator bookkeeping/cache '
                  'and bounded temporary CPU integer conversion are not measured GPU execution or idle time.')
