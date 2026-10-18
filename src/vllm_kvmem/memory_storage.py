"""Bounded immutable CPU authority using the existing publication/lease protocol.

There is no disk fallback, eviction, compression, or raw-K precision change.
The owner must reserve each request's complete history before admission.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, local
import time

from . import _native
from .storage import AsyncArchive


def gather_records_python(destination, sources, record_bytes, offset, length, begin, end):
    """Reference gather for immutable CPU records; also supports older extensions."""
    view = memoryview(destination)
    if (type(record_bytes) is not int or record_bytes <= 0 or
            any(type(value) is not int for value in (offset, length, begin, end)) or
            not 0 <= offset < offset+length <= record_bytes or
            not 0 <= begin <= end <= len(sources) or
            view.readonly or view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous or
            view.nbytes != len(sources)*length or
            any(type(source) is not bytes or len(source) != record_bytes for source in sources[begin:end])):
        raise ValueError('Invalid immutable CPU record gather')
    view = view.cast('B')
    for i in range(begin, end):
        view[i*length:(i+1)*length] = memoryview(sources[i])[offset:offset+length]
    return (end-begin)*length


def gather_records_native(destination, sources, record_bytes, offset, length, begin, end):
    """Copy disjoint ranges with the existing GIL-releasing native memcpy path.

    Missing functions in an older extension use the reference path. A present
    native function's validation or copy error must propagate, never be hidden
    by retrying a partly written destination through another backend.
    """
    if offset == 0 and length == record_bytes:
        native = getattr(_native, 'gather_cached_records', None)
        if callable(native):
            return native(destination, sources, record_bytes, begin, end)
    else:
        native = getattr(_native, 'gather_record_region', None)
        if callable(native):
            return native(destination, sources, record_bytes, offset, length, begin, end)
    return gather_records_python(destination, sources, record_bytes, offset, length, begin, end)


def copy_record_slices(source, offsets, record_bytes):
    """Own each sealed record without retaining its whole source batch.

    The native copy releases the GIL after allocating independent immutable
    bytes. Older extensions retain the reference implementation; errors from
    an available native implementation must propagate before publication.
    """
    native = getattr(_native, 'copy_record_slices', None)
    if callable(native):
        return native(source, offsets, record_bytes)
    return [bytes(source[offset:offset+record_bytes]) for offset in offsets]


@dataclass(frozen=True)
class MemoryAllocation:
    block_id: int
    slot: int
    generation: int


@dataclass(frozen=True)
class MemorySpan:
    slot: int
    offset: int
    size: int


class MemoryTier:
    def __init__(self, record_bytes, capacity):
        self.record_bytes, self.capacity = record_bytes, capacity
        self.allocations, self.records = {}, {}
        self.generation = 0
        self.lock = RLock()
        self.gather = None

    @property
    def used_slots(self):
        with self.lock:
            return len(self.allocations)

    def allocate(self, block_id):
        with self.lock:
            if not 0 <= block_id < self.capacity or block_id in self.allocations:
                raise ValueError('Invalid or already owned CPU archive slot')
            self.generation += 1
            value = MemoryAllocation(block_id, block_id, self.generation)
            self.allocations[block_id] = value
            return value

    def release(self, allocation):
        with self.lock:
            if self.allocations.get(allocation.slot) != allocation:
                return False
            del self.allocations[allocation.slot]
            self.records.pop(allocation.slot, None)
            return True

    def io(self, spans, buffer, write):
        start = time.perf_counter_ns()
        view = memoryview(buffer).cast('B')
        spans = tuple(spans)
        with self.lock:
            if len({s.slot for s in spans}) != len(spans):
                raise ValueError('Duplicate CPU archive I/O span')
            for span in spans:
                if (span.slot not in self.allocations or span.size != self.record_bytes
                        or span.offset < 0 or span.offset+span.size > view.nbytes):
                    raise ValueError('Invalid CPU archive I/O extent')
                if write and span.slot in self.records:
                    raise ValueError('Cannot overwrite immutable CPU authority')
                if not write and span.slot not in self.records:
                    raise KeyError('CPU archive record is not published')
            if write:
                allocations = tuple(self.allocations[span.slot] for span in spans)
            else:
                if view.readonly:
                    raise ValueError('CPU archive read destination is readonly')
                # Borrow immutable bytes under the publication lock, then let
                # native memcpy release the GIL outside it. Archive read leases
                # protect the generation; these references retain the payload
                # even if a direct tier caller releases its slot concurrently.
                sources = [self.records[span.slot] for span in spans]
        read_stats = {}
        if write:
            # Copying a long prefill snapshot must neither hold the publication
            # lock nor monopolize the engine's GIL. Keep exact per-record owners
            # rather than slices that retain an unaccounted full batch slab.
            started = time.perf_counter()
            records = copy_record_slices(view, [span.offset for span in spans], self.record_bytes)
            if (len(records) != len(spans) or
                    any(type(record) is not bytes or len(record) != self.record_bytes for record in records)):
                raise RuntimeError('Invalid owned CPU record copy')
            pending = {span.slot: record for span, record in zip(spans, records)}
            copy_seconds = time.perf_counter() - started
            with self.lock:
                # Archive leases ordinarily prevent this race, but direct tier
                # callers can release/reallocate a slot while copying is out of
                # the lock. Never publish bytes into a replacement generation.
                if any(self.allocations.get(span.slot) != allocation
                       for span, allocation in zip(spans, allocations)):
                    raise ValueError('CPU archive allocation changed during record copy')
                if any(span.slot in self.records for span in spans):
                    raise ValueError('Cannot overwrite immutable CPU authority')
                try:
                    self.records.update(pending)
                except BaseException:
                    # A failed insertion must not expose a partial transaction.
                    # All slots were absent and the publication lock is held.
                    for slot in pending:
                        self.records.pop(slot, None)
                    raise
            read_stats = dict(seal_record_copy_seconds=copy_seconds)
        else:
            copy_started = time.perf_counter()
            packed = (view.nbytes == len(spans)*self.record_bytes and
                      all(span.offset == i*self.record_bytes for i, span in enumerate(spans)))
            if not sources:
                tasks = 0
            elif packed:
                if self.gather is not None:
                    tasks = self.gather(view, sources, 0, self.record_bytes)
                else:
                    gather_records_native(view, sources, self.record_bytes, 0, self.record_bytes, 0, len(sources))
                    tasks = 1
            else:
                # The archive always requests packed records. Preserve the
                # tier's more general ordered/gapped destination-span contract.
                for span, source in zip(spans, sources):
                    view[span.offset:span.offset+span.size] = source
                tasks = 1
            read_stats = dict(cache_gather_tasks=tasks,
                              cache_copy_seconds=time.perf_counter()-copy_started)
        # Transfer/assembly callers sum these counters across packed regions.
        # Backend identity belongs in io_usage(), not this numeric contract.
        return {'bytes':sum(s.size for s in spans), 'syscalls':0,
                'duration_ns':time.perf_counter_ns()-start, **read_stats}


class MemoryArchive(AsyncArchive):
    def __init__(self, *args, gather_workers=8, **kwargs):
        if type(gather_workers) is not int or not 1 <= gather_workers <= 16:
            raise ValueError('CPU gather workers must be between 1 and 16')
        super().__init__(*args, **kwargs, memory_only=True)
        self.gather_workers = gather_workers
        # Archive jobs wait for their copies; using the archive executor for
        # these tasks would deadlock when its workers are all occupied by reads.
        self._gather_executor = ThreadPoolExecutor(max_workers=gather_workers,
                                                   thread_name_prefix='kvmem-cpu-gather')
        self._gather_control = RLock()
        self._gather_unusable = None
        self._gather_shutdown_started = False
        self._gather_shutdown_done = Event()
        self._gather_worker = local()
        self.tier.gather = self._gather

    def _copy_range(self, *args):
        self._gather_worker.active = True
        try:
            return gather_records_native(*args)
        finally:
            self._gather_worker.active = False

    def _stop_gather_pool(self):
        # Only archive I/O/caller threads may join the copy pool. A callback
        # cannot recursively gather or close its own pool and join itself.
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot close its own pool')
        with self._gather_control:
            self._gather_unusable = self._gather_unusable or 'closed'
            owner = not self._gather_shutdown_started
            self._gather_shutdown_started = True
        if owner:
            # submit() can put a task in the queue and then fail to start a
            # thread, without returning its Future. Cancel unseen queued tasks
            # and join all running copies before releasing the caller's buffers
            # or archive leases. wait=True alone leaves tasks queued when the
            # very first worker failed to start.
            self._gather_executor.shutdown(wait=True, cancel_futures=True)
            self._gather_shutdown_done.set()
        else:
            self._gather_shutdown_done.wait()

    def _gather(self, destination, sources, offset, length):
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot recursively gather')
        workers = min(self.gather_workers, len(sources))
        parallel = workers > 1 and len(sources)*length >= 8*1024**2
        jobs, error = [], None
        with self._gather_control:
            # Reject all subsequent reads, including the inline small-copy
            # path, after a submission failure. Never resurrect the executor.
            if self._gather_unusable is not None:
                raise RuntimeError('CPU gather pool unavailable: '+self._gather_unusable)
            if parallel:
                try:
                    for worker in range(workers):
                        jobs.append(self._gather_executor.submit(self._copy_range,
                            destination, sources, self.record_bytes, offset, length,
                            len(sources)*worker//workers, len(sources)*(worker+1)//workers))
                except BaseException as failure:
                    error = failure
                    # Store no traceback/source buffer in the permanent state.
                    # The lock prevents another read from submitting while the
                    # failing submit's unseen task is being drained/cancelled.
                    self._gather_unusable = 'submission failed ('+type(failure).__name__+')'
        if error is not None:
            self._stop_gather_pool()
        if not sources:
            return 0
        if not parallel:
            gather_records_native(destination, sources, self.record_bytes, offset, length, 0, len(sources))
            return 1
        # No destination, immutable source reference or archive lease may be
        # released while another worker is still copying, including on failure.
        for job in jobs:
            try:
                job.result()
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error
        return len(jobs)

    def _gather_region(self, destination, sources, offset, length):
        return self._gather(destination, sources, offset, length)

    def _region_sources(self, keys):
        # Native region gather reads immutable record buffers directly, avoiding
        # a second cache copy or a full-record fallback slab.
        with self.tier.lock:
            return [self.tier.records[self._entries[key].allocation.slot] for key in keys]

    def io_usage(self):
        return {**super().io_usage(), 'archive_backend':'cpu', 'disk_read_bytes':0,
                'disk_write_bytes':0, 'cpu_authority_bytes':self.tier.used_slots*self.record_bytes}

    def close(self):
        if getattr(self._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot close its own pool')
        if not super().close():
            return False
        self._stop_gather_pool()
        return True
