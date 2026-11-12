"""CPU-first immutable archive with bounded, lazy SSD writeback.

The existing AsyncArchive owns publication, namespace/generation leases and
I/O byte admission. This backend only changes a record's physical residency;
spilling never retires its logical repository record. Disk mutation is serialized
on archive workers, independently of the short allocation lock. Region reads use
the existing charged full-record fallback, not the CPU-only source-view path.

SSD files are ephemeral process-local segments. A successful writeback permits
same-process reads; it is deliberately not a restart/durability guarantee.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import Lock, RLock
import time
import uuid

from .memory_storage import MemoryAllocation, MemorySpan
from .storage import AsyncArchive, _submit_archive_job


@dataclass(frozen=True)
class _DiskRecord:
    segment: int
    slot: int
    generation: int
    digest: bytes


@dataclass
class _Segment:
    fd: int
    path: Path
    capacity: int
    free: list
    records: dict
    file_bytes: int = 0
    closing: bool = False


class _SegmentStore:
    """Extent capacity grows by segments; payload files grow only on write."""

    def __init__(self, directory, record_bytes, max_bytes, segment_bytes, *, drop_cache=False):
        self.directory = Path(directory)
        self.record_bytes = record_bytes
        # drop_cache: written and read records leave the page cache (flush()
        # writes them back first), so spilled bytes really leave RAM instead of
        # sitting dirty for up to vm.dirty_expire_centisecs.
        self.drop_cache = drop_cache and hasattr(os, 'posix_fadvise')
        self.capacity = max_bytes // record_bytes
        self.segment_records = max(1, segment_bytes // record_bytes)
        self.segments = {}
        self.prefix = 'kvmem-tiered-' + uuid.uuid4().hex
        self._sequence = self._generation = self._reserved_records = 0
        self._lock = RLock()
        self.read_bytes = self.write_bytes = self.syscalls = 0
        self.peak_file_bytes = 0
        self._close_error = None

    def _allocate(self):
        with self._lock:
            if self._close_error is not None:
                raise RuntimeError('Tiered SSD close failed') from self._close_error
            candidate = next(((sid, seg) for sid, seg in self.segments.items() if seg.free and not seg.closing), None)
            if candidate is None:
                count = min(self.segment_records, self.capacity - self._reserved_records)
                if count <= 0:
                    raise OSError('Tiered SSD byte quota exhausted')
                self.directory.mkdir(parents=True, exist_ok=True)
                sid = self._sequence
                self._sequence += 1
                path = self.directory / f'{self.prefix}-{sid}.kv'
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
                try:
                    # Unlinked open files release their storage even after a
                    # process crash. No manifest/restart contract is implied.
                    path.unlink()
                except BaseException:
                    os.close(fd)
                    raise
                if self.drop_cache:
                    # Records are read whole and once: readahead would only
                    # pull neighbouring records back into the page cache.
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                seg = _Segment(fd, path, count, list(reversed(range(count))), {})
                self.segments[sid] = seg
                self._reserved_records += count
            else:
                sid, seg = candidate
            slot = seg.free.pop()
            self._generation += 1
            generation = self._generation
            seg.records[slot] = generation
            return sid, seg, slot, generation

    def write(self, source):
        if len(source) != self.record_bytes:
            raise ValueError('SSD write requires one complete record')
        view = memoryview(source)
        location = None
        done = 0
        try:
            digest = hashlib.sha256(source).digest()
            sid, seg, slot, generation = self._allocate()
            location = _DiskRecord(sid, slot, generation, digest)
            while done < len(view):
                try:
                    count = os.pwrite(seg.fd, view[done:], slot*self.record_bytes + done)
                except InterruptedError:
                    continue
                if count <= 0:
                    raise OSError('Tiered SSD write made no progress')
                done += count
                with self._lock:
                    self.syscalls += 1
                    self.write_bytes += count
                    seg.file_bytes = max(seg.file_bytes, slot*self.record_bytes + done)
                    self.peak_file_bytes = max(self.peak_file_bytes,
                                              sum(s.file_bytes for s in self.segments.values()))
            if self.drop_cache:
                # Starts write-back of this record now (dirty pages are only
                # dropped once clean; flush() finishes the job).
                os.posix_fadvise(seg.fd, slot*self.record_bytes, self.record_bytes, os.POSIX_FADV_DONTNEED)
            return location
        except BaseException:
            if location is not None:
                self.release(location)
            raise
        finally:
            view.release()
            source = None

    def read_into(self, location, destination):
        if len(destination) != self.record_bytes:
            raise ValueError('SSD read requires one complete record destination')
        with self._lock:
            if self._close_error is not None:
                raise RuntimeError('Tiered SSD close failed') from self._close_error
            seg = self.segments.get(location.segment)
            if seg is None or seg.records.get(location.slot) != location.generation:
                raise KeyError('Stale tiered SSD extent')
            fd = seg.fd
        digest = hashlib.sha256()
        data = None
        try:
            done = 0
            while done < self.record_bytes:
                try:
                    data = os.pread(fd, min(1024*1024, self.record_bytes-done),
                                    location.slot*self.record_bytes+done)
                except InterruptedError:
                    continue
                if not data:
                    raise OSError('Tiered SSD record is truncated')
                destination[done:done+len(data)] = data
                digest.update(data)
                done += len(data)
                with self._lock:
                    self.syscalls += 1
                    self.read_bytes += len(data)
            if digest.digest() != location.digest:
                raise OSError('Tiered SSD record checksum mismatch')
            if self.drop_cache:
                os.posix_fadvise(fd, location.slot*self.record_bytes, self.record_bytes, os.POSIX_FADV_DONTNEED)
        finally:
            data = destination = None

    # Batched native I/O (session tiering): the store hands out extents and
    # file descriptors; vllm_kvmem._native moves the bytes without the GIL.
    def reserve(self, count):
        """Allocate up to ``count`` extents; fewer when the byte quota is
        exhausted. Returns [(segment, slot, generation, fd, offset)]."""
        out = []
        for _ in range(count):
            try:
                sid, seg, slot, generation = self._allocate()
            except OSError as error:
                if 'quota exhausted' not in str(error):
                    raise
                break
            out.append((sid, slot, generation, seg.fd, slot*self.record_bytes))
        return out

    def commit(self, extents, digests):
        """Natively written extents become locations (accounting included)."""
        out = []
        with self._lock:
            for (sid, slot, generation, fd, offset), digest in zip(extents, digests):
                seg = self.segments[sid]
                seg.file_bytes = max(seg.file_bytes, offset + self.record_bytes)
                self.syscalls += 1
                self.write_bytes += self.record_bytes
                out.append(_DiskRecord(sid, slot, generation, digest))
            self.peak_file_bytes = max(self.peak_file_bytes, sum(s.file_bytes for s in self.segments.values()))
        return out

    def abandon(self, extents):
        for sid, slot, generation, _, _ in extents:
            self.release(_DiskRecord(sid, slot, generation, None))

    def open_targets(self, locations):
        """Duplicated fds (one per segment) and offsets for reading
        ``locations``, so a concurrent release cannot close a file under a
        native read. Extents released meanwhile are skipped. Returns
        (dups, valid indices, fds, offsets); close dups with close_targets()."""
        dups, valid, fds, offsets = {}, [], [], []
        try:
            with self._lock:
                if self._close_error is not None:
                    raise RuntimeError('Tiered SSD close failed') from self._close_error
                for index, location in enumerate(locations):
                    seg = self.segments.get(location.segment)
                    if seg is None or seg.closing or seg.records.get(location.slot) != location.generation:
                        continue
                    if location.segment not in dups:
                        dups[location.segment] = os.dup(seg.fd)
                    valid.append(index)
                    fds.append(dups[location.segment])
                    offsets.append(location.slot*self.record_bytes)
                self.syscalls += len(valid)
                self.read_bytes += len(valid)*self.record_bytes
        except BaseException:
            self.close_targets(list(dups.values()))
            raise
        return list(dups.values()), valid, fds, offsets

    @staticmethod
    def close_targets(dups):
        for fd in dups:
            os.close(fd)

    def flush(self, locations):
        """Wait until ``locations`` are on the device, then drop their pages
        from the page cache (no-op unless drop_cache)."""
        if not self.drop_cache or not locations:
            return
        from . import _native
        native = getattr(_native, 'flush_spill_ranges', None)
        if native is not None:
            fds, offsets = [], []
            with self._lock:
                for location in locations:
                    seg = self.segments.get(location.segment)
                    if seg is not None and not seg.closing and seg.records.get(location.slot) == location.generation:
                        fds.append(seg.fd)
                        offsets.append(location.slot*self.record_bytes)
            native(fds, offsets, self.record_bytes)
            return
        by_segment = {}
        with self._lock:
            for location in locations:
                seg = self.segments.get(location.segment)
                if seg is not None and not seg.closing and seg.records.get(location.slot) == location.generation:
                    by_segment.setdefault(location.segment, (seg, []))[1].append(location.slot)
        for seg, slots in by_segment.values():
            try:
                os.fdatasync(seg.fd)
                for slot in slots:
                    os.posix_fadvise(seg.fd, slot*self.record_bytes, self.record_bytes, os.POSIX_FADV_DONTNEED)
            except OSError:
                if seg.closing:                 # released concurrently: nothing left to drop
                    continue
                raise

    def release(self, location):
        with self._lock:
            if self._close_error is not None:
                raise RuntimeError('Tiered SSD close failed') from self._close_error
            seg = self.segments.get(location.segment)
            if seg is None or seg.records.get(location.slot) != location.generation:
                return False
            if len(seg.records) > 1:
                del seg.records[location.slot]
                seg.free.append(location.slot)
                return True
            if seg.closing:
                raise RuntimeError('Tiered SSD segment already has an active close')
            seg.closing = True
        # Keep the last extent, segment and physical file charge visible while
        # close is in progress. No metadata lock spans a potentially slow close.
        try:
            os.close(seg.fd)
        except BaseException as error:
            with self._lock:
                seg.closing = False
                self._close_error = error
            raise
        with self._lock:
            del self.segments[location.segment]
            self._reserved_records -= seg.capacity
        return True

    def snapshot(self):
        with self._lock:
            return dict(ssd_used_bytes=sum(len(s.records) for s in self.segments.values())*self.record_bytes,
                        ssd_reserved_bytes=self._reserved_records*self.record_bytes,
                        ssd_file_bytes=sum(s.file_bytes for s in self.segments.values()),
                        ssd_peak_file_bytes=self.peak_file_bytes,
                        ssd_segments=len(self.segments), disk_read_bytes=self.read_bytes,
                        ssd_closing_bytes=sum(s.file_bytes for s in self.segments.values() if s.closing),
                        disk_write_bytes=self.write_bytes, disk_syscalls=self.syscalls)

    def close(self):
        with self._lock:
            if self._close_error is not None:
                raise RuntimeError('Tiered SSD close failed') from self._close_error
            locations = [_DiskRecord(sid, slot, generation, b'')
                         for sid, seg in self.segments.items() for slot, generation in seg.records.items()]
        for location in locations:
            self.release(location)


@dataclass
class _ResidentRecord:
    cpu: bytes | None
    disk: _DiskRecord | None = None


class _TieredTier:
    def __init__(self, directory, record_bytes, capacity, cpu_bytes, ssd_max_bytes,
                 segment_bytes, low_watermark, high_watermark):
        self.record_bytes, self.capacity = record_bytes, capacity
        self.cpu_capacity_bytes = cpu_bytes // record_bytes * record_bytes
        self.high_bytes = max(record_bytes, int(self.cpu_capacity_bytes*high_watermark)//record_bytes*record_bytes)
        self.low_bytes = min(self.high_bytes-record_bytes,
                             int(self.cpu_capacity_bytes*low_watermark)//record_bytes*record_bytes)
        self.allocations = {}
        self.records = OrderedDict()
        self.generation = 0
        self.lock = RLock()
        self._operation_lock = Lock()
        self._busy = set()
        self._disk_releases = deque()
        self._disk_release_error = None
        self.disk = _SegmentStore(directory, record_bytes, ssd_max_bytes, segment_bytes)
        self.cpu_bytes = self.cpu_peak_bytes = self.spilled_records = self.spill_failures = 0
        self.cpu_read_bytes = self.promoted_records = 0
        self.closed = False

    @property
    def used_slots(self):
        with self.lock:
            return len(self.allocations)

    def allocate(self, block_id):
        with self.lock:
            if self.closed or not 0 <= block_id < self.capacity or block_id in self.allocations:
                raise ValueError('Invalid or already owned tiered archive slot')
            self.generation += 1
            allocation = MemoryAllocation(block_id, block_id, self.generation)
            self.allocations[block_id] = allocation
            return allocation

    def release(self, allocation):
        # Repository callbacks must never wait for worker disk I/O. A busy
        # victim remains charged and the existing reclaimer retries its fence.
        with self.lock:
            if self.allocations.get(allocation.slot) != allocation or allocation.slot in self._busy:
                return False
            record = self.records.pop(allocation.slot, None)
            del self.allocations[allocation.slot]
            if record is not None:
                if record.cpu is not None:
                    record.cpu = None
                    self.cpu_bytes -= self.record_bytes
                if record.disk is not None:
                    self._disk_releases.append(record.disk)
            return True

    def _drain_disk_releases(self):
        if self._disk_release_error is not None:
            raise RuntimeError('Tiered SSD physical reclamation failed') from self._disk_release_error
        while True:
            with self.lock:
                if not self._disk_releases:
                    return
                location = self._disk_releases[0]
            try:
                if not self.disk.release(location):
                    raise RuntimeError('Retired SSD extent lost its physical generation owner')
            except BaseException as error:
                # An exception is not a close fence. Retain the exact queue head
                # and fail closed; blindly retrying close can hit a reused fd.
                self._disk_release_error = error
                raise
            with self.lock:
                assert self._disk_releases[0] is location
                self._disk_releases.popleft()

    def _spill_one(self, allowed_busy=frozenset()):
        with self.lock:
            candidate = next(((slot, rec) for slot, rec in self.records.items()
                              if rec.cpu is not None and (slot not in self._busy or slot in allowed_busy)), None)
            if candidate is None:
                return False
            slot, record = candidate
            already_busy = slot in self._busy
            self._busy.add(slot)
            source = record.cpu
        try:
            location = record.disk or self.disk.write(source)
            with self.lock:
                record.disk = location
                record.cpu = None
                source = None
                self.cpu_bytes -= self.record_bytes
                self.spilled_records += 1
            return True
        except BaseException:
            with self.lock:
                self.spill_failures += 1
            raise
        finally:
            source = None
            with self.lock:
                if not already_busy:
                    self._busy.discard(slot)

    def _make_room(self, allowed_busy):
        with self.lock:
            pressure = self.cpu_bytes+self.record_bytes > self.high_bytes
        if pressure:
            while True:
                with self.lock:
                    if self.cpu_bytes <= self.low_bytes:
                        break
                if not self._spill_one(allowed_busy):
                    break
        with self.lock:
            if self.cpu_bytes+self.record_bytes > self.cpu_capacity_bytes:
                raise BufferError('Tiered CPU capacity is pinned by in-flight readers')

    def io(self, spans, buffer, write):
        view = memoryview(buffer).cast('B')
        try:
            return self._io(tuple(spans), view, write)
        finally:
            view.release()
            buffer = None

    def _io(self, spans, view, write):
        started = time.perf_counter_ns()
        slots = {span.slot for span in spans}
        with self._operation_lock:
            self._drain_disk_releases()
            with self.lock:
                if self.closed or len(slots) != len(spans) or any(
                        span.slot not in self.allocations or span.size != self.record_bytes or
                        span.offset < 0 or span.offset+span.size > view.nbytes for span in spans):
                    raise ValueError('Invalid tiered archive I/O extent')
                if write and any(slot in self.records for slot in slots):
                    raise ValueError('Cannot overwrite immutable tiered archive record')
                if not write and (view.readonly or any(slot not in self.records for slot in slots)):
                    raise ValueError('Tiered read requires published records and writable destination')
                self._busy.update(slots)
            before = self.disk.snapshot()
            cpu_hits = 0
            try:
                for span in spans:
                    destination = view[span.offset:span.offset+span.size]
                    try:
                        if write:
                            self._make_room(slots)
                            record = _ResidentRecord(bytes(destination))
                            with self.lock:
                                self.records[span.slot] = record
                                self.cpu_bytes += self.record_bytes
                                self.cpu_peak_bytes = max(self.cpu_peak_bytes, self.cpu_bytes)
                        else:
                            with self.lock:
                                record = self.records[span.slot]
                                self.records.move_to_end(span.slot)
                            if record.cpu is not None:
                                destination[:] = record.cpu
                                cpu_hits += self.record_bytes
                            else:
                                self.disk.read_into(record.disk, destination)
                                # Read-through only borrows truly idle resident
                                # room. It never starts a writeback cascade just
                                # to cache a read. The SSD authority remains valid.
                                with self.lock:
                                    promote = self.cpu_bytes+self.record_bytes <= self.high_bytes
                                if promote:
                                    payload = bytes(destination)
                                    with self.lock:
                                        record.cpu = payload
                                        self.cpu_bytes += self.record_bytes
                                        self.cpu_peak_bytes = max(self.cpu_peak_bytes, self.cpu_bytes)
                                        self.promoted_records += 1
                                    payload = None
                    finally:
                        destination.release()
                after = self.disk.snapshot()
                with self.lock:
                    self.cpu_read_bytes += cpu_hits
                return dict(bytes=(after['disk_write_bytes']-before['disk_write_bytes'] if write else
                                   after['disk_read_bytes']-before['disk_read_bytes']),
                            syscalls=after['disk_syscalls']-before['disk_syscalls'],
                            duration_ns=time.perf_counter_ns()-started,
                            logical_bytes=len(spans)*self.record_bytes, cache_hit_bytes=cpu_hits)
            finally:
                with self.lock:
                    self._busy.difference_update(slots)

    def spill(self, target_bytes=None):
        target = self.low_bytes if target_bytes is None else target_bytes
        if type(target) is not int or not 0 <= target <= self.cpu_capacity_bytes:
            raise ValueError('Invalid tiered CPU spill target')
        started = time.perf_counter_ns()
        with self._operation_lock:
            self._drain_disk_releases()
            before = self.disk.snapshot()
            while True:
                with self.lock:
                    if self.closed:
                        raise RuntimeError('Tiered archive is closed')
                    if self.cpu_bytes <= target:
                        break
                if not self._spill_one():
                    break
            after = self.disk.snapshot()
            return dict(bytes=after['disk_write_bytes']-before['disk_write_bytes'],
                        syscalls=after['disk_syscalls']-before['disk_syscalls'],
                        duration_ns=time.perf_counter_ns()-started)

    def snapshot(self):
        with self.lock:
            state = dict(cpu_authority_bytes=self.cpu_bytes, cpu_resident_bytes=self.cpu_bytes,
                         cpu_capacity_bytes=self.cpu_capacity_bytes, cpu_peak_bytes=self.cpu_peak_bytes,
                         cpu_high_bytes=self.high_bytes, cpu_low_bytes=self.low_bytes,
                         logical_allocated_bytes=len(self.allocations)*self.record_bytes,
                         spilled_records=self.spilled_records, spill_failures=self.spill_failures,
                         cpu_read_bytes=self.cpu_read_bytes, promoted_records=self.promoted_records,
                         ssd_retired_pending_bytes=len(self._disk_releases)*self.record_bytes)
        return {**state, **self.disk.snapshot()}

    def close(self):
        with self._operation_lock:
            with self.lock:
                if self.closed:
                    return
            self.disk.close()
            with self.lock:
                self.closed = True
                self.records.clear()
                self.allocations.clear()
                self._disk_releases.clear()
                self.cpu_bytes = 0


class TieredArchive(AsyncArchive):
    """Bounded CPU residency over a separately limited logical SSD archive.

    ``capacity`` bounds all retained records, including CPU-only records. The
    complete extent must fit ``ssd_max_bytes`` so every live CPU record has a
    future backing slot. That quota is a logical guarantee, not upfront disk
    allocation. Temporary read/write payloads have the separate inherited
    ``io_max_bytes`` bound. Native historical GC and CPU-only source borrows
    are intentionally unsupported until their residency leases are integrated.
    """

    def __init__(self, directory, record_bytes, capacity, workers=2, max_pending=4, *,
                 cpu_bytes, ssd_max_bytes, segment_bytes=64*1024**2,
                 cpu_low_watermark=.7, cpu_high_watermark=.85, io_max_bytes=0):
        if (any(type(v) is not int or v <= 0 for v in
                (record_bytes, capacity, cpu_bytes, ssd_max_bytes, segment_bytes)) or
                cpu_bytes < record_bytes or segment_bytes < record_bytes):
            raise ValueError('Tiered capacities must fit at least one complete record')
        if capacity*record_bytes > ssd_max_bytes:
            raise ValueError('Tiered logical archive exceeds SSD byte quota')
        if (any(type(v) not in (int, float) for v in (cpu_low_watermark, cpu_high_watermark)) or
                not 0 <= cpu_low_watermark < cpu_high_watermark <= 1):
            raise ValueError('Tiered CPU watermarks require 0 <= low < high <= 1')
        super().__init__(directory, record_bytes, capacity, workers, max_pending,
                         io_max_bytes=io_max_bytes, memory_only=True)
        self.tier = _TieredTier(directory, record_bytes, capacity, cpu_bytes, ssd_max_bytes,
                                segment_bytes, cpu_low_watermark, cpu_high_watermark)
        self._tier = self.tier
        self.span = MemorySpan
        # Inline/prepared CPU readers may retain uncharged source views. The
        # inherited disk-aware fallback admits its own scratch and metadata leases.
        self.memory_only = False
        self.ssd_max_bytes = ssd_max_bytes
        self._disk_reap_pending = False
        self._disk_reap_job = None
        self._disk_reap_submissions = self._disk_reap_submit_failures = 0

    def _raise_reap_error(self):
        failure = self._tier._disk_release_error or self._tier.disk._close_error
        if failure is not None:
            raise RuntimeError('Tiered SSD physical reclamation failed') from failure

    def _queue_disk_reap_locked(self):
        """Caller owns the archive lock; one metadata-only job may be queued.

        This maintenance lane owns no payload buffer and has one bounded slot,
        independent of normal read/write byte admission. It participates in the
        existing pending-work/close fence so idle retirement can finish itself.
        """
        self._raise_reap_error()
        with self._tier.lock:
            needed = bool(self._tier._disk_releases)
        if self._disk_reap_pending or not needed:
            return
        if self._closed:
            raise RuntimeError('Closed archive still owns retired SSD extents')
        self._disk_reap_pending = True
        self._pending += 1

        def work():
            succeeded = False
            try:
                with self._tier._operation_lock:
                    self._tier._drain_disk_releases()
                succeeded = True
            finally:
                with self._lock:
                    self._disk_reap_pending = False
                    self._complete(0)
                    # A repository release can append after the drain's final
                    # empty check but before this completion acquires the lock.
                    if succeeded:
                        self._queue_disk_reap_locked()

        try:
            self._disk_reap_job = _submit_archive_job(self.executor, work)
            self._disk_reap_submissions += 1
        except BaseException:
            self._disk_reap_pending = False
            self._complete(0)
            self._disk_reap_submit_failures += 1
            raise

    def _release(self, key):
        released = super()._release(key)
        if released:
            self._queue_disk_reap_locked()
        return released

    def _admit(self, size):
        self._raise_reap_error()
        return super()._admit(size)

    def reclamation_ready(self):
        """Nonblocking terminal ACK fence for retired physical SSD ownership."""
        with self._lock:
            self._queue_disk_reap_locked()
            with self._tier.lock:
                return not self._disk_reap_pending and not self._tier._disk_releases

    def wait_for_idle(self, timeout):
        with self._lock:
            self._queue_disk_reap_locked()
        ready = super().wait_for_idle(timeout)
        self._raise_reap_error()
        return ready

    def history_used_bytes(self):
        return self._tier.used_slots*self.record_bytes

    def io_usage(self):
        reap_error = self._tier._disk_release_error or self._tier.disk._close_error
        return {**super().io_usage(), **self._tier.snapshot(), 'archive_backend': 'tiered',
                'cpu_peak_scope': 'lifetime_resident_payload_bytes',
                'ssd_reap_pending': self._disk_reap_pending,
                'ssd_reap_submissions': self._disk_reap_submissions,
                'ssd_reap_submit_failures': self._disk_reap_submit_failures,
                'ssd_reap_error': str(reap_error) if reap_error is not None else None,
                'ssd_max_bytes': self.ssd_max_bytes}

    def spill(self, target_bytes=None):
        """Queue bounded streaming writeback, without holding the scheduler."""
        if target_bytes is not None and (type(target_bytes) is not int or
                not 0 <= target_bytes <= self._tier.cpu_capacity_bytes):
            raise ValueError('Invalid tiered CPU spill target')
        size = self.record_bytes
        with self._lock:
            if not self._admit(size):
                return None

        def work():
            try:
                return self._tier.spill(target_bytes)
            finally:
                with self._lock:
                    self._complete(size)

        with self._lock:
            try:
                return _submit_archive_job(self.executor, work)
            except BaseException:
                self._complete(size)
                raise

    def close(self):
        self._raise_reap_error()
        if not super().close():
            return False
        self._tier.close()
        return True
