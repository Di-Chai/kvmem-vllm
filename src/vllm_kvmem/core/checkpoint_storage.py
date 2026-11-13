"""Bounded CPU staging and lazy process-local SSD backing for exact swaps.

One serialized I/O lane preserves a complete CPU source until write success.
Reads materialize at most one admitted whole checkpoint and retain the disk
authority through GPU restore. Files are unlinked immediately, not restart
checkpoints. CUDA copy fences remain the request worker's responsibility.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import RLock
import uuid

import torch

from ..serving.request_checkpoint import ExactRequestCheckpoint, CopyCompletion


@dataclass(frozen=True)
class _DiskCheckpoint:
    fd: int
    size: int
    digest: bytes


@dataclass(eq=False)
class StoredCheckpoint:
    identity: tuple
    size: int
    cpu_charge: int
    state: str = 'reserved'
    snapshot: ExactRequestCheckpoint | None = None
    schemas: tuple = ()
    pinned: bool = False
    disk: _DiskCheckpoint | None = None
    loading: object = None


class CheckpointStore:
    def __init__(self, directory, cpu_bytes, ssd_bytes, *, low_watermark=.7, high_watermark=.85):
        if (any(type(v) is not int or v <= 0 for v in (cpu_bytes, ssd_bytes)) or
                not 0 < low_watermark < high_watermark < 1):
            raise ValueError('Checkpoint storage needs bounded CPU/SSD bytes and ordered watermarks')
        self.directory = Path(directory)
        self.cpu_capacity, self.ssd_capacity = cpu_bytes, ssd_bytes
        self.low_bytes, self.high_bytes = int(cpu_bytes*low_watermark), int(cpu_bytes*high_watermark)
        self.entries = OrderedDict()
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kvmem-checkpoint-ssd')
        self._job = None
        self._pressure = False
        self._close_error = None
        self._uncertain_close = None
        self.closed = False
        self.cpu_peak_bytes = self.cpu_charge_peak_bytes = self.ssd_peak_bytes = 0
        self.disk_write_bytes = self.disk_read_bytes = self.spills = self.loads = self.io_failures = 0
        self.cpu_deferrals = 0

    def _check(self, entry):
        self._raise_close_error()
        if self.closed or self.entries.get(entry.identity) is not entry:
            raise ValueError('Stale or foreign stored checkpoint')

    def _raise_close_error(self):
        if self._close_error is not None:
            raise RuntimeError('Checkpoint SSD close failed; descriptor ownership is uncertain') from self._close_error

    def _close_file(self, fd, charged_bytes):
        try:
            os.close(fd)
        except BaseException as error:
            # An error may be reported after the OS releases the descriptor.
            # Preserve conservative ownership and never retry this integer.
            with self._lock:
                self._close_error = error
                self._uncertain_close = fd, charged_bytes
            raise

    def _cpu_usage(self):
        return sum(entry.cpu_charge for entry in self.entries.values())

    def _record_peak(self):
        self.cpu_charge_peak_bytes = max(self.cpu_charge_peak_bytes, self._cpu_usage())
        resident = sum(entry.size for entry in self.entries.values()
                       if entry.snapshot is not None or entry.loading is not None)
        self.cpu_peak_bytes = max(self.cpu_peak_bytes, resident)

    def _write(self, entry):
        fd, view, piece, payload = None, None, None, None
        try:
            payload = entry.snapshot.payload
            view = memoryview(payload.numpy()).cast('B')
            digest = hashlib.sha256(view).digest()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory/('checkpoint-'+uuid.uuid4().hex)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
            path.unlink()
            offset = 0
            while offset < len(view):
                piece = view[offset:]
                try:
                    count = os.pwrite(fd, piece, offset)
                except InterruptedError:
                    continue
                finally:
                    piece.release()
                    piece = None
                if count <= 0:
                    raise OSError('Checkpoint SSD write made no progress')
                offset += count
            result = _DiskCheckpoint(fd, len(view), digest)
            fd = None
            return result
        finally:
            try:
                if fd is not None:
                    self._close_file(fd, entry.size)
            finally:
                if view is not None:
                    view.release()
                view = payload = piece = None

    @staticmethod
    def _read(entry):
        view, piece, payload = None, None, None
        try:
            disk, payload = entry.disk, entry.loading
            view = memoryview(payload.numpy()).cast('B')
            offset = 0
            # preadv fills the owned slab directly: no uncharged full-size
            # bytes object, and no additional pinned/GPU snapshot.
            while offset < len(view):
                piece = view[offset:min(offset+1024*1024, len(view))]
                try:
                    count = os.preadv(disk.fd, [piece], offset)
                except InterruptedError:
                    continue
                finally:
                    piece.release()
                    piece = None
                if count <= 0:
                    raise OSError('Checkpoint SSD payload is truncated')
                offset += count
            if hashlib.sha256(view).digest() != disk.digest:
                raise OSError('Checkpoint SSD payload checksum mismatch')
            return offset
        finally:
            if piece is not None:
                piece.release()
            if view is not None:
                view.release()
            view = payload = piece = None

    def _finish_io(self, *, suppress_error=False):
        self._raise_close_error()
        if self._job is None or not self._job[2].done():
            return
        entry, kind, job = self._job
        self._job = None
        try:
            result = job.result()
        except BaseException:
            self.io_failures += 1
            if kind == 'write':
                entry.state = 'resident'  # CPU source remains authoritative.
            else:
                entry.loading = None
                entry.cpu_charge = 0
                entry.state = 'disk'  # Immutable disk source remains intact.
            if not suppress_error:
                raise
        else:
            if kind == 'write':
                entry.disk = result
                # Drop the real pinned owner before releasing its CPU charge.
                entry.snapshot = None
                entry.cpu_charge = 0
                entry.state = 'disk'
                self.disk_write_bytes += entry.size
                self.spills += 1
                self.ssd_peak_bytes = max(self.ssd_peak_bytes,
                    sum(e.disk.size for e in self.entries.values() if e.disk is not None))
                if self._pressure and self._cpu_usage() > self.low_bytes:
                    self._spill_one()
                else:
                    self._pressure = False
            else:
                entry.snapshot = ExactRequestCheckpoint(entry.identity, entry.schemas,
                    entry.loading, CopyCompletion())
                entry.loading = None
                entry.state = 'resident'
                self.disk_read_bytes += result
                self.loads += 1

    def _spill_one(self, protected=None):
        if self._job is not None:
            return False
        for entry in self.entries.values():
            if entry is protected or entry.state != 'resident':
                continue
            if entry.disk is not None:
                entry.snapshot = None
                entry.cpu_charge = 0
                entry.state = 'disk'
                return True
            # Executor WorkItem.args can outlive Future.done(). Retain only the
            # mutable owner descriptor, never another tensor reference whose
            # lifetime would escape the store's CPU-credit release fence.
            job = self._executor.submit(self._write, entry)
            entry.state = 'spilling'
            self._job = entry, 'write', job
            return False  # CPU charge is retained until the I/O fence.
        return False

    def _room(self, amount, protected=None):
        while self._cpu_usage()+amount > self.cpu_capacity:
            if not self._spill_one(protected):
                self.cpu_deferrals += 1
                return False
        return True

    def reserve_capture(self, identity, amount):
        """Reserve before capture; a full CPU pool initiates spill and defers."""
        if type(amount) is not int or not 0 < amount <= self.cpu_capacity:
            raise BufferError('One complete checkpoint must fit the CPU staging cap')
        with self._lock:
            self._finish_io()
            if self.closed or identity in self.entries:
                raise ValueError('Checkpoint identity is closed or already retained')
            if sum(entry.size for entry in self.entries.values())+amount > self.ssd_capacity:
                raise BufferError('Checkpoint logical SSD quota exhausted')
            if not self._room(amount):
                return None
            entry = StoredCheckpoint(identity, amount, amount)
            self.entries[identity] = entry
            self._record_peak()
            return entry

    def install(self, entry, snapshot):
        with self._lock:
            self._check(entry)
            if (entry.state != 'reserved' or not isinstance(snapshot, ExactRequestCheckpoint) or
                    snapshot.identity != entry.identity or not 0 < snapshot.bytes <= entry.size or
                    snapshot.payload.device.type != 'cpu' or snapshot.payload.dtype != torch.uint8 or
                    snapshot.payload.ndim != 1 or not snapshot.payload.is_contiguous()):
                raise ValueError('Checkpoint payload differs from its reservation')
            entry.size = entry.cpu_charge = snapshot.bytes
            entry.snapshot, entry.schemas = snapshot, snapshot.schemas
            entry.pinned = snapshot.payload.is_pinned()
            entry.state = 'capturing'
            self._record_peak()

    def abort_capture(self, entry):
        with self._lock:
            self._check(entry)
            if entry.state != 'reserved':
                raise ValueError('Only an unallocated checkpoint reservation can abort')
            del self.entries[entry.identity]

    def capture_ready(self, entry):
        with self._lock:
            self._finish_io()
            self._check(entry)
            if entry.state == 'capturing':
                if not entry.snapshot.query():
                    return False
                entry.state = 'resident'
            elif entry.state not in ('resident', 'spilling', 'disk'):
                raise ValueError('Checkpoint capture is not pending or complete')
            if self._cpu_usage() > self.high_bytes:
                self._pressure = True
                self._spill_one()
            return True

    def acquire_restore(self, entry):
        """Return an owned exact snapshot only after a verified bounded read."""
        with self._lock:
            self._finish_io()
            self._check(entry)
            if entry.state == 'resident':
                entry.state = 'restoring'
                self.entries.move_to_end(entry.identity)
                return entry.snapshot
            if entry.state in ('loading', 'spilling'):
                return None
            if entry.state != 'disk':
                raise ValueError('Checkpoint is not ready for exact restore')
            if self._job is not None or not self._room(entry.size, protected=entry):
                return None
            entry.cpu_charge = entry.size
            try:
                entry.loading = torch.empty(entry.size, dtype=torch.uint8, pin_memory=entry.pinned)
                self._record_peak()
                job = self._executor.submit(self._read, entry)
            except BaseException:
                entry.loading = None
                entry.cpu_charge = 0
                raise
            entry.state = 'loading'
            self._job = entry, 'read', job
            return None

    def restore_complete(self, entry):
        """Caller has observed the GPU H2D completion; release staging safely."""
        with self._lock:
            self._check(entry)
            if entry.state != 'restoring':
                raise ValueError('Checkpoint restore has not acquired its staging owner')
            if entry.disk is not None:
                entry.snapshot = None
                entry.cpu_charge = 0
                entry.state = 'disk'
            else:
                entry.state = 'resident'

    def drop(self, entry):
        """Cancellation drains I/O/capture; an active GPU restore cannot drop."""
        with self._lock:
            self._finish_io(suppress_error=True)
            self._check(entry)
            if entry.state in ('spilling', 'loading'):
                return False
            if entry.state == 'restoring':
                raise ValueError('Cannot drop an in-flight GPU restore')
            if entry.snapshot is not None and not entry.snapshot.query():
                return False
            if entry.disk is not None:
                self._close_file(entry.disk.fd, entry.disk.size)
                entry.disk = None
            entry.snapshot = entry.loading = None
            entry.cpu_charge = 0
            entry.state = 'dropped'
            del self.entries[entry.identity]
            return True

    def snapshot(self):
        with self._lock:
            return dict(cpu_capacity_bytes=self.cpu_capacity, ssd_capacity_bytes=self.ssd_capacity,
                cpu_resident_bytes=sum(e.size for e in self.entries.values()
                    if e.snapshot is not None or e.loading is not None),
                cpu_promised_bytes=sum(e.cpu_charge for e in self.entries.values() if e.state == 'reserved'),
                cpu_charged_bytes=self._cpu_usage(), cpu_peak_bytes=self.cpu_peak_bytes,
                cpu_charge_peak_bytes=self.cpu_charge_peak_bytes,
                logical_bytes=sum(e.size for e in self.entries.values()),
                ssd_used_bytes=sum(e.disk.size for e in self.entries.values() if e.disk is not None),
                ssd_pending_write_bytes=(self._job[0].size
                    if self._job is not None and self._job[1] == 'write' else 0),
                ssd_peak_bytes=self.ssd_peak_bytes, disk_write_bytes=self.disk_write_bytes,
                disk_read_bytes=self.disk_read_bytes, spills=self.spills, loads=self.loads,
                io_pending=int(self._job is not None), io_failures=self.io_failures,
                close_error=str(self._close_error) if self._close_error is not None else None,
                ssd_close_uncertain_bytes=self._uncertain_close[1] if self._uncertain_close is not None else 0,
                cpu_deferrals=self.cpu_deferrals, entries=len(self.entries))

    def poll(self):
        with self._lock:
            self._finish_io()
            return self._job is None

    def close(self):
        with self._lock:
            self._finish_io()
            if self.entries or self._job is not None:
                return False
            self.closed = True
        self._executor.shutdown(wait=True)
        return True
