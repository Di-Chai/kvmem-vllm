"""Bounded pinned staging slots with completion-owned source lifetimes.

This copies packed payloads only; allocator and KV layout remain engine-owned.
A producer must retain source_owners until poll() observes actual completion.
"""
from dataclasses import dataclass
from threading import RLock

import torch
from . import _native


@dataclass
class Transfer:
    event: torch.cuda.Event
    owners: tuple
    slot: int

    def query(self):
        return self.event.query()


@dataclass(eq=False)
class PreparedFill:
    """Exclusive pinned storage until its CPU producer and subsequent DMA drain."""
    slot: int
    payload: torch.Tensor
    job: object = None
    filled: object = None
    state: str = 'filling'
    owners: tuple = ()


class PackedTransfers:
    def __init__(self, slot_bytes: int, slots: int, device: torch.device):
        if slot_bytes <= 0 or slots <= 0 or device.type != "cuda":
            raise ValueError("Require positive bounded staging capacity on CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = [torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True)
                        for _ in range(slots)]
        self.pending: dict[int, Transfer] = {}
        self.preparing = set()
        self._prepared = {}
        self._lock = RLock()
        self.closed = False
        self.poisoned = False
        self.bandwidth_limiter = None

    def poll(self) -> int:
        with self._lock:
            if self.poisoned:
                return 0
            finished = [slot for slot, txn in self.pending.items() if txn.query()]
            for slot in finished:
                del self.pending[slot]
            return len(finished)

    def upload(self, payload: torch.Tensor, destination: torch.Tensor, source_owners=(),
               ready: torch.cuda.Event | None = None) -> Transfer | None:
        if payload.device.type != "cpu" or payload.dtype != torch.uint8 or not payload.is_contiguous():
            raise ValueError("Payload must be contiguous CPU bytes")
        return self.upload_from_reader(payload.numel(), destination,
                                       lambda staging: staging.copy_(payload.flatten()),
                                       source_owners=source_owners, ready=ready)

    def upload_from_reader(self, size, destination, fill, source_owners=(), ready=None):
        """Fill an exclusive pinned slot synchronously, then submit its H2D copy.

        The callback must finish all CPU writes before returning (including on
        failure). Busy slots return None without invoking it. Both CPU I/O and
        DMA retain the slot; close cannot release either operation's storage.
        """
        if destination.device != self.device or destination.dtype != torch.uint8 or not destination.is_contiguous():
            raise ValueError("Destination must be contiguous same-device CUDA bytes")
        with self._lock:
            if self.closed:
                raise RuntimeError("Transfer pool is closed")
            if self.poisoned:
                raise RuntimeError('Transfer pool has an unfenced failed DMA')
            if size != destination.numel() or not 0 < size <= self.buffers[0].numel():
                raise ValueError("Payload extent exceeds destination or bounded staging capacity")
            self.poll()
            slot = next((i for i in range(len(self.buffers))
                         if i not in self.pending and i not in self.preparing), None)
            if slot is None:
                return None
            self.preparing.add(slot)
        try:
            retained = tuple(owner.clone() if isinstance(owner, _native.Lease) else owner
                             for owner in source_owners)
            staging = self.buffers[slot][:size]
            filled = fill(staging)
            if self.bandwidth_limiter is not None:
                self.bandwidth_limiter.acquire(size)
            if ready is None:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
            # Serialize copy-stream submission with other producers. The
            # failure fence must run before the preparing slot is released.
            with self._lock:
                if self.poisoned:
                    raise RuntimeError('Transfer pool has an unfenced failed DMA')
                try:
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(ready)
                        destination.flatten().copy_(staging, non_blocking=True)
                        destination.record_stream(self.stream)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                except BaseException as error:
                    try:
                        self.stream.synchronize()
                    except BaseException as fence_error:
                        # A failed failure-fence grants no right to recycle
                        # either endpoint. Quarantine this entire scratch pool.
                        self.poisoned = True
                        self._prepared[slot] = PreparedFill(slot, staging, filled=filled,
                            state='poisoned', owners=(destination, ready, *retained))
                        error.add_note('DMA failure fence also failed: '+repr(fence_error))
                    raise
                txn = Transfer(event, (destination, staging, ready, filled, *retained), slot)
                self.pending[slot] = txn
                return txn
        finally:
            with self._lock:
                if slot not in self._prepared:
                    self.preparing.remove(slot)

    def prepare_from_reader(self, size, begin):
        """Try to reserve one existing slot and start a nonblocking CPU read.

        ``begin`` returns a completion-owned job, or None on admission pressure.
        If it raises, it must have drained any writes it started. The job's
        result() must finish every CPU write even when it raises. No CUDA call
        is made by the CPU producer, and no additional staging is allocated.
        """
        with self._lock:
            if self.closed:
                raise RuntimeError('Transfer pool is closed')
            if self.poisoned:
                raise RuntimeError('Transfer pool has an unfenced failed DMA')
            if not 0 < size <= self.buffers[0].numel():
                raise ValueError('Payload exceeds bounded staging capacity')
            self.poll()
            slot = next((i for i in range(len(self.buffers))
                         if i not in self.pending and i not in self.preparing), None)
            if slot is None:
                return None
            ticket = PreparedFill(slot, self.buffers[slot][:size])
            self.preparing.add(slot)
            self._prepared[slot] = ticket
        try:
            ticket.job = begin(ticket.payload)
            if ticket.job is not None:
                return ticket
        except BaseException:
            self._release_prepared(ticket)
            raise
        self._release_prepared(ticket)
        return None

    def _take_prepared(self, ticket):
        with self._lock:
            if self._prepared.get(ticket.slot) is not ticket or ticket.state != 'filling':
                raise ValueError('Stale or concurrently consumed prepared fill')
            ticket.state = 'consuming'

    def _release_prepared(self, ticket):
        with self._lock:
            if self._prepared.get(ticket.slot) is not ticket:
                raise ValueError('Stale prepared fill release')
            del self._prepared[ticket.slot]
            self.preparing.remove(ticket.slot)
            ticket.state = 'retired'

    def discard_prepared(self, ticket):
        """Drain only this CPU job before making its pinned slot reusable."""
        self._take_prepared(ticket)
        try:
            ticket.filled = ticket.job.result()
        finally:
            self._release_prepared(ticket)

    def upload_prepared(self, ticket, destination, *, ready=None):
        """Finish this fill, then transfer ownership to the original DMA fence."""
        self._take_prepared(ticket)
        try:
            # Always finish the CPU producer before any failure can release its
            # slot, including a malformed destination or a failed DMA enqueue.
            ticket.filled = ticket.job.result()
            staging = ticket.payload
            size = staging.numel()
            if self.poisoned:
                raise RuntimeError('Transfer pool has an unfenced failed DMA')
            if (destination.device != self.device or destination.dtype != torch.uint8 or
                    not destination.is_contiguous() or destination.numel() != size):
                raise ValueError('Destination must be matching contiguous same-device CUDA bytes')
            if self.bandwidth_limiter is not None:
                self.bandwidth_limiter.acquire(size)
            if ready is None:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
            with self._lock:
                if self.poisoned:
                    raise RuntimeError('Transfer pool has an unfenced failed DMA')
                try:
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(ready)
                        destination.flatten().copy_(staging, non_blocking=True)
                        destination.record_stream(self.stream)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                except BaseException as error:
                    try:
                        self.stream.synchronize()
                    except BaseException as fence_error:
                        self.poisoned = True
                        ticket.state = 'poisoned'
                        ticket.owners = (destination, ready, ticket.filled)
                        error.add_note('DMA failure fence also failed: '+repr(fence_error))
                    raise
                transfer = Transfer(event, (destination, staging, ready, ticket.filled), ticket.slot)
                self.pending[ticket.slot] = transfer
                return transfer
        finally:
            if ticket.state != 'poisoned':
                self._release_prepared(ticket)

    def close(self) -> bool:
        with self._lock:
            if self.poisoned:
                return False
            self.poll()
            if self.pending or self.preparing:
                return False
            self.closed = True
            self.buffers.clear()
            return True


@dataclass
class Download:
    event: torch.cuda.Event
    payload: torch.Tensor
    owners: tuple
    slot: int

    def query(self):
        return self.event.query()


class PackedDownloads:
    """D2H slots remain owned until CPU consumption explicitly releases them."""
    def __init__(self, slot_bytes: int, slots: int, device: torch.device):
        if min(slot_bytes, slots) <= 0 or device.type != "cuda":
            raise ValueError("Require bounded pinned D2H storage on CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        self.stream = torch.cuda.Stream(device=self.device)
        self.buffers = [torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(slots)]
        self.capacity_bytes = slot_bytes * slots
        self.pending = {}
        self.closed = False
        self._lock = RLock()

    def download(self, payload, source_owners=(), ready=None):
        with self._lock:
            return self._download_locked(payload, source_owners, ready)

    def _download_locked(self, payload, source_owners=(), ready=None):
        if self.closed:
            raise RuntimeError("Download pool is closed")
        if (payload.device != self.device or payload.dtype != torch.uint8 or
                not payload.is_contiguous() or payload.numel() > self.buffers[0].numel()):
            raise ValueError("Download requires contiguous same-device bytes within slot capacity")
        slot = next((i for i in range(len(self.buffers)) if i not in self.pending), None)
        if slot is None:
            return None
        if ready is None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        retained = tuple(owner.clone() if isinstance(owner, _native.Lease) else owner
                         for owner in source_owners)
        staging = self.buffers[slot][:payload.numel()]
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                staging.copy_(payload.flatten(), non_blocking=True)
                payload.record_stream(self.stream)
                event = torch.cuda.Event()
                event.record(self.stream)
        except BaseException:
            # An enqueue failure must not let the producer reuse a pinned slot
            # while an earlier copy still writes it. This is failure-only.
            self.stream.synchronize()
            raise
        result = Download(event, staging, (payload, ready, *retained), slot)
        self.pending[slot] = result
        return result

    def release(self, download):
        with self._lock:
            if self.pending.get(download.slot) is not download:
                raise ValueError("Stale download slot release")
            if not download.query():
                return False
            del self.pending[download.slot]
            download.owners = ()
            return True

    def close(self):
        with self._lock:
            if self.pending:
                return False
            self.closed = True
            self.buffers.clear()
            return True
