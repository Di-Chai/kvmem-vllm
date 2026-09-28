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
        self._lock = RLock()
        self.closed = False

    def poll(self) -> int:
        with self._lock:
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
            if ready is None:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
            # Serialize copy-stream submission with other producers. The
            # failure fence must run before the preparing slot is released.
            with self._lock:
                try:
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(ready)
                        destination.flatten().copy_(staging, non_blocking=True)
                        destination.record_stream(self.stream)
                        event = torch.cuda.Event()
                        event.record(self.stream)
                except BaseException:
                    self.stream.synchronize()
                    raise
                txn = Transfer(event, (destination, staging, ready, filled, *retained), slot)
                self.pending[slot] = txn
                return txn
        finally:
            with self._lock:
                self.preparing.remove(slot)

    def close(self) -> bool:
        with self._lock:
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
        self.pending = {}
        self.closed = False

    def download(self, payload, source_owners=(), ready=None):
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
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            staging.copy_(payload.flatten(), non_blocking=True)
            payload.record_stream(self.stream)
            event = torch.cuda.Event()
            event.record(self.stream)
        result = Download(event, staging, (payload, ready, *retained), slot)
        self.pending[slot] = result
        return result

    def release(self, download):
        if self.pending.get(download.slot) is not download:
            raise ValueError("Stale download slot release")
        if not download.query():
            return False
        del self.pending[download.slot]
        download.owners = ()
        return True

    def close(self):
        if self.pending:
            return False
        self.closed = True
        self.buffers.clear()
        return True
