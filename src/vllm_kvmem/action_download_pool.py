"""Bounded pinned downloads: recycle only after DMA and every CPU alias die."""
from dataclasses import dataclass
from threading import RLock
import numpy as np
import torch


class _Array(np.ndarray):
    def __array_finalize__(self, source):
        self._lease = getattr(source, '_lease', None)


@dataclass
class _Entry:
    backing: object
    event: object
    capacity: int
    active: bool = False
    cpu_done: bool = True
    recorded: bool = False
    poisoned: bool = False


class _Lease:
    def __init__(self, pool, entry):
        self.pool, self.entry = pool, entry

    def __del__(self):
        with self.pool.lock:
            self.entry.cpu_done = True


class _Event:
    """All consumers of the fence retain the same storage lease."""
    def __init__(self, lease):
        self._lease = lease

    def query(self):
        return self._lease.entry.event.query()

    def synchronize(self):
        self._lease.entry.event.synchronize()


class TorchDownloads:
    @staticmethod
    def allocate(size):
        return torch.empty(size, dtype=torch.uint8, pin_memory=True)

    @staticmethod
    def event():
        return torch.cuda.Event()

    @staticmethod
    def record(event, source):
        event.record(torch.cuda.current_stream(source.device))


class ActionDownloadPool:
    def __init__(self, max_bytes=67108864, max_slots=128, *, backend=None):
        if (type(max_bytes) is not int or max_bytes < 256 or
                type(max_slots) is not int or max_slots < 1):
            raise ValueError('Action download pool requires positive bounded capacity')
        self.max_bytes,self.max_slots = max_bytes,max_slots
        self.backend = backend or TorchDownloads()
        self.entries = []
        self.lock = RLock()
        self.allocations = self.reuses = self.exhaustions = 0
        self.peak_active_bytes = 0

    def _poll(self):
        for entry in self.entries:
            if entry.active and entry.cpu_done and entry.recorded and not entry.poisoned:
                try:
                    if entry.event.query():
                        entry.active = False
                except BaseException:
                    entry.poisoned = True
                    raise

    def copy(self, source):
        """Never waits for another owner and never allocates past either cap.

        A NumPy buffer owner is retained by Torch's from_numpy storage deleter.
        Torch views, detach(), and numpy() aliases therefore keep the lease,
        including after the original packet and tensor objects are destroyed.
        """
        dtypes={torch.float32:np.float32,torch.float64:np.float64,torch.int64:np.int64,
                torch.int32:np.int32,torch.float16:np.float16,torch.uint8:np.uint8}
        if source.dtype not in dtypes or source.numel()==0:
            raise ValueError('Unsupported action download dtype or empty shape')
        size=source.numel()*source.element_size()
        capacity=max(256,1 << (size-1).bit_length())
        with self.lock:
            self._poll()
            available=[e for e in self.entries if not e.active and e.capacity>=size]
            if available:
                entry=min(available,key=lambda e:e.capacity);self.reuses+=1
            else:
                # Completed idle slots may be evicted without affecting another
                # owner's in-flight DMA or retained CPU references.
                for idle in sorted((e for e in self.entries if not e.active),key=lambda e:e.capacity):
                    if len(self.entries)<self.max_slots and sum(e.capacity for e in self.entries)+capacity<=self.max_bytes:
                        break
                    self.entries.remove(idle)
                if (len(self.entries)>=self.max_slots or
                        sum(e.capacity for e in self.entries)+capacity>self.max_bytes):
                    self.exhaustions+=1
                    raise BufferError('Action download pool capacity exhausted; no batch wait or unbudgeted allocation')
                entry=_Entry(self.backend.allocate(capacity),self.backend.event(),capacity)
                self.entries.append(entry);self.allocations+=1
            entry.active,entry.cpu_done,entry.recorded=True,False,False
            lease=_Lease(self,entry)
            array=entry.backing.numpy()[:size].view(dtypes[source.dtype]).view(_Array)
            array._lease=lease
            host=torch.from_numpy(array).reshape(source.shape)
            event=_Event(lease)
            self.peak_active_bytes=max(self.peak_active_bytes,sum(e.capacity for e in self.entries if e.active))
        try:
            host.copy_(source,non_blocking=True)
        finally:
            # A failing copy may have submitted DMA. Fence it before letting
            # aliases disappear; an unrecordable fence quarantines this slot.
            try:
                self.backend.record(entry.event,source)
                entry.recorded=True
            except BaseException:
                entry.poisoned=True
                raise
        return host,event

    def snapshot(self):
        with self.lock:
            return dict(allocations=self.allocations,reuses=self.reuses,exhaustions=self.exhaustions,
                allocated_pinned_bytes=sum(e.capacity for e in self.entries),
                active_pinned_bytes=sum(e.capacity for e in self.entries if e.active),
                peak_active_pinned_bytes=self.peak_active_bytes,event_count=len(self.entries),
                max_pinned_bytes=self.max_bytes,max_events=self.max_slots,
                quarantined_slots=sum(e.poisoned for e in self.entries))
