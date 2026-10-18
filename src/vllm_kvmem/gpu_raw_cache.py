"""Bounded, non-authoritative raw-K cache shared by active requests.

Keys include the CPU authority namespace and repository generation. Eviction
never changes history or selection. All GPU reads/writes are serialized by the
runner's assembly producer and a CUDA fence, including across execution streams.
"""
from collections import OrderedDict

import torch


def retire_raw_cache(runner, authority):
    """Drop an authority after real retirement/eviction, not a tool wait.

    Cache entries pin no CPU pages or active GPU allocation. Idle retained
    sessions may keep entries within the existing shared byte/LRU quotas.
    The background index must be changed by its own sole producer.
    """
    cache = getattr(runner, '_kvmem_gpu_raw_cache', None)
    if cache is not None:
        cache.drop_namespace(authority.memory.request_id)
    refresh = getattr(runner, '_kvmem_async_refresh', None)
    if refresh is not None:
        refresh.retire_namespace(authority.memory.request_id)


class RawCacheIndex:
    def __init__(self, slots, namespace_slots):
        if min(slots, namespace_slots) <= 0 or namespace_slots > slots:
            raise ValueError('Raw cache needs a bounded positive namespace quota')
        self.slots, self.namespace_slots = slots, namespace_slots
        self.entries = OrderedDict()
        self.free = list(range(slots-1, -1, -1))
        self.counts = {}
        self.evictions = 0

    @staticmethod
    def validate(key):
        if (not isinstance(key, tuple) or len(key) != 3 or not isinstance(key[0], str) or
                not key[0] or any(type(x) is not int or x < 0 for x in key[1:])):
            raise ValueError('Raw cache key requires namespace, slot and generation')

    def get(self, key):
        self.validate(key)
        slot = self.entries.get(key)
        if slot is not None:
            self.entries.move_to_end(key)
        return slot

    def remove(self, key):
        slot = self.entries.pop(key, None)
        if slot is not None:
            self.free.append(slot)
            self.counts[key[0]] -= 1
            if not self.counts[key[0]]:
                del self.counts[key[0]]

    def reserve(self, key, protected=()):
        self.validate(key)
        found = self.get(key)
        if found is not None:
            return found
        if self.counts.get(key[0], 0) >= self.namespace_slots:
            victim = next((k for k in self.entries if k[0] == key[0] and k not in protected), None)
            if victim is None:
                return None
            self.remove(victim)
            self.evictions += 1
        if not self.free:
            victim = next((k for k in self.entries if k not in protected), None)
            if victim is None:
                return None
            self.remove(victim)
            self.evictions += 1
        slot = self.free.pop()
        self.entries[key] = slot
        self.counts[key[0]] = self.counts.get(key[0], 0) + 1
        return slot

    def drop_namespace(self, namespace):
        for key in list(self.entries):
            if key[0] == namespace:
                self.remove(key)


class GpuRawCache:
    def __init__(self, budget_bytes, shape, device, max_requests, window_blocks, *, dtype=torch.bfloat16):
        self.shape = tuple(shape)
        self.device = device
        if dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            raise ValueError('Unsupported immutable raw-K cache dtype')
        self.dtype = dtype
        self.record_bytes = dtype.itemsize * torch.Size(shape).numel()
        slots = min(budget_bytes // self.record_bytes, max_requests * window_blocks)
        quota = min(window_blocks, slots // max_requests)
        if not quota:
            raise ValueError('GPU raw-K budget cannot fit one record per active request')
        self.index = RawCacheIndex(slots, quota)
        self.capacity_bytes = slots * self.record_bytes
        self.tensor = None
        self.fence = None

    def begin(self):
        stream = torch.cuda.current_stream(self.device)
        if self.fence is not None:
            stream.wait_event(self.fence)
        if self.tensor is None:
            # This opt-in allocation is outside native KV pages. Do not silently
            # consume their reservation or exceed the caller's explicit budget.
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.capacity_bytes + 1024**3 > free:
                raise BufferError('Insufficient free GPU memory for explicit raw-K cache budget')
            self.tensor = torch.empty((self.index.slots, *self.shape),
                                      dtype=self.dtype, device=self.device)

    def finish(self, event):
        self.fence = event

    def drop_namespace(self, namespace):
        self.index.drop_namespace(namespace)
