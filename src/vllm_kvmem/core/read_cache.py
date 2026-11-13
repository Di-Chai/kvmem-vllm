"""Bounded non-authoritative immutable archive read cache.

Eviction affects performance only. Archive/repository generations remain the
source of truth. Each entry owns one bytes object so a small surviving view
cannot keep an unaccounted whole batch alive. Integration must remove entries
when the corresponding generation is reclaimed and clear on archive close.
"""
from collections import OrderedDict
from threading import RLock


class RecordReadCache:
    def __init__(self, capacity_bytes, namespace_bytes):
        if any(type(x) is not int or x < 0 for x in (capacity_bytes, namespace_bytes)):
            raise ValueError("Read-cache quotas must be nonnegative byte counts")
        self.capacity_bytes = capacity_bytes
        self.namespace_bytes = min(namespace_bytes, capacity_bytes)
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.by_namespace = {}
        self.hits = self.misses = self.evictions = 0
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            value = self.entries.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
                self.entries.move_to_end(key)
            return value

    def put(self, key, payload):
        if (not isinstance(key, tuple) or len(key) != 3 or not isinstance(key[0], str) or not key[0] or
                type(key[1]) is not int or type(key[2]) is not int or min(key[1:]) < 0):
            raise ValueError("Cache key requires namespace, slot and generation")
        if type(payload) is not bytes or not payload:
            raise ValueError("Cache entry must own immutable bytes, not a backing-buffer view")
        size = len(payload)
        with self.lock:
            previous = self.entries.get(key)
            if previous is not None:
                if previous != payload:
                    raise ValueError("Cannot replace immutable data within one generation")
                self.entries.move_to_end(key)
                return True
            if size > self.namespace_bytes:
                return False
            namespace = key[0]
            while self.by_namespace.get(namespace, 0) + size > self.namespace_bytes:
                victim = next(k for k in self.entries if k[0] == namespace)
                self.remove(victim)
                self.evictions += 1
            while self.used_bytes + size > self.capacity_bytes:
                self.remove(next(iter(self.entries)))
                self.evictions += 1
            self.entries[key] = payload
            self.used_bytes += size
            self.by_namespace[namespace] = self.by_namespace.get(namespace, 0) + size
            return True

    def remove(self, key):
        with self.lock:
            value = self.entries.pop(key, None)
            if value is not None:
                self.used_bytes -= len(value)
                namespace = key[0]
                self.by_namespace[namespace] -= len(value)
                if not self.by_namespace[namespace]:
                    del self.by_namespace[namespace]

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.by_namespace.clear()
            self.used_bytes = 0
