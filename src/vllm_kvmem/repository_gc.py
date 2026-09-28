"""CPU-only native GC ownership and deferred index invalidation.

The release callback owns only storage and a bounded tombstone queue. It never
captures the authority, Store or GC bridge, and never mutates the foreground
index. A released generation is removed from the index before the next append;
snapshot leases protect scoring through selection, not just the archive read.
"""
from queue import Empty, Queue

from . import _native


class AuthorityGc:
    def __init__(self, memory, archive, config):
        tombstones = Queue(maxsize=memory.config.repository_blocks)
        namespace = memory.request_id

        def release(record):
            if not archive.reclaim(namespace, record):
                return False
            tombstones.put_nowait((record.handle.slot, record.handle.generation))
            return True

        self.tombstones = tombstones
        self.worker = _native.RepositoryGc(config, memory.repository, release)
        self.reclaimed_records = 0
        self.stopped = False
        memory._store.attach_gc(self.worker)
        # One request step, not a manufactured step per prefill chunk/token.
        memory._store.gc_begin_step()

    def drain(self, records):
        while True:
            try:
                key = self.tombstones.get_nowait()
            except Empty:
                break
            record = records.pop(key, None)
            if record is None or not record.archived:
                raise RuntimeError('GC released a missing or unpublished authority generation')
            self.reclaimed_records += 1
        if self.worker.last_error:
            raise RuntimeError(f'Authority GC release failed: {self.worker.last_error}')

    def stop(self):
        if not self.stopped:
            self.worker.stop()
            self.stopped = True
