"""CPU-only native GC ownership and deferred index invalidation.

The release callback owns only storage and a bounded tombstone queue. It never
captures the authority, Store or GC bridge, and never mutates the foreground
index. A released generation is removed from the index before the next append;
snapshot leases protect scoring through selection, not just the archive read.
"""
from queue import Empty, Queue
from threading import Event

from . import _native


def serving_gc_config(memory, max_query_tokens):
    """Original qw3 reserve policy, sized for the native admitted batch bound.

    The reserve must coexist with K+G; a smaller repository cannot become safe
    by evicting working/checkpoint/IO leases. Ingest grace counts accepted tokens,
    never scheduler iterations or draft candidates.
    """
    if type(max_query_tokens) is not int or max_query_tokens <= 0:
        raise ValueError('GC requires a positive native query bound')
    append_blocks = (max_query_tokens + memory.block_tokens - 1) // memory.block_tokens + 1
    config = _native.RepositoryGcConfig()
    config.ready_low = max(memory.repository_blocks // 20, append_blocks * 2)
    config.ready_high = max(memory.repository_blocks // 10, config.ready_low + append_blocks)
    protected = (memory.select_budget + memory.gen_budget) // memory.block_tokens
    if protected + config.ready_high >= memory.repository_blocks:
        raise ValueError('Repository must exceed K+G plus the GC ready reserve')
    config.recent_tokens = memory.recent_blocks * memory.block_tokens
    config.candidate_ingest_tokens = config.ready_high * memory.block_tokens
    config.background = True
    return config


class AuthorityGc:
    def __init__(self, memory, archive, config, progress=None):
        tombstones = Queue(maxsize=memory.config.repository_blocks)
        namespace = memory.request_id
        progress = progress if progress is not None else Event()

        def release(record):
            if not archive.reclaim(namespace, record):
                return False
            tombstones.put_nowait((record.handle.slot, record.handle.generation))
            progress.set()
            return True

        self.tombstones = tombstones
        self.progress = progress
        self.worker = _native.RepositoryGc(config, memory.repository, release)
        self.reclaimed_records = 0
        self.stopped = False
        memory._store.attach_gc(self.worker)
        # One request step, not a manufactured step per prefill chunk/token.
        memory._store.gc_begin_step()

    def drain(self, records, discarded=None):
        discarded = {} if discarded is None else discarded
        while True:
            try:
                key = self.tombstones.get_nowait()
            except Empty:
                break
            if discarded.pop(key, None) is not None:
                # A rewind suffix is no longer accepted history. It can include
                # an unpublished mutable tail; do not count it as a GC hole in
                # the retained branch or demand a durable complete record.
                continue
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
