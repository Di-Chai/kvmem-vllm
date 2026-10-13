"""Branch rollback uses the real native repository, archive and GC worker."""
from threading import Event as ThreadEvent, get_ident
import time

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, _Ready
from vllm_kvmem.cached_storage import CachedArchive
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.repository_gc import serving_gc_config
from vllm_kvmem.storage import AsyncArchive
from test_authority import setup, payload, drain
from test_core import Event


def complete_rewind(authority):
    deadline = time.monotonic() + 5
    while not authority.poll_rewind():
        assert time.monotonic() < deadline
        time.sleep(.001)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('async_seal', [False, True])
def test_rewind_recycles_only_discarded_generations_and_preserves_shared_namespace(tmp_path, cached, async_seal):
    archive = (CachedArchive(str(tmp_path), 48, 16, cache_bytes=768, namespace_bytes=384)
               if cached else AsyncArchive(str(tmp_path), 48, 16))
    archive.async_seal = async_seal
    a, _ = setup(tmp_path, 'a', archive)
    b, _ = setup(tmp_path, 'b', archive)
    assert a.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    assert b.append(*payload(50, 8), history_begin=0, local_rope_begin=0)
    drain(a); drain(b)
    checkpoint = a.memory.checkpoint()
    prefix = next(iter(a.records.values())).handle
    other = [r.handle for r in b.records.values()]
    seen = set()
    for branch in range(12):
        assert a.append(*payload(10 + branch, 7), history_begin=4, local_rope_begin=4)
        drain(a)
        old = [r.handle for r in a.records.values() if r.history_begin >= 4]
        assert not any((h.slot, h.generation) in seen for h in old)
        seen.update((h.slot, h.generation) for h in old)
        spans = a.begin_rewind(checkpoint)
        assert sum(s[2] for s in spans) == 7
        event = Event()
        a.attach_rewind_completion(event)
        assert not a.poll_rewind() and a.history_end == 11
        with pytest.raises(RuntimeError, match='rewind'):
            a.append(*payload(11, 1), history_begin=11, local_rope_begin=11)
        with pytest.raises(RuntimeError, match='rewind'):
            a.mean_keys()
        event.done = True
        complete_rewind(a)
        assert a.history_end == 4 and len(a.records) == 1 and not a.discarded_records
        assert all(not a.memory.repository.pin(h) for h in old)
        assert torch.equal(a.read_record(prefix)[0], payload(0, 4)[0])
        assert torch.equal(b.read_records(other)[1][0], payload(54, 4)[0])
        assert archive.tier.used_slots == 3
    del checkpoint
    assert a.close() and b.close() and archive.close()


def test_rewind_publication_and_late_read_each_retain_ownership(tmp_path):
    a, archive = setup(tmp_path)
    assert a.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    drain(a)
    checkpoint = a.memory.checkpoint()
    tier = archive.tier
    started, proceed = ThreadEvent(), ThreadEvent()

    class Delayed:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            started.set()
            assert proceed.wait(5)
            return tier.io(*args)

    archive.tier = Delayed()
    try:
        assert a.append(*payload(4, 4), history_begin=4, local_rope_begin=4)
        assert started.wait(5)
        assert a.begin_rewind(checkpoint) is None
        assert a.history_end == 8
    finally:
        proceed.set()
    drain(a)
    suffix = next(r for r in a.records.values() if r.history_begin == 4).handle
    lease = a.memory.repository.pin(suffix)
    started.clear(); proceed.clear()
    job = archive.read('a', [lease])
    lease.reset()
    assert started.wait(5)
    try:
        assert a.begin_rewind(checkpoint)
        a.attach_rewind_completion(_Ready())
        assert not a.poll_rewind() and a.history_end == 4
        assert len(a.records) == 1 and len(a.discarded_records) == 1
        assert archive.tier.used_slots == 2
    finally:
        proceed.set()
    assert job.result(5)[0]
    complete_rewind(a)
    assert archive.tier.used_slots == 1
    del checkpoint
    assert a.close() and archive.close()


def test_rewind_with_background_gc_preserves_branch_accounting_and_can_continue(tmp_path):
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
                        repository_blocks=24, sink_blocks=1, recent_blocks=0)
    archive = AsyncArchive(str(tmp_path), 48, 24)
    authority = RequestAuthority('gc', config, archive, 2, 1, 2,
        resident_records=8, gc_config=serving_gc_config(config, 8))
    foreground = get_ident()
    original_reclaim = archive.reclaim
    require_background = False
    background_releases = []

    def release(namespace, record):
        if require_background and get_ident() == foreground:
            return False  # Real GC must finish these generation-bound bundles.
        done = original_reclaim(namespace, record)
        if done and require_background:
            background_releases.append((record.handle.slot, record.handle.generation))
        return done

    archive.reclaim = release
    checkpoint = None
    def select():
        plan = authority.select_for_queries(torch.ones(2, 1, 2, 2))
        authority.memory.attach_completion(_Ready())
        assert authority.memory.commit_selection()

    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        drain(authority)
        for _ in range(40):
            select()
            deadline = time.monotonic() + 5
            while not authority.append(*payload(authority.history_end, 8),
                    history_begin=authority.history_end, local_rope_begin=8):
                authority.poll()
                assert time.monotonic() < deadline
                time.sleep(.001)
            drain(authority)
        assert authority.gc_reclaimed_records > 50
        checkpoint = authority.memory.checkpoint()
        position = authority.history_end
        require_background = True
        for _ in range(20):
            select()
            assert authority.append(*payload(position, 7), history_begin=position, local_rope_begin=8)
            drain(authority)
            assert authority.begin_rewind(checkpoint)
            authority.attach_rewind_completion(_Ready())
            complete_rewind(authority)
            assert authority.history_end == position
            assert authority.gc_reclaimed_records + len(authority.records) == position // 4
            assert len(authority.records) <= 24 and archive.tier.used_slots <= 24
            assert not authority.gc.stopped and not authority.gc.worker.last_error
        assert len(background_releases) >= 40  # Sealed and unpublished partial suffixes.
    finally:
        require_background = False
        checkpoint = None
        assert authority.close()
        assert archive.tier.used_slots == 0 and archive.close()


def test_close_during_rewind_preserves_fence_and_checkpoint_ownership(tmp_path):
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    drain(authority)
    checkpoint = authority.memory.checkpoint()
    assert authority.append(*payload(4, 7), history_begin=4, local_rope_begin=4)
    drain(authority)
    assert authority.begin_rewind(checkpoint)
    fence = Event()
    authority.attach_rewind_completion(fence)
    assert not authority.close() and archive.tier.used_slots == 2
    fence.done = True
    assert not authority.close()  # Checkpoint still owns the unchanged prefix.
    del checkpoint
    assert authority.close() and archive.tier.used_slots == 0
    assert archive.close()


def test_partial_suffix_rewind_restores_single_record_staging_budget(tmp_path):
    authority, archive = setup(tmp_path, resident=1)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    drain(authority)
    checkpoint = authority.memory.checkpoint()
    generations = set()
    for branch in range(12):
        assert authority.append(*payload(10 + branch, 3), history_begin=4, local_rope_begin=4)
        tail = next(r for r in authority.records.values() if r.history_begin == 4)
        key = (tail.handle.slot, tail.handle.generation)
        assert key not in generations
        generations.add(key)
        lease = authority.memory.repository.pin(tail.handle)
        assert authority.begin_rewind(checkpoint)
        authority.attach_rewind_completion(_Ready())
        assert not authority.poll_rewind()
        assert authority.history_end == 4 and len(authority.discarded_records) == 1
        lease.reset()
        complete_rewind(authority)
        assert not authority.discarded_records
    del checkpoint
    # The same single staging reservation can now seal and publish a new page.
    assert authority.append(*payload(100, 4), history_begin=4, local_rope_begin=4)
    drain(authority)
    assert authority.close() and archive.close()
