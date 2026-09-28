"""Native policy plus actual authority/archive generation recycling (CPU only)."""
import time

import pytest

torch = pytest.importorskip('torch')
from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority, _Ready
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive
from test_authority import payload, drain


def setup(tmp_path, background=False, cached=False):
    memory = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
                        repository_blocks=8, sink_blocks=1, recent_blocks=0)
    if cached:
        from vllm_kvmem.cached_storage import CachedArchive
        archive = CachedArchive(str(tmp_path), 48, 8, cache_bytes=384, namespace_bytes=384)
    else:
        archive = AsyncArchive(str(tmp_path), 48, 8)
    gc = n.RepositoryGcConfig()
    gc.ready_low = 3
    gc.ready_high = 4
    gc.policy.scan_blocks = gc.policy.batch_blocks = 8
    gc.policy.poll_ms = 1
    gc.policy.grace_steps = 1
    gc.policy.candidate_steps = 1
    gc.candidate_ingest_tokens = 4
    gc.background = background
    gc.tick_budget_us = 100000
    authority = RequestAuthority('a', memory, archive, 2, 1, 2,
                                 resident_records=8, gc_config=gc)
    return authority, archive


def select(authority):
    plan = authority.select_for_queries(torch.ones(2, 1, 2, 2))
    authority.memory.attach_completion(_Ready())
    assert authority.memory.commit_selection()
    return plan


def advance(worker):
    # Explicit request epochs for deterministic policy tests. Serving does not
    # manufacture epochs per token; the ingest-grace test below uses one epoch.
    for _ in range(8):
        worker.begin_step()
        worker.poll()


@pytest.mark.parametrize('cached', [False, True])
def test_gc_reuses_archive_slots_and_removes_old_index_generations(tmp_path, cached):
    authority, archive = setup(tmp_path, cached=cached)
    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        drain(authority)
        sink = next(iter(authority.records.values())).handle
        generations = set()
        for _ in range(32):
            plan = select(authority)
            del plan
            advance(authority.gc.worker)
            authority.poll()
            assert authority.append(*payload(authority.history_end, 8),
                                    history_begin=authority.history_end, local_rope_begin=8)
            drain(authority)
            assert len(authority.records) <= 8 and archive.tier.used_slots <= 8
            if cached:
                assert archive.read_cache.used_bytes <= 384
                assert all((slot, generation) in authority.records
                           for _, slot, generation in archive.read_cache.entries)
            generations.update((r.handle.slot, r.handle.generation) for r in authority.records.values())
            assert torch.equal(authority.read_record(sink)[0], payload(0, 4)[0])
        assert authority.history_end == 272
        assert len(generations) > 50 and authority.gc.reclaimed_records > 50
        with authority.index_snapshot() as (records, means):
            assert len(records) == means.shape[1]
            assert all(authority.memory.repository.pin(r.handle) for r in records)
    finally:
        assert authority.close()
        assert archive.tier.used_slots == 0 and archive.close()


def test_full_scoring_snapshot_prevents_policy_retirement_until_selection(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path)
    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        drain(authority)
        del_plan = select(authority)
        del del_plan
        assert authority.append(*payload(16, 8), history_begin=16, local_rope_begin=8)
        drain(authority)
        original = __import__('vllm_kvmem.retrieval', fromlist=['mean_key_scores']).mean_key_scores
        calls = []
        def score(*args):
            before = authority.gc.worker.stats['retired']
            advance(authority.gc.worker)
            assert authority.gc.worker.stats['retired'] == before
            calls.append(True)
            return original(*args)
        monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', score)
        plan = select(authority)
        del plan
        assert calls and authority.gc.worker.stats['reader_conflicts'] > 0
        advance(authority.gc.worker)
        authority.poll()
        assert authority.gc.reclaimed_records >= 2
    finally:
        assert authority.close() and archive.close()


def test_gc_requires_foreground_observation_of_durable_publication(tmp_path):
    authority, archive = setup(tmp_path)
    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        # Disk jobs can finish without authority.poll observing their result.
        # Hide completion to exercise the stronger backing-publication rule.
        hidden = []
        class Unobserved:
            def done(self): return False
            def result(self, **kwargs): raise AssertionError('Not observed yet')
        for job, keys in authority.writes:
            job.result(timeout=5)
            hidden.append((job, keys))
        authority.writes = [(Unobserved(), keys) for _, keys in hidden]
        plan = select(authority)
        del plan
        assert authority.append(*payload(16, 8), history_begin=16, local_rope_begin=8)
        for job, keys in authority.writes[len(hidden):]:
            job.result(timeout=5)
            hidden.append((job, keys))
        authority.writes = [(Unobserved(), keys) for _, keys in hidden]
        advance(authority.gc.worker)
        assert authority.gc.worker.stats['reclaimed'] == 0
        assert authority.gc.worker.stats['incomplete_backing'] > 0
        authority.writes = hidden
        authority.poll()
        advance(authority.gc.worker)
        authority.poll()
        assert authority.gc_reclaimed_records == 2
    finally:
        assert authority.close() and archive.close()


def test_background_ingestion_reclaims_without_artificial_request_steps(tmp_path):
    authority, archive = setup(tmp_path, background=True)
    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        drain(authority)
        plan = select(authority)
        del plan
        # Let the worker observe cold candidates before more accepted history.
        time.sleep(0.03)
        assert authority.append(*payload(16, 8), history_begin=16, local_rope_begin=8)
        drain(authority)
        deadline = time.monotonic() + 5
        while authority.gc.reclaimed_records < 2:
            authority.poll()
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert authority.memory.repository.ready_slots == 4
    finally:
        assert authority.close() and archive.close()


def test_retirement_accounts_for_gc_without_claiming_deleted_records_verified(tmp_path):
    from vllm_kvmem.retirement import retire_authority
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
    drain(authority)
    plan = select(authority)
    del plan
    assert authority.append(*payload(16, 8), history_begin=16, local_rope_begin=8)
    drain(authority)
    advance(authority.gc.worker)
    authority.poll()
    assert authority.gc_reclaimed_records == 2
    report = retire_authority(authority, True)
    assert report['history_end'] == 24
    assert report['gc_reclaimed_records'] == 2
    assert report['sealed_records_archived'] == report['sealed_records_verified'] == 4
    assert archive.tier.used_slots == 0 and archive.close()


def test_scoring_exception_releases_snapshot_leases_for_gc(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path)
    try:
        assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
        drain(authority)
        plan = select(authority)
        del plan
        assert authority.append(*payload(16, 8), history_begin=16, local_rope_begin=8)
        drain(authority)
        def fail(*args):
            raise RuntimeError('injected scorer failure')
        monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', fail)
        with pytest.raises(RuntimeError, match='injected scorer'):
            authority.select_for_queries(torch.ones(2, 1, 2, 2))
        advance(authority.gc.worker)
        authority.poll()
        assert authority.gc_reclaimed_records == 2
    finally:
        assert authority.close() and archive.close()
