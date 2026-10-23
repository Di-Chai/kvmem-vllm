"""Semantic step completion drives real native background reclamation."""
import math
import os
import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.historical_gc import HistoricalGcSettings
from vllm_kvmem.memory_storage import MemoryArchive
from test_historical_authority import append, advance, close


def setup(tmp_path, **params):
    config = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=16,
        repository_blocks=32, sink_blocks=1, recent_blocks=1)
    archive = MemoryArchive(str(tmp_path), 48, 32, native_payloads=True)
    gc = n.HistoricalGcConfig()
    gc.mode, gc.background, gc.policy = n.HistoricalGcMode.DELETE, False, 'statistical'
    gc.policy_params = params
    gc.ready_low, gc.ready_high = 1, 2
    gc.scan_blocks, gc.delete_blocks = 32, 8
    worker = archive.enable_history_gc(gc)
    authority = RequestAuthority('a', config, archive, 2, 1, 2, resident_records=16)
    append(authority)
    authority.poll()
    ledger = authority.gc.open_evidence(authority.records, epoch='branch-1')
    records = sorted(authority.records.values(), key=lambda r: r.history_begin)
    assert ledger.register_span('step', 1, 13, kind='completed_step', epoch='branch-1')
    return authority, archive, worker, ledger, records


def observations(ledger, first=0, count=3):
    for i in range(first, first+count):
        assert ledger.observe('step', epoch='branch-1', observation_id=i, importance=.01, cold=True)


def test_completed_step_is_proactively_reclaimed_but_result_and_mixed_blocks_survive(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path)
    try:
        # Plenty of unused repository capacity: reclamation is proactive.
        assert a.memory.repository.ready_slots > 2
        observations(ledger)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 0  # no semantic completion
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 2
        assert [r.history_begin for r in a.records.values()] == [0, 12, 16, 20, 24, 28]
        assert archive.io_usage()['cpu_authority_bytes'] == 6*48
        assert a.memory.repository.pin(records[4].handle)
        assert len(a.memory._store.working_records()) == 8  # GPU refresh is separate.
        # Even native evidence cannot cause the retained result to disappear.
        assert a.gc.scope.publish_evidence(records[4].handle, 100, True, 10, 10, 0., None)
        advance(a, worker)
        assert a.memory.repository.pin(records[4].handle)
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


def test_completion_and_unknown_observation_are_not_a_deletion_proof(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path)
    try:
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        observations(ledger, count=2)
        assert not ledger.observe('step', epoch='branch-1', observation_id=1, importance=0., cold=True)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        assert ledger.observe('step', epoch='branch-1', observation_id=2, importance=.9, cold=False)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        observations(ledger, first=3)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 2
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


def test_partial_publication_holds_all_candidates_until_revision_complete(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path)
    try:
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        observations(ledger)
        assert not ledger.flush(max_blocks=1)
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        assert ledger.flush(max_blocks=1)
        advance(a, worker)
        assert a.gc_reclaimed_records == 2
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


def test_model_signal_is_revision_checked_and_cannot_skip_other_gates(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path, min_model_reclaim_score=.8)
    try:
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        old = ledger.revision('step', epoch='branch-1')
        observations(ledger)
        assert not ledger.model_signal('step', epoch='branch-1', based_on_revision=old,
                                       decision_id=0, reclaim_score=1.)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        current = ledger.revision('step', epoch='branch-1')
        assert not ledger.model_signal('step', epoch='old-branch', based_on_revision=current,
                                       decision_id=0, reclaim_score=1.)
        assert ledger.model_signal('step', epoch='branch-1', based_on_revision=current,
                                   decision_id=0, reclaim_score=.9)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 2
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


def test_rewind_revokes_pending_eligibility_and_rejects_old_signals(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path)
    try:
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        observations(ledger)
        assert not ledger.flush(max_blocks=1)
        old = ledger.revision('step', epoch='branch-1')
        assert a.gc.prepare_rewind()
        assert ledger.closed and a.gc.evidence is None
        assert not ledger.model_signal('step', epoch='branch-1', based_on_revision=old,
                                       decision_id=1, reclaim_score=1.)
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
        new = a.gc.open_evidence(a.records, epoch='branch-2')
        assert new.register_span('step', 1, 13, kind='thinking', epoch='branch-2')
        assert new.complete_span('step', [records[4]], epoch='branch-2')
        assert new.observe('step', epoch='branch-2', observation_id=0, importance=.01, cold=True)
        assert new.flush()  # Live prefix generations retain their old revision.
        advance(a, worker)
        assert a.gc_reclaimed_records == 0
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


def test_mixed_tail_overlap_and_witness_dependencies_are_conservative(tmp_path):
    a, archive, worker, ledger, records = setup(tmp_path)
    try:
        with pytest.raises(ValueError, match='overlapping'):
            ledger.register_span('overlap', 4, 8, kind='thinking', epoch='branch-1')
        with pytest.raises(ValueError, match='witnesses'):
            ledger.complete_span('step', [], epoch='branch-1')
        with pytest.raises(ValueError, match='source'):
            ledger.complete_span('step', [records[1]], epoch='branch-1')
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        with pytest.raises(ValueError, match='overlapping'):
            ledger.register_span('result', 16, 20, kind='tool_detail', epoch='branch-1')
        assert not ledger.register_span('mixed', 21, 23, kind='thinking', epoch='branch-1')
    finally:
        records.clear()  # Release test-owned native index storage before close.
        close(a, archive)


@pytest.mark.parametrize('params', [
    {'min_observations': True}, {'min_observations': 0},
    {'min_cold_observations': 1.5}, {'min_cold_observations': -1},
    {'max_importance': math.nan}, {'max_importance': math.inf},
    {'max_importance': True}, {'max_importance': 10**1000}, {'max_importance': -1}, {'max_importance': 1.1},
    {'proactive': 1}, {'min_model_reclaim_score': -.1}, {'keep_mandatory': False},
])
def test_statistical_settings_validate_thresholds_before_startup(params):
    with pytest.raises(ValueError, match='historical_gc.policy_params'):
        HistoricalGcSettings(policy='statistical', policy_params=params)


def test_statistical_settings_round_trip_to_native():
    settings = HistoricalGcSettings(policy='statistical',
        policy_params={'min_observations': 4, 'max_importance': 0, 'proactive': False})
    cfg = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=16, repository_blocks=32, sink_blocks=1, recent_blocks=1)
    native = settings.native(cfg, 1)
    assert native.policy == 'statistical'
    assert native.policy_params['min_observations'] == 4
    assert native.policy_params['max_importance'] == 0
    assert native.policy_params['proactive'] is False


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                    reason='explicit GPU opt-in required')
def test_statistical_source_retires_while_its_copied_gpu_bytes_are_in_flight(tmp_path):
    import torch
    a, archive, worker, ledger, records = setup(tmp_path)
    event = None
    try:
        chunks = a.read_records([records[1].handle])
        raw, values = chunks[0]
        expected = torch.cat((raw.contiguous().view(torch.uint8).flatten(),
                              values.contiguous().view(torch.uint8).flatten())).clone()
        staging = expected.pin_memory()
        del raw, values, chunks
        stream, event = torch.cuda.Stream(), torch.cuda.Event()
        with torch.cuda.stream(stream):
            working = staging.to('cuda', non_blocking=True)
            torch.cuda._sleep(1_000_000_000)
            result = working.clone()
            event.record(stream)
        assert ledger.complete_span('step', [records[4]], epoch='branch-1')
        observations(ledger)
        assert ledger.flush()
        advance(a, worker)
        assert a.gc_reclaimed_records == 2
        assert archive.io_usage()['cpu_authority_bytes'] == 6*48
        assert not event.query()
        event.synchronize()
        assert torch.equal(result.cpu(), expected)
        assert a.memory.repository.pin(records[4].handle)
    finally:
        if event is not None:
            event.synchronize()
        records.clear()
        close(a, archive)
