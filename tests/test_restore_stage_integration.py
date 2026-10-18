"""Restore diagnostics preserve request-local progress and error semantics."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_kvmem.async_refresh import RefreshJob
from vllm_kvmem.async_restore import step_restore
from vllm_kvmem.connector import RestoreCommand


def waiting_manager(tmp_path, *, enabled):
    return SimpleNamespace(options=SimpleNamespace(restore_stage_timing=enabled),
        runner=SimpleNamespace(_kvmem_probe_output=tmp_path, _kvmem_sessions=object(),
            req_states=SimpleNamespace(req_id_to_index={})))


def test_disabled_diagnostics_never_construct_a_timer(monkeypatch, tmp_path):
    from vllm_kvmem import restore_timing
    monkeypatch.setattr(restore_timing, 'RestoreTiming', lambda **kwargs: pytest.fail('disabled timer'))
    manager = waiting_manager(tmp_path, enabled=False)
    job = RefreshJob(RestoreCommand('new', -1, 1, 64, 32, ()))
    assert not step_restore(manager, job)
    assert not list(tmp_path.iterdir())


def test_unready_request_does_not_write_poll_rows(tmp_path):
    manager = waiting_manager(tmp_path, enabled=True)
    job = RefreshJob(RestoreCommand('new', -1, 1, 64, 32, ()))
    assert not step_restore(manager, job)
    assert not list(tmp_path.iterdir())
    assert job.stage == 'prepare' and job.checkpoint is None


def test_bad_lease_records_stage_and_preserves_original_error(tmp_path):
    manager = waiting_manager(tmp_path, enabled=True)
    job = RefreshJob(RestoreCommand('new', -1, 1, 64, 32, ()))
    manager.jobs = {'new': job}
    manager.runner.req_states.req_id_to_index = {'new': 0}
    manager.runner._kvmem_restore_allocations = {}
    manager.runner._kvmem_sessions = SimpleNamespace(imports={'new'}, bindings={
        'new': SimpleNamespace(generation=1, history=64, window=32)})
    with pytest.raises(ValueError, match='no fresh native allocation witness'):
        step_restore(manager, job)
    rows = [json.loads(line) for line in (tmp_path/'restore_host_stages.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['outcome'] == 'error'
    assert rows[0]['request_id'] == 'new' and rows[0]['phase'] == 'prepare'
    stages = {s['name'] for s in rows[0]['timing']['stages']}
    assert stages == {'allocation_lease_validate', 'restore_main_thread_total'}
    assert job.stage == 'prepare' and job.checkpoint is None


def test_diagnostic_write_failure_cannot_hide_restore_invariant(monkeypatch, tmp_path):
    manager = waiting_manager(tmp_path, enabled=True)
    manager.runner.req_states.req_id_to_index = {'new': 0}
    manager.runner._kvmem_sessions = SimpleNamespace(imports={'new'}, bindings={
        'new': SimpleNamespace(generation=99, history=64, window=32)})
    job = RefreshJob(RestoreCommand('new', -1, 1, 64, 32, ()))
    def cannot_write(*args, **kwargs):
        raise OSError('diagnostic storage unavailable')
    monkeypatch.setattr(Path, 'open', cannot_write)
    with pytest.raises(ValueError, match='Restore command differs'):
        step_restore(manager, job)
    assert job.checkpoint is None and job.stage == 'prepare'


def test_timing_keeps_hot_restore_independent_of_cold_producer(monkeypatch, tmp_path):
    from test_async_restore import two_lane_manager
    manager, checkpoints, claims, published, fences, release, hot_started = two_lane_manager(monkeypatch, tmp_path)
    manager.options.restore_stage_timing = True
    original_claim = manager.runner._kvmem_sessions.pool.claim
    manager.runner._kvmem_sessions.pool.claim = lambda *args, timing=None, **kwargs: original_claim(*args, **kwargs)
    cold, hot = manager.jobs['cold'], manager.jobs['hot']
    try:
        assert manager.step(cold)
        assert manager.step(hot) and hot_started.wait(2)
        hot.future.result(timeout=2)
        fences['hot'].done = True
        assert manager.step(hot) and hot.done and not cold.done
        assert published == ['hot']
        rows = [json.loads(line) for line in (tmp_path/'restore_host_stages.jsonl').read_text().splitlines()]
        completed = [r for r in rows if r['request_id'] == 'hot' and r['phase'] == 'restore_copy']
        assert len(completed) == 1 and completed[0]['outcome'] == 'progress'
        stages = {s['name'] for s in completed[0]['timing']['stages']}
        assert {'model_state_validate', 'commit_selection', 'finish_claim', 'publish_restore'} <= stages
    finally:
        release.set()
        for fence in fences.values():
            fence.done = True
        manager.shutdown()
