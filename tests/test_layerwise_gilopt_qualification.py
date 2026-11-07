from copy import deepcopy
import json
from pathlib import Path

import pytest

import run_layerwise_gilopt_qualification as q


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_subset_keeps_complete_rows_sampling_and_order_without_changing_source():
    fixture = dict(suite='longmemeval_s', rows=[dict(id=c, input_tokens=100) for c in 'abc'],
        bindings=[dict(id=c, max_tokens=16384, thinking_token_budget=4096,
                       sampling=dict(seed=20260722)) for c in 'abc'],
        cases=3, case_ids=list('abc'), qualification=False, performance_observation=True,
        execution_flags=dict(layerwise_rolling_decode=True))
    before = deepcopy(fixture)
    result = q.subset(fixture, ['c', 'a'])
    assert fixture == before and result['case_ids'] == ['c', 'a']
    assert result['rows'] == [before['rows'][2], before['rows'][0]]
    assert result['bindings'] == [before['bindings'][2], before['bindings'][0]]
    assert result['execution_flags'] == before['execution_flags']
    assert result['qualification'] and not result['performance_observation']
    result['rows'][0]['input_tokens'] = 1
    result['bindings'][0]['sampling']['seed'] = 42
    assert fixture == before
    with pytest.raises(ValueError): q.subset(fixture, ['a', 'a'])
    with pytest.raises(ValueError): q.subset(fixture, ['unknown'])


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    previous, pointer = tmp_path/'previous', tmp_path/'pointer.json'
    save(previous/'status.json', dict(status='completed'))
    save(previous/'run/cleanup.json', dict(gpu_processes=[], remaining_owned_groups=[]))
    save(pointer, dict(queue_root='stale-completed-campaign'))
    owner = dict(pid=123, start_ticks='4', command_sha256='identified')
    plan = dict(previous=str(previous), previous_processes=dict(pid=owner, server_pid=None),
                pointer=str(pointer), previous_pointer_sha256=q.sha(pointer))
    monkeypatch.setattr(q, 'identity', lambda pid: None)
    return plan, owner


def test_stale_global_pointer_cannot_hide_a_live_identified_owner(handoff, monkeypatch):
    plan, owner = handoff
    monkeypatch.setattr(q, 'identity', lambda pid: owner)
    with pytest.raises(RuntimeError, match='still live'):
        q.handoff_ready(plan, gpu_processes=lambda: set())


def test_pid_reuse_is_reported_without_operating_on_process(handoff, monkeypatch):
    plan, owner = handoff
    monkeypatch.setattr(q, 'identity', lambda pid: dict(owner, start_ticks='5'))
    with pytest.raises(RuntimeError, match='PID was reused'):
        q.handoff_ready(plan, gpu_processes=lambda: set())


def test_handoff_requires_terminal_success_actual_free_gpu_and_unchanged_pointer(handoff):
    plan, owner = handoff
    q.handoff_ready(plan, gpu_processes=lambda: set())
    with pytest.raises(RuntimeError, match='not released'):
        q.handoff_ready(plan, gpu_processes=lambda: {456})
    previous = Path(plan['previous'])
    save(previous/'status.json', dict(status='running'))
    with pytest.raises(RuntimeError, match='not successfully completed'):
        q.handoff_ready(plan, gpu_processes=lambda: set())
    save(previous/'status.json', dict(status='failed'))
    with pytest.raises(RuntimeError, match='no automatic wait/retry'):
        q.handoff_ready(plan, gpu_processes=lambda: set())
    save(previous/'status.json', dict(status='completed'))
    save(Path(plan['pointer']), dict(queue_root='foreign'))
    with pytest.raises(RuntimeError, match='pointer changed'):
        q.handoff_ready(plan, gpu_processes=lambda: set())


def test_candidate_is_blocked_if_baseline_is_unstable_or_refresh_uncovered(tmp_path):
    save(tmp_path/'baseline-b/status.json', dict(status='completed'))
    save(tmp_path/'baseline-b-equivalence.json', dict(passed=False, decode_coverage=[{'r': 1}, {'r': 1}]))
    with pytest.raises(ValueError, match='stability'):
        q.require_predecessor(tmp_path, 'optimized')
    save(tmp_path/'baseline-b-equivalence.json', dict(passed=True, decode_coverage=[{'r': 0}, {'r': 0}]))
    with pytest.raises(ValueError, match='coverage'):
        q.require_predecessor(tmp_path, 'optimized')
    save(tmp_path/'baseline-b-equivalence.json', dict(passed=True, decode_coverage=[{'r': 1}, {'r': 1}]))
    q.require_predecessor(tmp_path, 'optimized')


def test_refresh_coverage_expansion_never_resamples_previous_diagnostic_ids(tmp_path):
    save(tmp_path/'fixture.json', dict(rows=[dict(id=c) for c in ('ordinary', 'long1', 'long2', 'long3')]))
    for cid in ('ordinary', 'long1', 'long2', 'long3'):
        save(tmp_path/'run/answers/cases'/cid/'result.json', dict(layerwise=dict(events=[
            dict(event_type='prefill_pressure' if cid == 'ordinary' else 'decode_pressure')])))
    assert q.choose_cases(tmp_path) == ['ordinary', 'long1']
    assert q.choose_cases(tmp_path, exclude=['ordinary', 'long1'], refresh_only=True) == ['long2', 'long3']
    with pytest.raises(ValueError, match='never resample'):
        q.choose_cases(tmp_path, exclude=['long1', 'long2', 'long3'], refresh_only=True)


@pytest.mark.parametrize('workers', [0, 9, True, 1.5])
def test_client_worker_override_rejects_invalid_counts_before_verification(tmp_path, workers):
    import run_layerwise_memory as memory
    with pytest.raises(ValueError, match='workers'):
        memory.infer(tmp_path, tmp_path/'capture', workers=workers)
    with pytest.raises(ValueError, match='workers'):
        memory.run(tmp_path, workers=workers)
