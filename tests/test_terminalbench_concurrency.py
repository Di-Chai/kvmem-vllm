"""More tool-task owners may share the same eight native model slots."""
import array
from copy import deepcopy
import json

import pytest

from scripts import run_terminalbench_suite as suite


def profile(model=8, sessions=12):
    return dict(port=19000, max_num_seqs=model, served_model_name='eight-model-slots',
                max_model_len=10000000, model='model', tokenizer='tokenizer',
                kvmem=dict(select_budget=36864, gen_budget=16384,
                           repository_blocks=8192, block_tokens=32,
                           session_capacity=sessions, session_checkpoint_bytes=6*1024**3,
                           cpu_read_cache_bytes=96*1024**3))


@pytest.mark.parametrize('model,tasks,explicit,expected', [
    (8, 89, None, 8), (3, 89, None, 3), (8, 2, None, 2),
    (8, 89, 12, 12), (8, 4, 12, 4), (4, 89, 12, 12),
])
def test_task_slots_can_exceed_model_slots_without_changing_model_or_budgets(model, tasks, explicit, expected):
    value = profile(model)
    original = deepcopy(value)
    result = suite.concurrency_limits(value, tasks, explicit)
    assert result == dict(task_concurrency=expected, model_concurrency=model,
                          session_capacity=12, bridge_max_sessions=16)
    assert value == original


@pytest.mark.parametrize('setting,explicit,match', [
    (profile(sessions=8), 12, 'session_capacity'),
    (profile(sessions=True), 12, 'session_capacity'),
    (profile(sessions=16), 17, 'bridge owner'),
    (profile(model=0), 12, 'model concurrency'),
    (profile(model=True), 12, 'model concurrency'),
    (profile(), 0, 'positive integer'),
    (profile(), -1, 'positive integer'),
    (profile(), True, 'positive integer'),
    (profile(), 12.0, 'positive integer'),
])
def test_prepare_limits_reject_unowned_or_invalid_oversubscription(setting, explicit, match):
    with pytest.raises(ValueError, match=match):
        suite.concurrency_limits(setting, 89, explicit)


@pytest.mark.parametrize('explicit,expected', [(None, 8), (12, 12)])
@pytest.mark.parametrize('memory_gib', [None, 32])
def test_prepared_full_job_records_task_and_native_capacity_separately(tmp_path, monkeypatch, explicit, expected, memory_gib):
    repo, dataset, root = tmp_path/'repo', tmp_path/'dataset', tmp_path/'run'
    original = repo/'artifacts/rolling-long-output-20261026-r1'
    original.mkdir(parents=True)
    (original/'input.bin').write_bytes(array.array('i', range(5000)).tobytes())
    protocol = repo/'artifacts/deepswe-first16-20261015-r16/protocol/protocol.so'
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(b'protocol')
    for index in range(89):
        task = dataset/'tasks'/f'task-{index:02d}'
        task.mkdir(parents=True)
        (task/'task.toml').write_text('[agent]\ntimeout_sec = 900\n'
                                    '[verifier]\ntimeout_sec = 30\n'
                                    '[environment]\ncpus = 1\nmemory_mb = 2048\n')
        (task/'instruction.md').write_text(f'Task {index}')
    value = profile()
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(value))
    monkeypatch.setattr(suite, 'REPO', repo)
    monkeypatch.setattr(suite, 'copy_download_cache', lambda source, target: target.mkdir(parents=True))
    monkeypatch.setattr(suite, 'cache_compose_override', lambda path: {})

    def freeze(_repo, destination):
        destination.mkdir(parents=True)
        (destination/'current.py').write_text('CURRENT=True')
        return dict(origin='current-working-tree', native={'current.so': 'sha'})

    monkeypatch.setattr(suite, 'freeze_runtime', freeze)
    monkeypatch.setattr(suite, 'task_query_manifest', lambda root: (
        dict(tasks={}, upstream_sources={}), dict(recorder_transport=True)))
    monkeypatch.setattr(suite.subprocess, 'check_output', lambda command, **kwargs:
                        b'{"harbor":"test"}' if '-c' in command else 'commit\n')
    monkeypatch.setattr(suite.subprocess, 'run', lambda *args, **kwargs: None)
    extra = {} if memory_gib is None else dict(container_memory_admission_bytes=memory_gib*1024**3)
    suite.prepare(root, dataset, profile_path=path, trial_concurrency=explicit, **extra)
    job = json.loads((root/'job.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert job['n_concurrent_trials'] == expected
    assert len(job['tasks']) == 89
    assert manifest['max_concurrency'] == manifest['task_concurrency'] == expected
    assert manifest['model_concurrency'] == 8
    assert manifest['session_capacity'] == 12
    assert manifest['bridge_max_sessions'] == 16
    assert manifest['container_memory_admission_bytes'] == (48 if memory_gib is None else memory_gib)*1024**3
    assert manifest['container_memory_host_reserve_bytes'] == 12*1024**3
    assert json.loads((root/'profile.json').read_text()) == value
