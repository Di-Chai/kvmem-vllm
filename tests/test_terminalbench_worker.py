from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.prepare_terminalbench_worker import worker_config, unexecuted, canonical_image_config
from scripts.terminalbench_worker_driver import verify_memory_admission


def test_worker_only_changes_host_paths_endpoint_and_task_selection():
    value = dict(n_attempts=1, retry=dict(max_retries=0), n_concurrent_trials=12,
                 jobs_dir='/old/jobs', job_name='formal', tasks=[dict(path='/old/a')],
                 environment=dict(type='docker', delete=False, env={'http_proxy': ''},
                                  cpu_enforcement_policy='limit', memory_enforcement_policy='limit',
                                  extra_docker_compose=['/old/compose.json']),
                 agents=[dict(kwargs=dict(api_base='http://127.0.0.1:18388/v1',
                         temperature=.6, llm_call_kwargs=dict(max_tokens=131072, seed=1000)))])
    before = deepcopy(value)
    config = worker_config(value, Path('/worker'), ['a', 'b'], 'http://127.0.0.1:18389')
    assert value == before
    assert config['n_concurrent_trials'] == 2
    assert config['tasks'] == [dict(path='/worker/dataset/tasks/a'), dict(path='/worker/dataset/tasks/b')]
    assert config['agents'][0]['kwargs']['llm_call_kwargs'] == value['agents'][0]['kwargs']['llm_call_kwargs']
    assert config['agents'][0]['kwargs']['temperature'] == .6
    assert config['environment'] | {'extra_docker_compose': value['environment']['extra_docker_compose']} == value['environment']
    value['retry']['max_retries'] = 1
    with pytest.raises(ValueError, match='one attempt'):
        worker_config(value, Path('/worker'), ['a'], 'http://127.0.0.1:18389')


def test_handoff_excludes_running_finished_and_answer_files(tmp_path):
    (tmp_path/'formal-events').mkdir()
    (tmp_path/'jobs/formal').mkdir(parents=True)
    (tmp_path/'job.json').write_text(json.dumps(dict(tasks=[dict(path='/tasks/'+n)
                    for n in ['queued','waiting','running','done','has_answer']])))
    events = [dict(event='start',task='terminal-bench/waiting'),
              dict(event='environment-start',task='terminal-bench/running'),
              dict(event='end',task='terminal-bench/done')]
    (tmp_path/'formal-events/events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    trial = tmp_path/'jobs/formal/has_answer__suffix'
    (trial/'agent').mkdir(parents=True)
    (trial/'config.json').write_text(json.dumps(dict(task=dict(path='/tasks/has_answer'))))
    (trial/'agent/answer.txt').write_text('already executed')
    assert unexecuted(tmp_path) == ['queued','waiting']


def test_image_comparison_ignores_only_known_empty_inspect_defaults():
    source = dict(Env=['A=1'], Cmd=['bash'])
    worker = dict(source, User='', OpenStdin=False, Entrypoint=None)
    assert canonical_image_config(source) == canonical_image_config(worker)
    for key, value in [('User','root'), ('OpenStdin',True), ('Entrypoint',['sh']),
                       ('Env',['A=2']), ('unknown',None)]:
        assert canonical_image_config(dict(worker, **{key:value})) != canonical_image_config(source)


def test_packaged_admission_rejects_legacy_launcher_despite_48g_manifest(tmp_path):
    launcher = tmp_path/'runtime/scripts/run_terminalbench_job.py'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('LEGACY_HARDCODED_BUDGET = 24 * 1024**3\n')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        container_memory_admission_bytes=48*1024**3,
        container_memory_host_reserve_bytes=12*1024**3)))
    with pytest.raises(ValueError, match='Packaged launcher'):
        verify_memory_admission(tmp_path)
    assert not (tmp_path/'memory-admission-qualification.json').exists()


def test_packaged_admission_checks_actual_launcher_and_both_limits(tmp_path, monkeypatch):
    import shutil
    import sys
    from types import SimpleNamespace
    from scripts import terminalbench_network
    monkeypatch.setitem(sys.modules, 'terminalbench_network', terminalbench_network)
    monkeypatch.setattr('psutil.virtual_memory', lambda:SimpleNamespace(total=64*1024**3))
    launcher = tmp_path/'runtime/scripts/run_terminalbench_job.py'
    launcher.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parents[1]/'scripts/run_terminalbench_job.py', launcher)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        container_memory_admission_bytes=48*1024**3,
        container_memory_host_reserve_bytes=12*1024**3)))
    receipt = verify_memory_admission(tmp_path)
    assert receipt['passed'] and receipt['effective_bytes'] == 48*1024**3
    assert receipt['host_reserve_bytes'] == 12*1024**3
    (tmp_path/'memory-admission-control.json').write_text(json.dumps(dict(
        schema_version=1, budget_bytes=24*1024**3)))
    with pytest.raises(ValueError, match='differs from the manifest'):
        verify_memory_admission(tmp_path)
