"""Qualification controller regressions without GPU access or live subprocesses."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).parents[1]/'scripts'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load('graph_benchmark_controller', 'benchmark_block_graph.py')
sys.path.insert(0, str(SCRIPTS))
try:
    summary = load('graph_benchmark_summary', 'summarize_block_graph.py')
finally:
    sys.path.pop(0)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared(tmp_path):
    root = tmp_path/'attempt'
    (root/'runtime').mkdir(parents=True)
    save(root/'profile.json', dict(port=12345))
    save(root/'fixture.json', {'fixture': 'same input'})
    manifest = dict(arm='multi', synchronous=False, runtime_sha256={}, python=sys.executable,
        fixture=str(root/'fixture.json'),
        profile_sha256=hashlib.sha256((root/'profile.json').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((root/'fixture.json').read_bytes()).hexdigest())
    save(root/'manifest.json', manifest)
    save(root/'status.json', dict(status='prepared'))
    return SimpleNamespace(output=root), manifest


class Process:
    def __init__(self, pid, returncode=None):
        self.pid, self.returncode = pid, returncode

    def poll(self):
        return self.returncode


def fake_processes(monkeypatch, tmp_path, args, result=None, on_health=None):
    processes, signals = [], []
    result = result or dict(status='passed', all_owned_sessions_released=True)

    def spawn(command, **kwargs):
        assert kwargs['start_new_session'] is True
        process = Process(701001+len(processes), 0 if processes else None)
        if processes:
            save(args.output/'qualification/results.json', result)
        processes.append(process)
        return process

    def killpg(pid, signum):
        signals.append((pid, signum))
        process = next(p for p in processes if p.pid == pid)
        process.returncode = -signum

    class Health:
        status = 200

        def __enter__(self):
            if on_health:
                on_health()
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench, 'gpu_processes', lambda: set())
    monkeypatch.setattr(bench, 'open', lambda *a, **kw: (tmp_path/'lock').open('a+'), raising=False)
    monkeypatch.setattr(bench.fcntl, 'flock', lambda *a: None)
    monkeypatch.setattr(bench.subprocess, 'Popen', spawn)
    monkeypatch.setattr(bench.os, 'killpg', killpg)
    monkeypatch.setattr(bench, 'live_group_members', lambda pid:
        [pid] if next(p for p in processes if p.pid == pid).returncode is None else [])
    monkeypatch.setattr(bench.urllib.request, 'build_opener', lambda *a:
        SimpleNamespace(open=lambda *a, **kw: Health()))
    return processes, signals


def test_controller_reads_qualification_status_before_reporting_success(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    processes, signals = fake_processes(monkeypatch, tmp_path, args,
        result=dict(status='incomplete', all_owned_sessions_released=True))
    with pytest.raises(RuntimeError, match='did not pass: incomplete'):
        bench.run(args)
    assert json.loads((args.output/'status.json').read_text())['status'] == 'failed'
    assert {pid for pid, _ in signals} == {p.pid for p in processes}
    assert not json.loads((args.output/'cleanup.json').read_text())['remaining_owned_process_groups']


def test_controller_requires_owned_session_release_even_for_passed_client(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    fake_processes(monkeypatch, tmp_path, args,
        result=dict(status='passed', all_owned_sessions_released=False))
    with pytest.raises(RuntimeError, match='did not pass'):
        bench.run(args)


def test_sigterm_unwinds_into_owned_group_cleanup_and_restores_signal_handler(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)

    def terminate():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    processes, signals = fake_processes(monkeypatch, tmp_path, args, on_health=terminate)
    original_cleanup = bench.stop_owned_groups

    def cleanup(processes):
        # Repeated SIGTERM cannot interrupt cleanup after the first unwinds.
        terminate()
        return original_cleanup(processes)

    monkeypatch.setattr(bench, 'stop_owned_groups', cleanup)
    with pytest.raises(bench.QualificationTerminated, match='SIGTERM'):
        bench.run(args)
    assert signal.getsignal(signal.SIGTERM) == previous
    assert len(processes) == 1 and signals == [(processes[0].pid, signal.SIGTERM)]
    assert json.loads((args.output/'status.json').read_text())['status'] == 'failed'
    assert (args.output/'cleanup.json').exists()


def test_passed_is_written_only_after_both_groups_are_retired(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    fake_processes(monkeypatch, tmp_path, args)
    cleanup = bench.stop_owned_groups

    def checked(processes):
        assert json.loads((args.output/'status.json').read_text())['status'] == 'testing'
        return cleanup(processes)

    monkeypatch.setattr(bench, 'stop_owned_groups', checked)
    bench.run(args)
    assert json.loads((args.output/'status.json').read_text())['status'] == 'passed'


def test_cleanup_escalates_even_when_leader_has_already_exited(monkeypatch):
    leader = Process(701001, 0)
    clock, alive, signals = [0.], {701002}, []
    monkeypatch.setattr(bench.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(bench.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(bench, 'live_group_members', lambda pid: list(alive))

    def killpg(pid, signum):
        assert pid == leader.pid
        signals.append(signum)
        if signum == signal.SIGKILL:
            alive.clear()

    monkeypatch.setattr(bench.os, 'killpg', killpg)
    assert not bench.stop_owned_groups([leader], grace_seconds=.2, kill_seconds=.2)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_cleanup_residue_fails_instead_of_leaving_a_passed_status(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    fake_processes(monkeypatch, tmp_path, args)
    monkeypatch.setattr(bench, 'stop_owned_groups', lambda processes: {701001: [701002]})
    with pytest.raises(RuntimeError, match='cleanup left live'):
        bench.run(args)
    assert json.loads((args.output/'status.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('field', ['profile', 'fixture'])
def test_changed_frozen_inputs_are_rejected_before_any_gpu_access(monkeypatch, tmp_path, field):
    args, _ = prepared(tmp_path)
    save(args.output/(field+'.json'), dict(changed=True))
    monkeypatch.setattr(bench, 'gpu_processes', lambda: pytest.fail('Touched GPU before validation'))
    with pytest.raises(ValueError, match='Frozen qualification '+field+' changed'):
        bench.run(args)


def test_legacy_unhashed_manifest_cannot_be_launched(monkeypatch, tmp_path):
    args, manifest = prepared(tmp_path)
    manifest.pop('profile_sha256')
    save(args.output/'manifest.json', manifest)
    monkeypatch.setattr(bench, 'gpu_processes', lambda: pytest.fail('Touched GPU before validation'))
    with pytest.raises(ValueError, match='Unfrozen legacy manifest'):
        bench.run(args)


def test_prepare_copies_shared_fixture_and_hashes_both_inputs(monkeypatch, tmp_path):
    source = tmp_path/'source'
    (source/'src').mkdir(parents=True)
    (source/'src/module.py').write_text('VALUE = 1\n')
    (source/'scripts').mkdir()
    for name in ('qualify_block_graph.py', 'benchmark_block_graph.py'):
        (source/'scripts'/name).write_text('# frozen\n')
    save(source/'artifacts/capacity-chain-g16-release-performance-20261017-r3/rolling-candidate-v2.json',
        dict(kvmem={}))
    shared = tmp_path/'shared.json'
    save(shared, dict(initial_tokens=[1, 2], later_questions=[3, 4]))
    args = SimpleNamespace(output=tmp_path/'attempt', fixture=shared,
        arm='multi', synchronous=False, port=12345)
    monkeypatch.setattr(bench, 'ROOT', source)
    bench.prepare(args)
    original = shared.read_bytes()
    save(shared, dict(changed=True))
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert Path(manifest['fixture']).read_bytes() == original
    assert manifest['fixture_sha256'] == hashlib.sha256(original).hexdigest()
    assert manifest['profile_sha256'] == hashlib.sha256((args.output/'profile.json').read_bytes()).hexdigest()
    assert 'src/module.py' in manifest['runtime_sha256']


def test_prepare_refuses_to_generate_different_fixtures_at_launch(tmp_path):
    args = SimpleNamespace(output=tmp_path/'attempt', fixture=tmp_path/'missing.json')
    with pytest.raises(ValueError, match='shared fixture first'):
        bench.prepare(args)
    assert not args.output.exists()


def test_custom_profile_and_client_are_frozen_without_graph_overrides(monkeypatch, tmp_path):
    source = tmp_path/'source'
    (source/'src').mkdir(parents=True)
    (source/'src/example.py').write_text('X = 1\n')
    (source/'scripts').mkdir()
    for name in ('qualify_block_graph.py', 'benchmark_block_graph.py', 'qualify_alignment.py'):
        (source/'scripts'/name).write_text('# '+name+'\n')
    profile = dict(port=100, model='local-fp8', kvmem=dict(graph_mode='collect',
        query_history=True, guided_query_tokens=256, select_budget=8192))
    save(tmp_path/'profile.json', profile)
    save(tmp_path/'fixture.json', dict(tokens=[1, 2]))
    args = SimpleNamespace(output=tmp_path/'attempt', fixture=tmp_path/'fixture.json',
        arm='collect', synchronous=False, port=12345, profile=tmp_path/'profile.json',
        qualifier=source/'scripts/qualify_alignment.py')
    monkeypatch.setattr(bench, 'ROOT', source)
    bench.prepare(args)
    frozen = json.loads((args.output/'profile.json').read_text())
    assert frozen == dict(profile, port=12345)
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['qualification_script'] == 'qualify_alignment.py'
    assert 'scripts/qualify_alignment.py' in manifest['runtime_sha256']
    assert 'scripts/qualify_block_graph.py' in manifest['runtime_sha256']
    assert json.loads((tmp_path/'profile.json').read_text()) == profile


@pytest.mark.parametrize('name', ['../outside.py', 'scripts/nested.py', 'client.txt'])
def test_manifest_client_cannot_escape_frozen_scripts(monkeypatch, tmp_path, name):
    args, manifest = prepared(tmp_path)
    manifest['qualification_script'] = name
    save(args.output/'manifest.json', manifest)
    monkeypatch.setattr(bench, 'gpu_processes', lambda: pytest.fail('Touched GPU before validation'))
    with pytest.raises(ValueError, match='Python filename'):
        bench.run(args)


def test_custom_client_must_be_covered_by_runtime_hash(monkeypatch, tmp_path):
    args, manifest = prepared(tmp_path)
    manifest['qualification_script'] = 'qualify_alignment.py'
    save(args.output/'manifest.json', manifest)
    monkeypatch.setattr(bench, 'gpu_processes', lambda: pytest.fail('Touched GPU before validation'))
    with pytest.raises(ValueError, match='runtime hash'):
        bench.run(args)


def test_foreign_gpu_is_never_stopped_or_joined(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    processes, signals = fake_processes(monkeypatch, tmp_path, args)
    monkeypatch.setattr(bench, 'gpu_processes', lambda: {999999})
    with pytest.raises(RuntimeError, match='GPU is occupied'):
        bench.run(args)
    assert processes == signals == []


def test_foreign_gpu_appearing_during_run_fails_cleanup_without_signalling_it(monkeypatch, tmp_path):
    args, _ = prepared(tmp_path)
    processes, signals = fake_processes(monkeypatch, tmp_path, args)
    observations = iter([set(), {999999}])
    monkeypatch.setattr(bench, 'gpu_processes', lambda: next(observations))
    with pytest.raises(RuntimeError, match='cleanup left live'):
        bench.run(args)
    assert {pid for pid, _ in signals} == {p.pid for p in processes}
    assert json.loads((args.output/'cleanup.json').read_text())['remaining_gpu_pids'] == [999999]


def comparison_attempts(tmp_path):
    paths = []
    common = dict(port=123, served_model_name='test', max_model_len=32768,
        kvmem=dict(select_budget=4096, gen_budget=4096, graph_trace=True,
            async_refresh=True, async_restore=True, cpu_history_coordination=True,
            maintenance_max_steps=8, action_fraction=0.))
    fixture = dict(first_input=[1, 2], second_question=[3], edited_question=[4])
    fixture['sha256'] = summary.digest(fixture)
    for arm, sync in [(arm, False) for arm in summary.ARMS]+[('multi', True)]:
        directory = tmp_path/(arm+('-sync' if sync else ''))
        profile = deepcopy(common)
        profile['kvmem']['graph_mode'] = arm if arm in ('collect', 'locality', 'attention', 'multi') else 'off'
        profile['kvmem']['action_fraction'] = .25 if arm == 'legacy' else 0.
        if sync:
            profile['kvmem'].update(async_refresh=False, async_restore=False,
                cpu_history_coordination=False, maintenance_max_steps=0)
        save(directory/'profile.json', profile)
        save(directory/'qualification/profile.json', profile)
        save(directory/'qualification/fixture.json', fixture)
        save(directory/'qualification/results.json', dict(arm=arm, fixture_sha256=fixture['sha256']))
        save(directory/'manifest.json', dict(arm=arm, synchronous=sync))
        paths.append(directory)
    return paths


def test_all_arm_comparison_reports_posthoc_legacy_manifest_limit(tmp_path):
    paths = comparison_attempts(tmp_path)
    checked = summary.validate_comparison_profiles(paths)
    assert checked['all_arm_controls_verified']
    assert not checked['profile_fixture_hashed_before_run']
    assert len(checked['legacy_manifest_runs']) == 7


@pytest.mark.parametrize('arm,key,value', [('attention', 'select_budget', 2048),
    ('multi', 'gen_budget', 8192), ('legacy', 'action_fraction', .5),
    ('multi-sync', 'maintenance_max_steps', 1), ('multi-sync', 'select_budget', 2048)])
def test_non_graph_drift_in_every_scored_arm_is_rejected(tmp_path, arm, key, value):
    paths = comparison_attempts(tmp_path)
    target = next(path for path in paths if path.name == arm)
    profile = json.loads((target/'profile.json').read_text())
    profile['kvmem'][key] = value
    save(target/'profile.json', profile)
    save(target/'qualification/profile.json', profile)
    with pytest.raises(ValueError):
        summary.validate_comparison_profiles(paths)


def test_same_first_inputs_but_changed_later_questions_are_rejected(tmp_path):
    paths = comparison_attempts(tmp_path)
    target = paths[-1]
    fixture = json.loads((target/'qualification/fixture.json').read_text())
    fixture['second_question'] = [900]
    fixture['sha256'] = summary.digest({k:v for k,v in fixture.items() if k != 'sha256'})
    save(target/'qualification/fixture.json', fixture)
    save(target/'qualification/results.json', dict(arm='multi', fixture_sha256=fixture['sha256']))
    with pytest.raises(ValueError, match='later questions differ'):
        summary.validate_comparison_profiles(paths)


def test_missing_comparison_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='all six'):
        summary.validate_comparison_profiles(comparison_attempts(tmp_path)[1:])


def test_summary_can_verify_new_profile_and_fixture_freeze_metadata(tmp_path):
    paths = comparison_attempts(tmp_path)
    for directory in paths:
        manifest = json.loads((directory/'manifest.json').read_text())
        fixture = directory/'qualification/fixture.json'
        manifest.update(fixture=str(fixture),
            fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
            profile_sha256=hashlib.sha256((directory/'profile.json').read_bytes()).hexdigest())
        save(directory/'manifest.json', manifest)
    report = summary.validate_comparison_profiles(paths)
    assert report['profile_fixture_hashed_before_run'] and not report['legacy_manifest_runs']
