import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('restore_stage_diagnostic',
    Path(__file__).parents[1] / 'scripts/diagnose_restore_stages.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'reference-performance'
    root.mkdir()
    profile = dict(model='/original-model', port=18490, max_num_seqs=8,
        verify_session_restore=False, kvmem=dict(step_latency_interval=1,
            unusual_preserved_setting='same', **{name: True for name in diagnostic.FEATURES}))
    fixture = dict(eos=42, agent_plans=[dict(initial_tokens=[10, 20], turns=[{'tool_tokens': [30]}])])
    workload = dict(duration_seconds=300., modes=['saturated'], concurrency=8,
        warmup_turns=2, output_tokens=1024, context_tokens=141870,
        epoch_turns=16, idle_seconds=15., port=18490, pairs=2,
        prepare_only=True, run_prepared=False)
    (root / 'P1.json').write_text(json.dumps(profile, indent=1) + '\n')
    (root / 'fixture.json').write_text(json.dumps(fixture, indent=1) + '\n')
    diagnostic.benchmark.q.save(root / 'manifest.json', dict(
        comparison_kind='host_dependency_ablation', feature_switches=list(diagnostic.FEATURES),
        workload=workload, source_sha256={'original.py': 'old-source'},
        profile_sha256={'P1': diagnostic.sha256(root / 'P1.json')},
        fixture_sha256=diagnostic.benchmark.q.digest(fixture)))
    return root


@pytest.fixture
def experiment(source, tmp_path, monkeypatch):
    # Keep source freezing and profile validation observable without loading a
    # model/tokenizer or copying the entire development tree.
    project = tmp_path / 'project'
    for folder in ('src', 'scripts', 'csrc'):
        (project / folder).mkdir(parents=True)
        (project / folder / 'source.txt').write_text(folder)
    monkeypatch.setattr(diagnostic, 'ROOT', project)
    monkeypatch.setattr(diagnostic.benchmark, 'ROOT', project)

    def prepare(args):
        root = args.output
        root.mkdir()
        frozen = root / 'frozen'
        for folder in ('src', 'scripts', 'csrc'):
            shutil.copytree(project / folder, frozen / folder)
        profile = diagnostic.benchmark.read(args.config)
        profile['kvmem']['step_latency_interval'] = 4  # The real shared helper does this.
        for arm in ('A0', 'P1'):
            diagnostic.benchmark.q.save(root / (arm + '.json'), profile)
        fixture = diagnostic.benchmark.read(args.fixture)
        diagnostic.benchmark.q.save(root / 'fixture.json', fixture)
        diagnostic.benchmark.q.save(root / 'manifest.json', dict(
            source_sha256=diagnostic.benchmark.hashes(frozen),
            fixture_sha256=diagnostic.benchmark.q.digest(fixture)))

    import vllm_kvmem.serve
    monkeypatch.setattr(vllm_kvmem.serve, 'build_launch', lambda *a, **kw: None)
    monkeypatch.setattr(diagnostic.benchmark, 'prepare', prepare)
    output = tmp_path / 'diagnostic'
    diagnostic.prepare(SimpleNamespace(source_experiment=source, output=output, port=18491))
    return output


def test_preparation_preserves_original_profile_fixture_and_workload(source, experiment):
    manifest, profile, fixture = diagnostic.validate(experiment)
    reference = diagnostic.benchmark.read(source / 'P1.json')
    assert profile == diagnostic.diagnostic_profile(reference, 18491)
    assert profile['kvmem']['step_latency_interval'] == 1
    assert (experiment / 'fixture.json').read_bytes() == (source / 'fixture.json').read_bytes()
    assert (experiment / 'reference/P1.json').read_bytes() == (source / 'P1.json').read_bytes()
    assert manifest['workload'] == {**{key: diagnostic.benchmark.read(source / 'manifest.json')
        ['workload'][key] for key in diagnostic.WORKLOAD_KEYS}, 'port': 18491, 'pairs': 1}
    assert [row['path'] for row in manifest['profile_changes']] == ['port', 'kvmem.restore_stage_timing']
    assert manifest['diagnostic_only'] and not manifest['performance_claim_allowed']
    assert manifest['execution_changes']['reference_pairs'] == 2
    assert not (experiment / 'A0.json').exists()
    assert fixture['agent_plans'][0]['turns'][0]['tool_tokens'] == [30]
    assert not reference['kvmem'].get('restore_stage_timing', False)


@pytest.mark.parametrize('what', ['profile', 'fixture', 'source', 'reference', 'workload', 'hidden-setting'])
def test_changed_frozen_inputs_cannot_run_even_with_rehashed_profile(experiment, what):
    manifest = diagnostic.benchmark.read(experiment / 'manifest.json')
    if what in ('profile', 'hidden-setting'):
        profile = diagnostic.benchmark.read(experiment / 'P1.json')
        profile['kvmem']['step_latency_interval'] = 4
        diagnostic.benchmark.q.save(experiment / 'P1.json', profile)
        if what == 'hidden-setting':
            manifest['profile_sha256']['P1'] = diagnostic.sha256(experiment / 'P1.json')
    elif what == 'fixture':
        (experiment / 'fixture.json').write_text('{}')
    elif what == 'source':
        (experiment / 'frozen/src/source.txt').write_text('changed')
    elif what == 'reference':
        (experiment / 'reference/P1.json').write_text('{}')
    else:
        manifest['workload']['epoch_turns'] = 8
    diagnostic.benchmark.q.save(experiment / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        diagnostic.validate(experiment)


def test_live_script_cannot_execute(experiment):
    with pytest.raises(ValueError, match='prepared frozen'):
        asyncio.run(diagnostic.run(SimpleNamespace(output=experiment)))


def frozen_execution(monkeypatch, experiment, tmp_path):
    monkeypatch.setattr(diagnostic, 'ROOT', experiment / 'frozen')
    monkeypatch.setattr(diagnostic.benchmark, 'ROOT', experiment / 'frozen')
    monkeypatch.setattr(diagnostic, 'LOCK_PATH', str(tmp_path / 'gpu.lock'))


def test_execution_runs_only_p1_and_uses_saved_args_then_refuses_restart(experiment, monkeypatch, tmp_path):
    frozen_execution(monkeypatch, experiment, tmp_path)
    calls = []

    async def arm(root, profile, fixture, name, pair, args):
        calls.append((name, pair, args.concurrency, args.duration_seconds, args.epoch_turns, args.pairs))
        assert args.warmup_turns == 2 and args.modes == ['saturated']
        return {'saturated': {'status': 'completed'}}

    monkeypatch.setattr(diagnostic.benchmark, 'run_arm', arm)
    # Neither changed CLI values nor the shared parser's defaults are consulted.
    args = argparse.Namespace(output=experiment, concurrency=1, duration_seconds=1, pairs=99)
    asyncio.run(diagnostic.run(args))
    assert calls == [('P1', 0, 8, 300., 16, 1)]
    report = diagnostic.benchmark.read(experiment / 'diagnostic.json')
    assert report['diagnostic_only'] and not report['performance_claim_allowed']
    assert not (experiment / 'comparison.json').exists()
    assert diagnostic.benchmark.read(experiment / 'status.json')['status'] == 'diagnostic_completed'
    with pytest.raises(ValueError, match='no automatic restarts'):
        asyncio.run(diagnostic.run(args))
    assert len(calls) == 1


@pytest.mark.parametrize('cancelled', [False, True])
def test_failed_or_cancelled_arm_is_preserved_and_never_retried(experiment, monkeypatch, tmp_path, cancelled):
    frozen_execution(monkeypatch, experiment, tmp_path)
    calls = []
    error = asyncio.CancelledError('signal') if cancelled else RuntimeError('owned server failed')
    wait = diagnostic.benchmark.q.wait_idle

    async def arm(*args):
        calls.append(1)
        raise error

    monkeypatch.setattr(diagnostic.benchmark, 'run_arm', arm)
    with pytest.raises(type(error)) as caught:
        asyncio.run(diagnostic.run(SimpleNamespace(output=experiment)))
    assert caught.value is error
    status = diagnostic.benchmark.read(experiment / 'status.json')
    assert status['status'] == ('interrupted' if cancelled else 'failed')
    assert status['diagnostic_only'] and diagnostic.benchmark.q.wait_idle is wait
    with pytest.raises(ValueError, match='no automatic restarts'):
        asyncio.run(diagnostic.run(SimpleNamespace(output=experiment)))
    assert calls == [1]


@pytest.mark.parametrize('coherent_manifest_edit', [False, True])
def test_waiting_for_idle_revalidates_disk_inputs_before_server_launch(
        experiment, monkeypatch, tmp_path, coherent_manifest_edit):
    frozen_execution(monkeypatch, experiment, tmp_path)
    launched = []

    async def wait(root, stable_seconds):
        assert stable_seconds == 15.
        profile = diagnostic.benchmark.read(root / 'P1.json')
        profile['port'] += 1
        diagnostic.benchmark.q.save(root / 'P1.json', profile)
        if coherent_manifest_edit:
            manifest = diagnostic.benchmark.read(root / 'manifest.json')
            reference = diagnostic.benchmark.read(root / 'reference/P1.json')
            manifest['workload']['port'] = profile['port']
            manifest['profile_sha256']['P1'] = diagnostic.sha256(root / 'P1.json')
            manifest['profile_changes'] = diagnostic.profile_changes(reference, profile)
            diagnostic.benchmark.q.save(root / 'manifest.json', manifest)
            diagnostic.validate(root)  # Internally valid, but differs from the in-memory inputs.

    async def arm(root, profile, fixture, name, pair, args):
        await diagnostic.benchmark.q.wait_idle(root, args.idle_seconds)
        launched.append(True)

    monkeypatch.setattr(diagnostic.benchmark.q, 'wait_idle', wait)
    monkeypatch.setattr(diagnostic.benchmark, 'run_arm', arm)
    message = 'manifest changed while waiting' if coherent_manifest_edit else 'profile or fixture changed'
    with pytest.raises(ValueError, match=message):
        asyncio.run(diagnostic.run(SimpleNamespace(output=experiment)))
    assert not launched


def test_busy_cooperative_lock_does_not_launch_or_overwrite_another_status(experiment, monkeypatch, tmp_path):
    frozen_execution(monkeypatch, experiment, tmp_path)

    async def forbidden(*args):
        pytest.fail('Busy GPU lock must prevent launch')

    monkeypatch.setattr(diagnostic.benchmark, 'run_arm', forbidden)
    with open(diagnostic.LOCK_PATH, 'w') as lock:
        diagnostic.fcntl.flock(lock, diagnostic.fcntl.LOCK_EX | diagnostic.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            asyncio.run(diagnostic.run(SimpleNamespace(output=experiment)))
    assert diagnostic.benchmark.read(experiment / 'status.json')['status'] == 'prepared'


def test_reference_hash_mismatch_is_rejected_before_creating_output(source, tmp_path):
    (source / 'fixture.json').write_text('{}')
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='Reference profile or fixture'):
        diagnostic.prepare(SimpleNamespace(source_experiment=source, output=output, port=None))
    assert not output.exists()
