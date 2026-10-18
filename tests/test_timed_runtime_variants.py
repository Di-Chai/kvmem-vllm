import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('timed_runtime_variants',
    Path(__file__).parents[1]/'scripts/benchmark_timed_runtime_variants.py')
variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variants)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def source(root, value):
    write(root/'src/vllm_kvmem/serve.py', value)
    write(root/'scripts/serve-support.py', 'shared server support')
    write(root/'csrc/gather.cpp', value+'native')
    write(root/'src/vllm_kvmem/__pycache__/serve.pyc', 'excluded')
    write(root/'artifacts/unrelated.txt', 'not a runtime source')
    return root


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    baseline = source(tmp_path/'baseline', 'original')
    candidate = source(tmp_path/'candidate', 'optimized')
    root = tmp_path/'experiment'
    args = SimpleNamespace(output=root, baseline_runtime=baseline, candidate_runtime=candidate,
                           policy=getattr(request, 'param', 'P1'))
    calls = []

    def shared_prepare(arguments):
        calls.append(arguments)
        root.mkdir()
        write(root/'frozen/scripts/client.py', 'one shared timed client')
        for arm in ('A0', 'P1'):
            variants.benchmark.q.save(root/(arm+'.json'), dict(model='/same/model', kvmem=dict(
                offline_agent_policy=arm == 'P1', active_resident_pressure_only=arm == 'P1')))
        variants.benchmark.q.save(root/'manifest.json', dict(
            source_sha256=variants.benchmark.hashes(root/'frozen'), fixture_sha256='fixture',
            workload=dict(duration_seconds=300, pairs=1, modes=['saturated'])))
        variants.benchmark.q.save(root/'status.json', dict(status='prepared'))

    monkeypatch.setattr(variants.benchmark, 'prepare', shared_prepare)
    variants.prepare(args)
    return args, calls


def test_preparation_freezes_two_runtimes_with_identical_p1_profiles(prepared):
    args, calls = prepared
    root = args.output
    manifest = json.loads((root/'manifest.json').read_text())
    assert calls == [args]
    assert manifest['comparison_kind'] == 'runtime'
    assert manifest['policy'] == 'P1'
    assert (root/'A0.json').read_bytes() == (root/'P1.json').read_bytes()
    assert json.loads((root/'A0.json').read_text())['kvmem']['offline_agent_policy'] is True
    for arm, original in [('A0', args.baseline_runtime), ('P1', args.candidate_runtime)]:
        snapshot = root/'runtime'/arm
        assert manifest['runtime_sha256'][arm] == variants.benchmark.hashes(snapshot)
        assert set(manifest['runtime_sha256'][arm]) == {
            'src/vllm_kvmem/serve.py', 'scripts/serve-support.py', 'csrc/gather.cpp'}
        assert not (snapshot/'artifacts').exists()
        assert manifest['runtime_source'][arm]['path'] == str(original)
        assert len(manifest['runtime_source'][arm]['source_hash']) == 64
        assert manifest['profile_sha256'][arm] == hashlib.sha256((root/(arm+'.json')).read_bytes()).hexdigest()
    assert manifest['runtime_sha256']['A0'] != manifest['runtime_sha256']['P1']
    assert (args.baseline_runtime/'src/vllm_kvmem/serve.py').read_text() == 'original'
    assert (args.candidate_runtime/'src/vllm_kvmem/serve.py').read_text() == 'optimized'
    assert json.loads((root/'status.json').read_text())['status'] == 'prepared'


def frozen_entry(monkeypatch, root):
    monkeypatch.setattr(variants, 'ROOT', root/'frozen')
    monkeypatch.setattr(variants.benchmark, 'ROOT', root/'frozen')


def test_live_script_cannot_run_a_prepared_runtime_comparison(prepared):
    args, _ = prepared
    with pytest.raises(ValueError, match='live checkout'):
        asyncio.run(variants.run(args))


def test_frozen_execution_reuses_shared_run_and_its_cleanup(prepared, monkeypatch):
    args, _ = prepared
    frozen_entry(monkeypatch, args.output)
    calls = []

    async def shared_run(arguments):
        calls.append(arguments)
        return 'shared result'

    monkeypatch.setattr(variants.benchmark, 'run', shared_run)
    assert asyncio.run(variants.run(args)) == 'shared result'
    assert calls == [args]


@pytest.mark.parametrize('tamper', ['runtime', 'profile', 'comparison_kind'])
def test_changed_runtime_or_policy_cannot_enter_shared_gpu_run(prepared, monkeypatch, tamper):
    args, _ = prepared
    root = args.output
    frozen_entry(monkeypatch, root)
    calls = []

    async def shared_run(arguments):
        calls.append(arguments)

    monkeypatch.setattr(variants.benchmark, 'run', shared_run)
    if tamper == 'runtime':
        (root/'runtime/P1/src/vllm_kvmem/serve.py').write_text('changed after freeze')
    elif tamper == 'profile':
        p = json.loads((root/'A0.json').read_text())
        p['kvmem']['offline_agent_policy'] = False
        variants.benchmark.q.save(root/'A0.json', p)
    else:
        p = json.loads((root/'manifest.json').read_text())
        p['comparison_kind'] = 'policy'
        variants.benchmark.q.save(root/'manifest.json', p)
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(variants.run(args))
    assert calls == []


def test_missing_source_is_rejected_before_preparation(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path/'never-created', baseline_runtime=None,
                           candidate_runtime=tmp_path/'candidate')
    with pytest.raises(ValueError, match='explicit'):
        variants.prepare(args)
    assert not args.output.exists()


def test_output_inside_copied_tree_is_rejected(tmp_path):
    runtime = source(tmp_path/'runtime', 'original')
    args = SimpleNamespace(output=runtime/'src/recursive-copy', baseline_runtime=runtime,
                           candidate_runtime=runtime)
    with pytest.raises(ValueError, match='inside a copied'):
        variants.validate_sources(args)


def test_runtime_snapshot_survives_later_source_edit(prepared, monkeypatch):
    args, _ = prepared
    frozen_entry(monkeypatch, args.output)
    before = variants.benchmark.hashes(args.output/'runtime/P1')
    (args.candidate_runtime/'src/vllm_kvmem/serve.py').write_text('next optimization')
    assert variants.validate_prepared(args.output)['runtime_sha256']['P1'] == before


def test_cli_preparation_requires_explicit_sources_without_starting_a_server(tmp_path):
    with pytest.raises(SystemExit) as error:
        variants.main(['--output', str(tmp_path/'experiment'), '--prepare-only',
                       '--config', 'profile.json', '--fixture', 'fixture.json'])
    assert error.value.code == 2
    assert not (tmp_path/'experiment').exists()


def test_run_prepared_cli_rejects_replacing_saved_source(tmp_path):
    with pytest.raises(SystemExit) as error:
        variants.main(['--output', str(tmp_path), '--run-prepared', '--candidate-runtime', str(tmp_path)])
    assert error.value.code == 2


@pytest.mark.parametrize('prepared', ['A0'], indirect=True)
def test_cold_policy_uses_the_same_a0_settings_for_both_runtime_sources(prepared, monkeypatch):
    args, _ = prepared
    root = args.output
    frozen_entry(monkeypatch, root)
    manifest = variants.validate_prepared(root)
    assert manifest['policy'] == 'A0'
    assert (root/'A0.json').read_bytes() == (root/'P1.json').read_bytes()
    for arm in ('A0', 'P1'):
        settings = json.loads((root/(arm+'.json')).read_text())['kvmem']
        assert settings['offline_agent_policy'] is False
        assert settings['active_resident_pressure_only'] is False
    assert 'not a task-quality evaluation' in manifest['runtime_comparison_scope']
