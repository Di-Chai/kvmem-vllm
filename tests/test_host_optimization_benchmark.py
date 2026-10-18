import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('host_optimization_benchmark',
    Path(__file__).parents[1] / 'scripts/benchmark_host_optimizations.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.fixture
def experiment(tmp_path):
    profile = dict(model='/model', max_num_seqs=8, kvmem=dict(offline_agent_policy=True))
    manifest = dict(comparison_kind='host_dependency_ablation', feature_switches=list(host.FEATURES),
                    profile_sha256={})
    for arm in ('A0', 'P1'):
        copy = json.loads(json.dumps(profile))
        copy['kvmem'].update({name: arm == 'P1' for name in host.FEATURES})
        path = tmp_path / (arm + '.json')
        path.write_text(json.dumps(copy))
        manifest['profile_sha256'][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_only_explicit_switches_differ(experiment):
    assert host.validate_profiles(experiment)['feature_switches'] == list(host.FEATURES)
    # Validation must not pop flags out of the actual saved profiles.
    assert json.loads((experiment / 'P1.json').read_text())['kvmem']['split_capture_dma']


@pytest.mark.parametrize('change', ['other-setting', 'wrong-flag', 'missing-flag', 'unhashed-edit', 'two-runtimes'])
def test_unmatched_arms_cannot_enter_gpu_run(experiment, change):
    manifest = json.loads((experiment / 'manifest.json').read_text())
    path = experiment / 'P1.json'
    profile = json.loads(path.read_text())
    if change == 'other-setting':
        profile['max_num_seqs'] = 4
    elif change == 'wrong-flag':
        profile['kvmem']['split_capture_dma'] = False
    elif change == 'missing-flag':
        del profile['kvmem']['native_prefill_commit']
    elif change == 'two-runtimes':
        manifest['runtime_sha256'] = {}
    else:
        profile['model'] = '/different'
    path.write_text(json.dumps(profile))
    if change != 'unhashed-edit':
        manifest['profile_sha256']['P1'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (experiment / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        host.validate_profiles(experiment)


def test_live_checkout_cannot_run_frozen_experiment(experiment):
    with pytest.raises(ValueError, match='prepared frozen'):
        asyncio.run(host.run(SimpleNamespace(output=experiment)))


def test_observer_tolerates_incomplete_json_and_retains_publication_state(tmp_path):
    (tmp_path / 'progress.json').write_text('{')
    (tmp_path / 'action_attention.jsonl').write_text('{"prefill_commit_host_seconds": 1.5}\n{')
    (tmp_path / 'scheduler_resources.json').write_text(json.dumps({'publication_blocked': ['slow']}))
    row = host.observe(tmp_path)
    assert 'progress' not in row
    assert row['action']['prefill_commit_host_seconds'] == 1.5
    assert row['resources']['publication_blocked'] == ['slow']
    assert row['finished_at'] >= row['started_at']
