"""Preparation-only coverage: no tokenizer, GPU probe, lock or server launch."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


spec = importlib.util.spec_from_file_location('shared_prefill_qualification',
    Path(__file__).parents[1] / 'scripts/qualify_agent_scheduling.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def prepared_args(tmp_path, monkeypatch, *, arms=('P1', 'Ppressure'), cap=2048):
    from test_coordination_qualification import native_profile
    source = native_profile(tmp_path)
    base = json.loads(source.read_text())
    base['max_num_seqs'] = 4
    base['kvmem'].update(block_tokens=32, repository_blocks=32768, cpu_history_coordination=True,
        async_restore=True, maintenance_max_steps=8, prefill_batch_tokens=3072)
    source.write_text(json.dumps(base))
    output = tmp_path / 'qualification'
    output.mkdir()
    (output / 'fixture.json').write_text(json.dumps(dict(prompt=[1, 2, 3], tool=[4], eos=0)))

    def forbidden(*args, **kwargs):
        pytest.fail('Preparation must not acquire a GPU lock, probe GPU or start an arm')

    monkeypatch.setattr(qualification, 'open', forbidden, raising=False)
    monkeypatch.setattr(qualification.q, 'gpu_state', forbidden)
    monkeypatch.setattr(qualification.q, 'wait_idle', forbidden)
    monkeypatch.setattr(qualification.q, 'run_arm', forbidden)
    args = NS(config=source, output=output, arms=list(arms), cases=['agents'], pairs=1,
        concurrency=4, context_tokens=32768, output_tokens=257, turns=3,
        verify_restores=True, prepare_only=True, prefill_batch_tokens=cap,
        phase_policy=False, prefill_coalescing=False, coalescing_mixed_start=False,
        lifecycle_checks=False, retirement_barrier=True, distinct_users=False)
    return args


def test_explicit_cap_survives_arm_defaults_real_launcher_and_saved_profiles(tmp_path, monkeypatch):
    from vllm_kvmem.serve import build_launch
    args = prepared_args(tmp_path, monkeypatch)
    asyncio.run(qualification.main(args))
    assert json.loads((args.output / 'status.json').read_text()) == {'status': 'prepared', 'gpu_started': False}
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['workload']['prefill_batch_tokens'] == 2048
    for arm in args.arms:
        profile = json.loads((args.output / (arm + '.json')).read_text())
        assert profile['kvmem']['prefill_batch_tokens'] == 2048
        assert profile['verify_session_restore'] and profile['kvmem']['offline_agent_policy']
        plan = build_launch(profile, args.config.parent, args.output / ('roundtrip-' + arm))
        argv = plan['vllm_argv']
        transfer = json.loads(argv[argv.index('--kv-transfer-config') + 1])
        assert transfer['kv_connector_extra_config']['kvmem']['prefill_batch_tokens'] == 2048


@pytest.mark.parametrize('omit_argument', [False, True])
def test_default_retains_existing_arm_caps_and_ignores_base_profile_cap(tmp_path, monkeypatch, omit_argument):
    args = prepared_args(tmp_path, monkeypatch, arms=('A0', 'P1', 'A2', 'A3'), cap=0)
    if omit_argument:
        del args.prefill_batch_tokens  # Existing Python callers retain their behavior.
    asyncio.run(qualification.main(args))
    expected = {'A0': 0, 'P1': 0, 'A2': 512, 'A3': 512}
    assert {arm: json.loads((args.output / (arm + '.json')).read_text())['kvmem']['prefill_batch_tokens']
            for arm in args.arms} == expected


@pytest.mark.parametrize('change', [
    {'verify_restores': False}, {'arms': ['A0']}, {'arms': ['Pttl']},
    {'arms': ['P1', 'A2']}, {'phase_policy': True}, {'prefill_coalescing': True},
    {'prefill_batch_tokens': -32}, {'prefill_batch_tokens': True},
])
def test_explicit_cap_rejects_unqualified_mode_before_reading_config(tmp_path, change):
    args = NS(output=tmp_path, verify_restores=True, cases=['agents'],
        prefill_batch_tokens=2048, arms=['P1'], phase_policy=False, prefill_coalescing=False)
    args.__dict__.update(change)
    with pytest.raises(ValueError, match='Shared prefill qualification'):
        asyncio.run(qualification.main(args))


def test_explicit_cap_rejects_non_aligned_budget_before_profile_or_gpu(tmp_path, monkeypatch):
    args = prepared_args(tmp_path, monkeypatch, cap=2049)
    with pytest.raises(ValueError, match='block aligned'):
        asyncio.run(qualification.main(args))
    assert not (args.output / 'P1.json').exists()
