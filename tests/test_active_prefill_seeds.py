"""Native/KVMem dual-seed preparation and audit, without Torch or model launch."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import benchmark_active_prefill as b
import prefill_autotune_control as fi
import prefill_inductor_control as ic


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    reference = tmp_path / 'reference'
    reference.mkdir()
    b.save(reference / 'runtime-proof.json', {'files': {}})
    cases = []
    for count in (32768, 262045):
        source = tmp_path / f'{count}.i32'
        source.write_bytes(bytes(4 * count))
        cases.append(dict(input_path=str(source), sampling={'max_tokens': 8192},
            row=dict(served_input_tokens=count, served_sha256=b.sha(source))))
    shape_plan = dict(families={name: {'chunks': chunks} for name, chunks in b.FAMILIES.items()},
        full32_chunks=[2048] * 16, logical2k_control_chunks=[2048] * 16, max_chunk=2048)
    values = dict(reference=reference, profile={'kvmem': {'action_native_scores': 'reuse',
        'prefill_publication_pipeline': True, 'prefill_native_plan_arm': True}},
        cases32=cases[0], cases256=cases[1], geometry=[dict(query_tokens=2048, active_seq_len=2048,
        prefill_terminal=False)], native_shape_plan=shape_plan, output_controls={'samples': [], 'unavailable': []},
        preparation_overrides={}, case_sources={}, geometry_suite_index=3,
        launch={'vllm_argv': ['unchanged launch fixture']})
    monkeypatch.setattr(b, 'reference_inputs', lambda *args: deepcopy(values))
    def origins(repo, ref, names):
        return ({name: repo / name for name in names if name.startswith('scripts/')}, {})
    monkeypatch.setattr(b, 'runtime_origins', origins)
    return values


def compiler_source(tmp_path, monkeypatch):
    cache = tmp_path / 'torch_aot_compile' / ('a' * 64) / 'inductor_cache'
    source = cache / 'aa/kernel.py'
    source.parent.mkdir(parents=True)
    source.write_text("@triton_heuristics.reduction(filename=__file__, inductor_meta={'kernel_name':'kernel'})\n"
                      '@triton.jit\ndef kernel(x):\n    return x + 1\n')
    key = source.parent / ('b' * 64 + '.best_config')
    b.save(key, dict(XBLOCK=256, num_warps=4, num_stages=1, configs_hash='c' * 64,
                    found_by_coordesc=False, time_taken_ms=1))
    log = tmp_path / 'source.log'
    log.write_text('saved AOT compiled function to ' + str(cache.parent / 'rank_0_0/model') + '\n')
    monkeypatch.setattr(ic, '_installed_key_for_source', lambda path: path.parent / key.name)
    return dict(target=dict(cache_dir=str(cache), server_log=str(log), required_common=['aa/' + key.name])), {
        'target': ['d' * 64, 'e' * 64]}


@pytest.mark.parametrize('seeded', [False, True])
def test_prepare_freezes_both_arms_same_seeds_and_keeps_26_case_suite(tmp_path, monkeypatch, prepared_inputs, seeded):
    kwargs = {}
    if seeded:
        seed = tmp_path / 'autotune.json'
        b.save(seed, {'_metadata': {'device': 'synthetic'}, 'matmul-shape': {'tactic': 4}})
        sources, hashes = compiler_source(tmp_path, monkeypatch)
        kwargs = dict(flashinfer_autotune_seed=seed, flashinfer_autotune_hashes=['f' * 64],
                      inductor_seed_sources=sources, inductor_seed_hashes=hashes)
    root = tmp_path / 'prepared'
    b.prepare(root, SCRIPTS.parent, prepared_inputs['reference'], **kwargs)
    plan = b.read(root / 'plan.json')
    assert len(b.read(root / 'native/suite.json')['cases']) == 26
    for arm in ('native', 'kvmem'):
        b.assert_frozen(root, arm)
        manifest = b.read(root / arm / 'manifest.json')
        assert manifest.get('inductor_seed_manifest_sha256') == plan['inductor_seed_manifest_sha256'][arm]
        assert manifest.get('flashinfer_seed_manifest_sha256') == plan['flashinfer_seed_manifest_sha256'][arm]
        for helper in ('prefill_autotune_control.py', 'prefill_inductor_control.py',
                       'prefill_inductor_worker.py', 'prefill_inductor_serve.py', 'benchmark_native_prefill.py'):
            assert b.sha(root / arm / 'runtime/scripts' / helper) == b.sha(SCRIPTS / helper)
    assert plan['inductor_seed_manifest_sha256']['native'] == plan['inductor_seed_manifest_sha256']['kvmem']
    assert bool(plan['flashinfer_autotune_seed_sha256']) is seeded
    native = root / 'native'
    import benchmark_native_prefill as native_b
    env = {}
    receipts = native_b.install_native_seeds(native, env)
    assert bool(receipts['inductor']) is seeded
    if seeded:
        assert not (native / 'run').exists()
        assert env['PREFILL_INDUCTOR_RUN_DIR'] == str(native)
        assert Path(env['VLLM_CACHE_ROOT']).is_relative_to(native)
        assert Path(env['VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR']).is_relative_to(native)
        assert b.read(native / 'inductor_seed_install.json')['target_run'] == '.'
        # Parent-plan tampering must fail even when both child seed manifests are valid.
        plan['inductor_seed_manifest_sha256']['native'] = '0' * 64
        b.save(root / 'plan.json', plan)
        with pytest.raises(ValueError, match='Seed manifest differs|chains differ'):
            b.assert_seed_plan(root)


@pytest.mark.parametrize('kwargs', [dict(flashinfer_autotune_seed=Path('seed')),
    dict(flashinfer_autotune_hashes=['a' * 64]), dict(inductor_seed_sources={}), dict(inductor_seed_hashes={})])
def test_prepare_rejects_unpaired_seed_inputs_before_creating_output(tmp_path, kwargs):
    root = tmp_path / 'output'
    with pytest.raises(ValueError, match='together'):
        b.prepare(root, SCRIPTS.parent, **kwargs)
    assert not root.exists()


def test_historical_plan_needs_no_new_manifest_or_seed_files(tmp_path):
    b.save(tmp_path / 'plan.json', {'driver_sha256': 'old'})
    b.assert_seed_plan(tmp_path)
    assert not (tmp_path / 'native/manifest.json').exists()


def pair():
    return dict(flashinfer=dict(seed_sha256='seed', loaded_files_semantically_identical=True,
        actual_configs_sha256='actual'), inductor=dict(seed_manifest_sha256='inductor',
        launch_verified=True, common_configurations_unchanged=True, formal_dispatch_observer=False,
        modules={'target': {'common_kernels': [dict(key='kernel', source_sha256='source',
            function_source_sha256='function', configs_hash='configs', selected_config={'R0_BLOCK': 512})]}}))


def test_pair_requires_actual_same_choices_and_no_formal_observer():
    b.audit_seed_pair(pair(), pair())
    b.audit_seed_pair(dict(flashinfer=None, inductor=None), dict(flashinfer=None, inductor=None))
    for change in ('missing', 'fi', 'observer', 'configuration', 'coverage'):
        other = pair()
        if change == 'missing':
            other['inductor'] = None
        elif change == 'fi':
            other['flashinfer']['actual_configs_sha256'] = 'different'
        elif change == 'observer':
            other['inductor']['formal_dispatch_observer'] = True
        elif change == 'coverage':
            other['inductor']['modules']['target']['common_kernels'] = []
        else:
            other['inductor']['modules']['target']['common_kernels'][0]['selected_config']['R0_BLOCK'] = 1024
        with pytest.raises(ValueError):
            b.audit_seed_pair(pair(), other)


def test_prepare_cli_passes_explicit_seed_maps(monkeypatch, tmp_path):
    sources, hashes = {'target': {'cache_dir': '/source'}}, {'target': ['a' * 64]}
    calls = []
    monkeypatch.setattr(b, 'prepare', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['driver', 'prepare', '--root', str(tmp_path / 'out'),
        '--flashinfer-autotune-seed', str(tmp_path / 'seed'), '--flashinfer-autotune-hash', 'b' * 64,
        '--inductor-seed-sources', json.dumps(sources), '--inductor-seed-hashes', json.dumps(hashes)])
    b.main()
    kwargs = calls[0][1]
    assert kwargs['inductor_seed_sources'] == sources and kwargs['inductor_seed_hashes'] == hashes
    assert kwargs['flashinfer_autotune_hashes'] == ['b' * 64]
