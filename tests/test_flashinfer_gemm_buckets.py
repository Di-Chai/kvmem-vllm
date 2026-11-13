"""CPU qualification of GEMM-only warmup alignment and cache isolation."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import sys
from types import SimpleNamespace

import pytest

from vllm_kvmem.serving.flashinfer_gemm_buckets import (
    _AlignmentState, _CONFIG_NAMES, _STATE, _align_gemm_overrides,
    _aligned_buckets, _install_warmup_adapter, _seed_cache,
    install_flashinfer_gemm_bucket_alignment,
)
from vllm_kvmem.core.options import RuntimeOptions
from test_graph_options import model_profile, runtime, worker_payload
from vllm_kvmem.serving.serve import build_launch


def native_bucket(rows):
    if rows <= 256:
        return 1 << (rows - 1).bit_length()
    if rows <= 2048:
        return ((rows + 255) // 256) * 256
    if rows <= 4096:
        return ((rows + 511) // 512) * 512
    return 1 << (rows - 1).bit_length()


BUCKETS = tuple(sorted({native_bucket(n) for n in range(1, 4097)})) + (4136,)


@dataclass(frozen=True)
class Spec:
    map_to_tuning_buckets: object = native_bucket
    gen_tuning_buckets: object = BUCKETS


@dataclass(frozen=True)
class Config:
    dynamic_tensor_specs: tuple = (Spec(),)
    constraint_specs: tuple = ('preserved constraints',)
    tensor_initializers: tuple = ('preserved initializers',)
    inputs_pre_hook: object = object()


@pytest.fixture
def qualified():
    class Tuner:
        is_tuning_mode = True
        _override_tuning_buckets = BUCKETS
        _override_round_up = False

        def __init__(self):
            self.cache = {}

        @classmethod
        def get(cls):
            return tuner

        def _apply_tuning_overrides(self, config):
            key = (id(config), self._override_tuning_buckets, self._override_round_up)
            if key not in self.cache:
                buckets = self._override_tuning_buckets
                if buckets is None:
                    return config
                up = self._override_round_up

                def mapper(rows):
                    points = [x for x in buckets if x >= rows] if up else [x for x in buckets if x <= rows]
                    return (min(points) if points else max(buckets)) if up else (max(points) if points else min(buckets))

                self.cache[key] = replace(config, dynamic_tensor_specs=(Spec(mapper, buckets),))
            return self.cache[key]

        def get_effective_map_to_tuning_buckets(self, config):
            return self._apply_tuning_overrides(config).dynamic_tensor_specs[0].map_to_tuning_buckets

    tuner = Tuner()
    configs = {name: Config() for name in _CONFIG_NAMES}

    @contextmanager
    def autotune(*, tuning_buckets=None, round_up=None):
        saved = tuner._override_tuning_buckets, tuner._override_round_up
        if tuning_buckets is not None:
            tuner._override_tuning_buckets = tuning_buckets
        if round_up is not None:
            tuner._override_round_up = round_up
        try:
            yield
        finally:
            tuner._override_tuning_buckets, tuner._override_round_up = saved

    return tuner, configs, autotune


def test_default_off_and_native_payload_roundtrip(tmp_path):
    assert RuntimeOptions().flashinfer_gemm_bucket_alignment is False
    assert install_flashinfer_gemm_bucket_alignment(None, RuntimeOptions(), None) is None
    profile = model_profile(tmp_path)
    profile['kvmem']['flashinfer_gemm_bucket_alignment'] = True
    launch = build_launch(profile, tmp_path, tmp_path / 'launch')
    assert runtime(worker_payload(launch)).flashinfer_gemm_bucket_alignment is True


@pytest.mark.parametrize('enabled', [False, True])
def test_launcher_preserves_only_opted_in_read_only_seed(enabled, monkeypatch, tmp_path):
    profile = model_profile(tmp_path)
    profile['kvmem']['flashinfer_gemm_bucket_alignment'] = enabled
    seed = tmp_path/'baseline.json'
    seed.write_bytes(b'{"baseline": "unchanged"}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED', 'baseline.json')
    monkeypatch.setenv('KVMEM_UNRELATED_AMBIENT_VALUE', 'must not pass through')
    launch = build_launch(profile, tmp_path, tmp_path/'launch')
    env = launch['environment']
    name = 'KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED'
    assert (name in env) is enabled
    if enabled:
        assert env[name] == str(seed.resolve())
    assert 'KVMEM_UNRELATED_AMBIENT_VALUE' not in env
    assert seed.read_bytes() == b'{"baseline": "unchanged"}\n'


def test_launcher_rejects_non_file_seed(monkeypatch, tmp_path):
    profile = model_profile(tmp_path)
    profile['kvmem']['flashinfer_gemm_bucket_alignment'] = True
    monkeypatch.setenv('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED', str(tmp_path))
    with pytest.raises(ValueError, match='autotune seed must be a file'):
        build_launch(profile, tmp_path, tmp_path/'launch')


@pytest.mark.parametrize('value', [0, 1, 'true', None, [], {}])
def test_option_is_strict_boolean(value):
    with pytest.raises(ValueError, match='flashinfer_gemm_bucket_alignment must be boolean'):
        runtime(dict(flashinfer_gemm_bucket_alignment=value))


@pytest.mark.parametrize('enabled', [None, True, False])
def test_install_accepts_vllm_automatic_autotune_and_rejects_only_false(
        enabled, monkeypatch, tmp_path):
    import vllm_kvmem.serving.flashinfer_gemm_buckets as adapter
    installed = []
    monkeypatch.setattr(adapter, '_install_warmup_adapter', lambda *args: installed.append(args))
    monkeypatch.setitem(sys.modules, 'flashinfer.autotuner',
                        SimpleNamespace(AutoTuner=object(), autotune=object()))
    monkeypatch.setitem(sys.modules, 'flashinfer.gemm', SimpleNamespace(gemm_base=object()))
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.warmup',
                        SimpleNamespace(kernel_warmup=SimpleNamespace(
                            resolve_flashinfer_autotune_file=lambda _: tmp_path/'upstream.json')))
    monkeypatch.delenv('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED', raising=False)
    runner = SimpleNamespace(max_num_tokens=4136, vllm_config=SimpleNamespace(
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=enabled),
        compute_hash=lambda **kwargs: 'host-test-config'))
    options = RuntimeOptions(flashinfer_gemm_bucket_alignment=True)
    if enabled is False:
        with pytest.raises(ValueError, match='requires FlashInfer autotuning'):
            install_flashinfer_gemm_bucket_alignment(runner, options, tmp_path)
        assert installed == []
    else:
        state = install_flashinfer_gemm_bucket_alignment(runner, options, tmp_path)
        assert state.max_model_rows == 4136 and state.report_path.exists()
        assert len(installed) == 1
        assert install_flashinfer_gemm_bucket_alignment(runner, options, tmp_path) is state
        assert len(installed) == 1


@pytest.mark.parametrize('seed_case', ['upstream', 'missing', 'explicit'])
def test_install_seeds_private_cache_from_upstream_unless_overridden(
        seed_case, monkeypatch, tmp_path):
    import vllm_kvmem.serving.flashinfer_gemm_buckets as adapter
    upstream = tmp_path/'upstream.json'
    explicit = tmp_path/'explicit.json'
    upstream_bytes = b'{"4136": "upstream baseline tactic"}\n'
    explicit_bytes = b'{"4136": "explicit baseline tactic"}\n'
    if seed_case != 'missing':
        upstream.write_bytes(upstream_bytes)
    explicit.write_bytes(explicit_bytes)
    if seed_case == 'explicit':
        monkeypatch.setenv('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED', str(explicit))
    else:
        monkeypatch.delenv('KVMEM_FLASHINFER_GEMM_AUTOTUNE_SEED', raising=False)
    resolved = []

    def resolve(runner):
        assert not hasattr(runner, _STATE)
        resolved.append(upstream)
        return upstream

    warmup = SimpleNamespace(resolve_flashinfer_autotune_file=resolve)
    monkeypatch.setattr(adapter, '_install_warmup_adapter', lambda *args: None)
    monkeypatch.setitem(sys.modules, 'flashinfer.autotuner',
                        SimpleNamespace(AutoTuner=object(), autotune=object()))
    monkeypatch.setitem(sys.modules, 'flashinfer.gemm', SimpleNamespace(gemm_base=object()))
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.warmup',
                        SimpleNamespace(kernel_warmup=warmup))
    runner = SimpleNamespace(max_num_tokens=4136, vllm_config=SimpleNamespace(
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=None),
        compute_hash=lambda **kwargs: 'host-test-config'))
    state = install_flashinfer_gemm_bucket_alignment(
        runner, RuntimeOptions(flashinfer_gemm_bucket_alignment=True), tmp_path/'run')
    if seed_case == 'missing':
        assert state.seed is None and not state.cache_path.exists()
        assert not upstream.exists()
    else:
        expected = explicit_bytes if seed_case == 'explicit' else upstream_bytes
        assert state.cache_path.read_bytes() == expected
        assert state.seed['copied_to_private_cache'] is True
        # Neither the default seed nor the override is ever rewritten.
        assert upstream.read_bytes() == upstream_bytes
        assert explicit.read_bytes() == explicit_bytes
    assert len(resolved) == (0 if seed_case == 'explicit' else 1)


def test_all_actual_rows_match_native_mapper_and_config_is_cached(qualified):
    tuner, configs, autotune = qualified
    state = SimpleNamespace(alignments=[])
    config = configs[_CONFIG_NAMES[0]]
    with _align_gemm_overrides(tuner, configs, autotune, state):
        adjusted = tuner._apply_tuning_overrides(config)
        assert tuner._apply_tuning_overrides(config) is adjusted
        mapper = tuner.get_effective_map_to_tuning_buckets(config)
        assert all(mapper(n) == native_bucket(n) for n in range(1, 4137))
        assert adjusted.dynamic_tensor_specs[0].gen_tuning_buckets[-1] == 8192
        assert 4136 not in adjusted.dynamic_tensor_specs[0].gen_tuning_buckets
        assert adjusted.constraint_specs is config.constraint_specs
        assert adjusted.tensor_initializers is config.tensor_initializers
        assert adjusted.inputs_pre_hook is config.inputs_pre_hook
    assert len(state.alignments) == 1
    assert tuner._override_tuning_buckets == BUCKETS
    assert tuner._override_round_up is False
    assert '_apply_tuning_overrides' not in vars(tuner)
    assert config.dynamic_tensor_specs[0].gen_tuning_buckets == BUCKETS


@pytest.mark.parametrize('case', ['not_tuning', 'no_buckets', 'unrelated_identity'])
def test_inference_and_non_gemm_configs_are_unchanged(qualified, case):
    tuner, configs, autotune = qualified
    config = configs[_CONFIG_NAMES[0]]
    if case == 'not_tuning':
        tuner.is_tuning_mode = False
    elif case == 'no_buckets':
        tuner._override_tuning_buckets = None
    else:
        config = replace(config)  # structurally equal is insufficient
    expected = tuner._apply_tuning_overrides(config)
    state = SimpleNamespace(alignments=[])
    with _align_gemm_overrides(tuner, configs, autotune, state):
        assert tuner._apply_tuning_overrides(config) is expected
    assert state.alignments == []


def test_hook_and_nested_overrides_restore_on_error(qualified):
    tuner, configs, autotune = qualified
    old = tuner._apply_tuning_overrides
    tuner._apply_tuning_overrides = old
    with pytest.raises(RuntimeError, match='dummy run failed'):
        with _align_gemm_overrides(tuner, configs, autotune, SimpleNamespace(alignments=[])):
            tuner._apply_tuning_overrides(configs[_CONFIG_NAMES[0]])
            raise RuntimeError('dummy run failed')
    assert tuner._apply_tuning_overrides is old
    assert tuner._override_tuning_buckets == BUCKETS
    assert tuner._override_round_up is False


def test_old_4136_cache_does_not_supply_new_8192_tactic(qualified):
    tuner, configs, autotune = qualified
    config = configs[_CONFIG_NAMES[0]]
    old_cache = {bucket: f'old tactic {bucket}' for bucket in BUCKETS}
    before = dict(old_cache)
    profiled = []
    with _align_gemm_overrides(tuner, configs, autotune, SimpleNamespace(alignments=[])):
        spec = tuner._apply_tuning_overrides(config).dynamic_tensor_specs[0]
        # FlashInfer generates all profiles in tuning mode, then checks each
        # profile key separately; old 4136 must not stand in for new 8192.
        for bucket in spec.gen_tuning_buckets:
            if spec.map_to_tuning_buckets(bucket) not in old_cache:
                profiled.append(bucket)
                old_cache[bucket] = f'fresh tactic {bucket}'
        assert spec.map_to_tuning_buckets(4136) == 8192
    assert profiled == [8192]
    assert all(old_cache[key] == value for key, value in before.items())
    assert old_cache[native_bucket(4118)] == 'fresh tactic 8192'


def test_adapter_only_changes_enabled_runner_and_private_cache(tmp_path, qualified):
    tuner, configs, autotune = qualified
    config = configs[_CONFIG_NAMES[0]]
    calls = []

    def original_runs(runner, **kwargs):
        calls.append((runner.max_num_tokens, runner.scheduler_config.max_num_batched_tokens, kwargs))
        return tuner.get_effective_map_to_tuning_buckets(config)(4118)

    shared_cache = tmp_path / 'shared.json'
    warmup = SimpleNamespace(_run_flashinfer_autotune_dummy_runs=original_runs,
                             resolve_flashinfer_autotune_file=lambda runner: shared_cache)
    _install_warmup_adapter(warmup, type(tuner), SimpleNamespace(**configs), autotune)
    first = warmup._run_flashinfer_autotune_dummy_runs
    _install_warmup_adapter(warmup, type(tuner), SimpleNamespace(**configs), autotune)
    assert warmup._run_flashinfer_autotune_dummy_runs is first
    runner = SimpleNamespace(max_num_tokens=4136,
                             scheduler_config=SimpleNamespace(max_num_batched_tokens=4136))
    assert warmup._run_flashinfer_autotune_dummy_runs(runner) == 4096
    assert warmup.resolve_flashinfer_autotune_file(runner) == shared_cache
    state = _AlignmentState(4136, tmp_path/'private'/'cache.json', tmp_path/'report.json')
    setattr(runner, _STATE, state)
    assert warmup._run_flashinfer_autotune_dummy_runs(runner, skip_attn=True) == 8192
    assert warmup.resolve_flashinfer_autotune_file(runner) == state.cache_path
    assert calls == [(4136, 4136, {}), (4136, 4136, {'skip_attn': True})]
    assert state.status == 'complete'
    assert state.report_path.exists()
    assert not shared_cache.exists()


def test_seed_preserves_exact_bytes_and_does_not_overwrite_private_cache(tmp_path):
    source = tmp_path / 'baseline.json'
    payload = b'{"old4136":"keep this tactic unchanged"}\n'
    source.write_bytes(payload)
    state = _AlignmentState(4136, tmp_path/'private'/'cache.json', tmp_path/'report.json')
    _seed_cache(state, source)
    assert source.read_bytes() == state.cache_path.read_bytes() == payload
    assert state.seed['copied_to_private_cache'] is True
    state.cache_path.write_text('already tuned new 8192')
    _seed_cache(state, source)
    assert source.read_bytes() == payload
    assert state.cache_path.read_text() == 'already tuned new 8192'
    assert state.seed['copied_to_private_cache'] is False


def test_unsupported_gemm_abi_fails_closed():
    config = Config(dynamic_tensor_specs=(Spec(lambda n: n),))
    gemm = SimpleNamespace(**{name: config for name in _CONFIG_NAMES})
    with pytest.raises(RuntimeError, match='Unsupported FlashInfer GEMM bucket mapping'):
        _install_warmup_adapter(SimpleNamespace(), object, gemm, None)
