"""CPU-only synthetic cache files and fake launchers; no torch/GPU imports."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prefill_inductor_control.py'
spec = importlib.util.spec_from_file_location('prefill_inductor_control', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
A, B, C, D = (letter * 64 for letter in 'abcd')


def key_for_source(path):
    return path.parent / (c._sha(path.name.encode()) + '.best_config')


def source_text(name, *, extra=''):
    return f'''import triton
@triton_heuristics.reduction(filename=__file__, inductor_meta={{'kernel_name': '{name}', 'autotune_local_cache': True{extra}}})
@triton.jit
def {name}(x):
    return x + 1
'''


def make_source(root, hash_, name='kernel', *, block=512):
    cache = root / 'torch_aot_compile' / hash_ / 'inductor_cache'
    source = cache / 'aa' / (name + '.py')
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(source_text(name))
    config = key_for_source(source)
    config.write_text(json.dumps(dict(XBLOCK=block, num_warps=8, num_stages=1,
        configs_hash='e' * 64, found_by_coordesc=False, time_taken_ms=123)))
    log = root / (hash_ + '.log')
    log.write_text('saved AOT compiled function to ' + str(cache.parent / 'rank_0_0/model') + '\n')
    descriptor = dict(cache_dir=str(cache), server_log=str(log), required_common=[str(config.relative_to(cache))])
    return descriptor, source, config


def setup(tmp_path, *, generated=True, second=False):
    arm = tmp_path / 'arm'
    arm.mkdir()
    (arm / 'manifest.json').write_text('{"preserved":true}')
    target, ts, tc = make_source(tmp_path / 'global-target', A)
    mtp, ms, mc = make_source(tmp_path / 'global-mtp', B, block=256)
    c.freeze_seed(arm, dict(target=target, mtp=mtp), dict(target=[C], mtp=[D]), _key_for_source=key_for_source)
    run = arm / 'run'
    run.mkdir()
    env = {'UNCHANGED': 'yes', 'VLLM_CACHE_ROOT': 'old-global'}
    c.install_seed(arm, run, env)
    destinations = {}
    lines = []
    for role, h, src in [('target', C, ts), ('mtp', D, ms)]:
        cache = Path(env['VLLM_CACHE_ROOT']) / 'torch_compile_cache/torch_aot_compile' / h / 'inductor_cache'
        dest = cache / 'aa' / src.name
        if generated:
            dest.write_bytes(src.read_bytes())
        model = cache.parent / 'rank_0_0/model'
        model.mkdir(parents=True)
        lines.append('INFO saved AOT compiled function to ' + str(model))
        destinations[role] = dest
    (run / 'server.log').write_text('\n'.join(lines))
    return arm, env, destinations, (target, mtp), (ts, ms)


def fake_class():
    class Autotuner:
        def _pre_launch(self, launcher, *args, **kwargs):
            self.called += 1
            return 'native'
    return Autotuner


def launch(cls, path, *, config=None, state='hit', single=True, source=None):
    tuner = cls()
    chosen = c._configuration(json.loads(key_for_source(path).read_text()))
    chosen.update(config or {})
    launcher = SimpleNamespace(config=SimpleNamespace(kwargs={key: val for key, val in chosen.items() if key not in ('num_warps', 'num_stages')},
        num_warps=chosen['num_warps'], num_stages=chosen['num_stages']))
    tuner.filename = str(path)
    text = path.read_text()
    func = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef))
    tuner.fn = SimpleNamespace(__name__=func.name, src=source or ast.get_source_segment(text, func))
    tuner.inductor_meta = {}
    tuner.autotune_cache_info = {'autotune_cache_state': state}
    tuner.launchers = [launcher] if single else [launcher, launcher]
    tuner.called = 0
    assert tuner._pre_launch(launcher, object(), stream='stream') == 'native'
    return tuner, launcher


def record_startup(arm, destinations):
    cls = fake_class()
    original = cls._pre_launch
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    for path in destinations.values():
        tuner, launcher = launch(cls, path)
        tuner._pre_launch(launcher, stream='second')
    result = handle.close(success=True)
    assert cls._pre_launch is original
    assert len(result['records']) == len(destinations)
    return handle


def test_no_seed_legacy_noop(tmp_path):
    assert c.validate_seed(tmp_path) is None
    assert c.install_seed(tmp_path, tmp_path, {}) is None
    assert c.audit_seed(tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_isolated_namespace_sources_exact_and_actual_startup(tmp_path):
    arm, env, destinations, descriptors, originals = setup(tmp_path)
    assert env['UNCHANGED'] == 'yes'
    assert Path(env['TORCHINDUCTOR_CACHE_DIR']).is_dir()
    assert Path(env['VLLM_CACHE_ROOT']).is_relative_to(arm)
    assert json.loads((arm / 'manifest.json').read_text())['preserved'] is True
    manifest = c.validate_seed(arm)
    # Same key legitimately has distinct choices in independent module namespaces.
    assert manifest['modules']['target']['entries'][0]['selected_config']['XBLOCK'] == 512
    assert manifest['modules']['mtp']['entries'][0]['selected_config']['XBLOCK'] == 256
    assert {p.suffix for p in Path(env['VLLM_CACHE_ROOT']).rglob('*') if p.is_file()} == {'.py', '.best_config'}
    handle = record_startup(arm, destinations)
    assert handle.close(success=True) is handle.result
    audit = c.audit_seed(arm, _key_for_source=key_for_source)
    assert audit['launch_verified'] is True
    assert audit['common_configurations_unchanged'] is True
    assert audit['formal_dispatch_observer'] is False
    assert audit['launch_records'] == 2
    originals[0].write_text('global source changed after freezing')
    assert c.validate_seed(arm) == manifest


def test_installer_does_not_copy_source_graph_or_binary(tmp_path):
    arm, env, _, _, _ = setup(tmp_path, generated=False)
    assert {path.suffix for path in Path(env['VLLM_CACHE_ROOT']).rglob('*') if path.is_file()} == {'.best_config'}
    assert list(Path(env['TORCHINDUCTOR_CACHE_DIR']).iterdir()) == []
    with pytest.raises(ValueError, match='already exists'):
        c.install_seed(arm, arm / 'run', env)


@pytest.mark.parametrize('marker', ['saved AOT compiled function to ', 'Directly load AOT compilation from path ', 'Compiling model again due to a load failure from '])
def test_actual_log_markers(tmp_path, marker):
    log = tmp_path / 'server.log'
    path = tmp_path / 'torch_aot_compile' / A / 'rank_0_0/model'
    log.write_text('\x1b[31m' + marker + str(path) + ', reason: diagnostic\x1b[0m\n')
    assert c.logged_aot_paths(log) == [path]


@pytest.mark.parametrize('kind', ['duplicate-role-hash', 'unlogged-source', 'missing-source', 'required-unknown', 'wrong-config-hash', 'retuning', 'invalid-hash'])
def test_freeze_rejects_before_writes(tmp_path, kind):
    arm = tmp_path / 'arm'; arm.mkdir(); (arm / 'manifest.json').write_text('{}')
    target, source, config = make_source(tmp_path / 'global', A)
    sources, allowed = dict(target=target), dict(target=[C])
    if kind == 'duplicate-role-hash':
        sources['mtp'] = target; allowed['mtp'] = [C]
    elif kind == 'unlogged-source':
        Path(target['server_log']).write_text('model started')
    elif kind == 'missing-source':
        source.unlink()
    elif kind == 'required-unknown':
        target['required_common'] = ['aa/' + B + '.best_config']
    elif kind == 'wrong-config-hash':
        data = json.loads(config.read_text()); data['configs_hash'] = 'bad'; config.write_text(json.dumps(data))
    elif kind == 'retuning':
        source.write_text(source_text('kernel', extra=", 'coordinate_descent_tuning': True"))
    else:
        allowed['target'] = ['bad']
    with pytest.raises(ValueError):
        c.freeze_seed(arm, sources, allowed, _key_for_source=key_for_source)
    assert not (arm / c.DIRECTORY).exists()


@pytest.mark.parametrize('change', ['config', 'timing', 'source', 'configs_hash'])
def test_same_role_key_cross_directory_conflicts_fail(tmp_path, change):
    arm = tmp_path / 'arm'; arm.mkdir(); (arm / 'manifest.json').write_text('{}')
    d1, s1, c1 = make_source(tmp_path / 'one', A)
    d2, s2, c2 = make_source(tmp_path / 'two', B)
    if change == 'source':
        s2.write_text(s2.read_text().replace('x + 1', 'x + 2'))
    else:
        data = json.loads(c2.read_text()); data[{'config':'XBLOCK', 'timing':'time_taken_ms', 'configs_hash':'configs_hash'}[change]] = 'f' * 64 if change == 'configs_hash' else 2048
        c2.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Conflicting'):
        c.freeze_seed(arm, dict(target=[d1, d2]), dict(target=[C]), _key_for_source=key_for_source)
    assert not (arm / c.DIRECTORY).exists()


def test_identical_duplicate_source_is_deduplicated(tmp_path):
    arm = tmp_path / 'arm'; arm.mkdir(); (arm / 'manifest.json').write_text('{}')
    d1, _, _ = make_source(tmp_path / 'one', A)
    d2, _, _ = make_source(tmp_path / 'two', B)
    manifest = c.freeze_seed(arm, dict(target=[d1, d2]), dict(target=[C]), _key_for_source=key_for_source)
    assert len(manifest['modules']['target']['entries']) == 1
    assert len(manifest['modules']['target']['sources']) == 2


@pytest.mark.parametrize('change', ['config', 'source', 'manifest', 'chain'])
def test_frozen_mutation_rejected(tmp_path, change):
    arm, _, _, _, _ = setup(tmp_path)
    if change == 'config':
        path = next((arm / c.DIRECTORY / 'target/configs').rglob('*.best_config'))
    elif change == 'source':
        path = next((arm / c.DIRECTORY / 'target/sources').rglob('*.py'))
    elif change == 'manifest':
        path = arm / c.DIRECTORY / 'manifest.json'
    else:
        path = arm / 'manifest.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        c.validate_seed(arm)


@pytest.mark.parametrize('change', ['outside-log', 'unapproved-hash', 'no-log', 'source', 'config', 'missing-required', 'no-launch-proof'])
def test_actual_audit_is_fail_closed(tmp_path, change):
    arm, _, destinations, _, _ = setup(tmp_path)
    record_startup(arm, destinations)
    log = arm / 'run/server.log'
    if change == 'outside-log':
        log.write_text(log.read_text().replace(str(arm / 'run/vllm_cache'), str(tmp_path / 'global')))
    elif change == 'unapproved-hash':
        log.write_text(log.read_text().replace(C, A))
    elif change == 'no-log':
        log.write_text('no actual paths')
    elif change == 'source':
        destinations['target'].write_text(destinations['target'].read_text() + '# changed\n')
    elif change == 'config':
        path = key_for_source(destinations['target']); data = json.loads(path.read_text()); data['XBLOCK'] = 1; path.write_text(json.dumps(data))
    elif change == 'missing-required':
        destinations['target'].unlink()
    else:
        (arm / 'run/inductor-startup-123.json').unlink()
    with pytest.raises(ValueError):
        c.audit_seed(arm, _key_for_source=key_for_source)


@pytest.mark.parametrize('change', ['config', 'state', 'multiple', 'function'])
def test_observer_rejects_wrong_actual_launch_restores_on_failure(tmp_path, change):
    arm, _, destinations, _, _ = setup(tmp_path)
    cls = fake_class(); original = cls._pre_launch
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    kwargs = {'config': {'XBLOCK': 1024}} if change == 'config' else {'state':'miss'} if change == 'state' else {'single':False} if change == 'multiple' else {'source':'def kernel(x):\n    return x + 2'}
    with pytest.raises(ValueError, match='frozen configuration'):
        launch(cls, destinations['target'], **kwargs)
    report = handle.close(success=False)
    assert cls._pre_launch is original
    assert report['success'] is False and report['error']
    with pytest.raises(ValueError, match='close successfully'):
        c.audit_seed(arm, _key_for_source=key_for_source)


def test_missing_required_startup_fails_before_formal_and_always_restores(tmp_path):
    arm, _, destinations, _, _ = setup(tmp_path)
    cls = fake_class(); original = cls._pre_launch
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    launch(cls, destinations['target'])
    with pytest.raises(ValueError, match='Missing required'):
        handle.close(success=True)
    assert cls._pre_launch is original
    assert handle.close(success=False)['success'] is False
    with pytest.raises(ValueError, match='close successfully'):
        c.audit_seed(arm, _key_for_source=key_for_source)


def test_failed_model_startup_writes_failure_without_coverage_claim(tmp_path):
    arm, _, _, _, _ = setup(tmp_path)
    cls = fake_class(); original = cls._pre_launch
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    report = handle.close(success=False)
    assert cls._pre_launch is original and not report['success']
    assert handle.close(success=False) is report


def test_startup_observer_cannot_stack_or_overwrite_report(tmp_path):
    arm, _, destinations, _, _ = setup(tmp_path)
    cls = fake_class()
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    with pytest.raises(ValueError, match='already installed'):
        c.install_launch_audit(arm / 'run/inductor-startup-124.json', _autotuner_type=cls)
    handle.close(success=False)
    with pytest.raises(FileExistsError):
        c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)


def test_startup_cannot_be_installed_outside_run(tmp_path):
    arm, _, _, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='within target_run'):
        c.install_launch_audit(arm / 'inductor-startup-123.json', _autotuner_type=fake_class())


def test_actual_cache_without_hook_is_explicitly_unverified(tmp_path):
    arm, _, _, _, _ = setup(tmp_path)
    report = c.audit_seed(arm, require_launch_audit=False, _key_for_source=key_for_source)
    assert report['common_configurations_unchanged'] is True
    assert report['launch_verified'] is False


def test_required_missing_is_not_classified_as_unused(tmp_path):
    arm, _, destinations, _, _ = setup(tmp_path)
    record_startup(arm, destinations)
    # A different kernel still exists so the module isn't rejected just as empty.
    path = destinations['target']
    other = path.with_name('other.py'); other.write_text(source_text('other'))
    key_for_source(other).write_bytes(key_for_source(path).read_bytes())
    path.unlink()
    with pytest.raises(ValueError, match='No actual generated common|Required common'):
        c.audit_seed(arm, _key_for_source=key_for_source)


@pytest.mark.parametrize('kwargs', [{'state': 'miss'}, {'single': False}, {'source': 'def kernel(x):\n    return x + 99'}])
def test_second_autotuner_cannot_hide_invalid_state_behind_same_file_config(tmp_path, kwargs):
    arm, _, destinations, _, _ = setup(tmp_path)
    cls = fake_class()
    handle = c.install_launch_audit(arm / 'run/inductor-startup-123.json', _autotuner_type=cls)
    launch(cls, destinations['target'])
    with pytest.raises(ValueError, match='frozen configuration'):
        launch(cls, destinations['target'], **kwargs)
    assert len(handle.records) == 1
    assert handle.close(success=False)['success'] is False
