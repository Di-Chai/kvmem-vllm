"""Pure stdlib fixture files: no model, GPU, global cache, or native extension."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/prefill_autotune_control.py'
spec = importlib.util.spec_from_file_location('prefill_autotune_control', PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
A, B = 'a' * 64, 'b' * 64


def setup(tmp_path):
    arm = tmp_path / 'arm'
    arm.mkdir()
    (arm / 'manifest.json').write_text('{"profile_sha256":"preserved"}')
    source = tmp_path / 'canonical.json'
    source.write_text(json.dumps(dict(_metadata=dict(gpu='fixture', flashinfer='fixed'),
        _generation='old-generation', key_a=['Runner', 0], key_b=['Runner', 4])) + '\n')
    control.freeze_seed(arm, source, [B, A])
    run = arm / 'run'
    run.mkdir()
    env = {'KEEP': 'unchanged'}
    control.install_seed(arm, run, env)
    cache = Path(env[control.ENVIRONMENT_KEY]) / A / 'autotune_configs.json'
    (run / 'server.log').write_text('INFO ' + control.LOG_MARKER + ' ' + str(cache) + '\n')
    return arm, source, cache, env


def test_legacy_without_seed_is_noop_even_without_manifest(tmp_path):
    assert control.validate_seed(tmp_path) is None
    assert control.install_seed(tmp_path, tmp_path, {}) is None
    assert control.audit_seed(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_seed_exact_bytes_sorted_hashes_chain_and_isolated_install(tmp_path):
    arm, source, cache, env = setup(tmp_path)
    manifest = control.validate_seed(arm)
    assert manifest['allowed_hashes'] == [A, B]
    assert manifest['config_count'] == 2
    assert cache.read_bytes() == source.read_bytes() == (arm / 'flashinfer_seed/seed.json').read_bytes()
    assert env['KEEP'] == 'unchanged'
    assert json.loads((arm / 'manifest.json').read_text())['profile_sha256'] == 'preserved'
    result = control.audit_seed(arm)
    assert result['actual_configs_sha256'] == manifest['configs_sha256']
    assert result['files'][0]['added_configs'] == {}
    # A later global-cache change cannot modify the frozen experiment seed.
    source.write_text('{}')
    assert control.validate_seed(arm) == manifest


@pytest.mark.parametrize('hashes', [[A, '../outside'], [A.upper()], ['x' * 64], [], A, [A, A]])
def test_invalid_hashes_are_rejected_before_any_write(tmp_path, hashes):
    arm = tmp_path / 'arm'
    arm.mkdir()
    (arm / 'manifest.json').write_text('{}')
    source = tmp_path / 'seed.json'
    source.write_text('{"_metadata":{},"key":["R",0]}')
    with pytest.raises(ValueError, match='hash'):
        control.freeze_seed(arm, source, hashes)
    assert not (arm / 'flashinfer_seed').exists()


@pytest.mark.parametrize('target', ['seed', 'seed_manifest', 'arm_chain'])
def test_seed_manifest_and_arm_chain_tampering_is_detected(tmp_path, target):
    arm, _, _, _ = setup(tmp_path)
    if target == 'seed':
        (arm / 'flashinfer_seed/seed.json').write_text('{}')
    elif target == 'seed_manifest':
        path = arm / 'flashinfer_seed/manifest.json'
        path.write_text(path.read_text() + ' ')
    else:
        (arm / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='hash|chain'):
        control.validate_seed(arm)


def test_missing_seed_is_fail_closed(tmp_path):
    arm, _, _, _ = setup(tmp_path)
    (arm / 'flashinfer_seed/seed.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        control.validate_seed(arm)


@pytest.mark.parametrize('kind', ['missing_log', 'no_path', 'outside', 'unapproved', 'relative', 'traversal'])
def test_missing_or_wrong_actual_log_path_fails(tmp_path, kind):
    arm, _, cache, _ = setup(tmp_path)
    log = arm / 'run/server.log'
    if kind == 'missing_log':
        log.unlink()
    elif kind == 'no_path':
        log.write_text('model started without evidence')
    else:
        path = {'outside': tmp_path / 'global/autotune_configs.json',
            'unapproved': cache.parent.parent / ('c' * 64) / 'autotune_configs.json',
            'relative': Path('cache.json'),
            'traversal': cache.parent / '..' / A / 'autotune_configs.json'}[kind]
        log.write_text(control.LOG_MARKER + ' ' + str(path))
    with pytest.raises(ValueError, match='log|path|hash|escapes'):
        control.audit_seed(arm)


@pytest.mark.parametrize('kind', ['changed_tactic', 'missing_tactic', 'metadata'])
def test_old_tactic_or_environment_changes_are_rejected(tmp_path, kind):
    arm, _, cache, _ = setup(tmp_path)
    data = json.loads(cache.read_text())
    if kind == 'changed_tactic':
        data['key_a'][1] = 4
    elif kind == 'missing_tactic':
        del data['key_b']
    else:
        data['_metadata']['gpu'] = 'another-GPU'
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='tactics changed|metadata differs'):
        control.audit_seed(arm)


def test_new_keys_are_explicit_and_generation_is_not_tactic_semantics(tmp_path):
    arm, _, cache, _ = setup(tmp_path)
    data = json.loads(cache.read_text())
    data['_generation'] = 'new-cache-generation'
    data['new_shape'] = ['Runner', 2]
    cache.write_text(json.dumps(data))
    result = control.audit_seed(arm)
    assert result['files'][0]['added_configs'] == {'new_shape': ['Runner', 2]}
    assert result['files'][0]['generation'] == 'new-cache-generation'
    assert result['actual_configs_sha256'] != result['seed_configs_sha256']
    assert result['seed_tactics_unchanged']


def test_symlink_escape_and_duplicate_install_are_rejected(tmp_path):
    arm, _, cache, env = setup(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        control.install_seed(arm, arm / 'run', env)
    original = cache.read_bytes()
    other = tmp_path / 'outside.json'
    other.write_bytes(original)
    cache.unlink()
    cache.symlink_to(other)
    with pytest.raises(ValueError, match='escapes'):
        control.audit_seed(arm)


def test_arms_compare_semantics_independent_of_allowed_hash_used(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir()
    arm_a, _, cache_a, _ = setup(first)
    arm_b, _, cache_b, _ = setup(second)
    cache_b = cache_b.parent.parent / B / cache_b.name
    (arm_b / 'run/server.log').write_text(control.LOG_MARKER + ' ' + str(cache_b))
    a, b = control.audit_seed(arm_a), control.audit_seed(arm_b)
    assert a['seed_sha256'] == b['seed_sha256']
    assert a['actual_configs_sha256'] == b['actual_configs_sha256']
    assert a['files'][0]['configuration_hash'] != b['files'][0]['configuration_hash']


def test_multiple_logged_caches_are_all_checked_and_report_disagreement(tmp_path):
    arm, _, cache, _ = setup(tmp_path)
    second = cache.parent.parent / B / cache.name
    data = json.loads(second.read_text())
    data['new_key'] = ['Runner', 5]
    second.write_text(json.dumps(data))
    with (arm / 'run/server.log').open('a') as log:
        log.write(control.LOG_MARKER + ' ' + str(second) + '\n')
    result = control.audit_seed(arm)
    assert len(result['files']) == 2
    assert result['loaded_files_semantically_identical'] is False
    assert result['actual_configs_sha256'] is None
