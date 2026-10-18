"""Opt-in direct CPU sealing is validated and transported before model startup."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def enabled(**updates):
    return dict(archive_backend='cpu', cpu_read_cache_bytes=1024**3,
                async_seal=True, cpu_direct_record_seal=True, **updates)


def test_default_false_and_existing_positional_constructor_arguments_keep_meaning():
    baseline = RuntimeOptions()
    assert baseline.cpu_direct_record_seal is False
    names = [field.name for field in fields(RuntimeOptions)]
    assert names[-3:] == ['_policy_explicit_fields', '_policy_resolved_values', 'cpu_direct_record_seal']
    legacy_arguments = [getattr(baseline, name) for name in names[:-1]]
    restored = RuntimeOptions(*legacy_arguments)
    assert restored == baseline and restored.cpu_direct_record_seal is False
    assert runtime().cpu_direct_record_seal is False


def test_connector_parse_is_nonmutating_and_replacement_validates_dependencies():
    payload = enabled()
    original = deepcopy(payload)
    options = runtime(payload)
    assert options.cpu_direct_record_seal is True and payload == original
    assert runtime(json.loads(json.dumps(payload))) == options
    assert replace(options, cpu_direct_record_seal=False).cpu_direct_record_seal is False
    with pytest.raises(ValueError, match='requires archive_backend=cpu and async_seal=true'):
        replace(options, async_seal=False)


@pytest.mark.parametrize('value', [0, 1, 'false', 'true', None, [], {}])
def test_seal_boolean_is_strict_for_constructor_and_connector(value):
    with pytest.raises(ValueError, match='cpu_direct_record_seal must be boolean'):
        RuntimeOptions(cpu_direct_record_seal=value)
    payload = enabled()
    payload['cpu_direct_record_seal'] = value
    with pytest.raises(ValueError, match='cpu_direct_record_seal must be boolean'):
        runtime(payload)


@pytest.mark.parametrize('backend,async_seal', [('nvme', False), ('nvme', True), ('cpu', False)])
def test_opt_in_requires_cpu_backend_and_async_seal(backend, async_seal):
    payload = enabled()
    payload.update(archive_backend=backend, async_seal=async_seal)
    with pytest.raises(ValueError, match='requires archive_backend=cpu and async_seal=true'):
        runtime(payload)
    payload['cpu_direct_record_seal'] = False
    assert runtime(payload).cpu_direct_record_seal is False


@pytest.mark.parametrize('value', [False, True])
def test_direct_resolved_profile_and_worker_json_roundtrip(tmp_path, value):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(enabled())
    profile['kvmem']['cpu_direct_record_seal'] = value
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'direct')
    for payload in (launch['profile']['kvmem'], worker_payload(launch)):
        assert payload['cpu_direct_record_seal'] is value
        assert runtime(payload).cpu_direct_record_seal is value
    restored = build_launch(json.loads(json.dumps(launch['profile'])), tmp_path, tmp_path/'reload')
    assert worker_payload(restored) == worker_payload(launch)
    assert profile == original


@pytest.mark.parametrize('value', [None, False, True])
def test_elastic_default_explicit_boolean_and_worker_json_roundtrip(tmp_path, value):
    profile = elastic_profile(tmp_path)
    profile.update(archive_backend='cpu', async_seal=True)
    if value is not None:
        profile['cpu_direct_record_seal'] = value
    original = deepcopy(profile)
    config, _ = elastic_launch(profile, tmp_path, tmp_path/'elastic')
    backend = config['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path/'native'))
    for payload in (config, backend['kvmem'], native):
        assert payload['cpu_direct_record_seal'] is (value is True)
    restored, _ = elastic_launch(json.loads(json.dumps(config)), tmp_path, tmp_path/'reload')
    assert restored['mean_k_backend_profile'] == backend
    assert profile == original


@pytest.mark.parametrize('updates,match', [
    ({'cpu_direct_record_seal': 1}, 'must be boolean'),
    ({'cpu_direct_record_seal': 'true'}, 'must be boolean'),
    ({'cpu_direct_record_seal': None}, 'must be boolean'),
    ({'archive_backend': 'nvme'}, 'requires archive_backend=cpu and async_seal=true'),
    ({'async_seal': False}, 'requires archive_backend=cpu and async_seal=true'),
])
def test_elastic_rejects_invalid_opt_in_before_backend_start(tmp_path, updates, match):
    profile = elastic_profile(tmp_path)
    profile.update(archive_backend='cpu', async_seal=True, cpu_direct_record_seal=True)
    profile.update(updates)
    with pytest.raises(ValueError, match=match):
        elastic_launch(profile, tmp_path, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_elastic_full_context_cannot_silently_ignore_cpu_direct_sealing(tmp_path):
    profile = dict(model='unused', tokenizer='unused', users_file='unused',
        context_mode='full', archive_backend='cpu', async_seal=True, cpu_direct_record_seal=True)
    with pytest.raises(ValueError, match='requires MeanK context management'):
        elastic_launch(profile, tmp_path, tmp_path/'invalid')
