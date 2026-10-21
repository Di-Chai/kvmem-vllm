"""Parallel CPU packing is opt-in and survives every public launch boundary."""
from copy import deepcopy
from dataclasses import fields, replace
import json

import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from test_historical_options import settings as historical_options
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from vllm_kvmem.storage import make_shared_archive


def enabled(**updates):
    return dict(archive_backend='cpu', cpu_read_cache_bytes=1024**3,
                async_seal=True, cpu_direct_record_seal=True,
                cpu_seal_pack_workers=4, **updates)


def test_default_and_legacy_positional_constructor_are_unchanged():
    baseline = RuntimeOptions()
    assert baseline.cpu_seal_pack_workers == runtime().cpu_seal_pack_workers == 1
    names = [field.name for field in fields(RuntimeOptions)]
    index = names.index('cpu_seal_pack_workers')
    assert RuntimeOptions(*[getattr(baseline, key) for key in names[:index]]) == baseline


@pytest.mark.parametrize('value', [0, 9, True, False, 1.0, '4', None])
def test_invalid_workers_fail_during_configuration(value):
    with pytest.raises(ValueError, match='cpu_seal_pack_workers must be an integer'):
        RuntimeOptions(cpu_seal_pack_workers=value)
    with pytest.raises(ValueError, match='cpu_seal_pack_workers must be an integer'):
        runtime({**enabled(), 'cpu_seal_pack_workers': value})


def test_parallel_pack_requires_direct_sealing_and_rejects_native_payload_storage():
    with pytest.raises(ValueError, match='requires direct CPU record sealing'):
        runtime({**enabled(), 'cpu_direct_record_seal': False})
    options = runtime(enabled())
    with pytest.raises(ValueError, match='requires direct CPU record sealing'):
        replace(options, cpu_direct_record_seal=False)
    with pytest.raises(ValueError, match='without historical_gc'):
        replace(historical_options(), cpu_direct_record_seal=True, cpu_seal_pack_workers=4)


@pytest.mark.parametrize('workers', [1, 4, 8])
def test_profile_worker_and_shared_archive_roundtrip(tmp_path, workers):
    profile = model_profile(tmp_path)
    profile['kvmem'].update({**enabled(), 'cpu_seal_pack_workers': workers})
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'pack')
    for payload in (launch['profile']['kvmem'], worker_payload(launch)):
        assert runtime(payload).cpu_seal_pack_workers == workers
    assert profile == original
    restored = build_launch(json.loads(json.dumps(launch['profile'])), tmp_path, tmp_path/'reload')
    assert worker_payload(restored) == worker_payload(launch)
    archive = make_shared_archive(runtime(worker_payload(launch)), tmp_path/'unused', 4, 8)
    try:
        assert archive.cpu_seal_pack_workers == workers
        assert archive.io_usage()['cpu_seal_pack_workers'] == workers
    finally:
        assert archive.close()


@pytest.mark.parametrize('workers', [None, 1, 4, 8])
def test_elastic_roundtrip_keeps_explicit_workers_and_default(tmp_path, workers):
    profile = elastic_profile(tmp_path)
    profile.update(archive_backend='cpu', async_seal=True, cpu_direct_record_seal=True)
    if workers is not None:
        profile['cpu_seal_pack_workers'] = workers
    original = deepcopy(profile)
    config, _ = elastic_launch(profile, tmp_path, tmp_path/'elastic')
    backend = config['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path/'native'))
    for payload in (config, backend['kvmem'], native):
        assert payload['cpu_seal_pack_workers'] == (workers or 1)
    assert profile == original


def test_full_context_rejects_ignored_parallel_pack(tmp_path):
    profile = dict(model='unused', tokenizer='unused', users_file='unused', context_mode='full',
        archive_backend='cpu', async_seal=True, cpu_direct_record_seal=True, cpu_seal_pack_workers=4)
    with pytest.raises(ValueError, match='requires MeanK context management'):
        elastic_launch(profile, tmp_path, tmp_path/'invalid')
