"""CPU read prefetch is opt-in and requires an existing bounded pinned pool."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def settings(**kwargs):
    return dict(archive_backend='cpu', cpu_read_cache_bytes=1024**3,
                packed_assembly=True, direct_pinned_reads=True,
                assembly_read_prefetch=True, **kwargs)


def test_default_and_worker_roundtrip():
    assert RuntimeOptions().assembly_read_prefetch is False
    payload = settings()
    original = deepcopy(payload)
    options = runtime(payload)
    assert options.assembly_read_prefetch and payload == original
    assert runtime(json.loads(json.dumps(payload))) == options
    with pytest.raises(ValueError, match='requires CPU archive'):
        replace(options, archive_backend='nvme')


@pytest.mark.parametrize('value', [0, 1, None, 'true', [], {}])
def test_boolean_is_strict(value):
    with pytest.raises(ValueError, match='assembly_read_prefetch must be boolean'):
        RuntimeOptions(assembly_read_prefetch=value)
    payload = settings()
    payload['assembly_read_prefetch'] = value
    with pytest.raises(ValueError, match='assembly_read_prefetch must be boolean'):
        runtime(payload)


@pytest.mark.parametrize('updates', [dict(archive_backend='nvme'), dict(packed_assembly=False),
                                   dict(direct_pinned_reads=False)])
def test_missing_pool_or_backend_rejected(updates):
    payload = settings()
    payload.update(updates)
    with pytest.raises(ValueError):
        runtime(payload)


@pytest.mark.parametrize('enabled', [False, True])
def test_direct_and_elastic_transport(tmp_path, enabled):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(settings())
    profile['kvmem']['assembly_read_prefetch'] = enabled
    native = build_launch(profile, tmp_path, tmp_path/'native')
    assert runtime(worker_payload(native)).assembly_read_prefetch is enabled
    (tmp_path/'elastic-fixture').mkdir()
    profile = elastic_profile(tmp_path/'elastic-fixture')
    profile.update(archive_backend='cpu', assembly_read_prefetch=enabled)
    config, _ = elastic_launch(profile, tmp_path, tmp_path/'elastic')
    assert config['mean_k_backend_profile']['kvmem']['assembly_read_prefetch'] is enabled
    worker = build_launch(config['mean_k_backend_profile'], tmp_path, tmp_path/'worker')
    assert runtime(worker_payload(worker)).assembly_read_prefetch is enabled
