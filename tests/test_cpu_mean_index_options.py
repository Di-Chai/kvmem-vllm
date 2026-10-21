"""MeanK SIMD is opt-in and survives launcher/worker configuration transport."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def test_default_false_and_connector_roundtrip():
    assert RuntimeOptions().cpu_mean_index_simd is False
    payload = dict(mean_k_backend='qw3_cuda', cpu_mean_index_simd=True)
    original = deepcopy(payload)
    options = runtime(payload)
    assert options.cpu_mean_index_simd and payload == original
    assert runtime(json.loads(json.dumps(payload))) == options
    with pytest.raises(ValueError, match='requires mean_k_backend=qw3_cuda'):
        replace(options, mean_k_backend='torch_reference')


@pytest.mark.parametrize('value', [0, 1, None, 'true', [], {}])
def test_strict_boolean(value):
    with pytest.raises(ValueError, match='cpu_mean_index_simd must be boolean'):
        RuntimeOptions(cpu_mean_index_simd=value)
    with pytest.raises(ValueError, match='cpu_mean_index_simd must be boolean'):
        runtime(dict(cpu_mean_index_simd=value))


@pytest.mark.parametrize('enabled', [False, True])
def test_direct_and_elastic_transport(tmp_path, enabled):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(mean_k_backend='qw3_cuda', cpu_mean_index_simd=enabled)
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'direct')
    assert runtime(worker_payload(launch)).cpu_mean_index_simd is enabled
    assert profile == original
    (tmp_path/'elastic-fixture').mkdir()
    elastic = elastic_profile(tmp_path/'elastic-fixture')
    elastic.update(mean_k_backend='qw3_cuda', cpu_mean_index_simd=enabled)
    config, _ = elastic_launch(elastic, tmp_path, tmp_path/'elastic')
    backend = config['mean_k_backend_profile']
    worker = worker_payload(build_launch(backend, tmp_path, tmp_path/'worker'))
    assert worker['cpu_mean_index_simd'] is enabled
    assert runtime(worker).cpu_mean_index_simd is enabled


def test_invalid_backend_rejected_before_start(tmp_path):
    with pytest.raises(ValueError, match='requires mean_k_backend=qw3_cuda'):
        runtime(dict(cpu_mean_index_simd=True))
    profile = dict(model='unused', tokenizer='unused', users_file='unused',
                   context_mode='full', cpu_mean_index_simd=True)
    with pytest.raises(ValueError, match='requires MeanK context management'):
        elastic_launch(profile, tmp_path, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
