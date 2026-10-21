import json
import time
from copy import deepcopy
from dataclasses import replace

import pytest

from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.historical_gc import HistoricalGcSettings, historical_record_charge
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.storage import make_shared_archive
from test_options import config


def settings(mode='delete'):
    memory = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=12,
                         repository_blocks=32, sink_blocks=1, recent_blocks=0)
    return RuntimeOptions(memory=memory, archive_backend='cpu', cpu_read_cache_bytes=1536,
        async_archive_retirement=True, async_seal=True, request_planning=True,
        cpu_history_coordination=True, session_capacity=1, session_checkpoint_bytes=4096,
        historical_gc=dict(mode=mode, ready_low=2, ready_high=4))


@pytest.mark.parametrize('mode', ['off', 'shadow', 'delete'])
def test_public_modes_install_native_cpu_ownership_and_one_archive_worker(tmp_path, mode):
    options = settings(mode)
    archive = make_shared_archive(options, tmp_path, 24, 1, max_query_tokens=16)
    try:
        assert archive._native_payload_arena is not None and archive._history_worker is not None
        assert archive.async_seal and archive._history_background
        assert archive.history_record_charge == historical_record_charge(24, 4) == 42
        assert archive.tier.capacity == options.cpu_read_cache_bytes//42
    finally:
        deadline = time.monotonic()+5
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)


def test_connector_payload_is_parsed_without_mutating_user_configuration():
    payload = dict(block_tokens=4, select_budget=8, gen_budget=12, repository_blocks=32,
        sink_blocks=1, recent_blocks=0,
        archive_backend='cpu', cpu_read_cache_bytes=1536, async_archive_retirement=True,
        async_seal=True, request_planning=True, cpu_history_coordination=True,
        session_capacity=1, session_checkpoint_bytes=4096, historical_gc={'mode':'shadow'})
    parsed = RuntimeOptions.from_vllm(config({'kvmem':payload}))
    assert parsed.historical_gc.mode == 'shadow'
    assert payload['historical_gc'] == {'mode':'shadow'}
    assert RuntimeOptions().historical_gc is None


def test_policy_defaults_parameters_and_native_config_own_their_values():
    legacy = HistoricalGcSettings.parse({'mode': 'delete'})
    assert legacy.policy == 'lru' and legacy.policy_params == {'keep_mandatory': True}
    payload = {'mode': 'shadow', 'policy': 'lru', 'policy_params': {'keep_mandatory': False}}
    parsed = HistoricalGcSettings.parse(payload)
    payload['policy_params']['keep_mandatory'] = True
    native = parsed.native(settings().memory, 2)
    assert native.policy == 'lru' and native.policy_params == {'keep_mandatory': False}
    parsed.policy_params['keep_mandatory'] = True
    assert native.policy_params == {'keep_mandatory': False}
    assert legacy.policy_params == {'keep_mandatory': True}


@pytest.mark.parametrize('value', [
    {'policy': 'fifo'}, {'policy': ''}, {'policy': 'LRU'}, {'policy': None}, {'policy': []},
    {'policy_params': None}, {'policy_params': []}, {'policy_params': False},
    {'policy_params': {'unknown': True}}, {'policy_params': {1: True}},
    *[{'policy_params': {'keep_mandatory': v}} for v in (0, 1, 'true', None, [], 1.0)],
])
def test_invalid_policy_settings_fail_before_startup(value):
    with pytest.raises(ValueError, match='historical_gc.policy'):
        HistoricalGcSettings.parse(value)


@pytest.mark.parametrize('keep', [True, False])
def test_policy_settings_survive_public_launch_worker_json_and_profile_replay(tmp_path, keep):
    from test_graph_options import model_profile, worker_payload, runtime
    from vllm_kvmem.serve import build_launch

    profile = model_profile(tmp_path)
    gc = dict(mode='shadow', policy='lru', policy_params={'keep_mandatory': keep})
    profile['kvmem'].update(graph_mode='off', archive_backend='cpu', cpu_read_cache_bytes=1024**3,
        async_seal=True, cpu_history_coordination=True, historical_gc=gc)
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'serve')
    worker = worker_payload(launch)
    assert worker['historical_gc'] == gc
    parsed = runtime(worker).historical_gc
    assert parsed.policy == 'lru' and parsed.policy_params == {'keep_mandatory': keep}
    replay = build_launch(json.loads(json.dumps(launch['profile'])), tmp_path, tmp_path/'replay')
    assert worker_payload(replay)['historical_gc'] == gc
    assert profile == original


@pytest.mark.parametrize('value', [False, 'delete', {'mode':'latest'}, {'poll_ms':True},
    {'scan_blocks':0}, {'delete_blocks':100000}, {'ready_low':3,'ready_high':2},
    {'ready_low':True,'ready_high':2}, {'mdoe':'off'}])
def test_invalid_historical_settings_fail_before_startup(value):
    with pytest.raises(ValueError, match='historical_gc'):
        HistoricalGcSettings.parse(value)


@pytest.mark.parametrize('change', [dict(repository_gc=True), dict(archive_backend='nvme'),
    dict(request_planning=False), dict(async_seal=False), dict(async_archive_retirement=False),
    dict(cpu_history_coordination=False), dict(session_capacity=0)])
def test_modes_require_safe_serving_lifecycle(change):
    with pytest.raises(ValueError, match='historical_gc'):
        replace(settings(), **change)
