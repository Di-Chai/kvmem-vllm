import time
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
