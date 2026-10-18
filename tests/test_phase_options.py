import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.options import RuntimeOptions


def profile():
    return json.loads((Path(__file__).parents[1] /
        'examples/qwen-nvfp4-offline-agent-k128-g128.json').read_text())


def options(payload):
    return RuntimeOptions.from_vllm(NS(
        kv_transfer_config=NS(kv_connector_extra_config={'kvmem': payload}),
        cache_config=NS(cache_dtype='fp8_e4m3')))


def test_phase_is_explicit_opt_in_and_roundtrips_public_connector_settings():
    payload = profile()['kvmem']
    assert options(payload).offline_phase_policy is False
    expected = copy.deepcopy(payload)
    payload.update(offline_phase_policy=True, phase_prefill_wave_tokens=8192, phase_max_wait_ms=1000.)
    resolved = options(payload)
    assert resolved.offline_phase_policy is True
    assert resolved.phase_prefill_wave_tokens == 8192 and resolved.phase_max_wait_ms == 1000.
    assert all(payload[k] == v for k, v in expected.items())


def test_disabled_phase_does_not_restrict_existing_small_generation_budgets():
    from vllm_kvmem.config import KVMemConfig
    resolved = RuntimeOptions(memory=KVMemConfig(gen_budget=1024))
    assert not resolved.offline_phase_policy and resolved.memory.gen_budget == 1024


@pytest.mark.parametrize('changes,match', [
    ({'offline_phase_policy': 1}, 'boolean'),
    ({'offline_agent_policy': False}, 'requires'),
    ({'phase_prefill_wave_tokens': 0}, 'wave'),
    ({'phase_prefill_wave_tokens': 8193}, 'wave'),
    ({'phase_prefill_wave_tokens': 262144}, 'wave'),
    ({'phase_prefill_wave_tokens': True}, 'wave'),
    ({'phase_max_wait_ms': 0.}, 'wait'),
    ({'phase_max_wait_ms': float('inf')}, 'wait'),
    ({'phase_max_wait_ms': True}, 'wait'),
    ({'prefill_cadence': 2}, 'independent'),
    ({'prefill_batch_tokens': 512}, 'independent'),
    ({'adaptive_prefill': True}, 'independent'),
])
def test_invalid_or_competing_phase_controls_are_rejected(changes, match):
    payload = profile()['kvmem']
    payload.update(offline_phase_policy=True)
    payload.update(changes)
    with pytest.raises(ValueError, match=match):
        options(payload)
