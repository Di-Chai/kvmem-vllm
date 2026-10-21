"""The publication pipeline is explicitly bounded and opt-in."""
from copy import deepcopy

import pytest

from vllm_kvmem.options import RuntimeOptions
from test_options import config


def profile():
    return {'kvmem': dict(prefill_publication_pipeline=True,
        async_cpu_scatter=True, split_capture_dma=True,
        action_attention=True, prefill_attention=True,
        session_capacity=1, session_checkpoint_bytes=4096,
        async_archive_retirement=True)}


def test_pipeline_defaults_off_and_public_config_round_trips_without_mutation():
    assert not RuntimeOptions().prefill_publication_pipeline
    value = profile()
    original = deepcopy(value)
    assert RuntimeOptions.from_vllm(config(value)).prefill_publication_pipeline
    assert value == original


@pytest.mark.parametrize('key,value,match', [
    ('prefill_publication_pipeline', 1, 'boolean'),
    ('async_cpu_scatter', False, 'requires async'),
    ('split_capture_dma', False, 'requires async'),
    ('prefill_attention', False, 'requires async'),
    ('action_attention', False, 'requires'),
    ('early_capture_slot_release', True, 'completion-owned'),
])
def test_pipeline_rejects_unqualified_lifetimes(key, value, match):
    options = profile()
    options['kvmem'][key] = value
    with pytest.raises(ValueError, match=match):
        RuntimeOptions.from_vllm(config(options))
