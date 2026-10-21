from copy import deepcopy

import pytest

from vllm_kvmem.options import RuntimeOptions
from test_options import config
from test_prefill_pipeline_options import profile


def enabled():
    value = profile()
    value['kvmem'].update(prefill_native_plan_arm=True, action_native_scores='reuse')
    return value


def test_native_plan_arm_is_opt_in_and_round_trips_without_mutation():
    assert not RuntimeOptions().prefill_native_plan_arm
    value = enabled()
    original = deepcopy(value)
    assert RuntimeOptions.from_vllm(config(value)).prefill_native_plan_arm
    assert value == original


@pytest.mark.parametrize('key,value,match', [
    ('prefill_native_plan_arm', 1, 'boolean'),
    ('prefill_publication_pipeline', False, 'requires the publication'),
    ('action_native_scores', 'off', 'requires the publication'),
])
def test_native_plan_arm_requires_complete_prepared_plans(key, value, match):
    options = enabled()
    options['kvmem'][key] = value
    with pytest.raises(ValueError, match=match):
        RuntimeOptions.from_vllm(config(options))
