import pytest

from vllm_kvmem.options import RuntimeOptions
from test_options import config
from test_prefill_pipeline_options import profile


def test_aligned_pieces_is_explicit_boolean_native_only_and_roundtrips():
    assert RuntimeOptions().action_native_prefill_aligned_pieces is False
    payload = profile()['kvmem']
    for mode in ('reuse', 'verify'):
        options = RuntimeOptions.from_vllm(config({'kvmem': {**payload,
            'action_native_scores': mode, 'action_native_prefill_aligned_pieces': True}}))
        assert options.action_native_prefill_aligned_pieces is True
        assert options.action_native_scores == mode
    for mode, flag in (('off', True), ('reuse', 1), ('reuse', None), ('reuse', 'true')):
        with pytest.raises(ValueError, match='aligned pieces'):
            RuntimeOptions.from_vllm(config({'kvmem': {**payload,
                'action_native_scores': mode, 'action_native_prefill_aligned_pieces': flag}}))
