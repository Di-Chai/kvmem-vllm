"""GDN/checkpoint/oracle consumers retain the actual native forward extent."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.publication_step import native_publication_step


def row(retained, begin=8192, *, chunks=2):
    offset, count = (chunks-1)*2048, chunks*2048
    kept = min(2048, max(0, retained-offset))
    return dict(history_begin=begin+offset, scheduled=2048, accepted=kept,
        rejected=0, terminal_discarded=2048-kept,
        physical_step=dict(history_begin=begin, scheduled=count, accepted=retained,
                           rejected=0, terminal_discarded=count-retained))


@pytest.mark.parametrize('retained', [0, 1, 2048, 2049, 4095, 4096])
def test_physical_receipt_uses_full_prefix_even_when_final_logical_row_accepts_zero(retained):
    logical = row(retained)
    assert native_publication_step(logical) is logical['physical_step']
    assert native_publication_step(logical)['accepted'] == retained
    assert logical['scheduled'] == 2048


def test_legacy_receipt_remains_the_same_object_without_new_field_requirements():
    legacy = dict(history_begin=8, accepted=2, scheduled=4)
    assert native_publication_step(legacy) is legacy


@pytest.mark.parametrize('change', ['missing', 'bool', 'negative', 'sum', 'offset',
                                  'not_final', 'accepted', 'reject', 'recursive', 'type'])
def test_malformed_physical_receipt_fails_closed(change):
    logical = row(2049)
    physical = logical['physical_step']
    if change == 'missing': physical.pop('rejected')
    elif change == 'bool': physical['rejected'] = False
    elif change == 'negative': physical['history_begin'] = -1
    elif change == 'sum': physical['terminal_discarded'] += 1
    elif change == 'offset': physical['history_begin'] += 1
    elif change == 'not_final': logical['history_begin'] -= 32
    elif change == 'accepted': logical.update(accepted=0, terminal_discarded=2048)
    elif change == 'reject': physical.update(rejected=1, terminal_discarded=2046)
    elif change == 'recursive': physical['physical_step'] = {}
    elif change == 'type': logical['physical_step'] = None
    with pytest.raises(ValueError):
        native_publication_step(logical)


@pytest.mark.parametrize('retained', [0, 1, 2048, 2049, 4095, 4096])
def test_native_continuation_uses_non_speculative_4k_extent_or_requires_replay(monkeypatch, retained):
    from test_session_state import fixture
    from vllm_kvmem.native_state import canonicalize_before_non_speculative_batch
    adapter, conv, ssm, _, _ = fixture(draft=False)
    runner = adapter.runner
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    runner.speculative_config = object()
    runner.model_state.num_accepted_tokens_gpu[0] = 3
    runner._kvmem_capture_runtime = NS(drain_requests=lambda *_a, **_kw: True,
                                      last_steps={'a': [row(retained)]})
    monkeypatch.setattr('vllm_kvmem.native_state.NativeRecurrentState', lambda _: adapter.recurrent)
    batch = NS(num_reqs=1, req_ids=['a'], num_draft_tokens_per_req=[0],
        is_prefilling_np=[False], num_computed_tokens_np=[8192+retained])
    before_conv, before_ssm = conv.clone(), ssm.clone()
    if retained != 4096:
        with pytest.raises(ValueError, match='checkpoint replay'):
            canonicalize_before_non_speculative_batch(runner, batch)
        assert runner.model_state.num_accepted_tokens_gpu[0] == 3
    else:
        canonicalize_before_non_speculative_batch(runner, batch)
        assert runner.model_state.num_accepted_tokens_gpu[0] == 1
    assert torch.equal(conv, before_conv) and torch.equal(ssm, before_ssm)


@pytest.mark.parametrize('retained', [0, 1, 2048, 2049, 4095, 4096])
def test_terminal_checkpoint_preserves_physical_non_speculative_full_or_replay_rule(retained):
    from test_session_state import fixture
    adapter, conv, ssm, _, _ = fixture(draft=False)
    runner = adapter.runner
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    runner.req_states.prefill_len.np[0] = 16384
    position = 8192+retained
    runner._kvmem_capture_runtime = NS(last_steps={'a': [row(retained)]},
                                      authorities={'a': NS(history_end=position)})
    before_conv, before_ssm = conv.clone(), ssm.clone()
    if retained != 4096:
        with pytest.raises(ValueError, match='checkpoint replay'):
            adapter.capture('a', position, max_bytes=4096, terminal=True)
    else:
        snapshot = adapter.capture('a', position, max_bytes=4096, terminal=True)
        assert torch.equal(snapshot.recurrent['gdn', 'ssm'], before_ssm[1])
    assert torch.equal(conv, before_conv) and torch.equal(ssm, before_ssm)


def test_rolling_oracle_reports_actual_forward_not_last_logical_transaction(monkeypatch, tmp_path):
    from test_rolling_verification import oracle_fixture
    from vllm_kvmem.rolling_verification import RollingPressureOracle
    runner, job, *_ = oracle_fixture(monkeypatch, tmp_path)
    runner._kvmem_capture_runtime.last_steps['a'] = [row(64, begin=0)]
    # The final logical child begins at 2048 but has no retained rows; the
    # accepted physical prefix still ends exactly at this pressure history.
    oracle = RollingPressureOracle(runner, job)
    assert oracle.last_step == dict(history_begin=0, scheduled=4096, accepted=64,
                                    rejected=0, terminal_discarded=4032)
    oracle.verify(job)
