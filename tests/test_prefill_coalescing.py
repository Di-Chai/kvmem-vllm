import json

import pytest

from vllm_kvmem.prefill_coalescing import OfflinePrefillCoalescer


SHAPE = dict(decode_requests=3, prefill_requests=1,
             decode_context_bucket=131072, prefill_context_bucket=131072)


def row(sequence, tokens=2048, ms=400., **kwargs):
    return dict(sequence=sequence, submitted_at=10., model_ms=ms,
                maintenance_ms=10000., step_ms=ms + 10000., prefill_tokens=tokens,
                ordinary_prefill_only=True, **SHAPE) | kwargs


def control():
    return OfflinePrefillCoalescer(2048, 4096, 32, clock=lambda: 10.)


def cap(c, **kwargs):
    return c.cap(**(SHAPE | dict(remaining_tokens=20000) | kwargs))


def baseline(c):
    c.observe([row(i) for i in range(1, 4)])


def calibrate(c, burst_ms=600., **kwargs):
    for i in range(4, 7):
        assert cap(c) == 4096
        receipt = c.last_authorization_id
        c.reconcile(receipt, actual_tokens=4096)
        c.observe([row(i, 4096, burst_ms, coalescing_authorization_id=receipt, **kwargs)])


def test_exact_context_requires_three_baselines_then_three_measured_bursts():
    c = control()
    for i in range(1, 4):
        assert cap(c) == 2048
        c.observe([row(i)])
    calibrate(c)
    assert cap(c) == 4096 and c.last_authorization_id is None
    assert c.snapshot()['banks'][0]['state'] == 'approved'
    assert c.snapshot()['banks'][0]['measured_gain'] == .25
    json.dumps(c.snapshot())


def test_async_caps_cannot_authorize_unlimited_unmeasured_experiments():
    c = control(); baseline(c)
    receipts = []
    for _ in range(3):
        assert cap(c) == 4096
        receipts.append(c.last_authorization_id)
    assert len(set(receipts)) == 3
    for _ in range(20):
        assert cap(c) == 2048 and c.last_authorization_id is None
    assert c.snapshot()['pending_authorizations'] == 3
    assert c.snapshot()['banks'][0]['state'] == 'awaiting_feedback'


def test_unsubmitted_or_trimmed_grants_release_once_and_tail_does_not_train():
    c = control(); baseline(c)
    for actual in (0, 1024, 2048):
        assert cap(c) == 4096
        receipt = c.last_authorization_id
        assert c.reconcile(receipt, actual_tokens=actual)
        assert not c.reconcile(receipt, actual_tokens=actual)
    assert c.snapshot()['banks'][0]['calibration_grants'] == 0
    assert cap(c) == 4096
    receipt = c.last_authorization_id
    c.observe([row(4, 1000, 100., coalescing_authorization_id=receipt)])
    assert c.snapshot()['banks'][0]['burst_samples'] == 0
    assert c.snapshot()['banks'][0]['calibration_grants'] == 0


@pytest.mark.parametrize('change', [dict(decode_requests=2), dict(prefill_requests=2),
    dict(decode_context_bucket=65536), dict(prefill_context_bucket=65536)])
def test_approved_result_is_never_reused_across_occupancy_or_context(change):
    c = control(); baseline(c); calibrate(c)
    assert cap(c, **change) == 2048
    assert cap(c) == 4096


def test_actual_feedback_shape_mismatch_consumes_quota_but_does_not_train_other_bank():
    c = control(); baseline(c)
    assert cap(c) == 4096
    receipt = c.last_authorization_id
    c.reconcile(receipt, actual_tokens=4096)
    c.observe([row(4, 4096, 600., prefill_context_bucket=262144,
                   coalescing_authorization_id=receipt)])
    state = c.snapshot()
    assert state['pending_authorizations'] == 0 and state['shape_mismatches'] == 1
    assert len(state['banks']) == 1 and state['banks'][0]['burst_samples'] == 0
    assert state['banks'][0]['calibration_grants'] == 1
    assert state['discarded_grants'] == 1 and state['released_grants'] == 0


def test_reconcile_can_reject_actual_shape_before_feedback_without_refunding_executed_work():
    c = control(); baseline(c)
    assert cap(c) == 4096
    receipt = c.last_authorization_id
    assert c.reconcile(receipt, actual_tokens=4096,
                       **(SHAPE | dict(prefill_context_bucket=262144)))
    assert c.snapshot()['banks'][0]['calibration_grants'] == 1


@pytest.mark.parametrize('actual', [2049, 3072, 3686, 4096])
def test_executed_extra_work_cannot_reopen_calibration_quota_via_reconciliation(actual):
    c = control(); baseline(c)
    for _ in range(3):
        assert cap(c) == 4096
        receipt = c.last_authorization_id
        kw = (SHAPE | dict(prefill_context_bucket=262144)) if actual == 4096 else {}
        assert c.reconcile(receipt, actual_tokens=actual, **kw)
        assert not c.reconcile(receipt, actual_tokens=actual, **kw)
    for _ in range(10):
        assert cap(c) == 2048
    state = c.snapshot()
    assert state['pending_authorizations'] == 0
    assert state['banks'][0]['calibration_grants'] == 3
    assert state['discarded_grants'] == 3 and state['released_grants'] == 0


def test_trimmed_feedback_above_baseline_consumes_quota_without_training():
    c = control(); baseline(c)
    for i in range(4, 7):
        assert cap(c) == 4096
        c.observe([row(i, 3072, 500., coalescing_authorization_id=c.last_authorization_id)])
    assert cap(c) == 2048
    state = c.snapshot()
    assert state['banks'][0]['burst_samples'] == 0
    assert state['banks'][0]['calibration_grants'] == 3


@pytest.mark.parametrize('ms,state', [(790., 'insufficient_gain'), (900., 'insufficient_gain')])
def test_failure_returns_to_baseline_instead_of_infinite_bursts(ms, state):
    c = control(); baseline(c); calibrate(c, burst_ms=ms)
    for _ in range(10):
        assert cap(c) == 2048
    assert c.snapshot()['banks'][0]['state'] == state
    assert c.snapshot()['calibration_grants'] == 3


def test_five_percent_boundary_and_model_step_growth_guard_use_actual_tokens():
    c = control(); baseline(c); calibrate(c, burst_ms=760.)
    assert cap(c) == 4096  # Exactly 5% cheaper per public token.
    # A token-count mismatch can improve unit cost yet violate the model-step guard.
    c = control()
    c.observe([row(i, 1844, 400.) for i in range(1, 4)])
    calibrate(c, burst_ms=820.)
    assert cap(c) == 2048
    assert c.snapshot()['banks'][0]['state'] == 'model_step_growth'


def test_high_maintenance_does_not_enter_model_cost_comparison():
    c = control(); baseline(c)
    calibrate(c, maintenance_ms=1e8, step_ms=1e8)
    assert cap(c) == 4096
    assert c.snapshot()['banks'][0]['burst_model_ms'] == 600.


def test_query_unknown_nonordinary_or_bad_feedback_never_trains():
    c = control()
    invalid = [dict(ordinary_prefill_only=False), dict(ordinary_prefill_only=1),
               dict(ordinary_prefill_only=None), dict(submitted_at=4.9),
               dict(submitted_at=11.), dict(model_ms=float('nan')), dict(model_ms=0),
               dict(model_ms=float('inf')), dict(prefill_tokens=2048.),
               dict(prefill_requests=2), dict(decode_context_bucket=None)]
    c.observe([row(i, **kw) for i, kw in enumerate(invalid, 1)])
    assert cap(c) == 2048 and c.snapshot()['observations'] == 0


def test_duplicate_and_out_of_order_sequences_do_not_retrain():
    c = control()
    c.observe([row(1), row(1), row(3), row(2)])
    assert cap(c) == 2048
    assert c.snapshot()['observations'] == 2
    c.observe([row(4)])
    assert cap(c) == 4096


def test_tail_blocks_do_not_train_baseline_or_consume_burst_authorization():
    c = control()
    c.observe([row(i, 1000) for i in range(1, 10)])
    assert cap(c) == 2048
    c.observe([row(i) for i in range(10, 13)])
    for remaining in (0, 2048, 4095):
        assert cap(c, remaining_tokens=remaining) == 2048
    assert c.snapshot()['calibration_grants'] == 0
    assert cap(c, remaining_tokens=4096) == 4096


def test_unstable_baseline_or_burst_keeps_baseline():
    c = control()
    c.observe([row(1), row(2), row(3, ms=1000.)])
    assert cap(c) == 2048
    assert c.snapshot()['banks'][0]['state'] == 'unstable_baseline'
    c = control(); baseline(c)
    for i, ms in enumerate((600., 600., 1000.), 4):
        assert cap(c) == 4096
        c.observe([row(i, 4096, ms, coalescing_authorization_id=c.last_authorization_id)])
    assert cap(c) == 2048
    assert c.snapshot()['banks'][0]['state'] == 'unstable_burst'


def test_expiry_resets_calibration_and_rejects_old_receipts():
    now = [10.]
    c = OfflinePrefillCoalescer(2048, 4096, 32, clock=lambda: now[0])
    baseline(c); assert cap(c) == 4096
    receipt = c.last_authorization_id
    now[0] = 70.
    assert cap(c) == 2048 and c.snapshot()['pending_authorizations'] == 0
    assert not c.reconcile(receipt, actual_tokens=0)
    c.observe([row(i, submitted_at=70.) for i in range(4, 7)])
    assert cap(c) == 4096


def test_sample_retention_uses_submission_time_not_delayed_feedback_arrival():
    now = [10.]
    c = OfflinePrefillCoalescer(2048, 4096, 32, clock=lambda: now[0])
    c.observe([row(i, submitted_at=5.) for i in range(1, 4)])
    calibrate(c)
    assert cap(c) == 4096
    now[0] = 65.
    # The bank epoch is only 55s old, but its baseline samples are already 60s old.
    assert cap(c) == 2048
    assert c.snapshot()['banks'][0]['baseline_samples'] == 0


def test_banks_samples_and_pending_receipts_are_bounded():
    c = control()
    sequence = 0
    for context in range(1, 81):
        for _ in range(40):
            sequence += 1
            c.observe([row(sequence, prefill_context_bucket=context)])
        for _ in range(3):
            assert cap(c, prefill_context_bucket=context) == 4096
    assert len(c.banks) == 64
    assert all(len(bank.samples) == 32 for bank in c.banks.values())
    assert c.snapshot()['pending_authorizations'] == 64 * 3
    assert cap(c, prefill_context_bucket=1) == 2048


def test_no_decode_is_still_eligible_without_a_fake_decode_context():
    c = control()
    c.observe([row(i, decode_requests=0, decode_context_bucket=None) for i in range(1, 4)])
    assert cap(c, decode_requests=0, decode_context_bucket=None) == 4096


@pytest.mark.parametrize('sizes', [(0, 4096, 32), (2048, 8192, 32),
    (2048, 4097, 32), (2048, 1024, 32), (True, 2, 1), (2048, 4096, 0)])
def test_invalid_configuration_is_rejected(sizes):
    with pytest.raises(ValueError):
        OfflinePrefillCoalescer(*sizes)


def test_equal_maximum_disables_experiments_and_invalid_cap_is_rejected():
    c = OfflinePrefillCoalescer(2048, 2048, 32, clock=lambda:10.)
    baseline(c)
    assert cap(c) == 2048 and c.snapshot()['banks'] == []
    with pytest.raises(ValueError):
        cap(c, remaining_tokens=-1)
    with pytest.raises(ValueError):
        cap(c, decode_requests=True)
