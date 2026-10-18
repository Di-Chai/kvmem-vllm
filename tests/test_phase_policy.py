import pytest

from vllm_kvmem.phase_policy import PhasePolicy


def policy(max_seqs=4, wave=8192, max_wait=1000):
    now = [10.]
    return PhasePolicy(max_seqs, wave, max_wait, clock=lambda: now[0]), now


def test_default_decode_then_aged_single_prefill():
    p, now = policy()
    d = p.decide(['d'], {'p': 16000}, [])
    assert (d.phase, d.reason, d.prefill_budget) == ('decode', 'decode_ready', 0)
    p.record({})
    now[0] += .999
    assert p.decide(['d'], {'p': 16000}, []).phase == 'decode'
    now[0] = 11.
    d = p.decide(['d'], {'p': 16000}, [])
    assert (d.phase, d.reason, d.changed, d.oldest_wait_ms) == ('prefill', 'prefill_aged', True, 1000.)
    assert d.prefill_budget == 8192


@pytest.mark.parametrize(('max_seqs', 'trigger'), [(1, 1), (4, 2), (5, 3), (8, 4)])
def test_backlog_threshold_is_ceil_half_and_counts_both_prefill_queues(max_seqs, trigger):
    p, _ = policy(max_seqs)
    assert p.prefill_trigger == trigger
    waiting = [str(i) for i in range(trigger-1)]
    decode = ['d'] if max_seqs > 1 else []
    d = p.decide(decode, {'running': 8192}, waiting)
    assert d.phase == 'prefill'
    assert d.reason == ('prefill_backlog' if decode else 'no_decode_ready')


def test_wave_uses_actual_tokens_and_yields_a_decode_step_before_reentering():
    p, _ = policy()
    d = p.decide(['d'], {'p': 20000}, ['q'])
    assert d.prefill_budget == 8192
    p.record({'p': 1024})
    d = p.decide(['d'], {'p': 18976}, ['q'])
    assert d.prefill_budget == 7168 and d.wave_tokens == 1024 and d.wave_batches == 1
    p.record({'p': 4096, 'q': 3072})
    assert p.phase == 'prefill'  # Feedback never changes this batch's phase.
    d = p.decide(['d'], {'p': 14880, 'q': 10000}, [])
    assert (d.phase, d.reason, d.prefill_budget) == ('decode', 'prefill_wave_complete', 0)
    p.record({}, decode_tokens_by_request={'d': 5})
    d = p.decide(['d'], {'p': 14880, 'q': 10000}, [])
    assert d.phase == 'prefill' and d.wave_tokens == 0 and d.wave_batches == 0


def test_zero_admitted_tokens_do_not_consume_wave_or_claim_a_batch():
    p, _ = policy()
    p.decide(['d'], {'p': 20000}, ['q'])
    p.record({})
    d = p.decide(['d'], {'p': 20000}, ['q'])
    assert d.prefill_budget == 8192 and d.wave_tokens == d.wave_batches == 0


def test_transaction_strips_all_prefill_then_decode_resumes_without_first_batch():
    p, _ = policy()
    p.decide(['d'], {'p': 20000}, ['q'])
    p.record({})
    d = p.decide(['d'], {}, [])
    assert (d.phase, d.reason) == ('decode', 'prefill_unavailable')
    assert not p.snapshot()['eligible_wait_ms']


def test_completed_prefill_returns_to_decode_before_budget_is_exhausted():
    p, _ = policy()
    p.decide([], {'p': 128}, [])
    p.record({'p': 128})
    d = p.decide(['p'], {}, [])
    assert (d.phase, d.reason, d.wave_batches) == ('decode', 'prefill_unavailable', 1)


def test_no_decode_never_inserts_empty_step_at_wave_boundary():
    p, _ = policy(wave=1024)
    for remaining in (4096, 3072, 2048):
        d = p.decide([], {'p': remaining}, [])
        assert d.phase == 'prefill' and d.prefill_budget == 1024
        p.record({'p': 1024})
    assert p.snapshot()['wave_tokens'] == 1024
    assert p.snapshot()['waves'] == 3
    assert p.snapshot()['total_prefill_tokens'] == 3072
    assert p.snapshot()['prefill_budget_overshoot_tokens'] == 0
    d = p.decide(['d'], {'p': 1024}, [])
    assert (d.phase, d.reason) == ('decode', 'prefill_wave_complete')


def test_no_decode_partial_steps_share_one_wave_budget():
    p, _ = policy(wave=1024)
    p.decide([], {'p': 10000}, [])
    p.record({'p': 768})
    d = p.decide([], {'p': 9232}, [])
    assert d.prefill_budget == 256 and d.wave_id == 1
    p.record({'p': 256})
    d = p.decide([], {'p': 8976}, [])
    assert d.phase == 'prefill' and not d.changed and d.wave_id == 2
    assert d.prefill_budget == 1024 and d.reason == 'prefill_wave_renewed_no_decode'


def test_backlog_must_wait_for_actual_decode_progress_after_a_wave():
    p, now = policy(wave=1024)
    p.decide(['d'], {'p': 10000}, ['q'])
    p.record({'p': 1024})
    assert p.decide(['d'], {'p': 8976}, ['q']).phase == 'decode'
    p.record({})  # Native allocation or cancellation produced no model work.
    now[0] += 10.
    for _ in range(3):
        d = p.decide(['d'], {'p': 8976}, ['q'])
        assert d.phase == 'decode' and d.reason == 'awaiting_decode_progress'
        p.record({})
    p.decide(['d'], {'p': 8976}, ['q'])
    p.record({}, decode_tokens_by_request={'d': 5})
    assert p.decide(['d'], {'p': 8976}, ['q']).phase == 'prefill'


def test_missing_eligible_decode_does_not_block_prefill_waiting_for_decode_feedback():
    p, _ = policy(wave=1024)
    p.decide(['d'], {'p': 8192}, ['q'])
    p.record({'p': 1024})
    p.decide(['d'], {'p': 7168}, ['q'])
    p.record({})
    d = p.decide([], {'p': 7168}, ['q'])
    assert d.phase == 'prefill' and d.prefill_budget == 1024


def test_wave_wall_limit_protects_decode_after_real_prefill_progress():
    p, now = policy()
    p.decide(['d'], {'p': 20000}, ['q'])
    p.record({'p': 32})
    now[0] += 1.
    d = p.decide(['d'], {'p': 19968}, ['q'])
    assert d.phase == 'decode' and d.reason == 'prefill_wave_time_limit'
    assert d.wave_tokens == 32 and d.wave_batches == 1


def test_atomic_budget_tail_ends_wave_without_fabricating_progress():
    p, _ = policy(wave=1024)
    p.decide(['d'], {'p': 10000}, ['q'])
    p.record({'p': 1010})
    assert p.decide(['d'], {'p': 8990}, ['q']).prefill_budget == 14
    p.record({}, blocked_by_budget=True)
    d = p.decide(['d'], {'p': 8990}, ['q'])
    assert d.phase == 'decode' and d.reason == 'prefill_alignment_tail'
    assert p.snapshot()['total_prefill_tokens'] == 1010


def test_alignment_tail_without_decode_renews_prefill_without_empty_decode_step():
    p, _ = policy(wave=1024)
    p.decide([], {'p': 10000}, [])
    p.record({'p': 1010})
    assert p.decide([], {'p': 8990}, []).prefill_budget == 14
    p.record({}, blocked_by_budget=True)
    d = p.decide([], {'p': 8990}, [])
    assert d.phase == 'prefill' and d.prefill_budget == 1024 and d.wave_id == 2
    assert p.snapshot()['last_wave']['tokens'] == 1010


def test_successful_budget_cap_does_not_end_a_wave_early():
    p, _ = policy()
    p.decide(['d'], {'p': 10000}, ['q'])
    p.record({'p': 1024}, blocked_by_budget=True)
    d = p.decide(['d'], {'p': 8976}, ['q'])
    assert d.phase == 'prefill' and d.prefill_budget == 7168


def test_no_admission_slot_never_stops_decode_even_after_long_wait():
    p, now = policy()
    for _ in range(4):
        # Blocked external waiting requests are intentionally not supplied.
        assert p.decide(['a', 'b', 'c', 'd'], {}, []).phase == 'decode'
        p.record({})
        now[0] += 100.
    assert p.snapshot()['transitions'] == 0
    assert p.snapshot()['max_prefill_wait_ms'] == 0.


def test_aging_resets_only_for_progressing_request_and_survives_waiting_to_running():
    p, now = policy()
    p.decide([], {}, ['a', 'b'])
    now[0] += .75
    p.record({'a': 1024})
    now[0] += .25
    p.decide(['d'], {'a': 8192, 'b': 8192}, [])
    ages = p.snapshot()['eligible_wait_ms']
    assert ages == {'a': 250., 'b': 1000.}
    assert p.snapshot()['max_prefill_wait_ms'] == 1000.


def test_recorded_progress_does_not_reset_age_again_when_snapshot_catches_up():
    p, now = policy()
    p.decide([], {'p': 10000}, [])
    p.record({'p': 1024})
    now[0] += .75
    p.decide(['d'], {'p': 8976}, [])
    assert p.snapshot()['eligible_wait_ms'] == {'p': 750.}


def test_external_prefill_progress_and_temporary_ineligibility_reset_age():
    p, now = policy()
    p.decide(['d'], {'p': 10000}, [])
    now[0] += .75
    p.decide(['d'], {'p': 9000}, [])
    assert p.snapshot()['eligible_wait_ms'] == {'p': 0.}
    now[0] += .75
    p.decide(['d'], {}, [])  # Async transaction or cancellation, not an eligible wait.
    now[0] += 10.
    d = p.decide(['d'], {'p': 9000}, [])
    assert d.phase == 'decode' and d.oldest_wait_ms == 0.


def test_atomic_query_overrun_is_counted_once_and_phase_changes_next_step():
    p, _ = policy(wave=1024)
    p.decide(['d'], {'p': 10000}, ['q'])
    f = p.record({'p': 4096}, atomic_query_tokens=4096)
    assert f['prefill_tokens'] == 4096 and f['budget_overshoot_tokens'] == 3072
    assert f['atomic_query_overshoot_tokens'] == 3072 and f['ordinary_budget_overshoot_tokens'] == 0
    assert p.snapshot()['total_prefill_tokens'] == 4096
    assert p.phase == 'prefill'
    assert p.decide(['d'], {'p': 5904}, ['q']).phase == 'decode'


def test_ordinary_overrun_is_visible_rather_than_blended_into_query():
    p, _ = policy(wave=1024)
    p.decide([], {'p': 8192}, [])
    f = p.record({'p': 4096}, atomic_query_tokens=512)
    assert f['budget_overshoot_tokens'] == 3072
    assert f['ordinary_budget_overshoot_tokens'] == 2560
    assert f['atomic_query_overshoot_tokens'] == 512


def test_idle_after_prefill_does_not_grant_phantom_work():
    p, _ = policy()
    p.decide([], {}, ['p'])
    p.record({})
    d = p.decide([], {}, [])
    assert d.reason == 'idle' and d.prefill_budget == 0
    d = p.decide([], {}, ['q'])
    assert d.phase == 'prefill' and d.prefill_budget == 8192


def test_transition_snapshot_records_reason_and_starvation_without_mutating_inputs():
    p, now = policy()
    running = {'p': 1024}
    p.decide(['d'], running, [])
    now[0] += 2.
    d = p.decide(['d'], running, [])
    s = p.snapshot()
    assert running == {'p': 1024} and d.reason == 'prefill_aged'
    assert s['last_transition']['eligible_wait_ms'] == {'p': 2000.}
    assert s['last_transition']['previous'] == 'decode'
    assert s['max_prefill_wait_ms'] == 2000.


@pytest.mark.parametrize('args', [(0, 8192, 1000), (True, 8192, 1000), (4, 0, 1000),
                                 (4, 8192, 0), (4, 8192, float('inf')), (4, 8192, True)])
def test_invalid_configuration_fails_before_decisions(args):
    with pytest.raises(ValueError):
        PhasePolicy(*args)


def test_invalid_inputs_and_duplicate_feedback_do_not_silently_double_count():
    p, now = policy()
    with pytest.raises(RuntimeError):
        p.record({})
    for decode, running, waiting in [(['d', 'd'], {}, []), (['d'], {'d': 1}, []),
                                     ([], {'p': 0}, []), ([], {'p': 1}, ['p']),
                                     (['a', 'b', 'c', 'd', 'e'], {}, [])]:
        with pytest.raises(ValueError):
            p.decide(decode, running, waiting)
    p.decide([], {'p': 8192}, [])
    for tokens, atomic in [({'unknown': 1}, 0), ({'p': -1}, 0), ({'p': True}, 0),
                           ({'p': 1}, 2)]:
        with pytest.raises(ValueError):
            p.record(tokens, atomic)
    p.record({'p': 1024})
    with pytest.raises(RuntimeError):
        p.record({'p': 1024})
    assert p.snapshot()['total_prefill_tokens'] == 1024
    now[0] -= 1.
    with pytest.raises(ValueError, match='monotonic'):
        p.decide([], {'p': 7168}, [])
