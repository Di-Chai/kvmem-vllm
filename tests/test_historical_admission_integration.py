"""Real connector ordering for whole-turn reservations and idle GC feedback."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.outputs import KVConnectorOutput
from vllm_kvmem.connector import ControlCompletions
from vllm_kvmem.historical_admission import HistoricalHistoryAdmission
from vllm_kvmem.session_control import SessionNotice
from test_connector import pair
from test_session_control import ledger, request


def setup(capacity=32):
    scheduler, _ = pair()
    scheduler.sessions, _ = ledger()
    scheduler.history_admission = HistoricalHistoryAdmission(capacity*24, 24, 16, 4, scheduler.sessions)
    return scheduler


def notice(connector, value):
    connector.update_connector_output(KVConnectorOutput(
        kv_connector_worker_meta=ControlCompletions(sessions=[value])))


def first_turn(connector, rid='a', session='s'):
    req = request(rid, session=session)
    req.max_tokens = 8
    req.num_computed_tokens = 24
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    connector.update_state_after_alloc(req, [], 0)
    entry = connector.sessions.requests[rid]
    connector.sessions.capture_before_final_prefill(req, 1)
    connector.request_finished_all_groups(req, [])
    assert connector.history_admission.snapshot()['retiring'] == 1
    retired = SessionNotice('retired', entry.namespace, entry.generation, rid,
        history=24, window=12, ready_records=10, allocated_records=6)
    notice(connector, retired)
    return retired


def test_exact_prefix_is_resolved_before_whole_turn_reservation():
    connector = setup()
    retired = first_turn(connector)
    assert connector.history_admission.snapshot()['reserved_records'] == 6
    req = request('next', tokens=list(range(29)))
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 8
    assert connector.get_num_new_matched_tokens(req, 0) == (24, False)
    connector.update_state_after_alloc(req, [], 24)
    reservation = next(iter(connector.history_admission.ledger.reservations.values()))
    assert reservation.base_history == 24 and reservation.history_limit == 37
    assert reservation.reserved_records == 14
    # An old same-generation ACK cannot settle its successor's reservation.
    with pytest.raises(ValueError, match='another native request'):
        notice(connector, retired)
    assert connector.history_admission.snapshot()['active'] == 1


def test_idle_gc_feedback_unblocks_lookup_before_any_model_execution():
    connector = setup(20)
    first = first_turn(connector)
    first_turn(connector, 'peer', 'other')
    req = request('next', tokens=list(range(29)))
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 8
    assert connector.get_num_new_matched_tokens(req, 0) == (24, False)
    # Increasing this turn's complete public output stays within per-repository
    # geometry but exceeds the shared CPU budget until idle GC actually frees.
    req.max_tokens = 16
    assert connector.get_num_new_matched_tokens(req, 0) == (None, False)
    assert connector.sessions.requests['next'].phase == 'lookup'
    assert connector.history_admission.snapshot()['active'] == 0
    notice(connector, SessionNotice('history_capacity', first.namespace, first.generation, '',
        history=24, ready_records=12, allocated_records=4))
    assert connector.get_num_new_matched_tokens(req, 0) == (24, False)
    connector.update_state_after_alloc(req, [], 24)
    assert connector.history_admission.snapshot()['reserved_records'] == 20


def test_cancelled_lookup_keeps_old_idle_capacity_and_permanent_oversize_is_request_local():
    connector = setup()
    retired = first_turn(connector)
    req = request('oversize', tokens=list(range(29)))
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 100
    assert connector.get_num_new_matched_tokens(req, 0) == (None, False)
    assert 'oversize' in connector.capacity_rejections
    connector.request_finished_all_groups(req, [])
    assert connector.sessions.entries[retired.namespace].phase == 'idle'
    assert connector.history_admission.snapshot()['reserved_records'] == 6
    assert not connector.history_admission.requests


def test_no_session_credits_survive_eos_until_finished_sending_fence():
    connector = setup()
    req = request('plain', session=None)
    req.max_tokens = 8
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    connector.update_state_after_alloc(req, [], 0)
    connector.request_finished_all_groups(req, [])
    assert connector.history_admission.snapshot()['reserved_records'] == 13
    connector.update_connector_output(KVConnectorOutput(finished_sending={'plain'}))
    assert connector.history_admission.snapshot()['reserved_records'] == 0


def test_waiting_turn_requests_a_larger_ready_reserve_and_clears_it_before_execution():
    connector = setup()
    first = first_turn(connector)
    connector.sessions.take_commands()
    req = request('large', tokens=list(range(29)))
    req.num_prompt_tokens, req.max_tokens = 29, 20
    assert connector.get_num_new_matched_tokens(req, 0) == (None, False)
    commands = connector.sessions.take_commands().history_capacity
    assert len(commands) == 1 and commands[0].ready_records == 11
    assert commands[0].generation == first.generation and commands[0].retained_records is None
    notice(connector, SessionNotice('history_capacity', first.namespace, first.generation, '',
        history=24, ready_records=12, allocated_records=4))
    assert connector.get_num_new_matched_tokens(req, 0) == (24, False)
    reset, = connector.sessions.take_commands().history_capacity
    assert reset.ready_records == 0 and reset.retained_records is None
    connector.update_state_after_alloc(req, [], 24)
    assert connector.history_admission.snapshot()['active'] == 1


def test_shared_cpu_pressure_uses_only_idle_unprotected_records_and_cancel_clears_hint():
    connector = setup(20)
    first = first_turn(connector)
    first_turn(connector, 'peer', 'other')
    connector.sessions.take_commands()
    req = request('next', tokens=list(range(29)))
    req.num_prompt_tokens, req.max_tokens = 29, 16
    assert connector.get_num_new_matched_tokens(req, 0) == (None, False)
    pressure, = connector.sessions.take_commands().history_capacity
    assert pressure.namespace == first.namespace and pressure.retained_records == 4
    assert pressure.ready_records == 0
    connector.request_finished_all_groups(req, [])
    reset, = connector.sessions.take_commands().history_capacity
    assert reset.retained_records is None and reset.ready_records == 0
    assert connector.history_admission.snapshot()['active'] == 0


def test_idle_feedback_in_flight_cannot_replace_the_next_active_turn_reservation():
    connector = setup()
    first = first_turn(connector)
    req = request('next', tokens=list(range(29)))
    req.num_prompt_tokens, req.max_tokens = 29, 8
    assert connector.get_num_new_matched_tokens(req, 0) == (24, False)
    connector.update_state_after_alloc(req, [], 24)
    before = connector.history_admission.snapshot()['reserved_records']
    notice(connector, SessionNotice('history_capacity', first.namespace, first.generation, '',
        history=24, ready_records=12, allocated_records=4))
    assert connector.history_admission.snapshot()['reserved_records'] == before
    assert connector.history_admission.snapshot()['active'] == 1


def test_oldest_index_only_tail_is_not_supply_and_cannot_starve_a_later_idle_victim():
    connector = setup(20)
    first = first_turn(connector)
    peer = first_turn(connector,'peer','other')
    notice(connector,SessionNotice('history_capacity',first.namespace,first.generation,'',
        history=24,ready_records=14,allocated_records=2,protected_records=2))
    connector.sessions.take_commands()
    req = request('next',tokens=list(range(29)))
    req.num_prompt_tokens,req.max_tokens = 29,28
    assert connector.get_num_new_matched_tokens(req,0) == (None,False)
    pressure, = connector.sessions.take_commands().history_capacity
    assert pressure.namespace == peer.namespace and pressure.retained_records == 5
