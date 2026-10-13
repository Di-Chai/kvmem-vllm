from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.session_control import SessionLedger, SessionNotice


def request(name='a', session='s', tokens=None, salt='tenant'):
    return SimpleNamespace(request_id=name, session_id=session, cache_salt=salt,
        prompt_token_ids=list(range(25)) if tokens is None else tokens,
        num_computed_tokens=16, num_prompt_tokens=25)


def ledger(capacity=2):
    events = []
    result = SessionLedger(capacity, 4,
        lambda *args: events.append(('prepare', args)),
        lambda *args: events.append(('abort', args)),
        lambda *args: events.append(('commit', args)))
    return result, events


def first_turn(control, req=None):
    req = req or request()
    assert control.lookup(req, 0) == 0
    control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    binding, capture = commands.bindings[0], commands.captures[0]
    control.finished(req)
    notice = SessionNotice('retired', binding.namespace, binding.generation, req.request_id, capture.position, 12)
    control.consume(notice)
    return binding, notice


def test_hit_requires_worker_retirement_then_fresh_exact_prefix_allocation():
    control, events = ledger()
    a, b = request(), request('b')
    assert control.lookup(a, 0) == 0
    control.allocated(a, 0)
    control.capture_before_final_prefill(a, 9)
    commands = control.take_commands()
    assert len(commands.captures) == 1
    assert control.lookup(b, 0) is None
    control.finished(a)
    assert control.lookup(b, 0) is None and control.has_pending_work()
    binding = commands.bindings[0]
    retired = SessionNotice('retired', binding.namespace, binding.generation, 'a', 16, 12)
    control.consume(retired)
    assert not control.has_pending_work()
    assert control.lookup(b, 0) == 16
    assert control.lookup(b, 0) == 16 and len(events) == 1
    control.allocated(b, 16)
    restored = replace(retired, kind='restored', request_id='b')
    control.consume(restored)
    assert events[-1] == ('commit', ('b', 16))
    with pytest.raises(ValueError, match='import completion'):
        control.consume(restored)
    with pytest.raises(ValueError, match='another native'):
        control.consume(retired)
    control.finished(b)
    control.consume(replace(retired, request_id='b', history=0, window=0))
    assert not control.entries and not control.requests


def test_decode_checkpoint_retains_accepted_output_and_failed_replacement_keeps_previous_prefix():
    control, events = ledger()
    req = request()
    control.lookup(req, 0); control.allocated(req, 0)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    binding = commands.bindings[0]
    def notice(kind, position, window=0):
        return SessionNotice(kind, binding.namespace, binding.generation, req.request_id,
                             position, window)
    control.consume(notice('captured', 16))
    req.num_computed_tokens = 32
    req.all_token_ids = req.prompt_token_ids + [101, 102, 103, 104, 105, 106, 107, 108]
    control.capture_decode_boundary(req)
    command = control.take_commands().captures[0]
    assert command.position == 32 and command.tokens[-8:] == tuple(range(101, 109))
    control.consume(notice('captured', 32))
    req.num_computed_tokens = 36; req.all_token_ids += [201, 202, 203, 204]
    control.capture_decode_boundary(req)
    control.take_commands()
    control.consume(notice('capture_deferred', 36))
    control.finished(req)
    control.consume(notice('retired', 32, 12))
    continuation = request('new', tokens=req.all_token_ids + [400, 500])
    assert control.lookup(continuation, 0) == 32
    assert events[-1] == ('prepare', ('new', 32, 12))


def test_terminal_checkpoint_uses_full_accepted_identity_at_a_partial_block():
    control, events = ledger()
    control.terminal_capture = True
    req = request(); control.lookup(req, 0); control.allocated(req, 0)
    binding = control.take_commands().bindings[0]
    req.all_token_ids = req.prompt_token_ids + [700, 701, 702, 703, 704]
    req.num_computed_tokens = 29; req.num_in_flight_tokens = 0
    req.num_tokens = 30; req.num_output_tokens = 5
    req.status = SimpleNamespace(name='FINISHED_LENGTH_CAPPED')
    req.is_finished = lambda: True
    control.finished(req)
    terminal = control.take_commands().terminals[0]
    assert terminal.position == 29 and terminal.tokens[-5:] == tuple(range(700, 705))
    control.consume(SessionNotice('captured', binding.namespace, binding.generation, 'a', 29))
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, 'a', 29, 13))
    assert control.lookup(request('new', tokens=req.all_token_ids + [999]), 0) == 29
    assert events[-1] == ('prepare', ('new', 29, 13))


def test_prefix_miss_and_lru_eviction_hold_capacity_until_generation_ack():
    control, _ = ledger(capacity=1)
    binding, retired = first_turn(control)
    changed = list(range(25)); changed[3] = 999
    req = request('b', tokens=changed)
    assert control.lookup(req, 0) is None
    assert control.lookup(req, 0) is None
    commands = control.take_commands()
    assert len(commands.evictions) == 1 and control.has_pending_work()
    with pytest.raises(ValueError, match='Stale'):
        control.consume(SessionNotice('evicted', binding.namespace, binding.generation + 1, ''))
    control.consume(SessionNotice('evicted', binding.namespace, binding.generation, ''))
    assert control.lookup(req, 0) == 0
    assert control.entries[binding.namespace].generation > binding.generation
    with pytest.raises(ValueError, match='Stale'):
        control.consume(retired)
    # Cancelling a waiting cold request creates no phantom worker ownership.
    control.finished(req)
    assert not control.entries and not control.has_pending_work()
    first_turn(control)
    assert control.lookup(request('other', session='new'), 0) is None
    assert len(control.take_commands().evictions) == 1


def test_cancel_before_allocation_restores_idle_lease_and_aborts_only_preparation():
    control, events = ledger()
    binding, _ = first_turn(control)
    req = request('b')
    assert control.lookup(req, 0) == 16
    control.finished(req)
    assert events[-1] == ('abort', ('b',))
    assert control.entries[binding.namespace].phase == 'idle'
    assert control.lookup(request('c'), 0) == 16


def test_full_busy_cache_allows_uncached_session_but_serializes_its_turns():
    control, _ = ledger(capacity=1)
    a = request()
    assert control.lookup(a, 0) == 0
    control.allocated(a, 0)
    assert control.lookup(request('b'), 0) is None
    unrelated = request('c', salt='other-tenant')
    assert control.lookup(unrelated, 0) == 0
    control.allocated(unrelated, 0)
    assert len(control.entries) == 1 and 'c' not in control.requests
    assert control.lookup(unrelated, 0) == 0
    continuation = request('d', salt='other-tenant')
    assert control.lookup(continuation, 0) is None
    assert [binding.request_id for binding in control.take_commands().bindings] == ['a']
    control.finished(unrelated)
    assert not control.uncached_sessions and not control.uncached_requests
    assert control.lookup(continuation, 0) == 0
    control.finished(continuation)


def test_uncached_session_lease_survives_a_checkpoint_slot_becoming_available():
    control, _ = ledger(capacity=1)
    cached = request()
    assert control.lookup(cached, 0) == 0
    control.allocated(cached, 0)
    binding = control.take_commands().bindings[0]
    uncached = request('b', session='overflow')
    assert control.lookup(uncached, 0) == 0
    control.allocated(uncached, 0)
    control.finished(cached)
    control.consume(SessionNotice('retired', binding.namespace, binding.generation, 'a'))
    assert not control.entries
    assert control.lookup(request('c', session='overflow'), 0) is None
    # A retry after admission pressure stays uncached for this whole request.
    assert control.lookup(uncached, 0) == 0 and not control.entries
    control.finished(uncached)
    continuation = request('c', session='overflow')
    assert control.lookup(continuation, 0) == 0
    assert ('tenant', 'overflow') in control.entries
    control.finished(continuation)
    assert not control.entries and not control.uncached_sessions and not control.uncached_requests


def test_cancel_uncached_request_before_allocation_releases_only_its_lease():
    control, events = ledger(capacity=1)
    cached = request()
    assert control.lookup(cached, 0) == 0
    uncached = request('b', session='overflow')
    assert control.lookup(uncached, 0) == 0
    # Waiting requests that never acquired ownership cannot release the owner.
    waiting = request('c', session='overflow')
    assert control.lookup(waiting, 0) is None
    control.finished(waiting)
    assert control.lookup(waiting, 0) is None
    control.finished(uncached)
    assert control.lookup(waiting, 0) == 0
    control.finished(waiting)
    assert list(control.requests) == ['a'] and not events
    assert not control.uncached_sessions and not control.uncached_requests


def test_checkpoint_candidate_requires_sealed_final_prefill_boundary():
    control, _ = ledger()
    req = request()
    control.lookup(req, 0); control.allocated(req, 0)
    control.capture_before_final_prefill(req, 4)
    req.num_computed_tokens = 17
    control.capture_before_final_prefill(req, 8)
    req.num_computed_tokens = 16
    control.capture_before_final_prefill(req, 9)
    control.capture_before_final_prefill(req, 9)
    commands = control.take_commands()
    assert [c.position for c in commands.captures] == [16]
    control.finished(req)
    b = commands.bindings[0]
    with pytest.raises(ValueError, match='different prompt'):
        control.consume(SessionNotice('retired', b.namespace, b.generation, req.request_id, 12, 12))
    control.consume(SessionNotice('retired', b.namespace, b.generation, req.request_id))
    assert not control.entries  # Worker may decline a checkpoint on byte pressure.


def test_native_connector_metadata_roundtrip_for_exact_checkpoint_ack(monkeypatch):
    from test_connector import pair
    from vllm.v1.outputs import KVConnectorOutput
    from vllm_kvmem.connector import ControlCompletions
    scheduler, _ = pair()
    control, events = ledger()
    scheduler.sessions = control
    req = request()
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)
    scheduler.update_state_after_alloc(req, None, 0)
    control.capture_before_final_prefill(req, 9)
    metadata = scheduler.build_connector_meta(None)
    binding = metadata.sessions.bindings[0]
    scheduler.request_finished_all_groups(req, ())
    assert scheduler.has_pending_push_work()
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        sessions=[SessionNotice('retired', binding.namespace, binding.generation, 'a', 16, 12)])))
    assert scheduler.get_num_new_matched_tokens(request('b'), 0) == (16, False)
    assert events == [('prepare', ('b', 16, 12))]
