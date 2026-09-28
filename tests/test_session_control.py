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


def test_full_busy_cache_allows_untracked_session_but_not_same_session_race():
    control, _ = ledger(capacity=1)
    a = request()
    assert control.lookup(a, 0) == 0
    control.allocated(a, 0)
    assert control.lookup(request('b'), 0) is None
    unrelated = request('c', salt='other-tenant')
    assert control.lookup(unrelated, 0) == 0
    control.allocated(unrelated, 0)
    assert len(control.entries) == 1 and 'c' not in control.requests
    control.finished(unrelated)


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
