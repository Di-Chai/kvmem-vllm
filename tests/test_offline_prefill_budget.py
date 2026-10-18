"""Shared ordinary-prefill admission with continuous offline ready-work ordering.

CPU-only candidates: these tests do not create a model runner or CUDA context.
Native allocation/restore ownership remains covered by the existing integration
tests; here we exercise real ordering, plan admission and native MTP lookahead.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_kvmem.offline_policy import OfflineAgentPolicy
from vllm_kvmem.prefill_control import BatchPrefillBudget
from test_phase_scheduler import NATIVE_RESERVE, request, scheduler as phase_scheduler


def setup(monkeypatch, tmp_path, *, cap=2048):
    s = phase_scheduler(monkeypatch, tmp_path)
    s.phase_policy = s._phase_decision = None
    s.scheduler_config.max_num_seqs = 8
    s.request_planning = s.async_restore = True
    s.prefill_batch = BatchPrefillBudget(cap, 32)
    s.connector.active_residency = None
    s.connector.publication_blocked = set()
    s.waiting, s.skipped_waiting = FCFSRequestQueue(), FCFSRequestQueue()
    s._is_blocked_waiting_status = lambda status: status == 'blocked'
    s.offline_policy = OfflineAgentPolicy(s, NS(maintenance_aging_ms=250,
        prefill_max_tokens=2048), None, clock=lambda: s.now)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS', '2048')
    return s


def prepare(s):
    s.offline_policy.prepare()
    s._prepare_coalescing()
    s._prepare_prefill_budget()


def prefill(rid, *, history=32768, prompt=100000, plan=None):
    return request(rid, computed=history, prompt=prompt, output=0, plan=plan)


def reserve(s, r, n=10000):
    return s._reserve_prefill_lookahead(r, r.num_computed_tokens, n)


def test_five_decodes_and_three_prefills_share_one_cap_then_idle_prefill_recovers(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    prefills = [prefill('p'+str(i)) for i in range(3)]
    decodes = [request('d'+str(i), computed=65536, prompt=65536) for i in range(5)]
    # Native increments the step before admitting these newly ready decodes.
    for r in decodes:
        r.next_decode_eligible_step = s.current_step + 1
    s.running = prefills + decodes
    prepare(s)
    assert s.running == decodes + prefills
    assert not s._decode_ready() and s._decode_ready(step=s.current_step + 1)
    assert [reserve(s, r, 5) for r in decodes] == [5] * 5
    assert [reserve(s, r) for r in prefills] == [672] * 3
    assert s.prefill_batch.remaining == 32
    assert s.prefill_batch.snapshot()['contenders'] == 3
    # No ready decode: recover the existing per-request chunk, not an idle cap.
    s.connector.publication_blocked.update(r.request_id for r in decodes)
    prepare(s)
    assert [reserve(s, r) for r in prefills] == [2048] * 3
    assert s.prefill_batch.remaining is None


@pytest.mark.parametrize('blocked_kind', ['publication', 'transaction', 'inflight', 'future_step', 'finished'])
def test_unready_decode_does_not_enable_shared_cap(monkeypatch, tmp_path, blocked_kind):
    s = setup(monkeypatch, tmp_path)
    d, a, b = request('decode'), prefill('a'), prefill('b')
    s.running = [d, a, b]
    if blocked_kind == 'publication': s.connector.publication_blocked.add(d.request_id)
    elif blocked_kind == 'transaction': s.transactions[d.request_id] = object()
    elif blocked_kind == 'inflight': d.num_in_flight_tokens = 1
    elif blocked_kind == 'future_step': d.next_decode_eligible_step = s.current_step + 2
    elif blocked_kind == 'finished': d.is_finished = lambda: True
    prepare(s)
    assert s.prefill_batch.remaining is None
    assert reserve(s, a) == reserve(s, b) == 2048


def test_native_false_prefill_flag_does_not_turn_unfinished_suffix_into_decode(monkeypatch, tmp_path):
    from test_native_readiness_admission import request as native_request
    s = setup(monkeypatch, tmp_path)
    suffix = native_request('restored-suffix', prompt=49)
    suffix.num_computed_tokens = 32
    assert suffix.is_prefill_chunk is False  # Native default, unchanged by a restore control step.
    s.running = [suffix, prefill('other')]
    prepare(s)
    assert not s._decode_ready(step=s.current_step + 1)
    assert s.prefill_batch.remaining is None


@pytest.mark.parametrize('blocked_kind', ['history', 'gpu', 'native',
    'transaction', 'inflight', 'future_step', 'finished'])
def test_blocked_waiter_never_dilutes_ready_prefill_share(monkeypatch, tmp_path, blocked_kind):
    s = setup(monkeypatch, tmp_path)
    d, p, b = request('decode'), prefill('ready'), prefill('blocked', history=0)
    s.running = [p, d]
    s.waiting.add_request(b)
    if blocked_kind == 'history':
        s.connector.history_admission.can_admit = lambda r: r is not b
    elif blocked_kind == 'gpu':
        s.kv_cache_manager.admission_preview = lambda r: 'gpu_lifetime_budget' if r is b else None
    elif blocked_kind == 'native': b.status = 'blocked'
    elif blocked_kind == 'transaction': s.transactions[b.request_id] = object()
    elif blocked_kind == 'inflight': b.num_in_flight_tokens = 1
    elif blocked_kind == 'future_step': b.next_decode_eligible_step = s.current_step + 2
    elif blocked_kind == 'finished': b.is_finished = lambda: True
    prepare(s)
    assert s.prefill_batch.contenders == 1
    assert reserve(s, p) == 2048
    assert list(s.waiting) == [b] and not s.connector.sessions.entries
    assert b.num_computed_tokens == 0


def test_running_publication_never_dilutes_ready_prefill_share(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    p, b = prefill('ready'), prefill('publishing')
    b.status = RequestStatus.RUNNING
    s.running = [request('decode'), p, b]
    s.connector.publication_blocked.add(b.request_id)
    prepare(s)
    assert s.prefill_batch.contenders == 1 and reserve(s, p) == 2048
    assert reserve(s, b) == 0 and b.num_computed_tokens == 32768


def test_skipped_ready_waiter_counts_and_aging_order_is_unchanged(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    old = prefill('old', history=0)
    s.running = [request('decode')]
    s.skipped_waiting.add_request(old)
    prepare(s)
    s.now += .3
    new = prefill('new', history=0)
    s.waiting.add_request(new)
    s.offline_policy.prepare()
    before = (list(s.running), list(s.waiting), list(s.skipped_waiting), dict(s.offline_policy.ranks))
    s._prepare_prefill_budget()
    assert s.prefill_batch.contenders == 2
    assert before == (list(s.running), list(s.waiting), list(s.skipped_waiting), s.offline_policy.ranks)
    assert s.offline_policy.waiting_queue().peek_request() is old
    assert reserve(s, old) == reserve(s, new) == 1024


def test_ready_restore_is_control_only_and_does_not_dilute_public_prefill(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    p, restored = prefill('p'), prefill('restored', history=0)
    s.running = [request('decode'), p]
    s.waiting.add_request(restored)
    # The exact read-only checkpoint hint is normally produced by _hint().
    s.offline_policy._hint = lambda r: (32768, 16384, 'cpu', None)
    prepare(s)
    assert s.offline_policy.rows['restored'].restore_tokens == 16384
    assert s.prefill_batch.contenders == 1
    assert reserve(s, p) == 2048
    assert not s.connector.sessions.entries and restored.num_computed_tokens == 0


def test_atomic_query_is_not_a_fair_share_and_preserves_native_mtp_lookahead(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    p = prefill('ordinary')
    q = prefill('query', history=19008, prompt=23024,
                plan={'kvmem_query_span': [19028, 23022]})
    s.running = [request('decode'), q, p]
    s.num_prefill_lookahead = 4
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', NATIVE_RESERVE)
    prepare(s)
    assert s.prefill_batch.contenders == 1 and reserve(s, p) == 2048
    assert reserve(s, q, 4016) == 4014  # Complete Query even after ordinary credits run out.
    assert s.prefill_batch.remaining == 0
    assert s._cost_admitted['query']['atomic_range'] == (19008, 23022)


def test_prepared_replay_is_bounded_by_original_reservation_and_not_reselected(monkeypatch, tmp_path):
    from dataclasses import replace
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_kvmem.admission import QueryPolicy
    from test_scheduler_transactions import setup_scheduler
    s, r, _ = setup_scheduler()
    s._bounded_target = s.request_planning = True
    s.query_policy = QueryPolicy(16, 32, 64)
    s.async_refresh = True
    s.prepared_refreshes = {'a': (48, 32)}
    r.num_computed_tokens, r.num_prompt_tokens = 48, 96
    r.kv_transfer_params = {'kvmem_request_plan': {'kvmem_reselect': 'off'}}
    s.prefill_batch = BatchPrefillBudget(16, 16)
    s.prefill_batch.begin(decode_ready=True, contenders=1)
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self, r, h, n: n)
    tokens = s._reserve_prefill_lookahead(r, 48, 32)
    assert tokens == 16 and s.prepared_refreshes == {'a': (48, 32)}
    out = replace(SchedulerOutput.make_empty(), num_scheduled_tokens={'a': tokens},
                  total_num_scheduled_tokens=tokens)
    assert not s._build_kv_connector_meta(s.connector, out).commands
    assert not s.prepared_refreshes and not s.transactions
    assert out.num_scheduled_tokens == {'a': 16} and r.num_computed_tokens == 48


@pytest.mark.parametrize('decode_ready,cap', [(False, 64), (True, 64), (True, 32)])
def test_shared_cap_does_not_strand_mtp_before_required_generation_checkpoint(monkeypatch, tmp_path, decode_ready, cap):
    from vllm_kvmem.admission import QueryPolicy
    s = setup(monkeypatch, tmp_path, cap=cap)
    s.query_policy = QueryPolicy(32, 128, 128)
    s.num_prefill_lookahead = 4
    r = prefill('checkpoint', history=96, prompt=131)
    s.running = [r]
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', NATIVE_RESERVE)
    s.prefill_batch.begin(decode_ready=decode_ready, contenders=1)
    # This is a multi-module lookahead edge contract, not evidence about the
    # currently qualified one-module MTP GPU profile (whose lookahead is 1).
    assert reserve(s, r, 35) == 32  # Visit 128, never an unresumable 127.
    r.num_computed_tokens = 128
    from vllm_kvmem.request_plan import request_plan
    plan = request_plan(r, 32, 128)
    assert plan.action(128, 131, 0, 256) == 'pressure'
    # Simulate the completed selection's mapped offset, not a model forward.
    s.window_managers[0].offsets[r.request_id] = 64
    assert plan.action(128, 131, 64, 256) is None
    s.prefill_batch.begin(decode_ready=decode_ready, contenders=1)
    assert reserve(s, r, 3) == 3


def native_restore_setup(monkeypatch, tmp_path, *, hot=False):
    from test_native_readiness_admission import native_fixture, checkpoint, decode, enqueue, request as native_request
    from vllm_kvmem.admission import QueryPolicy
    s, window, sessions, residency, gate, _ = native_fixture(monkeypatch, tmp_path, max_seqs=4)
    # The fixture normally exercises allocation without bounded forwarding.
    # Enable the actual bounded metadata hook so restore is stripped to control.
    del s._build_kv_connector_meta
    s._bounded_target = s.request_planning = s.async_restore = True
    s.async_refresh = s.session_capture_decode = False
    s.prepared_refreshes, s._deferred_native_padding = {}, {}
    s._restore_sequence = 0
    s.query_policy = QueryPolicy(16, 256, 256)
    window.select_budget, window.capacity = 256, 512
    s.prefill_batch = BatchPrefillBudget(64, 16)  # Fits one block per native slot.
    s.connector.scheduler_completion = s._complete_refresh
    # History accounting is independently exercised by archive/admission tests;
    # here it permits the native GPU/session/restore allocation paths to run.
    s.connector.history_admission = NS(can_admit=lambda r: True, allows=lambda r: True,
        allocated=lambda *args: None, consume=lambda *args: None, finished=lambda *args: None)
    s.offline_policy = OfflineAgentPolicy(s, NS(maintenance_aging_ms=250,
        prefill_max_tokens=2048), None, clock=lambda: 10.)
    checkpoint(sessions, window, residency, 'restore-session', hot=hot)
    d = native_request('decode', prompt=16)
    decode(s, d)
    p = native_request('prefill', prompt=129)
    assert s.kv_cache_manager.allocate_slots(p, 16) is not None
    p.num_computed_tokens, p.is_prefill_chunk, p.status = 16, True, RequestStatus.RUNNING
    s.requests[p.request_id] = p
    s.running.append(p)
    restored = native_request('restored', session='restore-session', prompt=49)
    enqueue(s, restored)
    return s, window, sessions, residency, gate, restored


@pytest.mark.parametrize('hot', [False, True])
@pytest.mark.parametrize('allocation_fails', [False, True])
def test_native_restore_control_bypasses_spent_forward_cap_but_not_allocation(monkeypatch, tmp_path, hot, allocation_fails):
    from vllm.v1.outputs import KVConnectorOutput
    from vllm_kvmem.connector import ControlCompletions, RestoreCommand
    from vllm_kvmem.session_control import SessionNotice
    from test_core import Event
    s, window, sessions, residency, gate, r = native_restore_setup(monkeypatch, tmp_path, hot=hot)
    if allocation_fails:
        original = s.kv_cache_manager.native.allocate_slots
        monkeypatch.setattr(s.kv_cache_manager.native, 'allocate_slots',
            lambda req, *a, **kw: None if req is r else original(req, *a, **kw))
    output = s.schedule()  # Real pinned native loop, actual lookup/permit/allocate/metadata.
    assert output.num_scheduled_tokens == {'decode': 1, 'prefill': 64}
    assert s.prefill_batch.remaining == 0 and s.prefill_batch.reserved == {'prefill': 64}
    if allocation_fails:
        assert not output.kv_connector_metadata.commands
        assert r.status is RequestStatus.WAITING and r.num_computed_tokens == 0
        assert r.request_id not in gate.permits and r.request_id not in s.kv_cache_manager.reservations
        assert not window.req_to_blocks.get(r.request_id) and not window.resident_imports
        assert sessions.requests[r.request_id].phase == 'lookup'
        assert bool(residency.entries) is hot
        return
    command, = output.kv_connector_metadata.commands
    assert isinstance(command, RestoreCommand) and command.request_id == r.request_id
    assert gate.permits[r.request_id].allocated and command.resident is hot
    assert r.num_computed_tokens == 32 and r.num_in_flight_tokens == 0
    assert r.is_prefill_chunk is False and r.request_id in s.transactions
    assert window.commit_import(r.request_id, 32, Event(True))
    s.connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        completed=[(r.request_id, command.command_id)], sessions=[SessionNotice('restored',
            sessions.namespace(r), command.generation, r.request_id, 32, 32, retained=True)])))
    assert not s.transactions and r.request_id not in gate.permits
    # Peers still have in-flight model rows. The restored suffix is not decode
    # just because its initial native flag survived this control-only step.
    assert not s._decode_ready() and r.num_computed_tokens < r.num_prompt_tokens


def test_cancelled_restore_lookup_loses_cap_exemption_and_returns_provisional_resources(monkeypatch, tmp_path):
    s, window, sessions, residency, gate, r = native_restore_setup(monkeypatch, tmp_path, hot=True)
    assert s.connector.get_num_new_matched_tokens(r, 0) == (32, False)
    assert s._restore_control_reservation(r, 32)
    assert not s._restore_control_reservation(r, 31)
    assert s.connector.request_finished_all_groups(r, ()) == (False, None)
    assert not s._restore_control_reservation(r, 32)
    assert not gate.permits and not window.resident_imports and not window.imports
    assert sessions.entries[sessions.namespace(r)].phase == 'idle' and residency.entries
    s.prefill_batch.begin(decode_ready=True, contenders=1)
    s.prefill_batch.charge('peer', 64)
    assert s._reserve_prefill_lookahead(r, 32, 16) == 0
    assert r.num_computed_tokens == 0 and not window.req_to_blocks.get(r.request_id)


def test_stale_waiting_publication_preserves_share_and_does_not_hide_restore_behind_it(monkeypatch, tmp_path):
    from test_native_readiness_admission import enqueue, request as native_request
    from vllm_kvmem.connector import RestoreCommand
    s, _, _, _, gate, restored = native_restore_setup(monkeypatch, tmp_path)
    older = native_request('older-prefill', prompt=129)
    older.arrival_time = restored.arrival_time - 1
    enqueue(s, older)
    s.connector.publication_blocked = {older.request_id}
    assert older.status is RequestStatus.WAITING
    output = s.schedule()
    # Existing FIFO/aging still puts the older ordinary request first. Its
    # native WAITING status ignores stale publication feedback, so the same
    # rule must retain its fair share instead of letting running P spend all 64.
    assert output.num_scheduled_tokens == {'decode': 1, 'prefill': 32, 'older-prefill': 32}
    assert s.prefill_batch.contenders == 2 and s.prefill_batch.remaining == 0
    assert s.prefill_batch.reserved == {'prefill': 32, 'older-prefill': 32}
    assert [r.request_id for r in s.running] == ['decode', 'prefill', 'older-prefill', 'restored']
    command, = output.kv_connector_metadata.commands
    assert isinstance(command, RestoreCommand) and command.request_id == restored.request_id
    assert gate.permits[restored.request_id].allocated
    assert restored.num_computed_tokens == 32 and restored.num_in_flight_tokens == 0
    assert not s.waiting and not s.skipped_waiting


def test_deferred_mtp_tail_requires_all_placeholders_and_does_not_advance_when_deferred(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path, cap=32)
    s.num_spec_tokens = 4
    tail = prefill('tail', history=64, prompt=65)
    tail.spec_token_ids = [-1] * 4
    s._deferred_native_padding = {'tail': (64, 5)}
    s.running = [request('decode'), tail]
    s.prepared_refreshes['tail'] = (64, 5)
    prepare(s)
    s.prefill_batch.charge('peer', 28)
    before = deepcopy((tail.num_computed_tokens, tail.spec_token_ids, s.prepared_refreshes))
    assert reserve(s, tail, 5) == 0
    assert before == (tail.num_computed_tokens, tail.spec_token_ids, s.prepared_refreshes)
    prepare(s)
    assert reserve(s, tail, 4) == 0
    assert reserve(s, tail, 5) == 5
    assert s.prefill_batch.remaining == 27


def test_rejected_or_cancelled_reservation_never_advances_counters_and_expires_next_step(monkeypatch, tmp_path):
    s = setup(monkeypatch, tmp_path)
    a, b = prefill('a'), prefill('b')
    s.running = [request('decode'), a, b]
    prepare(s)
    s.archive_admission = NS(allows=lambda rid, *_: rid != 'a')
    assert reserve(s, a) == 0
    assert 'a' not in s.prefill_batch.reserved
    assert reserve(s, b) == reserve(s, b) == 1024
    assert s.prefill_batch.remaining == 1024  # Repeated native reservation is idempotent.
    # Native may subsequently fail allocation or cancel B. Do not recycle its
    # credits into an uncertain same-step allocation; reset only at begin().
    b.is_finished = lambda: True
    s.running.remove(b)
    s.archive_admission = None
    prepare(s)
    assert s.prefill_batch.reserved == {} and reserve(s, a) == 2048
    assert a.num_computed_tokens == b.num_computed_tokens == 32768


def test_shared_budget_options_remain_opt_in_and_reject_unqualified_offline_combinations():
    from test_phase_options import profile, options
    payload = profile()['kvmem']
    assert options(payload).prefill_batch_tokens == 0
    payload['prefill_batch_tokens'] = 2048
    assert options(payload).offline_agent_policy and options(payload).prefill_batch_tokens == 2048
    for extra in ({'offline_phase_policy': True}, {'offline_prefill_coalescing': True},
                  {'adaptive_prefill': True}):
        with pytest.raises(ValueError, match='independent'):
            options(payload | extra)
