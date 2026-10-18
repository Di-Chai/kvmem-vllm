from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_kvmem.admission import QueryPolicy
from vllm_kvmem.scheduler import KVMemScheduler

NATIVE_RESERVE = Scheduler._reserve_prefill_lookahead


def request(rid, *, computed=128, prompt=128, output=1, plan=None):
    return SimpleNamespace(request_id=rid, num_computed_tokens=computed,
        num_prompt_tokens=prompt, num_tokens=prompt + output,
        num_tokens_with_spec=prompt + output, num_in_flight_tokens=0,
        num_output_placeholders=0, num_stale_output_tokens=0,
        next_decode_eligible_step=0, is_prefill_chunk=computed < prompt,
        is_finished=lambda: False, status='waiting', spec_token_ids=[],
        max_tokens=128, prompt_token_ids=[1] * prompt,
        kv_transfer_params={'kvmem_request_plan': plan or {'kvmem_reselect': 'off'}})


def scheduler(monkeypatch, tmp_path, *, budget=8192):
    from vllm_kvmem.phase_policy import PhasePolicy
    s = KVMemScheduler.__new__(KVMemScheduler)
    s._bounded_target = True
    s.phase_policy = PhasePolicy(4, budget, 1000., clock=lambda: s.now)
    s._phase_decision = None
    s.now = 0.
    s.current_step = 0
    s.scheduler_config = SimpleNamespace(max_num_seqs=4)
    s.running, s.waiting, s.skipped_waiting = [], [], []
    s.transactions, s.prepared_refreshes = {}, {}
    s.async_refresh = s.async_restore = False
    s.session_capture_decode = False
    s.query_policy = QueryPolicy(32, 131072, 131072)
    s.window_managers = {0: SimpleNamespace(offsets={})}
    s.connector = SimpleNamespace(pending={}, sessions=SimpleNamespace(
        entries={}, uncached_sessions={}, namespace=lambda r: r.request_id,
        capacity=32, preserve_history=True),
        history_admission=SimpleNamespace(can_admit=lambda r: True))
    s.kv_cache_manager = SimpleNamespace(admission_preview=lambda r: None)
    s.offline_policy = SimpleNamespace(ranks={}, _hint=lambda r: (0, 0, 'none', None))
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self, r, h, n: n)
    return s


def record(s, tokens):
    s._record_phase(SimpleNamespace(num_scheduled_tokens=tokens))


def test_four_running_decodes_never_yield_to_waiting_without_native_slot(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    s.running = [request(str(i)) for i in range(4)]
    s.waiting = [request('new', computed=0, prompt=10000, output=0)]
    for now in (0., 2., 100.):
        s.now = now
        s._prepare_phase()
        assert s._phase_decision.phase == 'decode'
        assert not s._phase_prefills
        assert s._reserve_prefill_lookahead(s.running[0], 128, 1) == 1
        assert s._reserve_prefill_lookahead(s.waiting[0], 0, 1024) == 0
        record(s, {'0': 1})
    assert len(s.running) == 4
    assert all(r.next_decode_eligible_step == 0 for r in s.running)


@pytest.mark.parametrize('blocked', ['transaction', 'connector', 'inflight', 'native_fence', 'stale', 'publication'])
def test_native_and_kvmem_fences_are_never_decode_readiness(monkeypatch, tmp_path, blocked):
    s = scheduler(monkeypatch, tmp_path)
    r = request('decode')
    s.running = [r]
    if blocked == 'transaction': s.transactions[r.request_id] = object()
    if blocked == 'connector': s.connector.pending[r.request_id] = object()
    if blocked == 'inflight': r.num_in_flight_tokens = 1
    if blocked == 'native_fence': r.next_decode_eligible_step = 2
    if blocked == 'stale': r.num_stale_output_tokens = 1
    if blocked == 'publication':
        from vllm.v1.request import RequestStatus
        r.status = RequestStatus.RUNNING
        s.connector.publication_blocked = {r.request_id}
    s._prepare_phase()
    assert not s._phase_decodes
    assert s._reserve_prefill_lookahead(r, 128, 1) == 0


def test_pending_publication_cannot_hold_decode_phase_against_ready_prefill(monkeypatch, tmp_path):
    from vllm.v1.request import RequestStatus
    s = scheduler(monkeypatch, tmp_path)
    decode = request('decode')
    decode.status = RequestStatus.RUNNING
    s.running = [decode]
    s.connector.publication_blocked = {'decode'}
    s.waiting = [request('fresh', computed=0, prompt=10000, output=0)]
    s._is_blocked_waiting_status = lambda status: False
    s._prepare_phase()
    assert not s._phase_decodes and not s._decode_ready()
    assert s._phase_decision.phase == 'prefill'
    assert s._phase_prefills == {'fresh'}


@pytest.mark.parametrize('blocked', ['cpu', 'gpu', 'session', 'capacity', 'native'])
def test_blocked_waiting_work_cannot_force_prefill_phase(monkeypatch, tmp_path, blocked):
    s = scheduler(monkeypatch, tmp_path)
    s.running = [request('decode')]
    r = request('prefill', computed=0, prompt=10000, output=0)
    s.waiting = [r]
    s._is_blocked_waiting_status = lambda status: blocked == 'native'
    if blocked == 'cpu': s.connector.history_admission.can_admit = lambda r: False
    if blocked == 'gpu': s.kv_cache_manager.admission_preview = lambda r: 'gpu_lifetime_budget'
    if blocked == 'session':
        s.connector.sessions.entries[r.request_id] = SimpleNamespace(phase='retiring', request_id='old')
    if blocked == 'capacity': s.connector.sessions.capacity = 0
    s.now = 20.
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert not s._phase_prefills


def test_final_prefill_keeps_native_sampling_shape_then_parks_next_decode(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    tail = request('tail', computed=64, prompt=65, output=0)
    other = request('long', computed=0, prompt=10000, output=0)
    s.running = [tail, other]
    s.num_spec_tokens = 4
    s._prepare_phase()
    assert s._phase_decision.phase == 'prefill'
    assert s._reserve_prefill_lookahead(tail, 64, 5) == 5
    record(s, {'tail': 5})
    assert s.phase_policy.snapshot()['total_prefill_tokens'] == 1
    # Native update/sampling has consumed the prompt and produced its first
    # token. The policy never suppresses that token or edits these counters.
    tail.num_computed_tokens = 65
    tail.num_tokens = tail.num_tokens_with_spec = 66
    tail.is_prefill_chunk = False
    s._prepare_phase()
    assert s._phase_decision.phase == 'prefill'
    assert s._reserve_prefill_lookahead(tail, 65, 5) == 0
    assert tail.num_tokens == 66 and tail.next_decode_eligible_step == 0


def test_prefill_wave_shares_budget_but_query_replay_may_overshoot_once(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path, budget=64)
    ordinary = request('ordinary', computed=0, prompt=10000, output=0)
    replay = request('query', computed=19008, prompt=100000, output=0,
                     plan={'kvmem_query_span': [19028, 23022]})
    second = request('query2', computed=19008, prompt=100000, output=0,
                     plan={'kvmem_query_span': [19028, 23022]})
    s.running = [ordinary, replay, second]
    s._prepare_phase()
    assert s._reserve_prefill_lookahead(ordinary, 0, 1024) == 64
    assert s._reserve_prefill_lookahead(replay, 19008, 10000) == 4014
    assert s._reserve_prefill_lookahead(second, 19008, 10000) == 0
    record(s, {'ordinary': 64, 'query': 4014})
    assert s.phase_policy.snapshot()['total_prefill_tokens'] == 4078


def test_atomic_query_is_not_clipped_again_by_native_mtp_lookahead(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path, budget=64)
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', NATIVE_RESERVE)
    s.num_prefill_lookahead = 4
    replay = request('query', computed=19008, prompt=23024, output=0,
                     plan={'kvmem_query_span': [19028, 23022]})
    s.running = [replay]
    s._prepare_phase()
    assert s._reserve_prefill_lookahead(replay, 19008, 4016) == 4014
    record(s, {'query': 4014})


def test_full_session_table_does_not_block_stateless_native_request(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    r = request('stateless', computed=0, prompt=10000, output=0)
    s.waiting = [r]
    s.connector.sessions.namespace = lambda r: None
    s.connector.sessions.capacity = 0
    s._is_blocked_waiting_status = lambda status: False
    s._prepare_phase()
    assert s._phase_prefills == {'stateless'}
    assert s._phase_decision.phase == 'prefill'


def test_async_refresh_only_does_not_pause_decode_or_charge_public_prefill(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    decode = request('decode')
    refresh = request('refresh', computed=262080, prompt=300000, output=0)
    s.running = [decode, refresh]
    s.async_refresh = True
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert s._phase_maintenance == {'refresh'}
    assert s._reserve_prefill_lookahead(refresh, 262080, 32) == 32
    assert s._reserve_prefill_lookahead(decode, 128, 1) == 1
    # The existing connector removes refresh's public forward before recording.
    record(s, {'decode': 1})
    assert s.phase_policy.snapshot()['total_prefill_tokens'] == 0
    with pytest.raises(RuntimeError, match='maintenance admitted public prefill'):
        record(s, {'refresh': 32, 'decode': 1})


def test_exact_async_import_is_maintenance_until_public_tail_ready(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    s.async_restore = True
    s.running = [request('decode')]
    r = request('restore', computed=0, prompt=129, output=0)
    s.waiting = [r]
    s._is_blocked_waiting_status = lambda status: False
    s.offline_policy._hint = lambda req: (128, 128, 'cpu', None)
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert s._phase_maintenance == {'restore'}
    assert s._reserve_prefill_lookahead(r, 128, 1) == 1
    record(s, {})
    s.waiting.clear()
    r.num_computed_tokens = 128
    s.running.append(r)
    s._prepare_phase()
    assert 'restore' in s._phase_prefills
    assert not s._phase_maintenance


def test_deferred_waiting_head_cannot_hide_later_restore_or_consume_slot(monkeypatch, tmp_path):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    s = scheduler(monkeypatch, tmp_path)
    s.async_restore = True
    s.running = [request(str(i)) for i in range(3)]
    cold = request('cold', computed=0, prompt=10000, output=0)
    restored = request('restored', computed=0, prompt=129, output=0)
    s.waiting, s.skipped_waiting = FCFSRequestQueue(), FCFSRequestQueue()
    s.waiting.extend([cold, restored])
    s._is_blocked_waiting_status = lambda status: False
    s.offline_policy.ranks = {'cold': 0, 'restored': 1}
    s.offline_policy.waiting_queue = lambda: s.waiting
    s.offline_policy._hint = lambda r: (128, 128, 'cpu', None) if r is restored else (0, 0, 'none', None)
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert s._phase_prefills == {'cold'}
    assert s._phase_maintenance == {'restored'}
    queue = s._select_waiting_queue_for_scheduling()
    assert queue.peek_request() is restored
    assert list(queue) == [restored, cold]
    assert len(s.running) == 3
    assert s._reserve_prefill_lookahead(restored, 128, 1) == 1
    assert s._reserve_prefill_lookahead(cold, 0, 1024) == 0


def test_fragmented_wave_tail_yields_only_after_recording_no_admitted_prefill(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path, budget=32)
    tail = request('tail', computed=64, prompt=65, output=0)
    long = request('long', computed=0, prompt=10000, output=0)
    decode = request('decode')
    s.running = [decode, tail, long]
    s._prepare_phase()
    assert s._reserve_prefill_lookahead(tail, 64, 1) == 1
    record(s, {'tail': 1})
    tail.num_computed_tokens, tail.num_tokens, tail.num_tokens_with_spec = 65, 66, 66
    tail.is_prefill_chunk = False
    s._prepare_phase()
    assert s._phase_decision.prefill_budget == 31
    assert s._reserve_prefill_lookahead(long, 0, 1024) == 0
    record(s, {})
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert s._reserve_prefill_lookahead(decode, 128, 1) == 1
    record(s, {'decode': 1})
    assert s.phase_policy.snapshot()['total_prefill_tokens'] == 1


def test_admission_losing_readonly_preview_yields_decode_after_empty_wave(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    decode = request('decode')
    first = request('p1', computed=0, prompt=10000, output=0)
    second = request('p2', computed=0, prompt=10000, output=0)
    s.running = [decode, first, second]
    s._prepare_phase()
    assert s._phase_decision.phase == 'prefill'
    assert s._reserve_prefill_lookahead(first, 0, 1024) == 1024
    # A later allocation/transaction fence can still reject the proposal.
    record(s, {})
    s._prepare_phase()
    assert s._phase_decision.phase == 'decode'
    assert s._reserve_prefill_lookahead(decode, 128, 1) == 1
    record(s, {'decode': 1})


def test_steady_decode_logs_at_most_once_per_second_but_records_every_batch(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    r = request('decode')
    s.running = [r]
    monkeypatch.setattr('vllm_kvmem.scheduler.time.monotonic', lambda: s.now)
    for at in (0., .1, .2, .9, 1.):
        s.now = at
        s._prepare_phase()
        assert s._reserve_prefill_lookahead(r, 128, 1) == 1
        record(s, {'decode': 1})
    assert len((tmp_path / 'phase_decisions.jsonl').read_text().splitlines()) == 2
    assert s.phase_policy.snapshot()['total_decode_tokens'] == 5


def test_off_mode_preserves_original_admission_without_phase_state(monkeypatch, tmp_path):
    s = scheduler(monkeypatch, tmp_path)
    s.phase_policy = None
    r = request('p', computed=0, prompt=10000, output=0)
    s._prepare_phase()
    assert s._reserve_prefill_lookahead(r, 0, 1024) == 1024
    s._record_phase(SimpleNamespace())
    assert not (tmp_path / 'phase_decisions.jsonl').exists()
