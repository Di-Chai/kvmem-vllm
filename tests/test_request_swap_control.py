"""Real native page pools with deterministic worker completion boundaries."""
from types import SimpleNamespace
from collections import deque
from random import Random

import pytest

pytest.importorskip('vllm')
from vllm.v1.request import RequestStatus

from test_incremental_capacity import make_gate, request
from vllm_kvmem.request_swap_control import RequestSwapController


class Connector:
    def __init__(self):
        self.pending, self.commands, self.publication_blocked = {}, [], set()

    def queue(self, command):
        assert command.request_id not in self.pending
        self.pending[command.request_id] = command.command_id
        self.commands.append(command)


def fixture(*, blocks=5, capacity=65536, state=False, mtp=0):
    scheduler = SimpleNamespace(current_step=100, requests={}, running=[], transactions={},
        prepared_refreshes={}, connector=Connector(), scheduler_config=SimpleNamespace(max_num_seqs=8))
    finished = []

    def finish(req, *, delay_free_blocks=False):
        finished.append((req.request_id, delay_free_blocks))
        gate.free(req)
        scheduler.requests.pop(req.request_id)
        scheduler.running.remove(req)

    controller = RequestSwapController(scheduler, checkpoint_bytes=capacity, page_bytes=1024,
        finish_callback=finish, min_resident_steps=2, max_pause_steps=4)
    gate, groups, pool, _ = make_gate(blocks, state=state, mtp=mtp, controller=controller)
    controller.bind(gate)
    scheduler.kv_cache_manager = gate
    return controller, gate, groups, pool, scheduler, finished


def admit(f, name, count):
    controller, gate, groups, pool, scheduler, finished = f
    req = request(name)
    assert gate.allocate_slots(req, count) is not None
    req.num_computed_tokens = count
    req.status = RequestStatus.RUNNING
    scheduler.requests[name] = req
    scheduler.running.append(req)
    return req


def ack(f, action, rid, *, clear_pending=True):
    controller, _, _, _, scheduler, _ = f
    command = scheduler.connector.commands.pop(0)
    assert (command.action, command.request_id) == (action, rid)
    assert controller.complete(rid, command.command_id)
    if clear_pending:
        del scheduler.connector.pending[rid]
    return command


def pressure_pair():
    f = fixture()
    a, b = admit(f, 'a', 48), admit(f, 'b', 16)
    assert not f[1].can_schedule(b, 16)
    return f, a, b


def test_hint_only_and_capture_fence_keep_pages_until_verified_worker_ack():
    f, a, b = pressure_pair()
    controller, gate, groups, pool, scheduler, _ = f
    before = tuple(groups[0].req_to_blocks['a'])
    assert not controller.is_suspended('a') and not scheduler.connector.commands
    controller.before_schedule()
    assert controller.is_suspended('a') and not controller.is_suspended('b')
    assert pool.get_num_free_blocks() == 0
    assert tuple(groups[0].req_to_blocks['a']) == before
    command = scheduler.connector.commands[0]
    assert command.checkpoint_bytes == 3 * 1024 + 4096
    with pytest.raises(ValueError, match='Stale'):
        controller.complete('a', command.command_id + 1)
    assert pool.get_num_free_blocks() == 0
    ack(f, 'capture', 'a')
    assert pool.get_num_free_blocks() == 3
    controller.before_schedule()
    assert scheduler.connector.commands == []  # Give blocked b its growth turn.
    assert gate.allocate_slots(b, 16) is not None
    b.num_computed_tokens += 16
    assert pool.get_num_free_blocks() == 2
    assert controller.snapshot()['resources']['resources']['checkpoint_bytes']['live'] == 7168


def test_restore_waits_for_pages_protects_reclaimed_capacity_and_requires_forward_progress():
    f, a, b = pressure_pair()
    controller, gate, groups, pool, scheduler, _ = f
    controller.before_schedule(); ack(f, 'capture', 'a')
    assert gate.allocate_slots(b, 16) is not None
    b.num_computed_tokens += 16
    controller.before_schedule()
    assert not scheduler.connector.commands  # a needs three pages, only two are free.
    for _ in range(3):
        scheduler.current_step += 1
        assert gate.allocate_slots(b, 1) is not None
        b.num_computed_tokens += 1
        controller.before_schedule()
    assert scheduler.connector.commands[0].request_id == 'b'
    assert controller.reserved_pages('newcomer') == 1
    assert not gate.can_schedule(request('newcomer'), 16)
    assert gate.allocate_slots(request('newcomer'), 16) is None
    ack(f, 'capture', 'b')
    assert pool.get_num_free_blocks() == 4
    controller.before_schedule()
    assert pool.get_num_free_blocks() == 1
    assert controller.is_suspended('a')
    ack(f, 'restore', 'a', clear_pending=False)
    controller.before_schedule()
    assert not scheduler.connector.commands  # Connector removes pending after callback.
    del scheduler.connector.pending['a']
    controller.before_schedule()
    assert controller.is_suspended('a') and not gate.can_schedule(a, 1)
    ack(f, 'drop', 'a')
    controller.before_schedule()
    assert not controller.is_suspended('a')
    assert a.num_computed_tokens == 48
    assert len(groups[0].req_to_blocks['a']) == 3
    scheduler.current_step += 100
    assert not controller._eligible(a)  # Idle scheduler turns cannot permit repeated swapping.
    assert gate.allocate_slots(a, 1) is not None
    a.num_computed_tokens += 1
    controller.before_schedule()
    snapshot = controller.snapshot()
    assert snapshot['resume_forward_completions'] == 1
    assert snapshot['resume_forward_tokens'] == 1
    assert snapshot['max_pause_progress_rounds_observed'] >= 4
    assert snapshot['recent_transitions'][0]['action'] == 'capture'


@pytest.mark.parametrize('phase', ['capturing', 'paused', 'restoring', 'restored', 'dropping'])
def test_cancellation_waits_for_all_checkpoint_and_drop_fences_before_native_free(phase):
    f = fixture(blocks=6)
    a, b = admit(f, 'a', 48), admit(f, 'b', 16)
    controller, gate, groups, pool, scheduler, finished = f
    assert controller._capture(a)
    if phase != 'capturing':
        ack(f, 'capture', 'a')
    if phase in ('restoring', 'restored', 'dropping'):
        controller._pressure.clear()
        controller.before_schedule()
        assert scheduler.connector.commands[0].action == 'restore'
    if phase in ('restored', 'dropping'):
        ack(f, 'restore', 'a')
    if phase == 'dropping':
        controller.before_schedule()
    a.status = RequestStatus.FINISHED_ABORTED
    assert controller.defer_free(a, delay_free_blocks=True)
    assert not finished
    if phase == 'capturing':
        assert pool.get_num_free_blocks() == 1
        ack(f, 'capture', 'a')
    if phase == 'restoring':
        assert pool.get_num_free_blocks() == 1
        ack(f, 'restore', 'a')
    if phase != 'dropping':
        controller.before_schedule()
    assert not finished and controller.is_suspended('a')
    ack(f, 'drop', 'a', clear_pending=False)
    controller.before_schedule()
    assert not finished
    del scheduler.connector.pending['a']
    controller.before_schedule()
    assert finished == [('a', True)]
    assert pool.get_num_free_blocks() == 4
    assert not controller.is_suspended('a')
    assert controller.snapshot()['resources']['resources']['checkpoint_bytes']['live'] == 0
    assert not controller.defer_free(b)


def test_serial_restore_reserves_first_forward_headroom_and_cannot_deadlock_two_protected_owners():
    f, a, b = pressure_pair()
    controller, gate, _, pool, scheduler, _ = f
    assert controller._capture(a); ack(f, 'capture', 'a')
    assert controller._capture(b); ack(f, 'capture', 'b')
    controller._pressure.clear()
    scheduler.current_step += 10
    assert pool.get_num_free_blocks() == 4
    assert controller._restore(controller._states['b'])
    assert pool.get_num_free_blocks() == 3
    ack(f, 'restore', 'b')
    controller.before_schedule(); ack(f, 'drop', 'b'); controller.before_schedule()
    assert not scheduler.connector.commands  # Overdue a cannot steal b's first-forward pages.
    assert controller.snapshot()['awaiting_forward_progress'] == ['b']
    assert controller.reserved_pages('newcomer') == 3
    snapshot = gate.snapshot()
    assert snapshot['future_reserved_pages'] == 0
    assert snapshot['restore_progress_reserved_pages'] == 3
    assert snapshot['free_after_growth_reserve_pages'] == 0
    assert not gate.can_schedule(request('newcomer'), 1)
    assert gate.can_schedule(b, 1)
    assert gate.allocate_slots(b, 1) is not None
    b.num_computed_tokens += 1
    controller.before_schedule()
    assert controller.snapshot()['resume_forward_completions'] == 1
    assert controller.snapshot()['awaiting_forward_progress'] == []


def test_cancel_after_drop_releases_first_forward_credit_before_native_free():
    f, a, b = pressure_pair()
    controller, gate, _, pool, scheduler, _ = f
    assert controller._capture(a); ack(f, 'capture', 'a')
    assert controller._capture(b); ack(f, 'capture', 'b')
    controller._pressure.clear()
    assert controller._restore(controller._states['b'])
    ack(f, 'restore', 'b'); controller.before_schedule()
    ack(f, 'drop', 'b'); controller.before_schedule()
    assert controller.reserved_pages('newcomer') == 3
    b.status = RequestStatus.FINISHED_ABORTED
    assert not controller.defer_free(b)
    gate.free(b)
    scheduler.requests.pop('b'); scheduler.running.remove(b)
    assert gate.snapshot()['restore_progress_reserved_pages'] == 0
    assert pool.get_num_free_blocks() == 4
    controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'restore'
    assert scheduler.connector.commands[0].request_id == 'a'


@pytest.mark.parametrize('busy', ['inflight', 'pending', 'transaction', 'publication', 'refresh'])
def test_only_quiescent_victims_can_enter_capture(busy):
    f, a, b = pressure_pair()
    controller, _, _, pool, scheduler, _ = f
    if busy == 'inflight':
        a.num_in_flight_tokens = 1
    elif busy == 'pending':
        scheduler.connector.pending['a'] = 5
    elif busy == 'transaction':
        scheduler.transactions['a'] = object()
    elif busy == 'publication':
        scheduler.connector.publication_blocked.add('a')
    else:
        scheduler.prepared_refreshes['a'] = object()
    controller.before_schedule()
    assert not scheduler.connector.commands
    assert pool.get_num_free_blocks() == 0


def test_insufficient_host_quota_defers_without_a_control_command_or_partial_free():
    f, a, b = pressure_pair()
    controller, _, groups, pool, scheduler, _ = f
    # Model other already promised captures without allocating their backing.
    available = controller.ledger.snapshot()['resources']['checkpoint_bytes']['available']
    held = controller.ledger.prepare('held', 0, {'checkpoint_bytes': available})
    assert held is not None
    for _ in range(10):
        controller.before_schedule()
    assert scheduler.connector.commands == [] and scheduler.connector.pending == {}
    assert controller.snapshot()['capacity_deferrals'] == 10
    assert pool.get_num_free_blocks() == 0 and len(groups[0].req_to_blocks['a']) == 3
    controller.ledger.abort(held)
    controller.before_schedule()
    assert len(scheduler.connector.commands) == 1
    for _ in range(10):
        controller.before_schedule()
    assert len(scheduler.connector.commands) == 1


def test_pressure_for_cancelled_owner_does_not_pause_a_peer():
    f, a, b = pressure_pair()
    controller, gate, _, _, scheduler, _ = f
    b.status = RequestStatus.FINISHED_ABORTED
    assert not controller.defer_free(b)
    gate.free(b)
    scheduler.requests.pop('b'); scheduler.running.remove(b)
    controller.before_schedule()
    assert not scheduler.connector.commands and not controller.has_pending_work()
    assert not controller.is_suspended('a')


def test_native_state_and_mtp_rows_are_in_checkpoint_and_restore_commands():
    f = fixture(blocks=19, mtp=4, capacity=131072)
    a, b = admit(f, 'a', 32), admit(f, 'b', 16)
    controller, _, groups, pool, scheduler, _ = f
    assert controller._capture(a)
    command = ack(f, 'capture', 'a')
    assert tuple(map(len, command.block_ids)) == (2, 5)
    assert all('a' not in group.req_to_blocks for group in groups)
    controller.before_schedule()
    command = ack(f, 'restore', 'a')
    assert tuple(map(len, command.block_ids)) == (2, 5)
    controller.before_schedule(); ack(f, 'drop', 'a'); controller.before_schedule()
    assert not controller.is_suspended('a') and a.num_computed_tokens == 32


def test_unpublished_capture_failure_rolls_back_owned_pages_and_host_credit():
    f, a, b = pressure_pair()
    controller, _, groups, pool, scheduler, _ = f
    def reject(command):
        raise RuntimeError('queue unavailable')
    scheduler.connector.queue = reject
    with pytest.raises(RuntimeError, match='queue unavailable'):
        controller.before_schedule()
    assert not controller.is_suspended('a') and a.next_decode_eligible_step < 2**62
    assert pool.get_num_free_blocks() == 0 and len(groups[0].req_to_blocks['a']) == 3
    resources = controller.snapshot()['resources']['resources']['checkpoint_bytes']
    assert resources['live'] == resources['promised'] == 0


def test_startup_requires_retained_checkpoint_plus_bounded_progress_capacity():
    with pytest.raises(ValueError, match='need at least'):
        fixture(capacity=57343)


def test_empty_polling_and_delayed_ack_do_not_age_newcomer_but_actual_progress_bounds_wait():
    f = fixture()
    controller, gate, _, _, scheduler, _ = f
    a, b, c = admit(f, 'a', 8), admit(f, 'b', 8), admit(f, 'c', 17)
    waiter = request('waiter')
    scheduler.requests['waiter'] = waiter
    assert not gate.can_schedule(waiter, 16)
    controller.before_schedule()
    start = controller.snapshot()['progress_rounds']
    for _ in range(10000):
        scheduler.current_step += 1
        controller.before_schedule()
    assert not scheduler.connector.commands
    assert controller.snapshot()['progress_rounds'] == start
    assert controller.snapshot()['idle_poll_calls'] >= 10000
    for _ in range(controller.max_pause_steps - start):
        assert gate.allocate_slots(a, 1) is not None
        a.num_computed_tokens += 1
        controller.before_schedule()
    command = scheduler.connector.commands[0]
    assert command.action == 'capture'
    assert controller.snapshot()['recent_transitions'][-1]['reason'] == 'waiter_fairness'
    before = controller.snapshot()['progress_rounds']
    for _ in range(1000):
        scheduler.current_step += 1
        controller.before_schedule()
    assert len(scheduler.connector.commands) == 1
    assert controller.snapshot()['progress_rounds'] == before


def test_restored_cooldown_counts_own_forward_turns_instead_of_peer_or_idle_steps():
    f = fixture(blocks=9)
    controller, gate, _, _, scheduler, _ = f
    controller.min_resident_steps = 4
    a, b = admit(f, 'a', 8), admit(f, 'b', 8)
    controller.before_schedule()
    assert controller._capture(a); ack(f, 'capture', 'a')
    controller.before_schedule(); ack(f, 'restore', 'a')
    controller.before_schedule(); ack(f, 'drop', 'a'); controller.before_schedule()
    assert gate.allocate_slots(a, 1) is not None
    a.num_computed_tokens += 1
    controller.before_schedule()
    for _ in range(6):
        assert gate.allocate_slots(b, 1) is not None
        b.num_computed_tokens += 1
        scheduler.current_step += 1000
        controller.before_schedule()
    assert not controller._eligible(a)
    for _ in range(3):
        assert gate.allocate_slots(a, 1) is not None
        a.num_computed_tokens += 1
        controller.before_schedule()
    assert controller._eligible(a)


def test_imported_history_and_inflight_work_are_not_counted_as_completed_progress():
    f = fixture(blocks=9)
    controller, gate, groups, _, scheduler, _ = f
    req = request('import')
    scheduler.requests['import'] = req
    groups[0].prepare_import(req.request_id, 128, 32)
    assert gate.allocate_slots(req, 8, num_external_computed_tokens=128) is not None
    req.num_computed_tokens = 128
    controller.before_schedule()
    assert controller.snapshot()['progress_rounds'] == 0
    req.num_computed_tokens = 136
    req.num_in_flight_tokens = 8
    controller.before_schedule()
    assert controller.snapshot()['progress_rounds'] == 0
    req.num_in_flight_tokens = 0
    assert groups[0].commit_import('import', 128, SimpleNamespace(query=lambda: True))
    controller.before_schedule()
    assert controller.snapshot()['observed_forward_tokens'] == 8
    assert controller.snapshot()['observed_forward_turns'] == 1


def test_growth_reclaims_smallest_sufficient_snapshot_instead_of_largest_owner():
    f = fixture(blocks=7)
    controller, gate, _, _, scheduler, _ = f
    admit(f, 'large', 48)
    admit(f, 'small', 16)
    grow = admit(f, 'grow', 32)
    assert not gate.can_schedule(grow, 1)
    controller.before_schedule()
    command = scheduler.connector.commands[0]
    assert command.request_id == 'small'
    assert command.checkpoint_bytes == 1024 + 4096
    assert controller.snapshot()['recent_transitions'][-1]['reason'] == 'resident_growth'


def test_growth_preserves_nearly_finished_decode_when_another_victim_fits():
    from vllm import SamplingParams
    from vllm.v1.request import Request
    f = fixture(blocks=6)
    controller, gate, _, _, scheduler, _ = f
    near = Request('near', [1] * 8, SamplingParams(max_tokens=1), None)
    assert gate.allocate_slots(near, 8) is not None
    near.num_computed_tokens = 8
    near.status = RequestStatus.RUNNING
    scheduler.requests['near'] = near; scheduler.running.append(near)
    admit(f, 'far', 32)
    grow = admit(f, 'grow', 32)
    assert not gate.can_schedule(grow, 1)
    controller.before_schedule()
    assert scheduler.connector.commands[0].request_id == 'far'


def bind_session(f, req):
    from vllm_kvmem.session_control import SessionCommands
    req.session_id, req.cache_salt = 'session', 'salt'
    entry = SimpleNamespace(namespace=('salt', 'session'), generation=7, request_id=req.request_id,
        phase='active', allocated=True, history=0, candidate_position=32)
    f[4].connector.sessions = SimpleNamespace(entries={entry.namespace: entry},
        requests={req.request_id: entry}, commands=SessionCommands())
    return entry


@pytest.mark.parametrize('phase', ['capturing', 'paused', 'restoring', 'restored', 'dropping'])
def test_session_cancellation_restores_exact_pages_before_original_finish(phase):
    f = fixture(blocks=6)
    controller, _, groups, pool, scheduler, finished = f
    a, b = admit(f, 'a', 48), admit(f, 'b', 16)
    entry = bind_session(f, a)
    assert controller._capture(a)
    identity = scheduler.connector.commands[0].session_identity
    if phase != 'capturing':
        ack(f, 'capture', 'a')
    if phase in ('restoring', 'restored', 'dropping'):
        controller.before_schedule()
    if phase in ('restored', 'dropping'):
        ack(f, 'restore', 'a')
    if phase == 'dropping':
        controller.before_schedule()
    original_finish = controller.finish_callback
    def finish(req, **kwargs):
        assert len(groups[0].req_to_blocks[req.request_id]) == 3
        assert entry.phase == 'active' and entry.generation == 7
        original_finish(req, **kwargs)
    controller.finish_callback = finish
    a.status = RequestStatus.FINISHED_ABORTED
    assert controller.defer_free(a)
    if phase == 'capturing':
        ack(f, 'capture', 'a')
    if phase in ('capturing', 'paused'):
        controller.before_schedule()
        assert scheduler.connector.commands[0].action == 'restore'
        assert not finished
    if phase not in ('restored', 'dropping'):
        command = ack(f, 'restore', 'a')
        assert command.session_identity == identity
    if phase != 'dropping':
        controller.before_schedule()
    assert not finished
    command = ack(f, 'drop', 'a')
    assert command.session_identity == identity
    controller.before_schedule()
    assert finished == [('a', False)] and pool.get_num_free_blocks() == 4


def test_session_generation_change_rejects_capture_ack_before_page_release():
    f, a, b = pressure_pair()
    entry = bind_session(f, a)
    controller, _, groups, pool, scheduler, _ = f
    controller.before_schedule()
    entry.generation += 1
    with pytest.raises(ValueError, match='changed'):
        ack(f, 'capture', 'a')
    assert pool.get_num_free_blocks() == 0 and len(groups[0].req_to_blocks['a']) == 3


def test_logical_ssd_quota_can_exceed_one_checkpoint_cpu_staging_capacity():
    scheduler = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=8))
    controller = RequestSwapController(scheduler, checkpoint_bytes=65536, checkpoint_cpu_bytes=8192,
        page_bytes=1024, finish_callback=lambda *args, **kwargs: None)
    gate, _, _, _ = make_gate(controller=controller)
    controller.bind(gate)
    assert controller.snapshot()['checkpoint_cpu_capacity_bytes'] == 8192
    assert controller.snapshot()['checkpoint_capacity_bytes'] == 65536
    controller = RequestSwapController(scheduler, checkpoint_bytes=65536, checkpoint_cpu_bytes=8191,
        page_bytes=1024, finish_callback=lambda *args, **kwargs: None)
    gate, _, _, _ = make_gate(controller=controller)
    with pytest.raises(ValueError, match='CPU capacity'):
        controller.bind(gate)


def completion_grace_fixture(*, usable=5, session=False):
    from vllm import SamplingParams
    from vllm.v1.request import Request
    f = fixture(blocks=usable + 1)
    controller, gate, _, _, scheduler, _ = f
    controller.min_resident_steps = 4
    target = admit(f, 'target', 8)
    near = Request('near', [1] * 16, SamplingParams(max_tokens=4), None)
    assert gate.allocate_slots(near, 17) is not None
    near.num_computed_tokens = 17
    near.append_output_token_ids([7, 8])
    near.status = RequestStatus.RUNNING
    scheduler.requests['near'] = near; scheduler.running.append(near)
    peer = admit(f, 'peer', 8)
    if session:
        bind_session(f, target)
    controller.before_schedule()
    assert controller._capture(target); ack(f, 'capture', 'target')
    return f, target, near, peer


def advance_peer_to_fairness(f, peer):
    controller, gate, _, _, scheduler, _ = f
    for _ in range(controller.max_pause_steps):
        assert gate.allocate_slots(peer, 1) is not None
        peer.num_computed_tokens += 1
        scheduler.current_step += 1
        controller.before_schedule()


def test_fairness_completion_grace_uses_real_free_pages_and_avoids_redundant_capture():
    f, target, near, peer = completion_grace_fixture()
    controller, gate, _, pool, scheduler, finished = f
    advance_peer_to_fairness(f, peer)
    assert controller.snapshot()['completion_grace_rounds'] == 1
    assert not scheduler.connector.commands and pool.get_num_free_blocks() == 2
    # A forecast free is never an allocation credit: the paused owner still
    # cannot restore its full first-forward ceiling before the finisher frees.
    assert not controller._restore(controller._states['target'])
    for token in (9, 10):
        assert gate.allocate_slots(near, 1) is not None
        near.num_computed_tokens += 1
        near.append_output_token_ids([token])
        if near.num_output_tokens < near.max_tokens:
            controller.before_schedule()
    near.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert not controller.defer_free(near)
    controller.finish_callback(near)
    assert pool.get_num_free_blocks() == 4
    controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'restore'
    assert scheduler.connector.commands[0].request_id == 'target'
    assert controller.captures == 1 and finished == [('near', False)]


def test_completion_grace_expires_after_bounded_executed_rounds_even_if_finisher_stalls():
    f, target, near, peer = completion_grace_fixture()
    controller, gate, _, _, scheduler, _ = f
    advance_peer_to_fairness(f, peer)
    for _ in range(1000):
        scheduler.current_step += 1
        controller.before_schedule()
    assert not scheduler.connector.commands
    assert controller.snapshot()['completion_grace_rounds'] == 1
    for _ in range(controller.min_resident_steps):
        assert gate.allocate_slots(peer, 1) is not None
        peer.num_computed_tokens += 1
        controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'capture'
    snapshot = controller.snapshot()
    assert snapshot['completion_grace_expirations'] == 1
    assert snapshot['recent_transitions'][-1]['reason'] == 'fair_restore_after_grace'
    assert snapshot['completion_grace_rounds'] == controller.min_resident_steps


@pytest.mark.parametrize('blocked', ['finisher', 'all', 'insufficient_pages'])
def test_completion_grace_never_waits_on_blocked_or_insufficient_potential_frees(blocked):
    f, target, near, peer = completion_grace_fixture(usable=4 if blocked == 'insufficient_pages' else 5)
    controller, gate, _, pool, scheduler, _ = f
    # Age with real peer forwards, inserting pressure only before the fairness decision.
    for _ in range(controller.max_pause_steps - 1):
        assert gate.allocate_slots(peer, 1) is not None
        peer.num_computed_tokens += 1
        controller.before_schedule()
    assert gate.allocate_slots(peer, 1) is not None
    peer.num_computed_tokens += 1
    if blocked != 'insufficient_pages':
        reserve = pool.get_num_free_blocks() + 1
        assert not gate.can_schedule(near, 1, reserved_blocks=reserve)
        if blocked == 'all':
            assert not gate.can_schedule(peer, 1, reserved_blocks=reserve)
    controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'capture'
    assert controller.snapshot()['completion_grace_rounds'] == 0


def test_cancelled_session_restore_bypasses_completion_grace():
    f, target, near, peer = completion_grace_fixture(session=True)
    controller, _, _, _, scheduler, finished = f
    advance_peer_to_fairness(f, peer)
    assert controller.snapshot()['completion_grace_rounds'] == 1
    target.status = RequestStatus.FINISHED_ABORTED
    assert controller.defer_free(target)
    controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'capture'
    assert controller.snapshot()['recent_transitions'][-1]['reason'] == 'cancelled_session_restore'
    assert not finished


def finish_grace_owner_without_native_ack(f, near):
    controller, gate, _, _, scheduler, _ = f
    assert gate.allocate_slots(near, 2) is not None
    near.num_computed_tokens += 2
    near.append_output_token_ids([9, 10])
    near.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.running.remove(near)
    assert not controller.defer_free(near)
    controller.before_schedule()


def test_completion_grace_bridges_delayed_terminal_ack_without_lending_pages():
    f, target, near, peer = completion_grace_fixture()
    controller, gate, _, pool, scheduler, _ = f
    advance_peer_to_fairness(f, peer)
    finish_grace_owner_without_native_ack(f, near)
    assert not scheduler.connector.commands
    assert controller.snapshot()['completion_grace_retirement_rounds'] == 1
    assert controller.snapshot()['recent_transitions'][-1]['retiring_finishers'] == ['near']
    assert pool.get_num_free_blocks() == 2
    assert not controller._restore(controller._states['target'])
    for _ in range(1000):
        scheduler.current_step += 1
        controller.before_schedule()
    assert not scheduler.connector.commands and controller.captures == 1
    assert controller.snapshot()['completion_grace_retirement_rounds'] == 1
    assert controller.snapshot()['completion_grace_retirement_polls'] == 1001
    gate.free(near)  # The original terminal ACK finally releases native pages.
    scheduler.requests.pop('near')
    controller.before_schedule()
    assert scheduler.connector.commands[0].action == 'restore'
    assert scheduler.connector.commands[0].request_id == 'target'
    assert controller.captures == 1


@pytest.mark.parametrize('replacement', ['request_id', 'reservation_owner', 'freed_pages'])
def test_retirement_grace_rejects_recycled_ids_foreign_reservations_and_released_pages(replacement):
    f, target, near, peer = completion_grace_fixture()
    controller, gate, _, _, scheduler, _ = f
    advance_peer_to_fairness(f, peer)
    finish_grace_owner_without_native_ack(f, near)
    assert controller._completion_grace(controller._states['target'])
    if replacement == 'request_id':
        scheduler.requests['near'] = request('near')
    elif replacement == 'reservation_owner':
        gate.reservations['near'] = request('near'), gate.reservations['near'][1]
    else:
        gate.free(near)
    assert not controller._completion_grace(controller._states['target'])
    assert not controller._states['target'].grace_finishers


def test_retirement_grace_keeps_original_deadline_and_eventually_captures_a_live_peer():
    f, target, near, peer = completion_grace_fixture(usable=6)
    controller, gate, _, _, scheduler, _ = f
    assert gate.allocate_slots(peer, 9) is not None
    peer.num_computed_tokens += 9  # Two live peer pages can satisfy the urgent restore.
    advance_peer_to_fairness(f, peer)
    finish_grace_owner_without_native_ack(f, near)
    state = controller._states['target']
    deadline = state.paused_progress + controller.max_pause_steps + controller.min_resident_steps
    while controller._progress_round < deadline:
        assert gate.allocate_slots(peer, 1) is not None
        peer.num_computed_tokens += 1
        controller.before_schedule()
        if controller._progress_round < deadline:
            assert not scheduler.connector.commands
    assert scheduler.connector.commands[0].action == 'capture'
    assert scheduler.connector.commands[0].request_id == 'peer'
    assert controller.snapshot()['completion_grace_expirations'] == 1
    assert controller.snapshot()['recent_transitions'][-1]['reason'] == 'fair_restore_after_grace'


@pytest.mark.parametrize('bypass', ['all_blocked', 'cancelled_target'])
def test_retirement_grace_preserves_pressure_and_cancelled_session_bypasses(bypass):
    f, target, near, peer = completion_grace_fixture(session=True)
    controller, gate, _, pool, scheduler, _ = f
    advance_peer_to_fairness(f, peer)
    finish_grace_owner_without_native_ack(f, near)
    state = controller._states['target']
    assert controller._completion_grace(state)
    if bypass == 'all_blocked':
        assert not gate.can_schedule(peer, 1, reserved_blocks=pool.get_num_free_blocks()+1)
    else:
        target.status = RequestStatus.FINISHED_ABORTED
        assert controller.defer_free(target)
    assert not controller._completion_grace(state)


def test_finished_owner_cannot_first_qualify_for_completion_grace():
    f, target, near, peer = completion_grace_fixture()
    controller, _, _, _, scheduler, _ = f
    near.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.running.remove(near)
    assert not controller._completion_grace(controller._states['target'])
    assert not controller._states['target'].grace_finishers


@pytest.mark.parametrize('usable,mtp', [(4, 0), (6, 0), (8, 0), (9, 4), (12, 4), (18, 4)])
@pytest.mark.parametrize('seed', [3, 17, 43, 109, 2, 11, 71, 127])
@pytest.mark.parametrize('delayed_terminal_ack', [False, True])
def test_mixed_owner_soak_makes_bounded_progress_and_returns_all_native_leases(usable, mtp, seed, delayed_terminal_ack, record_property):
    rng = Random(seed)
    worst_bytes = (4 + (mtp + 1 if mtp else 0)) * 1024 + 4096
    f = fixture(blocks=usable + 1, mtp=mtp, capacity=4 * worst_bytes + 8 * 4096)
    controller, gate, groups, pool, scheduler, _ = f
    gate.max_authorities = 4
    jobs = [request(f'job-{index}') for index in range(8)]
    goals = {job.request_id: rng.choice((32, 48, 60)) for job in jobs}
    joins = {job.request_id: (0 if index < 4 else rng.randrange(3, 18)) for index, job in enumerate(jobs)}
    completed, cancelled, deadlines, last_progress, terminal_deadlines = set(), set(), {}, {}, {}
    trace = deque(maxlen=18)
    max_stall, max_controls, migrations = 0, 0, 0

    def finish(req, *, delay_free_blocks=False):
        rid = req.request_id
        assert rid not in completed
        if rid in gate.reservations:
            gate.free(req)
        scheduler.requests.pop(rid, None)
        if req in scheduler.running:
            scheduler.running.remove(req)
        completed.add(rid)
        trace.append((tick, 'finish', rid, req.num_computed_tokens))

    def cancel(req):
        req.status = RequestStatus.FINISHED_ABORTED
        cancelled.add(req.request_id)
        if req in scheduler.running:
            scheduler.running.remove(req)  # Native abort removes running rows before deferred free.
        if not controller.defer_free(req):
            finish(req)

    controller.finish_callback = finish
    for tick in range(700):
        scheduler.current_step = 100 + tick
        for job in jobs:
            if joins[job.request_id] == tick:
                scheduler.requests[job.request_id] = job
                last_progress[job.request_id] = tick
                trace.append((tick, 'join', job.request_id, 0))
        if scheduler.connector.commands:
            command = scheduler.connector.commands[0]
            deadline = deadlines.setdefault(command.command_id, tick + rng.randrange(1, 5))
            if tick >= deadline:
                ack(f, command.action, command.request_id)
                migrations += 1
                trace.append((tick, command.action + '-ack', command.request_id, command.history_end))
        for rid, deadline in list(terminal_deadlines.items()):
            if tick >= deadline:
                finish(scheduler.requests[rid])
                del terminal_deadlines[rid]
        # Cancel one migrating owner and one ordinary/waiting owner. Cancellation
        # intentionally races delayed ACKs while preserving the native slot lease.
        if tick >= 9 and not cancelled and controller._states:
            state = next(iter(controller._states.values()))
            cancel(state.ticket.request)
        if tick == 21:
            candidates = [r for r in scheduler.requests.values() if not r.is_finished()
                          and r.request_id not in controller._states]
            if candidates:
                cancel(candidates[-1])
        controller.before_schedule()
        gate.begin_step()
        try:
            # Rotate the scheduler order, but keep each complete atomic operation
            # intact. The final forward never crosses the unchanged K+G boundary.
            active = [r for r in scheduler.requests.values() if not r.is_finished()]
            rng.shuffle(active)
            for req in active:
                if controller.is_suspended(req.request_id):
                    continue
                count = min(rng.choice((1, 7, 16, 17)), goals[req.request_id] - req.num_computed_tokens)
                options = dict(num_lookahead_tokens=mtp)
                if not gate.can_schedule(req, count, **options):
                    continue
                assert gate.allocate_slots(req, count, **options) is not None
                req.status = RequestStatus.RUNNING
                if req not in scheduler.running:
                    scheduler.running.append(req)
                req.num_computed_tokens += count
                last_progress[req.request_id] = tick
                trace.append((tick, 'forward', req.request_id, req.num_computed_tokens))
                if req.num_computed_tokens == goals[req.request_id]:
                    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    assert not controller.defer_free(req)
                    if delayed_terminal_ack:
                        scheduler.running.remove(req)
                        terminal_deadlines[req.request_id] = tick + rng.randrange(1, 5)
                    else:
                        finish(req)
        finally:
            gate.end_step()
        native_rows = [block for manager in groups for row in manager.req_to_blocks.values() for block in row]
        assert len(native_rows) == len({block.block_id for block in native_rows})
        assert all(not block.is_null and block.ref_cnt == 1 for block in native_rows)
        assert len(native_rows) + pool.get_num_free_blocks() == usable
        resources = controller.snapshot()['resources']['resources']['checkpoint_bytes']
        assert 0 <= resources['live'] + resources['promised'] <= controller.checkpoint_bytes
        max_controls = max(max_controls, len(scheduler.connector.commands))
        assert len(scheduler.connector.commands) <= 1 and len(scheduler.connector.pending) <= 1
        assert len(gate.reservations) <= 4
        for req in scheduler.requests.values():
            max_stall = max(max_stall, tick - last_progress[req.request_id])
            assert tick - last_progress[req.request_id] < 300, (
                seed, usable, mtp, list(trace), controller.snapshot())
        if len(completed) == len(jobs):
            break
    assert len(completed) == len(jobs), (seed, usable, mtp, list(trace), controller.snapshot())
    controller.before_schedule()
    assert pool.get_num_free_blocks() == usable
    assert not gate.reservations and not controller.has_pending_work()
    assert not scheduler.connector.pending and not scheduler.connector.commands
    assert not terminal_deadlines
    assert not any(manager.req_to_blocks for manager in groups)
    resources = controller.snapshot()['resources']['resources']['checkpoint_bytes']
    assert resources['live'] == resources['promised'] == 0
    assert max_stall < 300 and max_controls <= 1
    if usable == 4 + (mtp + 1 if mtp else 0):
        assert migrations > 0
    record_property('ticks', tick + 1)
    record_property('max_stall', max_stall)
    record_property('migration_acks', migrations)
    record_property('cancelled', len(cancelled))

