"""Swap control uses the native connector even when no model token is scheduled."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import kv_connector as native

from vllm_kvmem.serving.connector import ControlMetadata, RefreshCommand
from vllm_kvmem.serving.request_swap_worker import SwapCommand
from test_connector import Event, pair, poll, send


def command(action='capture', command_id=1):
    return SwapCommand('a', command_id, 1, action, 32, ((1,),),
                       checkpoint_bytes=128 if action == 'capture' else 0)


class Worker:
    def __init__(self):
        self.commands = []
        self.checkpoints = {}
        self.event = Event()

    def execute(self, cmd):
        self.commands.append(cmd)
        if cmd.action == 'capture':
            self.checkpoints[cmd.request_id] = object()
        def query():
            if self.event.done and cmd.action == 'drop':
                del self.checkpoints[cmd.request_id]
            return self.event.done
        return NS(query=query)


def test_native_idle_control_runs_capture_restore_drop_without_refresh_runtime(monkeypatch):
    scheduler, connector = pair()
    worker = connector.swap_worker = Worker()
    committed = []
    scheduler.scheduler_completion = lambda *value: committed.append(value)
    wrapper = native.ActiveKVConnector.__new__(native.ActiveKVConnector)
    wrapper.kv_connector, wrapper._disabled, wrapper._pending_load_kwargs = connector, False, None
    monkeypatch.setattr(native, 'is_forward_context_available', lambda: True)
    monkeypatch.setattr(native, 'get_forward_context', lambda: None)
    def idle():
        out = replace(SchedulerOutput.make_empty(),
                      kv_connector_metadata=scheduler.build_connector_meta(None))
        result = wrapper.no_forward(out).kv_connector_output
        if result is not None:
            scheduler.update_connector_output(result)
    for number, action in enumerate(('capture', 'restore', 'drop'), 1):
        worker.event.done = False
        scheduler.queue(command(action, number))
        idle()
        assert committed == [('a', n) for n in range(1, number)]
        assert scheduler.has_pending_push_work()
        worker.event.done = True
        idle()
        assert committed[-1] == ('a', number)
        assert bool(worker.checkpoints) == (action != 'drop')
    assert [cmd.action for cmd in worker.commands] == ['capture', 'restore', 'drop']
    assert not scheduler.has_pending_push_work()


def test_cancellation_ack_waits_for_explicit_checkpoint_drop():
    scheduler, connector = pair()
    worker = connector.swap_worker = Worker()
    scheduler.scheduler_completion = lambda *_: None
    scheduler.queue(command())
    send(scheduler, connector)
    assert scheduler.request_finished_all_groups(NS(request_id='a'), ()) == (True, None)
    send(scheduler, connector)
    worker.event.done = True
    assert poll(scheduler, connector).finished_sending is None
    assert not scheduler.pending and scheduler.has_pending_push_work()
    assert poll(scheduler, connector).finished_sending is None
    scheduler.queue(command('drop', 2))
    send(scheduler, connector)
    assert poll(scheduler, connector).finished_sending == {'a'}
    assert not scheduler.has_pending_push_work()


def test_swap_handler_is_explicit_and_normal_refresh_still_uses_runtime():
    scheduler, connector = pair()
    scheduler.queue(command())
    with pytest.raises(RuntimeError, match='without its worker'):
        send(scheduler, connector)
    scheduler, connector = pair()
    worker = connector.swap_worker = Worker()
    refresh = Event()
    connector.runtime = lambda _: refresh
    scheduler.queue(RefreshCommand('b', 2, ()))
    send(scheduler, connector)
    assert not worker.commands and connector.worker_jobs['b'][1] is refresh


def test_incremental_sessions_without_owned_ledger_rejected_before_admission_side_effect():
    scheduler, _ = pair()
    scheduler.gpu_memory_admission = 'incremental'
    scheduler.history_admission = NS(allows=lambda _: pytest.fail('history credit changed'), matched_admission=False)
    scheduler.sessions = NS(lookup=lambda *_: pytest.fail('session lookup changed'))
    assert scheduler.get_num_new_matched_tokens(NS(request_id='a', session_id='retained'), 0) == (None, False)
    assert 'owned session ledger' in scheduler.capacity_rejections['a']
    scheduler.gpu_memory_admission = 'static'
    scheduler.history_admission = None
    scheduler.sessions = NS(lookup=lambda *_: 24)
    assert scheduler.get_num_new_matched_tokens(NS(request_id='b', session_id='retained'), 0) == (24, False)


def test_paused_gap_keeps_engine_control_loop_live():
    scheduler, _ = pair()
    scheduler.request_swap_pending = lambda: True
    assert scheduler.has_pending_push_work()
    scheduler.request_swap_pending = lambda: False
    assert not scheduler.has_pending_push_work()


def test_async_refresh_never_interprets_swap_as_refresh():
    from test_async_refresh import manager
    refresh = manager()
    out = replace(SchedulerOutput.make_empty(),
                  kv_connector_metadata=ControlMetadata(commands=[command()], cancelled={'a'}))
    result = refresh.defer_terminals(out)
    assert result.kv_connector_metadata.commands == [command()]
    assert not refresh.jobs and not refresh.deferred


def test_native_cancel_keeps_slot_admission_until_drop_and_deferred_free():
    from types import MethodType
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus
    from vllm_kvmem.serving.scheduler import KVMemScheduler
    from test_request_swap_control import fixture, admit, ack
    from test_incremental_capacity import request

    f = fixture(blocks=9, capacity=131072)
    controller, gate, _, pool, scheduler, finished = f
    gate.max_authorities = 2
    a, b = admit(f, 'a', 16), admit(f, 'b', 16)
    assert controller._capture(a)
    ack(f, 'capture', 'a')
    # Use native external-cancel processing: it removes a from running BEFORE
    # our _free_request can defer recycling its retained worker slot.
    scheduler.swap_controller = controller
    scheduler._free_request = MethodType(KVMemScheduler._free_request, scheduler)
    released = set()
    def actual_free(req, *, delay_free_blocks=False):
        assert req not in scheduler.running
        gate.free(req)
        del scheduler.requests[req.request_id]
        released.add(req.request_id)
    controller.finish_callback = actual_free
    assert Scheduler.finish_requests(scheduler, ['a'], RequestStatus.FINISHED_ABORTED) == [a]
    assert scheduler.running == [b] and a.is_finished()
    assert not released and 'a' in gate.reservations
    fresh = request('replacement')
    physical_free = pool.get_num_free_blocks()
    assert physical_free >= 1  # The blocker is the live slot, not GPU capacity.
    assert not gate.can_schedule(fresh, 16)
    assert gate.allocate_slots(fresh, 16) is None
    assert pool.get_num_free_blocks() == physical_free
    controller.before_schedule()
    ack(f, 'drop', 'a', clear_pending=False)
    controller.before_schedule()
    assert not released and not gate.can_schedule(fresh, 16)
    del scheduler.connector.pending['a']
    controller.before_schedule()
    assert released == {'a'} and 'a' not in gate.reservations
    assert gate.can_schedule(fresh, 16)
    assert gate.allocate_slots(fresh, 16) is not None
