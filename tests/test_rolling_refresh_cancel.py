"""Cancellation fences for accepted-history, zero-replay pressure refreshes."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip('vllm')
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_kvmem.async_refresh import AsyncRefresh
from vllm_kvmem.connector import ControlMetadata
from vllm_kvmem.window_batch import WindowBatchAdapter
from vllm_kvmem.window_manager import RefreshPending
from test_async_refresh import manager
from test_scheduler_transactions import setup_scheduler
from test_session_worker import worker_fixture, wait_retirement
from test_authority import payload, drain
from test_core import Event


class Fence(Event):
    def record(self):
        self.recorded = True


def rolling_transaction():
    scheduler, request, pools = setup_scheduler()
    request.num_computed_tokens = 48
    request.num_prompt_tokens = 32
    scheduler.prepared_refreshes = {}
    scheduler.begin_refresh(request, 32, replay_tokens=0, selection='pressure',
                            asynchronous=True)
    command = scheduler.connector.commands[0]
    runtime = manager()
    runtime.options.rolling_active_window = True
    runtime.runner.req_states = NS(req_id_to_index={'a': 0, 'b': 1})
    runtime.runner._kvmem_window_adapter = WindowBatchAdapter((0, 1), 64)
    runtime.runner._kvmem_window_adapter.commit('a', 48, 48)
    runtime.runner._kvmem_sessions = None
    runtime.report = lambda *args: None
    return scheduler, request, pools, command, runtime


def cancel_output(command=None):
    return replace(SchedulerOutput.make_empty(), finished_req_ids={'a'},
        num_scheduled_tokens={'b': 1}, total_num_scheduled_tokens=1,
        kv_connector_metadata=ControlMetadata(
            commands=[command] if command is not None else [], cancelled={'a'},
            terminal_prefixes={'a': 'accepted-prefix'}))


def assert_native_leased(scheduler, runtime):
    assert not scheduler._request_blocks_can_be_freed(scheduler.requests['a'])
    assert runtime.runner.req_states.req_id_to_index == {'a': 0, 'b': 1}
    for owner in scheduler.window_managers.values():
        assert len(owner.req_to_blocks['a']) == 4
        assert len(owner.req_to_blocks['b']) == 2
        with pytest.raises(RefreshPending):
            owner.free('a')


def acknowledge_cancel(scheduler, request, pools, command):
    request.is_finished = lambda: True
    scheduler._complete_refresh('a', command.command_id)
    # ACK releases the transaction, not the original physical pages. They
    # remain owned until the ordinary native finished-request free.
    assert [pool.get_num_free_blocks() for pool in pools] == [10, 10]
    assert not scheduler.table_updates and not scheduler.prepared_refreshes
    assert not scheduler.transactions and request.next_decode_eligible_step == 7
    for owner in scheduler.window_managers.values():
        owner.free('a')
        assert len(owner.req_to_blocks['b']) == 2
    assert [pool.get_num_free_blocks() for pool in pools] == [14, 14]


@pytest.mark.parametrize('before_dispatch', [True, False])
def test_cancel_before_submission_holds_index_and_pages_until_own_fence(monkeypatch, before_dispatch):
    scheduler, request, pools, command, runtime = rolling_transaction()
    fence = Fence()
    monkeypatch.setattr(torch.cuda, 'Event', lambda: fence)
    runtime.executor = NS(submit=lambda *args: pytest.fail('Cancelled job submitted assembly'))
    if not before_dispatch:
        runtime.enqueue(command)
    held = runtime.defer_terminals(cancel_output(command if before_dispatch else None))
    if before_dispatch:
        runtime.enqueue(command)
    assert not held.finished_req_ids
    assert held.num_scheduled_tokens == {'b': 1}
    assert held.total_num_scheduled_tokens == 1
    runtime.pump(('b',))
    job = runtime.jobs['a']
    assert fence.recorded and not job.done and job.cancelled and job.peers == {'b'}
    assert_native_leased(scheduler, runtime)
    assert runtime.runner._kvmem_window_adapter.offsets['a'] == 0
    fence.done = True
    runtime.pump(('b',))
    assert job.done and job.selected_tokens is None
    released = runtime.defer_terminals(replace(SchedulerOutput.make_empty(),
        kv_connector_metadata=ControlMetadata()))
    assert released.finished_req_ids == {'a'}
    assert released.kv_connector_metadata.terminal_prefixes == {'a': 'accepted-prefix'}
    acknowledge_cancel(scheduler, request, pools, command)


def assembled_job(runtime, command, fence):
    job = runtime.enqueue(command)
    calls = []
    memory = NS(attach_completion=lambda value: calls.append('attach'),
                commit_selection=lambda: calls.append('commit') or True)
    job.authority = NS(memory=memory, _refresh_background=True)
    assembly = NS(event=fence, uploaded_bytes=32)
    assembler = NS(close=lambda: calls.append('close') or True)
    result = (assembler, assembly, NS(total_window_tokens=32), {}, 0.001)
    job.future = NS(done=lambda: True, result=lambda: result)
    job.stage = 'assembly'
    return job, calls


def test_cancel_pending_assembly_waits_only_owner_and_never_publishes_new_offset():
    scheduler, request, pools, command, runtime = rolling_transaction()
    fence = Fence()
    job, calls = assembled_job(runtime, command, fence)
    held = runtime.defer_terminals(cancel_output())
    assert not held.finished_req_ids and held.num_scheduled_tokens == {'b': 1}
    runtime.pump(('b',))
    assert job.peers == {'b'} and not job.done and not calls
    assert job.authority._refresh_background
    assert_native_leased(scheduler, runtime)
    fence.done = True
    runtime.pump(('b',))
    assert job.done and job.cancelled and calls == ['attach', 'commit', 'close']
    assert not job.authority._refresh_background
    # Completed CPU assembly can commit its frame for orderly retirement, but
    # cancellation must not advertise a new running GPU view.
    assert runtime.runner._kvmem_window_adapter.offsets['a'] == 0
    released = runtime.defer_terminals(replace(SchedulerOutput.make_empty(),
        kv_connector_metadata=ControlMetadata()))
    assert released.finished_req_ids == {'a'}
    acknowledge_cancel(scheduler, request, pools, command)


def test_cancel_after_worker_commit_before_scheduler_ack_retains_original_allocation():
    scheduler, request, pools, command, runtime = rolling_transaction()
    job, calls = assembled_job(runtime, command, Fence(True))
    runtime.pump(('b',))
    assert job.done and calls == ['attach', 'commit', 'close']
    assert runtime.runner._kvmem_window_adapter.offsets['a'] == 16
    assert_native_leased(scheduler, runtime)
    # The worker's fence permits index retirement, while the scheduler still
    # owns the original page table until it consumes the acknowledgement.
    released = runtime.defer_terminals(cancel_output())
    assert released.finished_req_ids == {'a'}
    assert released.num_scheduled_tokens == {'b': 1}
    assert not runtime.deferred
    assert_native_leased(scheduler, runtime)
    acknowledge_cancel(scheduler, request, pools, command)


@pytest.mark.parametrize('new_suffix', [False, True])
def test_cancelled_terminal_checkpoint_never_claims_hot_pages_and_cold_restore_is_exact(tmp_path, new_suffix):
    worker, binding, authority, archive, (conv, ssm, cache) = worker_fixture(tmp_path)
    adapter = worker.model_adapter
    # Use the native-MTP checkpoint contract: its KV is in the historical
    # authority; model snapshots retain GDN state and acceptance separately.
    adapter.contract = NS(method='mtp', validate_native_groups=lambda _: ())
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    worker.runtime.last_steps['a'] = [{'history_begin': 0, 'accepted': 8, 'scheduled': 8}]
    checkpoint = worker.pool.capture(binding.namespace, 'a', authority,
        list(range(20)), adapter, replace=True, terminal=True)
    before = conv.clone(), ssm.clone(), cache.clone()
    old_records = checkpoint.window.records
    if new_suffix:
        assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
        drain(authority)
        records = list(authority.records.values())
        plan = authority.memory.select({record.handle.slot: float(i) for i, record in enumerate(records)})
        authority.memory.attach_completion(Event(True))
        assert authority.memory.commit_selection()
        assert tuple(authority.memory._store.working_records()) != old_records
        del plan
    original_end = authority.history_end
    assert worker.detach('a', authority, {'terminal_prefix': {'status': 'FINISHED_ABORTED'}})
    del worker.runtime.authorities['a']
    wait_retirement(worker, 'a')
    notice, = worker.notices()
    assert notice.kind == 'retired' and not notice.resident_ready
    assert notice.history == authority.history_end == checkpoint.position == 8
    assert worker.runtime.completed[-1]['history_end'] == original_end
    assert checkpoint.rewind_started is new_suffix
    # Original immutable generations and the exact saved model prefix survive
    # even if the cancelled assembly already changed the live selected frame.
    for record in authority.records.values():
        raw, values = authority.read_record(record.handle)
        expected = payload(record.history_begin, record.count)
        assert torch.equal(raw, expected[0]) and torch.equal(values, expected[1])
    assert worker.pool.claim(binding.namespace, 'new', list(range(20))) is checkpoint
    plan = worker.pool.prepare_claim_window(checkpoint)
    assert all(not row.skip and not row.working_k_resident for row in plan.remaps)
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    conv[1:3].fill_(-11)
    ssm[1:3].fill_(-11)
    assert adapter.restore(checkpoint.model, 'new') is None
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(conv[3:5], before[0][3:5]) and torch.equal(ssm[3:5], before[1][3:5])
    assert torch.equal(cache, before[2])
    assert worker.pool.finish_claim(checkpoint, Event(True)) is authority
    del plan
    assert worker.close() and authority.close() and archive.close()
