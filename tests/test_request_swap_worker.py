from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.request_swap_worker import RequestSwapWorker, SwapCommand


class Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


class Adapter:
    def __init__(self, calls):
        self.calls = calls
        self.capture_event = Event()
        self.restore_event = Event()
        self.fail_capture = False

    def capture(self, request_id, operation_id, block_ids, max_bytes):
        self.calls.append(('capture', request_id, operation_id, block_ids, max_bytes))
        if self.fail_capture:
            raise RuntimeError('capture failed after drained submission')
        return NS(bytes=12, query=self.capture_event.query)

    def restore(self, snapshot, request_id, operation_id, block_ids):
        self.calls.append(('restore', request_id, operation_id, block_ids))
        return self.restore_event

    def release(self, request_id):
        self.calls.append(('release', request_id))


def fixture(capacity=64):
    calls = []
    state = NS(ready=True)
    def drain(request_ids, *, wait):
        calls.append(('drain', tuple(request_ids), wait))
        return state.ready
    runner = NS(device='cpu', req_states=NS(req_id_to_index={'a': 0, 'b': 1},
        num_computed_tokens_np=[160, 128], num_computed_tokens=NS(gpu=torch.tensor([160, 128]))),
        _kvmem_request_generations={'a': 2, 'b': 3},
        _kvmem_capture_runtime=NS(drain_requests=drain,
            authorities={'a': NS(history_end=160), 'b': NS(history_end=128)}),
        block_tables=NS(append_block_ids=lambda index, rows, overwrite: calls.append(('table', index, rows, overwrite)),
                        apply_staged_writes=lambda: calls.append(('apply',))))
    adapter = Adapter(calls)
    worker = RequestSwapWorker(runner, capacity, adapter=adapter)
    return worker, adapter, state, calls


def command(action='capture', **changes):
    value = SwapCommand('a', {'capture': 10, 'restore': 11, 'drop': 12}[action],
                        1, action, 160, ((1, 2), (3,)), 32 if action == 'capture' else 0)
    return replace(value, **changes)


def usage(worker):
    return worker.ledger.snapshot()['resources']['checkpoint_bytes']


@pytest.mark.parametrize('mutation', ['scheduled', 'finished', 'preempted', 'cached', 'table', 'new'])
@pytest.mark.parametrize('before_dispatch', [False, True])
def test_suspended_slot_is_protected_before_any_public_update(mutation, before_dispatch):
    worker, _, _, _ = fixture()
    if not before_dispatch:
        worker.execute(command())
    output = NS(num_scheduled_tokens={'b': 1}, finished_req_ids=set(),
                preempted_req_ids=set(), scheduled_cached_reqs=NS(req_ids=['b']),
                scheduled_new_reqs=[], kv_connector_metadata=NS(
                    commands=[command()] if before_dispatch else [], block_table_updates={}))
    worker.validate_public_step(output)
    if mutation == 'scheduled': output.num_scheduled_tokens['a'] = 1
    if mutation == 'finished': output.finished_req_ids.add('a')
    if mutation == 'preempted': output.preempted_req_ids.add('a')
    if mutation == 'cached': output.scheduled_cached_reqs.req_ids.append('a')
    if mutation == 'table': output.kv_connector_metadata.block_table_updates['a'] = ((1,),)
    if mutation == 'new': output.scheduled_new_reqs.append(NS(req_id='a'))
    with pytest.raises(RuntimeError, match='drop acknowledgement'):
        worker.validate_public_step(output)


def test_capture_restore_and_explicit_drop_retain_slot_and_fence_byte_credit():
    worker, adapter, state, calls = fixture()
    captured = worker.execute(command())
    assert usage(worker)['promised'] == 32
    assert not captured.query()
    assert usage(worker)['live'] == 12 and usage(worker)['promised'] == 20
    adapter.capture_event.done = True
    assert captured.query() and captured.query()
    assert usage(worker)['live'] == 12 and usage(worker)['promised'] == 0
    restored = worker.execute(command('restore', block_ids=((5, 4), (6,))))
    assert calls[-3:] == [('table', 0, ((5, 4), (6,)), True), ('apply',),
                         ('restore', 'a', 1, ((5, 4), (6,)))]
    assert not restored.query() and usage(worker)['live'] == 12
    with pytest.raises(ValueError, match='completed'): worker.execute(command('drop'))
    adapter.restore_event.done = True
    assert restored.query() and usage(worker)['live'] == 12
    assert worker.runner.req_states.req_id_to_index == {'a': 0, 'b': 1}
    dropped = worker.execute(command('drop'))
    assert usage(worker)['live'] == 12
    assert dropped.query() and dropped.query()
    assert usage(worker)['live'] == 0 and not worker.checkpoints
    assert worker.capture_count == worker.restore_count == worker.drop_count == 1


def test_unconfirmed_capture_waits_without_allocating_or_synchronizing(monkeypatch):
    worker, adapter, state, calls = fixture()
    worker.runner.device = 'cuda:0'
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(('sync', device)))
    state.ready = False
    job = worker.execute(command())
    assert not job.query() and calls == [('drain', ('a',), True)]
    assert usage(worker)['live'] == 0 and usage(worker)['promised'] == 32
    state.ready = True
    assert not job.query()
    assert calls[1:3] == [('drain', ('a',), True), ('sync', 'cuda:0')]
    assert calls[3][0] == 'capture'


def test_cancelled_paused_request_can_drop_without_restoring_or_accessing_old_slot():
    worker, adapter, state, calls = fixture()
    job = worker.execute(command())
    adapter.capture_event.done = True
    assert job.query()
    del worker.runner.req_states.req_id_to_index['a']
    del worker.runner._kvmem_request_generations['a']
    assert worker.execute(command('drop')).query()
    assert not any(call[0] in ('restore', 'table') for call in calls)
    assert usage(worker)['live'] == 0


def test_capacity_is_reserved_before_any_capture_and_returns_only_after_drop():
    worker, adapter, state, calls = fixture(32)
    job = worker.execute(command())
    peer = command(request_id='b', history_end=128, command_id=20, operation_id=2)
    with pytest.raises(BufferError): worker.execute(peer)
    assert calls == []
    adapter.capture_event.done = True
    assert job.query()
    with pytest.raises(BufferError): worker.execute(peer)
    assert worker.execute(command('drop')).query()
    assert worker.execute(peer).query()


def test_capture_failure_drains_and_returns_dispatched_credit():
    worker, adapter, state, calls = fixture()
    adapter.fail_capture = True
    job = worker.execute(command())
    with pytest.raises(RuntimeError, match='capture failed'): job.query()
    assert usage(worker)['live'] == usage(worker)['promised'] == 0
    assert not worker.checkpoints


def test_capture_reconciles_stale_native_cpu_hint_only_after_confirmed_gpu_boundary():
    worker, adapter, state, calls = fixture()
    worker.runner.req_states.num_computed_tokens_np[0] = 155
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    assert worker.runner.req_states.num_computed_tokens_np[0] == 160


@pytest.mark.parametrize('witness', ['gpu', 'authority'])
def test_capture_refuses_unconfirmed_history_before_copy(witness):
    worker, adapter, state, calls = fixture()
    if witness == 'gpu': worker.runner.req_states.num_computed_tokens.gpu[0] = 159
    else: worker.runner._kvmem_capture_runtime.authorities['a'].history_end = 159
    job = worker.execute(command())
    with pytest.raises(ValueError, match='confirmed authority'): job.query()
    assert not any(call[0] == 'capture' for call in calls)


@pytest.mark.parametrize('change', ['generation', 'history', 'slot', 'operation', 'transport'])
def test_restore_refuses_stale_identity_before_table_write(change):
    worker, adapter, state, calls = fixture()
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    cmd = command('restore')
    if change == 'generation': worker.runner._kvmem_request_generations['a'] += 1
    if change == 'history': worker.runner.req_states.num_computed_tokens_np[0] += 1
    if change == 'slot': worker.runner.req_states.req_id_to_index['a'] = 1
    if change == 'operation': cmd = replace(cmd, operation_id=2)
    if change == 'transport': cmd = replace(cmd, command_id=10)
    calls.clear()
    with pytest.raises(ValueError): worker.execute(cmd)
    assert calls == [] and usage(worker)['live'] == 12


def test_restore_cannot_be_replayed_after_completion():
    worker, adapter, _, _ = fixture()
    adapter.capture_event.done = adapter.restore_event.done = True
    assert worker.execute(command()).query()
    assert worker.execute(command('restore')).query()
    with pytest.raises(ValueError, match='already restored'):
        worker.execute(command('restore', command_id=14))
