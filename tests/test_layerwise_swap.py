from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.layerwise_swap import capture, ready, validate_modes
from test_request_swap_worker import Event, command, fixture as ordinary_fixture, usage


class History:
    def __init__(self):
        self.values = {0: [2., 3.], 128: [1., 2.]}
        self.end, self.samples, self.chunks = 160, 2, 1
        self._buffered_state = object()

    def snapshot(self):
        return self.values, self.end, self.samples, self.chunks


def fixture():
    worker, adapter, state, calls = ordinary_fixture()
    runner = worker.runner
    runner._kvmem_request_swap_options = NS(request_swap_layerwise=True)
    query = NS(mode='sequential', optimized=True, native_batches=True,
               names=('l0', 'l1'), pending={}, active_owners={}, active=None,
               finished=set(), no_probe=False, mtp_selection='global')
    pressure = NS(query=query, single_assembly=True, names=query.names,
                  rolling_decode=False,
                  maps={'a': (160, [[(0, 1, 0, 32, 0)], [(4, 1, 128, 32, 0)]])},
                  histories={'a': [History(), History()]}, decode_histories={})
    query.pressure = pressure
    runner._kvmem_layerwise_query = query
    runner._kvmem_layerwise_pressure = pressure
    runner._kvmem_async_refresh = NS(jobs={}, deferred={})
    runtime = runner._kvmem_capture_runtime
    runtime.publication_ready = lambda ids: state.ready
    runtime.authorities['a'].memory = NS(phase=NS(name='READY'), _submitted=False,
        _fence=None, _plan=None, _store=NS(working_records=lambda: ((0, 1, 0, 32, 0),)))
    runner._kvmem_window_adapter = NS(offsets={'a': 128}, history_ends={'a': 160},
                                    generations={'a': 1}, private_capacities={})
    return worker, adapter, state, calls


def test_opt_in_requires_actual_controller_and_plain_swaps_are_unchanged():
    worker, _, _, _ = ordinary_fixture()
    assert ready(worker.runner, 'a') and capture(worker.runner, 'a') is None
    worker.runner._kvmem_request_swap_options = NS(request_swap_layerwise=True)
    with pytest.raises(RuntimeError, match='qualified live controllers'):
        worker.execute(command()).query()
    assert not worker.capture_count


@pytest.mark.parametrize('mode', ['original_rope', 'no_probe', 'rolling_decode', 'mtp_selection'])
def test_direct_worker_modes_cannot_bypass_launch_guards(mode, monkeypatch):
    worker, _, _, _ = fixture()
    runner = worker.runner
    query, pressure = runner._kvmem_layerwise_query, runner._kvmem_layerwise_pressure
    monkeypatch.delenv('KVMEM_LAYERWISE_ORIGINAL_ROPE', raising=False)
    validate_modes(query, pressure)
    if mode == 'original_rope':
        monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE', '1')
    elif mode == 'no_probe': query.no_probe = True
    elif mode == 'rolling_decode': pressure.rolling_decode = True
    else: query.mtp_selection = 'layer0'
    with pytest.raises(ValueError, match='Layerwise request swap requires compact positions'):
        validate_modes(query, pressure)
    with pytest.raises(ValueError, match='Layerwise request swap requires compact positions'):
        ready(runner, 'a')


@pytest.mark.parametrize('busy', ['future', 'pending', 'active', 'serial_active', 'publication',
                                 'background', 'transaction', 'borrowed', 'fused_fence'])
def test_capture_waits_for_owner_quiescence_without_copy_or_gpu_sync(busy, monkeypatch):
    worker, adapter, state, calls = fixture()
    runner = worker.runner
    query = runner._kvmem_layerwise_query
    authority = runner._kvmem_capture_runtime.authorities['a']
    if busy == 'future':
        # CPU producer completion is insufficient: job.done covers CUDA/commit.
        runner._kvmem_async_refresh.jobs['a'] = NS(done=False, future=NS(done=lambda: True))
    elif busy == 'pending': query.pending['a'] = {}
    elif busy == 'active': query.active_owners['a'] = {}
    elif busy == 'serial_active': query.active = dict(request_id='a')
    elif busy == 'publication': state.ready = False
    elif busy == 'background': authority._refresh_background = True
    elif busy == 'transaction': authority.memory._fence = Event(False)
    elif busy == 'borrowed':
        parent = NS(_runner_binding=(runner, 'a'))
        runner._kvmem_pressure_assembly_staging = NS(owner=NS(_validated_parent=parent))
    else:
        runner._kvmem_pressure_assembly_staging = NS(owner=None,
            fused_binding=(id(authority), 'a'), fused_event=Event(False))
    monkeypatch.setattr('torch.cuda.synchronize', lambda *_: pytest.fail('Must not wait on GPU'))
    done = worker.execute(command())
    assert not done.query()
    assert not any(call[0] in ('capture', 'table', 'restore') for call in calls)
    assert usage(worker)['promised'] == 32 and usage(worker)['live'] == 0
    assert worker.capture_waits == 1


def test_ready_peer_work_does_not_globally_block_capture_and_state_is_retained():
    worker, adapter, state, calls = fixture()
    runner = worker.runner
    query, pressure = runner._kvmem_layerwise_query, runner._kvmem_layerwise_pressure
    runner._kvmem_async_refresh.jobs['b'] = NS(done=False)
    query.active_owners['b'] = {}
    runner._kvmem_pressure_assembly_staging = NS(owner=NS(_runner_binding=(runner, 'b')))
    frame, histories = pressure.maps['a'], pressure.histories['a']
    captured = worker.execute(command())
    assert not captured.query()
    assert worker.checkpoints['a'].layerwise is not None
    assert usage(worker)['live'] == 12
    adapter.capture_event.done = True
    assert captured.query()
    # Unrelated owners may publish histories/windows while a is suspended.
    pressure.maps['b'] = (200, ())
    runner._kvmem_window_adapter.generations['b'] = 99
    restored = worker.execute(command('restore', block_ids=((8, 9), (10,))))
    assert not restored.query()
    assert pressure.maps['a'] is frame and pressure.histories['a'] is histories
    adapter.restore_event.done = True
    assert restored.query()
    assert worker.execute(command('drop')).query()
    assert pressure.maps['a'] is frame and pressure.histories['a'] is histories
    assert usage(worker)['live'] == 0


@pytest.mark.parametrize('change', ['map', 'history_value', 'history_replaced', 'buffer_replaced',
                                   'window_version', 'window_offset', 'finished', 'controller',
                                   'authority', 'disabled'])
def test_restore_refuses_logical_drift_before_any_page_write(change):
    worker, adapter, _, calls = fixture()
    runner = worker.runner
    pressure = runner._kvmem_layerwise_pressure
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    if change == 'map': pressure.maps['a'][1][1][0] = (5, 1, 128, 32, 0)
    elif change == 'history_value': pressure.histories['a'][0].values[0][0] += 1
    elif change == 'history_replaced': pressure.histories['a'][0] = History()
    elif change == 'buffer_replaced': pressure.histories['a'][0]._buffered_state = object()
    elif change == 'window_version': runner._kvmem_window_adapter.generations['a'] += 1
    elif change == 'window_offset': runner._kvmem_window_adapter.offsets['a'] += 1
    elif change == 'finished': runner._kvmem_layerwise_query.finished.add('a')
    elif change == 'controller': runner._kvmem_layerwise_query = None
    elif change == 'authority':
        old = runner._kvmem_capture_runtime.authorities['a']
        runner._kvmem_capture_runtime.authorities['a'] = NS(**vars(old))
    else: runner._kvmem_request_swap_options.request_swap_layerwise = False
    calls.clear()
    with pytest.raises(RuntimeError, match='Layerwise swap'):
        worker.execute(command('restore'))
    assert calls == []
    assert usage(worker)['live'] == 12 and worker.checkpoints['a'].phase == 'paused'


@pytest.mark.parametrize('phase', ['capture', 'restore'])
def test_state_mutation_while_dma_unfinished_never_acknowledges_or_frees(phase):
    worker, adapter, _, _ = fixture()
    completion = worker.execute(command())
    assert not completion.query()
    if phase == 'restore':
        adapter.capture_event.done = True
        assert completion.query()
        completion = worker.execute(command('restore'))
        assert not completion.query()
    worker.runner._kvmem_window_adapter.generations['a'] += 1
    with pytest.raises(RuntimeError, match='window generation changed'):
        completion.query()
    assert usage(worker)['live'] == 12 and worker.checkpoints
    assert worker.restore_count == 0


def test_cancel_drop_releases_finished_snapshot_without_reviving_retired_layer_state():
    worker, adapter, _, calls = fixture()
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    runner = worker.runner
    del runner.req_states.req_id_to_index['a']
    del runner._kvmem_request_generations['a']
    del runner._kvmem_layerwise_pressure.maps['a']
    del runner._kvmem_layerwise_pressure.histories['a']
    del runner._kvmem_capture_runtime.authorities['a']
    assert worker.execute(command('drop')).query()
    assert usage(worker)['live'] == 0 and not worker.checkpoints
    assert not any(call[0] in ('restore', 'table') for call in calls)


def test_finished_query_and_decode_histories_survive_without_reselection():
    worker, adapter, _, _ = fixture()
    runner = worker.runner
    runner._kvmem_layerwise_query.finished.add('a')
    decode = [NS(mass={0: 2.}, end=160, samples=1, chunks=1) for _ in range(2)]
    runner._kvmem_layerwise_pressure.decode_histories['a'] = decode
    witness = capture(runner, 'a')
    witness.validate(runner)
    decode[1].mass[0] += 1
    witness.validate(runner)  # Waiting polls inspect identity/counters only.
    with pytest.raises(RuntimeError, match='logical state'):
        witness.validate(runner, deep=True)


def test_waiting_polls_do_not_materialize_score_snapshots():
    worker, adapter, _, _ = fixture()
    rows = worker.runner._kvmem_layerwise_pressure.histories['a']
    reads = []
    for history in rows:
        original = history.snapshot
        history.snapshot = lambda original=original: (reads.append(1), original())[1]
    completion = worker.execute(command())
    assert not completion.query()
    assert len(reads) == 2
    for _ in range(10):
        assert not completion.query()
    assert len(reads) == 2
    adapter.capture_event.done = True
    assert completion.query()
    assert len(reads) == 4
    row = worker.snapshot()['requests']['a']
    assert row['layerwise_deep_checks'] == 1
    assert 0 <= row['layerwise_deep_check_seconds'] <= row['layerwise_check_seconds']
    assert worker.execute(command('drop')).query()
    assert not worker.snapshot()['requests']
    assert worker.snapshot()['layerwise_deep_checks'] == 1
    assert worker.snapshot()['layerwise_capture_seconds'] > 0


def test_capture_drains_final_publication_before_requiring_ready_memory():
    worker, adapter, state, calls = fixture()
    runner = worker.runner
    memory = runner._kvmem_capture_runtime.authorities['a'].memory
    memory.phase.name = 'APPENDING'
    memory._submitted, memory._fence = True, Event(False)
    state.ready = False
    def drain(request_ids, *, wait):
        calls.append(('drain', tuple(request_ids), wait))
        memory.phase.name = 'READY'
        memory._submitted, memory._fence = False, None
        state.ready = True
        return True
    runner._kvmem_capture_runtime.drain_requests = drain
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    assert calls[0] == ('drain', ('a',), True)
    assert worker.capture_count == 1


@pytest.mark.parametrize('initial_adapter', [False, True])
def test_peer_first_window_creation_does_not_invalidate_untouched_owner(initial_adapter):
    worker, adapter, _, _ = fixture()
    runner = worker.runner
    runner._kvmem_layerwise_pressure.maps.clear()
    runner._kvmem_layerwise_pressure.histories.clear()
    runner._kvmem_window_adapter = (NS(offsets={}, history_ends={}, generations={},
                                       private_capacities={}) if initial_adapter else None)
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    runner._kvmem_window_adapter = NS(offsets={'b': 128}, history_ends={'b': 160},
                                     generations={'b': 1}, private_capacities={})
    completion = worker.execute(command('restore', block_ids=((8, 9), (10,))))
    assert not completion.query()
    adapter.restore_event.done = True
    assert completion.query()
    assert worker.execute(command('drop')).query()


@pytest.mark.parametrize('field', ['offsets', 'history_ends', 'generations', 'private_capacities'])
@pytest.mark.parametrize('change', ['add', 'remove', 'modify'])
def test_normalized_window_still_rejects_owner_entry_changes(field, change):
    worker, _, _, _ = fixture()
    runner = worker.runner
    window = NS(offsets={}, history_ends={}, generations={}, private_capacities={})
    runner._kvmem_window_adapter = window
    if change != 'add':
        getattr(window, field)['a'] = 1
    witness = capture(runner, 'a')
    if change == 'remove':
        del getattr(window, field)['a']
    else:
        getattr(window, field)['a'] = 2
    with pytest.raises(RuntimeError, match='window generation changed'):
        witness.validate(runner)
