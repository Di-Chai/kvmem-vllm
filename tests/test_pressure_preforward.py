from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.connector import ControlMetadata, RefreshCommand
from vllm_kvmem.pressure_commit import prepare_commit
from test_pressure_pipeline_handoff import handoff


def pref():
    pipeline, refresh, job, trace = handoff()
    pipeline.preforward = True
    pipeline.runner._kvmem_async_refresh = refresh
    command = RefreshCommand('r', 1, (), asynchronous=True, selection='pressure', pressure_stage='prefill')
    output = NS(kv_connector_metadata=ControlMetadata(commands=[command]),
                num_scheduled_tokens={'peer': 1024}, finished_req_ids=set())
    native = NS(_disabled=False, _pending_load_kwargs=None)
    def before(output, **kwargs):
        trace.append('native-pre')
        native._pending_load_kwargs = kwargs
    def load():
        assert native._pending_load_kwargs is not None
        trace.append('native-load')
        native._pending_load_kwargs = None
    native.pre_forward, native._start_load_kv = before, load
    pipeline.runner.kv_connector = native
    return pipeline, refresh, job, trace, native, output


def test_early_load_uses_native_receipt_once_and_excludes_current_peer():
    p, refresh, job, trace, native, output = pref()
    p.install_preforward()
    native.pre_forward(output, num_tokens=1024)
    assert [item for item in trace if isinstance(item, str)][:3] == [
        'native-pre', 'native-load', 'existing-step']
    assert job.done and not job.future.done()
    assert native._pending_load_kwargs is None  # Native post-forward skips load.
    assert trace[-1][1]['boundary'] == 'pre_forward'
    # Original metadata is still available for native feedback.
    assert len(output.kv_connector_metadata.commands) == 1


@pytest.mark.parametrize('kind', ['query','decode','guided','query_span','sync','current',
                                 'cancel','finished','eviction','disabled','mixed'])
def test_unqualified_control_keeps_original_load_order(kind):
    p, refresh, job, trace, native, output = pref()
    control = output.kv_connector_metadata
    changes = dict(query={'selection':'query'}, decode={'pressure_stage':'decode'},
                   guided={'guided_reason':'prefill-pressure'}, query_span={'query_span':(1,2)},
                   sync={'asynchronous':False})
    if kind in changes:control.commands[0] = replace(control.commands[0], **changes[kind])
    if kind == 'current':output.num_scheduled_tokens['r'] = 1024
    if kind == 'cancel':control.cancelled.add('r')
    if kind == 'finished':output.finished_req_ids.add('unrelated')
    if kind == 'eviction':control.sessions.evictions.append('checkpoint')
    if kind == 'disabled':native._disabled = True
    if kind == 'mixed':control.commands.append(replace(control.commands[0], request_id='q', selection='query'))
    native._pending_load_kwargs = {}
    p.start_preforward(native, output)
    assert trace == [] and native._pending_load_kwargs == {} and not job.done


def test_preforward_uses_current_schedule_instead_of_previous_batch_owner():
    p, refresh, job, trace, native, output = pref()
    p.runner._kvmem_last_batch.req_ids = ('r',)
    native._pending_load_kwargs = {}
    p.start_preforward(native, output)
    assert 'existing-step' in trace


def test_ordinary_pressure_replay_reservation_keeps_original_prepare_path():
    p, refresh, job, trace, native, output = pref()
    output.kv_connector_metadata.commands[0] = replace(
        output.kv_connector_metadata.commands[0], input_tokens=tuple(range(32)))
    native._pending_load_kwargs = {}
    p.start_preforward(native, output)
    assert trace[0] == 'native-load' and 'existing-step' in trace


def test_preforward_never_waits_for_plan_even_when_post_forward_polls():
    p, refresh, job, trace, native, output = pref()
    job.stage = 'assembly'; job.future = Future()
    p.dispatch = lambda *args: False
    class NoWait:
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def wait(self, *args):raise AssertionError('foreground plan wait')
    p.prepared_condition = NoWait()
    p.prime_post_forward(refresh)
    assert not job.done and trace[-1][1]['prepared'] is False


def selection():
    calls = []
    class Handle:
        def __init__(self, slot):self.slot = slot
        @property
        def generation(self):
            calls.append(self.slot)
            return 7 + self.slot
    records = [NS(handle=Handle(i), history_begin=begin, count=n)
               for i, (begin, n) in enumerate([(0, 32), (96, 32), (192, 11), (224, 32)])]
    chosen = [[records[i] for i in ids] for ids in [[0, 1, 2], [1, 2, 3]]]
    job = NS(authority=object(), command=NS(request_id='r'), history=256, selected_tokens=75,
             layerwise_pressure_plan=(chosen, []))
    prepared = prepare_commit(job, chosen, records, [0, 1, 2, 3], [[0, 1, 2], [1, 2, 3]])
    return job, prepared, calls


def test_prepared_maps_preserve_layer_order_partial_offsets_and_shared_handle_identity():
    job, prepared, calls = selection()
    assert prepared.maps == (
        ((0, 7, 0, 32, 0), (1, 8, 96, 32, 32), (2, 9, 192, 11, 64)),
        ((1, 8, 96, 32, 0), (2, 9, 192, 11, 32), (3, 10, 224, 32, 43)))
    assert sorted(calls) == [0, 1, 2, 3]
    prepared.validate(job)


@pytest.mark.parametrize('kind', ['owner','request','history','selection','extent'])
def test_prepared_map_cannot_publish_for_another_transaction(kind):
    job, prepared, _ = selection()
    if kind == 'owner':job.authority = object()
    if kind == 'request':job.command.request_id = 'other'
    if kind == 'history':job.history += 32
    if kind == 'selection':job.layerwise_pressure_plan = (list(job.layerwise_pressure_plan[0]), [])
    if kind == 'extent':job.selected_tokens += 32
    with pytest.raises(RuntimeError, match='accepted owner/selection'):prepared.validate(job)


def test_controller_prepared_commit_matches_original_native_selection_and_evidence():
    from test_layerwise_pressure import controller, options, native_authority, finish
    c = controller(); c.prepared_commit = True
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); hs = c._history('r', o)
    rows = list(a.records.values())
    for i, history in enumerate(hs):
        history.values = {r.history_begin:[1. if j in ((2, 3) if i == 0 else (5, 6)) else .01, 1.]
                          for j, r in enumerate(rows)}
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    emitted = []; c.query.emit = lambda event, **fields: emitted.append((event, fields))
    try:
        c.prepare(NS(options=o), job)
        chosen = job.layerwise_pressure_plan[0]
        slots = {r.handle.slot for r in chosen[0]}
        a.memory.select({}, pressure_scores=[(r.handle.slot, float(r.handle.slot in slots)) for r in rows],
                        max_tokens=96)
        finish(a.memory)
        job.selected_tokens = 96
        with pytest.raises(RuntimeError, match='completed assembly or guarded'):
            c.committed(job)
        job.layerwise_single_assembly_done = True
        c.prepared_commit = False; c.committed(job)
        original = c.maps.pop('r'); old_event = emitted.pop()
        c.prepared_commit = True; c.committed(job)
        assert c.maps['r'] == original and emitted.pop() == old_event
    finally:
        assert a.memory.close()
