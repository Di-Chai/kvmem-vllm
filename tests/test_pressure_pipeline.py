"""Guarded partial KV must not be advertised as completed/recyclable KV."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.memory import Phase
from vllm_kvmem.pressure_pipeline import LayerMaterialization, PressurePipeline
from test_recall_experiment import native_authority


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


class Stream:
    cuda_stream = 7

    def __init__(self):
        self.events = []

    def wait_event(self, event):
        self.events.append(event)


def state():
    authority = native_authority(count=256, budget=96, sink=1)
    plan = authority.memory.select({}, max_tokens=96)
    controller = NS(installed=True)
    job = NS(authority=authority)
    lease = LayerMaterialization(controller, job, plan, ('a', 'b', 'draft'))
    return authority, lease


def dispose(authority, lease):
    if authority.memory.phase is Phase.ASSEMBLING:
        authority.memory.attach_completion(Fence(True))
        assert authority.memory.commit_selection()
    lease.final_event = Fence(True)
    if authority.memory._layerwise_materialization is lease:
        authority.memory.finish_layerwise_selection(lease)
    assert authority.memory.close()


def test_first_ready_publishes_same_selection_without_recycling_pending_kv():
    a, lease = state()
    selected = {r.block_id for r in lease.plan.remaps}
    try:
        assert not lease.query() and not lease.first_ready()
        first = Fence()
        lease.publish_layer(0, first)
        with pytest.raises(RuntimeError, match='guarded'):
            a.memory.commit_layerwise_selection(lease)
        first.ready = True
        a.memory.commit_layerwise_selection(lease)
        lease.dispatched = True
        assert a.memory.phase is Phase.READY
        assert {r[0] for r in a.memory._store.working_records()} == selected
        assert not lease.query() and not a.memory.close()
        with pytest.raises(RuntimeError, match='Whole-window'):
            a.memory.checkpoint()
        with pytest.raises(RuntimeError, match='final fence'):
            a.memory.finish_layerwise_selection(lease)
        assert not lease.dispatched or a.memory._layerwise_materialization is lease
    finally:
        dispose(a, lease)


def test_materialization_requires_exact_owner_plan_installed_guards_and_no_prior_submission():
    for change in ('owner', 'plan', 'guards', 'submitted'):
        a, lease = state()
        lease.publish_layer(0, Fence(True))
        if change == 'owner': lease.job = NS(authority=NS(memory=object()))
        if change == 'plan': lease.plan = object()
        if change == 'guards': lease.controller.installed = False
        if change == 'submitted': a.memory.attach_completion(Fence(True))
        with pytest.raises(RuntimeError):
            a.memory.commit_layerwise_selection(lease)
        assert a.memory.phase is Phase.ASSEMBLING
        if change == 'submitted':
            assert a.memory.commit_selection()
        dispose(a, lease)


def test_reader_lease_is_installed_before_native_publication_can_release_gil():
    a, lease = state()
    lease.publish_layer(0, Fence(True))
    original = a.memory._publish_selection
    observed = []
    def publish():
        observed.append(a.memory._layerwise_materialization is lease)
        original()
    a.memory._publish_selection = publish
    try:
        a.memory.commit_layerwise_selection(lease)
        assert observed == [True]
    finally:
        dispose(a, lease)


def test_layer_wait_does_not_enqueue_wait_on_unrecorded_cuda_event():
    a, lease = state()
    stream = Stream()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lease.wait_layer, 1, stream)
            assert not future.done() and not stream.events
            lease.publish_layer(0, Fence(True))
            assert not future.done() and not stream.events
            second = Fence(False)
            lease.publish_layer(1, second)
            future.result(timeout=2)
        assert stream.events == [second]
        lease.wait_layer(1, stream)
        assert stream.events == [second]  # ordered stream carries the dependency
        assert not lease.query()
        other = Stream(); other.cuda_stream = 8
        with pytest.raises(RuntimeError, match='stream changed'):
            lease.wait_layer(1, other)
    finally:
        dispose(a, lease)


def test_failed_producer_wakes_consumer_and_cannot_publish_success():
    a, lease = state()
    try:
        stream = Stream()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lease.wait_layer, 0, stream)
            lease.fail(OSError('archive read failed'), Fence(False))
            with pytest.raises(RuntimeError, match='producer failed'):
                future.result(timeout=2)
        with pytest.raises(RuntimeError, match='producer failed'):
            lease.first_ready()
        assert not stream.events and not lease.query()
    finally:
        dispose(a, lease)


def test_layer_submission_order_and_duplicate_are_rejected():
    a, lease = state()
    try:
        with pytest.raises(RuntimeError, match='order'):
            lease.publish_layer(1, Fence())
        lease.publish_layer(0, Fence())
        with pytest.raises(RuntimeError, match='order'):
            lease.publish_layer(0, Fence())
    finally:
        dispose(a, lease)


def test_foreign_public_batch_does_not_wait_for_suspended_owner():
    pipeline = PressurePipeline(NS(runner=NS(_kvmem_last_batch=NS(req_ids=('other',)))))
    pipeline.active['pending'] = NS(query=lambda: pytest.fail('Foreign owner must not wait'))
    pipeline.wait_layer(0)


def test_reap_retains_all_resources_until_future_and_final_cuda_completion():
    a, lease = state()
    log = []
    pressure = NS(runner=NS())
    pipeline = PressurePipeline(pressure)
    lease.publish_layer(0, Fence(True))
    a.memory.commit_layerwise_selection(lease)
    lease.dispatched = True
    event = Fence(False)
    lease.final_event = event
    result = NS(event=event, uploaded_bytes=1, read_stats={})
    job = lease.job
    job.result = None
    job.layerwise_pipeline = lease
    done = [False]
    job.future = NS(done=lambda:done[0], result=lambda:(
        NS(close=lambda:log.append('close') or True), result, lease.plan, {}, .1))
    pipeline.active['r'] = lease
    refresh = NS(maintenance=NS(observe=lambda *a:log.append('observe'),
        release=lambda *a:log.append('release')), report=lambda *a:log.append('report'),
        emit=lambda *a, **k:log.append('completed'))
    try:
        pipeline.reap(refresh)
        assert not log
        done[0] = True
        pipeline.reap(refresh)
        assert not log and not a.memory.close()
        event.ready = True
        pipeline.reap(refresh)
        assert log == ['close', 'observe', 'release', 'report', 'completed']
        assert not pipeline.active and a.memory._layerwise_materialization is None
        assert job.future is None and lease.plan is None
    finally:
        dispose(a, lease)


def test_cancel_after_early_ack_still_defers_page_and_request_index_recycling():
    from dataclasses import replace
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_kvmem.connector import ControlMetadata
    from test_async_refresh import manager
    refresh = manager()
    job = NS(done=True, cancelled=False)
    pipeline = NS(active={'a': NS(job=job)})
    refresh.runner._kvmem_layerwise_pressure = NS(pipeline=pipeline)
    # pump already removed the acknowledged command. The materialization
    # independently owns its lease until all producer work is complete.
    assert not refresh.jobs
    output = replace(SchedulerOutput.make_empty(), finished_req_ids={'a','b'},
        kv_connector_metadata=ControlMetadata(cancelled={'a'}, terminal_prefixes={'a':'P'}))
    held = refresh.defer_terminals(output)
    assert held.finished_req_ids == {'b'} and job.cancelled
    empty = replace(SchedulerOutput.make_empty(), kv_connector_metadata=ControlMetadata())
    assert not refresh.defer_terminals(empty).finished_req_ids
    pipeline.active.clear()
    released = refresh.defer_terminals(empty)
    assert released.finished_req_ids == {'a'}
    assert released.kv_connector_metadata.terminal_prefixes == {'a':'P'}


@pytest.mark.parametrize('mode', ['NONE', 'PIECEWISE', 'FULL'])
def test_pipeline_preserves_native_piecewise_compute_graphs(monkeypatch, mode):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu import model_runner
    from vllm_kvmem.layerwise_query import install_native_query_dispatch
    calls = []
    def dispatch(*args, **kwargs):
        calls.append(kwargs.copy())
        return NS(cg_mode=CUDAGraphMode.NONE if kwargs.get('need_eager') else
                  getattr(CUDAGraphMode, mode)), None
    monkeypatch.setattr(model_runner, 'dispatch_cg_and_sync_dp', dispatch)
    monkeypatch.setattr(model_runner, 'set_forward_context', lambda *a, **k:k)
    controller = NS(has_active=False, pressure=NS(pipeline=NS(needs_dispatch_boundary=True)))
    install_native_query_dispatch(controller)
    result = model_runner.dispatch_cg_and_sync_dp()
    if mode == 'FULL':
        assert result[0].cg_mode == CUDAGraphMode.NONE
        assert calls == [{}, {'need_eager': True}]
    else:
        assert result[0].cg_mode == getattr(CUDAGraphMode, mode) and calls == [{}]
    assert not model_runner.set_forward_context().get('skip_compiled', False)
