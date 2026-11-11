"""Early scheduling still requires the exact immutable plan and layer guards."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.pressure_pipeline import PressurePipeline
from test_pressure_pipeline import state, dispose, Fence, Stream


def test_plan_handoff_does_not_claim_gpu_completion_or_allow_unguarded_access():
    authority, lease = state()
    lease.controller.early_handoff = True
    try:
        assert lease.dispatchable() and not lease.first_ready() and not lease.query()
        authority.memory.commit_layerwise_selection(lease)
        lease.dispatched = True
        assert not authority.memory.close()
        with pytest.raises(RuntimeError, match='Whole-window'):
            authority.memory.checkpoint()
        lease.publish_layer(0, Fence(False))
        stream = Stream()
        lease.wait_layer(0, stream)
        assert stream.events == [lease.events[0]] and not lease.query()
    finally:
        dispose(authority, lease)


def handoff():
    authority = NS(history_end=100)
    command = NS(request_id='r', selection='pressure', pressure_stage='prefill')
    job = NS(command=command, done=False, cancelled=False, guided_pressure=False,
             stage='prepare', history=100, authority=authority, future=None)
    runner = NS(_kvmem_last_batch=NS(req_ids=('peer',)),
                _kvmem_capture_runtime=NS(authorities={'r':authority},
                                         publication_ready=lambda ids:True))
    pipeline = PressurePipeline(NS(runner=runner, pipeline_handoff=True))
    trace = []
    def step(j):
        assert j is job
        trace.append('existing-step')
        j.stage = 'assembly'; j.future = Future()
        j.layerwise_pipeline = object()
    def dispatch(refresh,j):
        assert j.layerwise_pipeline is not None
        trace.append('dispatch');j.done=True
        return True
    pipeline.dispatch = dispatch
    refresh = NS(jobs={'r':job},step=step,emit=lambda j,s,**k:trace.append((s,k)))
    return pipeline, refresh, job, trace


def test_post_forward_uses_original_prepare_then_current_feedback_without_waiting_for_all_layers():
    pipeline, refresh, job, trace = handoff()
    pipeline.prime_post_forward(refresh)
    assert job.done and not job.future.done()
    assert trace[0]=='existing-step' and trace[2]=='dispatch'
    assert trace[-1][0]=='pipeline_handoff_attempt' and trace[-1][1]['dispatched']


@pytest.mark.parametrize('fault', ['off','cancelled','done','query','decode','guided','current',
                                 'unpublished','foreign_prefix','missing_authority','no_capture'])
def test_handoff_does_not_prepare_or_ack_unqualified_owners(fault):
    pipeline, refresh, job, trace = handoff()
    if fault=='off': pipeline.early_handoff=False
    if fault=='cancelled': job.cancelled=True
    if fault=='done': job.done=True
    if fault=='query': job.command.selection='query'
    if fault=='decode': job.command.pressure_stage='decode'
    if fault=='guided': job.guided_pressure=True
    if fault=='current': pipeline.runner._kvmem_last_batch.req_ids=('r',)
    if fault=='unpublished': pipeline.runner._kvmem_capture_runtime.publication_ready=lambda ids:False
    if fault=='foreign_prefix': job.authority.history_end=99
    if fault=='missing_authority': pipeline.runner._kvmem_capture_runtime.authorities={}
    if fault=='no_capture': pipeline.runner._kvmem_capture_runtime=None
    pipeline.prime_post_forward(refresh)
    assert trace==[]


def test_unready_metadata_has_bounded_wait_and_is_not_acknowledged():
    pipeline, refresh, job, trace = handoff()
    job.stage='assembly';job.future=Future()
    pipeline.dispatch=lambda r,j:False
    pipeline.prime_post_forward(refresh,max_wait_seconds=0)
    assert not job.done and trace[-1][1]['prepared'] is False
    assert trace[-1][1]['dispatched'] is False


def test_preparation_error_propagates_before_any_dispatch():
    pipeline, refresh, job, trace = handoff()
    job.stage='assembly';job.future=Future();job.future.set_exception(OSError('failed source'))
    with pytest.raises(OSError, match='failed source'):
        pipeline.prime_post_forward(refresh)
    assert trace==[] and not job.done


def test_lane_or_byte_admission_refusal_does_not_wait_or_force_producer():
    pipeline, refresh, job, trace = handoff()
    def no_credit(j):
        trace.append('admission-deferred');j.stage='assembly_wait'
    refresh.step=no_credit
    pipeline.prime_post_forward(refresh)
    assert job.future is None and not job.done
    assert trace[0]=='admission-deferred' and not any(x=='dispatch' for x in trace)


class Unwaited:
    """A condition that fails the test when anything waits on it."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def wait(self, *a): raise AssertionError('A just-submitted producer must not be waited for')


def test_fresh_wait_flag_is_validated_and_defaults_to_the_bounded_wait(monkeypatch):
    from vllm_kvmem import pressure_pipeline
    monkeypatch.delenv('KVMEM_PRESSURE_HANDOFF_FRESH_WAIT', raising=False)
    assert pressure_pipeline.fresh_handoff_wait() and handoff()[0].fresh_handoff_wait
    monkeypatch.setenv('KVMEM_PRESSURE_HANDOFF_FRESH_WAIT', '0')
    assert not pressure_pipeline.fresh_handoff_wait() and not handoff()[0].fresh_handoff_wait
    monkeypatch.setenv('KVMEM_PRESSURE_HANDOFF_FRESH_WAIT', '2')
    with pytest.raises(ValueError, match='FRESH_WAIT'):
        pressure_pipeline.fresh_handoff_wait()


def test_without_fresh_wait_a_just_submitted_producer_is_not_waited_for():
    pipeline, refresh, job, trace = handoff()
    pipeline.fresh_handoff_wait = False
    def step(j):
        trace.append('existing-step'); j.stage = 'assembly'; j.future = Future()
    refresh.step = step
    pipeline.dispatch = lambda r, j: False        # the plan is not prepared yet
    pipeline.prepared_condition = Unwaited()
    pipeline.prime_post_forward(refresh, max_wait_seconds=10.)
    assert not job.done and not job.future.done()
    assert trace[0] == 'existing-step' and trace[1][0] == 'pipeline_handoff_prime'
    assert trace[-1][0] == 'pipeline_handoff_attempt'
    assert trace[-1][1]['dispatched'] is False and trace[-1][1]['prepared'] is False


def test_without_fresh_wait_an_already_prepared_plan_is_still_acknowledged_at_once():
    pipeline, refresh, job, trace = handoff()     # its step publishes the plan synchronously
    pipeline.fresh_handoff_wait = False
    pipeline.prepared_condition = Unwaited()
    pipeline.prime_post_forward(refresh)
    assert job.done and trace[-1][1]['dispatched'] and trace[-1][1]['prepared']


def test_without_fresh_wait_an_earlier_producer_keeps_the_bounded_wait(monkeypatch):
    from vllm_kvmem import pressure_pipeline
    pipeline, refresh, job, trace = handoff()
    pipeline.fresh_handoff_wait = False
    job.stage = 'assembly'; job.future = Future()   # submitted at an earlier boundary
    now = [0.]; monkeypatch.setattr(pressure_pipeline.time, 'perf_counter', lambda: now[0])
    waits = []
    class Clock:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def wait(self, seconds):
            waits.append(seconds); now[0] += .01
            if len(waits) == 3: job.layerwise_pipeline = object()
    pipeline.prepared_condition = Clock()
    pipeline.prime_post_forward(refresh, max_wait_seconds=.064)
    assert len(waits) == 3 and job.done and trace[-1][1]['dispatched']


def test_without_fresh_wait_a_failed_just_submitted_producer_still_raises():
    pipeline, refresh, job, trace = handoff()
    pipeline.fresh_handoff_wait = False
    def step(j):
        j.stage = 'assembly'; j.future = Future(); j.future.set_exception(OSError('failed source'))
    refresh.step = step
    with pytest.raises(OSError, match='failed source'):
        pipeline.prime_post_forward(refresh)
    assert not job.done
