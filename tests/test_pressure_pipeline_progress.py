"""Native boundary progress cannot bypass ownership, readiness or lane credit."""
from concurrent.futures import Future
from copy import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.connector import ControlMetadata, RefreshCommand
from test_pressure_pipeline_handoff import handoff


def progress():
    pipeline, refresh, job, trace = handoff()
    pipeline.progress = True
    pipeline.runner._kvmem_async_refresh = refresh
    pipeline.reap = lambda r, **kwargs:trace.append('reap')
    native = NS(_disabled=False, _pending_load_kwargs=None)
    def pre(scheduler_output, **kwargs):
        trace.append('native-pre')
        native._pending_load_kwargs = kwargs
    native.pre_forward = pre
    native.finish_forward = lambda:trace.append('native-finish')
    native._start_load_kv = lambda:pytest.fail('New command receipt must remain native-owned')
    pipeline.runner.kv_connector = native
    output = NS(kv_connector_metadata=ControlMetadata(),num_scheduled_tokens={'peer':1024},
                finished_req_ids=set())
    pipeline.install_progress()
    return pipeline, refresh, job, trace, native, output


def test_native_keyword_boundary_reaps_before_prepare_and_preserves_load_receipt():
    p, refresh, job, trace, native, output = progress()
    native.pre_forward(scheduler_output=output,num_tokens=1024)
    assert trace[:3] == ['native-pre','reap','existing-step']
    assert job.done and native._pending_load_kwargs == {'num_tokens':1024}
    attempts=[fields for event,fields in (r for r in trace if isinstance(r,tuple))
              if event=='pipeline_handoff_attempt']
    assert attempts[0]['boundary'] == 'native_pre_forward'
    native.finish_forward()
    assert trace[-2:] == ['native-finish','reap']
    assert p.progress_requests is None


@pytest.mark.parametrize('kind',['cancel','finished','eviction','disabled','query','decode','sync','guided','mixed'])
def test_lifecycle_or_mixed_controls_keep_existing_native_order(kind):
    p, refresh, job, trace, native, output = progress()
    c=RefreshCommand('other',2,(),asynchronous=True,selection='pressure',pressure_stage='prefill')
    control=output.kv_connector_metadata
    if kind=='cancel':control.cancelled.add('r')
    elif kind=='finished':output.finished_req_ids.add('r')
    elif kind=='eviction':control.sessions.evictions.append('snapshot')
    elif kind=='disabled':native._disabled=True
    else:
        changes={'query':{'selection':'query'},'decode':{'pressure_stage':'decode'},
                 'sync':{'asynchronous':False},'guided':{'guided_reason':'prefill-pressure'}}
        control.commands.append(replace(c,**changes.get(kind,{})))
        if kind=='mixed':control.commands.append(replace(c,selection='query'))
    native.pre_forward(scheduler_output=output)
    native.finish_forward()
    assert trace == ['native-pre','native-finish'] and not job.done


def test_current_batch_owner_cannot_be_prepared_at_either_boundary():
    p, refresh, job, trace, native, output = progress()
    p.runner._kvmem_last_batch.req_ids=()  # Current scheduler output is authoritative.
    output.num_scheduled_tokens={'r':1024}
    native.pre_forward(scheduler_output=output)
    native.finish_forward()
    assert trace == ['native-pre','reap','native-finish','reap'] and not job.done


def test_finished_boundary_never_uses_stale_context_after_unqualified_forward():
    p, refresh, job, trace, native, output = progress()
    native.pre_forward(scheduler_output=output)
    assert p.progress_requests == {'peer'}
    output.finished_req_ids.add('retiring')
    native.pre_forward(scheduler_output=output)
    trace.clear();native.finish_forward()
    assert trace == ['native-finish'] and p.progress_requests is None


def test_ready_ack_does_not_consume_next_original_lane_admission_opportunity():
    p, refresh, job, trace, native, output = progress()
    second=copy(job);second.command=NS(request_id='s',selection='pressure',pressure_stage='prefill')
    refresh.jobs['s']=second
    p.runner._kvmem_capture_runtime.authorities['s']=second.authority
    def step(j):
        trace.append(('step',j.command.request_id));j.stage='assembly';j.future=Future()
        j.layerwise_pipeline=object()
    def dispatch(r,j):
        trace.append(('dispatch',j.command.request_id));j.done=True
        return True
    refresh.step=step;p.dispatch=dispatch
    native.pre_forward(scheduler_output=output)
    assert job.done and second.done
    assert [r for r in trace if isinstance(r,tuple) and r[0]=='step'] == [('step','r'),('step','s')]
    assert trace.count('reap') == 2


def test_native_polls_never_wait_for_unprepared_plan_or_cuda():
    p, refresh, job, trace, native, output = progress()
    job.stage='assembly';job.future=Future()
    p.dispatch=lambda *a:False
    class NoWait:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def wait(self,*a):raise AssertionError('Native boundary must not wait for a plan')
    p.prepared_condition=NoWait()
    native.pre_forward(scheduler_output=output)
    native.finish_forward()
    assert not job.done and not job.future.done()


def test_acknowledged_but_unfinished_copy_still_owns_the_only_lane():
    p, refresh, job, trace, native, output = progress()
    p.active['prior']=object()  # Original jobs may have dropped its early ACK.
    refresh.step=lambda *args:pytest.fail('Unfinished copy still owns scratch')
    p.prime_post_forward(refresh,max_wait_seconds=0)
    assert not job.done and job.future is None and not trace


def test_entire_feedback_turn_shares_original_bounded_wait_budget(monkeypatch):
    from vllm_kvmem import pressure_pipeline
    p, refresh, job, trace, native, output = progress()
    now=[0.];monkeypatch.setattr(pressure_pipeline.time,'perf_counter',lambda:now[0])
    second=copy(job);second.command=NS(request_id='s',selection='pressure',pressure_stage='prefill')
    second.stage='assembly';second.future=Future()
    refresh.jobs['s']=second;p.runner._kvmem_capture_runtime.authorities['s']=second.authority
    def reap(r,**kwargs):now[0]+=.1
    p.reap=reap
    def dispatch(r,j):
        if j is job:j.done=True;return True
        return False
    p.dispatch=dispatch
    class NoWait:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def wait(self,*a):raise AssertionError('A second job must not get a fresh wait budget')
    p.prepared_condition=NoWait()
    p.prime_post_forward(refresh,max_wait_seconds=.064)
    assert job.done and not second.done
