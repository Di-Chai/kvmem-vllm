"""Opt-in schedule-time pressure ACK: dispatch only prepared plans, report only them."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.connector import ControlCompletions, KVMemConnector
from test_pressure_pipeline_handoff import handoff


def prepared():
    pipeline, refresh, job, trace = handoff()
    pipeline.runner._kvmem_async_refresh = refresh
    pipeline.reap = lambda r, **kwargs: trace.append(('reap', kwargs.get('boundary')))
    job.stage = 'assembly'; job.future = Future(); job.layerwise_pipeline = object()
    refresh.step = lambda j: pytest.fail('A prepared plan needs no engine-thread step')
    return pipeline, refresh, job, trace


def test_prepared_plan_is_dispatched_and_reported():
    p, refresh, job, trace = prepared()
    assert p.schedule_boundary() == ('r',)
    assert job.done and trace[0] == ('reap', 'schedule') and 'dispatch' in trace
    attempt = [f for e, f in (t for t in trace if isinstance(t, tuple) and len(t) == 2
                             and isinstance(t[1], dict)) if e == 'pipeline_handoff_attempt']
    assert attempt and attempt[0]['boundary'] == 'schedule' and attempt[0]['dispatched']


def test_unprepared_job_is_never_prepared_or_waited_for():
    p, refresh, job, trace = prepared()
    job.stage = 'prepare'; job.future = None; job.layerwise_pipeline = None
    assert p.schedule_boundary() == () and not job.done
    job.stage = 'assembly'; job.future = Future()  # Producer still selecting.
    assert p.schedule_boundary() == () and not job.done and not job.future.done()


def test_failed_preparation_is_not_swallowed():
    p, refresh, job, trace = prepared()
    job.layerwise_pipeline = None
    job.future.set_exception(RuntimeError('producer failed'))
    with pytest.raises(RuntimeError, match='producer failed'):
        p.schedule_boundary()


@pytest.mark.parametrize('fault', ['off', 'cancelled', 'done', 'query', 'decode', 'guided',
                                   'unpublished', 'foreign_prefix', 'missing_authority'])
def test_unqualified_owner_is_not_acknowledged(fault):
    p, refresh, job, trace = prepared()
    if fault == 'off': p.early_handoff = False
    if fault == 'cancelled': job.cancelled = True
    if fault == 'done': job.done = True
    if fault == 'query': job.command.selection = 'query'
    if fault == 'decode': job.command.pressure_stage = 'decode'
    if fault == 'guided': job.guided_pressure = True
    if fault == 'unpublished': p.runner._kvmem_capture_runtime.publication_ready = lambda ids: False
    if fault == 'foreign_prefix': job.authority.history_end = 99
    if fault == 'missing_authority': p.runner._kvmem_capture_runtime.authorities = {}
    assert p.schedule_boundary() == ()
    assert 'dispatch' not in trace and (fault == 'done' or not job.done)


def test_waiter_admission_respects_the_busy_lane():
    p, refresh, job, trace = prepared()
    job.stage = 'assembly_wait'; job.future = None; job.layerwise_pipeline = None
    p.active['other'] = object()
    refresh.step = lambda j: pytest.fail('Busy lane must not admit another producer')
    assert p.schedule_boundary() == ()
    p.active.clear()
    def admit(j):
        trace.append('admit'); j.stage = 'assembly'; j.future = Future()
    refresh.step = admit
    assert p.schedule_boundary() == () and trace.count('admit') == 1 and not job.done


def worker(poll, jobs):
    return NS(pressure_schedule_poll=poll, worker_jobs=jobs)


def test_connector_reports_only_handoffs_from_this_poll():
    command = NS(command_id=7)
    done = NS(query=lambda: True, selected_tokens=36864)
    other = NS(query=lambda: True, selected_tokens=None)
    w = worker(lambda: ('r',), {'r': (command, done), 'restore': (NS(command_id=3), other)})
    meta = KVMemConnector.take_schedule_completions(w)
    assert isinstance(meta, ControlCompletions)
    assert meta.completed == [('r', 7)] and meta.selected_windows == {'r': 36864}
    assert set(w.worker_jobs) == {'restore'}  # Other completions keep the native path.
    assert meta.publication_sequence is None and meta.archive_sequence is None


def test_connector_without_poll_or_dispatch_reports_nothing():
    assert KVMemConnector.take_schedule_completions(worker(None, {})) is None
    assert KVMemConnector.take_schedule_completions(worker(lambda: (), {})) is None


def test_connector_rejects_dispatch_without_completed_job():
    w = worker(lambda: ('r',), {'r': (NS(command_id=1), NS(query=lambda: False))})
    with pytest.raises(RuntimeError, match='did not complete'):
        KVMemConnector.take_schedule_completions(w)


def test_scheduler_applies_completions_through_native_connector_path(monkeypatch):
    from vllm.distributed import kv_transfer
    from vllm_kvmem.serving.scheduler import KVMemScheduler
    meta = ControlCompletions([('r', 7)], {'r': 36864})
    seen = []
    w = NS(take_schedule_completions=lambda: meta)
    monkeypatch.setattr(kv_transfer, 'has_kv_transfer_group', lambda: True)
    monkeypatch.setattr(kv_transfer, 'get_kv_transfer_group', lambda: w)
    sched = NS(transactions={'r': object()},
               connector=NS(update_connector_output=lambda out: seen.append(out)))
    KVMemScheduler._schedule_time_acks(sched)
    assert len(seen) == 1 and seen[0].kv_connector_worker_meta is meta
    assert seen[0].finished_sending is None
    seen.clear(); sched.transactions = {}
    KVMemScheduler._schedule_time_acks(sched)  # Nothing suspended: no worker call.
    assert not seen


def test_scheduler_refuses_its_own_connector(monkeypatch):
    from vllm.distributed import kv_transfer
    from vllm_kvmem.serving.scheduler import KVMemScheduler
    own = NS(take_schedule_completions=lambda: None)
    monkeypatch.setattr(kv_transfer, 'has_kv_transfer_group', lambda: True)
    monkeypatch.setattr(kv_transfer, 'get_kv_transfer_group', lambda: own)
    with pytest.raises(RuntimeError, match='scheduler connector'):
        KVMemScheduler._schedule_time_acks(NS(transactions={'r': 1}, connector=own))


def test_idle_wait_dispatches_a_plan_that_becomes_ready_during_the_wait():
    import threading, time
    p, refresh, job, trace = prepared()
    job.layerwise_pipeline = None
    def publish():
        time.sleep(.02)
        with p.prepared_condition:
            job.layerwise_pipeline = object()
            p.prepared_condition.notify_all()
    threading.Thread(target=publish).start()
    started = time.perf_counter()
    assert p.schedule_boundary(wait_seconds=.5) == ('r',)
    assert job.done and time.perf_counter() - started < .4


def test_idle_wait_is_bounded_and_does_not_dispatch_unprepared_plans():
    import time
    p, refresh, job, trace = prepared()
    job.layerwise_pipeline = None
    started = time.perf_counter()
    assert p.schedule_boundary(wait_seconds=.03) == ()
    assert not job.done and .02 < time.perf_counter() - started < .5


@pytest.mark.parametrize('runnable', [False, True])
def test_scheduler_waits_only_when_nothing_else_is_schedulable(monkeypatch, runnable):
    from vllm.distributed import kv_transfer
    from vllm_kvmem.serving.scheduler import KVMemScheduler
    waits = []
    def take(wait=0.):
        waits.append(wait)
        return None
    w = NS(take_schedule_completions=take)
    monkeypatch.setattr(kv_transfer, 'has_kv_transfer_group', lambda: True)
    monkeypatch.setattr(kv_transfer, 'get_kv_transfer_group', lambda: w)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_IDLE_ACK_WAIT_MS', '40')
    running = [NS(request_id='r')] + ([NS(request_id='peer')] if runnable else [])
    sched = NS(transactions={'r': object()}, connector=NS(update_connector_output=lambda out: None),
               running=running, _waiting_queues=lambda: ())
    KVMemScheduler._schedule_time_acks(sched)
    assert waits == ([0.] if runnable else [.04])
