"""Query completion handoff must preserve fences, ownership and exclusion."""
from types import SimpleNamespace as NS

import pytest

from test_pressure_completion_poll import inflight
from vllm_kvmem.memory import Phase


def query_inflight():
    manager, job, future, fence, authority, trace = inflight()
    manager.query_post_forward_reap = True
    job.command.selection = 'query'
    job.command.query_span = (256, 257)
    job.command.input_tokens = (1,)
    return manager, job, future, fence, authority, trace


def test_query_ack_keeps_cuda_fence_and_excludes_probe_before_commit(monkeypatch):
    from test_connector import pair
    from vllm_kvmem import prefill_attention
    manager, job, future, fence, authority, trace = query_inflight()
    monkeypatch.setattr(prefill_attention, 'exclude_query_replay',
                        lambda *args: trace.append(('exclude', args[2:])))
    _, connector = pair()
    connector.query_completion_poll = manager.poll_query_completions
    connector.worker_jobs = {'r': (job.command, job)}
    try:
        connector.get_finished(set())
        assert future.reads == 0 and not trace
        future.ready = True
        connector.get_finished(set())
        assert future.reads == 1 and not trace and not job.done
        fence.ready = True
        connector.get_finished(set())
        assert job.done and authority.memory.phase is Phase.READY
        assert trace[:2] == [('exclude', ('r', 256, 257, True)), ('commit', ('r', 256, 96))]
        assert 'query_completion_reaped' in trace
        assert connector.worker_completions == [('r', 1)]
        connector.get_finished(set())
        assert connector.worker_completions == [('r', 1)] and future.reads == 1
    finally:
        fence.ready = True
        if not job.done:
            manager.step(job)
        assert authority.memory.close()


@pytest.mark.parametrize('fault', ['disabled', 'done', 'cancelled', 'pressure',
    'prepare', 'query', 'assembly_wait', 'no_capture', 'no_future', 'unready_future',
    'unpublished', 'foreign_authority'])
def test_query_poll_never_prepares_or_bypasses_ownership(fault):
    manager, job, future, fence, authority, trace = query_inflight()
    future.ready = fence.ready = True
    if fault == 'disabled': manager.query_post_forward_reap = False
    if fault == 'done': job.done = True
    if fault == 'cancelled': job.cancelled = True
    if fault == 'pressure': job.command.selection = 'pressure'
    if fault in ('prepare', 'query', 'assembly_wait'): job.stage = fault
    if fault == 'no_capture': manager.runner._kvmem_capture_runtime = None
    if fault == 'no_future': job.future = None
    if fault == 'unready_future': future.ready = False
    if fault == 'unpublished': manager.runner._kvmem_capture_runtime.publication_ready = lambda _: False
    if fault == 'foreign_authority': manager.runner._kvmem_capture_runtime.authorities = {'r': object()}
    manager.step = lambda *a: pytest.fail('Unexpected work/commit')
    manager.poll_query_completions()
    assert future.reads == 0 and not trace
    authority.memory.attach_completion(NS(query=lambda: True))
    assert authority.memory.commit_selection() and authority.memory.close()
