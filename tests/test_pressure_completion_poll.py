"""Ready-only post-forward pressure ACK: no waits, new work or early page reuse."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.core.memory import Phase
from test_recall_experiment import native_authority


def inflight():
    authority=native_authority(count=256,budget=96,sink=1)
    plan=authority.memory.select({},max_tokens=96)
    fence=NS(ready=False)
    def query():return fence.ready
    fence.query=query
    fence.synchronize=lambda:pytest.fail('Post-forward must not synchronize CUDA')
    assembly=NS(event=fence,query=query)
    asm=NS(close=lambda:True)
    future=NS(ready=False,reads=0)
    def result():
        assert future.ready
        future.reads+=1
        return asm,assembly,plan,{},.01
    future.done=lambda:future.ready
    future.result=result
    command=NS(request_id='r',command_id=1,selection='pressure',pressure_stage='prefill',
        groups=((0,NS(history_end=256)),),input_tokens=())
    job=RefreshJob(command,stage='assembly',authority=authority,future=future)
    job.layerwise_single_assembly_done=True
    trace=[];manager=AsyncRefresh.__new__(AsyncRefresh)
    manager.jobs={'r':job};manager.options=NS();manager.maintenance=None
    manager.runner=NS(_kvmem_layerwise_pressure=NS(single_assembly=True,post_forward_reap=True),
        _kvmem_capture_runtime=NS(authorities={'r':authority},publication_ready=lambda ids:True),
        _kvmem_window_adapter=NS(commit=lambda *args:trace.append(('commit',args))),_kvmem_sessions=None)
    manager.emit=lambda job,event,**fields:trace.append(event)
    manager.report=lambda *args:trace.append('report')
    return manager,job,future,fence,authority,trace


def test_post_forward_ack_requires_future_cuda_and_original_transaction_commit():
    from test_connector import pair
    manager,job,future,fence,authority,trace=inflight()
    _,connector=pair()
    connector.pressure_completion_poll=manager.poll_pressure_completions
    connector.worker_jobs={'r':(job.command,job)}
    try:
        connector.get_finished(set())
        assert future.reads==0 and not trace and not job.done
        assert authority.memory.phase is Phase.ASSEMBLING
        future.ready=True
        connector.get_finished(set())
        assert future.reads==1 and not trace and not job.done
        assert authority.memory.phase is Phase.ASSEMBLING
        fence.ready=True
        connector.get_finished(set())
        assert job.done and authority.memory.phase is Phase.READY
        assert connector.worker_completions==[('r',1)]
        assert connector.worker_selected_windows=={'r':96}
        assert trace[0]==('commit',('r',256,96))
        assert 'pressure_completion_reaped' in trace
        connector.get_finished(set())
        assert connector.worker_completions==[('r',1)] and future.reads==1
    finally:
        fence.ready=True
        if not job.done:manager.step(job)
        assert authority.memory.close()


@pytest.mark.parametrize('fault',['cancelled','query','decode','prepare','assembly_wait','not_single',
    'disabled','no_controller','no_capture','no_future','unready_future','missing_single_done','unpublished','foreign_authority'])
def test_post_forward_never_starts_new_work_or_changes_other_paths(fault):
    manager,job,future,fence,authority,trace=inflight()
    future.ready=fence.ready=True
    if fault=='cancelled':job.cancelled=True
    if fault=='query':job.command.selection='query'
    if fault=='decode':job.command.pressure_stage='decode'
    if fault in ('prepare','assembly_wait'):job.stage=fault
    if fault=='not_single':manager.runner._kvmem_layerwise_pressure.single_assembly=False
    if fault=='disabled':manager.runner._kvmem_layerwise_pressure.post_forward_reap=False
    if fault=='no_controller':manager.runner._kvmem_layerwise_pressure=None
    if fault=='no_capture':manager.runner._kvmem_capture_runtime=None
    if fault=='no_future':job.future=None
    if fault=='unready_future':future.ready=False
    if fault=='missing_single_done':job.layerwise_single_assembly_done=False
    if fault=='unpublished':manager.runner._kvmem_capture_runtime.publication_ready=lambda ids:False
    if fault=='foreign_authority':manager.runner._kvmem_capture_runtime.authorities={'r':object()}
    manager.step=lambda *a:pytest.fail('Forbidden post-forward work or commit')
    manager.poll_pressure_completions()
    assert future.reads==0 and not trace and authority.memory.phase is Phase.ASSEMBLING
    authority.memory.attach_completion(NS(query=lambda:True))
    assert authority.memory.commit_selection() and authority.memory.close()
