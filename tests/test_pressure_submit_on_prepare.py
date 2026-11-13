from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.async_refresh import AsyncRefresh, RefreshJob


def setup(monkeypatch):
    manager = AsyncRefresh.__new__(AsyncRefresh)
    trace = []
    manager.runner = NS(_kvmem_layerwise_pressure=NS(submit_on_prepare=True,single_assembly=True))
    manager.options = NS()
    manager.stream = object()
    manager._restore_state = lambda job:trace.append('state-restored')
    manager.emit = lambda job,stage,**kw:trace.append(stage)
    fence = NS(record=lambda:trace.append('fence-recorded'))
    monkeypatch.setattr(torch.cuda,'Event',lambda:fence)
    job = RefreshJob(NS(request_id='r',command_id=1,selection='pressure',pressure_stage='prefill',guided_reason=None),authority=NS())
    manager.jobs = {'r':job}
    manager.maintenance = NS(acquire=lambda j:True)
    def submit(fn,actual,query,ready):
        assert actual is job and query is None and ready is fence
        assert job.stage == 'assembly' and job.authority._refresh_background
        trace.append('executor-submit')
        return object()
    manager.executor = NS(submit=submit)
    return manager,job,trace


def test_pressure_prepare_enqueues_once_without_executing_assembly(monkeypatch):
    manager,job,trace = setup(monkeypatch)
    manager._assemble = lambda *args:pytest.fail('Assembly must stay on the producer')
    manager._submit(job)
    assert trace == ['state-restored','fence-recorded','assembly_queued','executor-submit','pressure_assembly_submitted_inline']
    assert job.future is not None and job.assembly_inputs is None


@pytest.mark.parametrize('blocked',['lane','credit'])
def test_inline_submission_preserves_existing_lane_and_transfer_admission(monkeypatch,blocked):
    manager,job,trace = setup(monkeypatch)
    if blocked=='lane':manager.jobs['other']=NS(future=object(),done=False,command=NS(resident=False))
    else:manager.maintenance.acquire=lambda job:False
    manager._submit(job)
    assert job.stage=='assembly_wait' and job.future is None and job.assembly_inputs is not None
    assert trace == ['state-restored','fence-recorded','assembly_queued']


@pytest.mark.parametrize('fault',['off','no_controller','not_single','cancelled','query','decode','guided'])
def test_other_refresh_paths_do_not_start_inline_work(monkeypatch,fault):
    manager,job,trace = setup(monkeypatch)
    if fault=='off':manager.runner._kvmem_layerwise_pressure.submit_on_prepare=False
    if fault=='no_controller':manager.runner._kvmem_layerwise_pressure=None
    if fault=='not_single':manager.runner._kvmem_layerwise_pressure.single_assembly=False
    if fault=='cancelled':job.cancelled=True
    if fault=='query':job.command.selection='query'
    if fault=='decode':job.command.pressure_stage='decode'
    if fault=='guided':job.command.guided_reason='prefill-pressure'
    manager.step=lambda job:pytest.fail('Unexpected inline step')
    manager._submit(job)
    assert trace==['state-restored','fence-recorded','assembly_queued']
