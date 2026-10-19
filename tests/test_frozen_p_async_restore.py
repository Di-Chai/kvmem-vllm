"""Frozen input checkpoints survive async imports without answer feedback."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.session_control import SessionLedger, SessionNotice, SessionBinding, SessionCommands, SessionEviction
from vllm_kvmem.async_restore import RestoreContext, step_restore
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.connector import RestoreCommand
from vllm_kvmem.prompt_checkpoints import CheckpointPhase
from test_core import Event
from test_session_control import request, first_turn
from test_session_worker import worker_fixture, wait_retirement
from test_authority import payload, drain
from test_graph_lifecycle import attach
from test_prefill_attention import options
from vllm_kvmem.action_attention import ActionHistory
from vllm_kvmem.block_graph import GraphSample


def settings(**changes):
    values=dict(request_planning=True, session_capacity=3, session_checkpoint_bytes=4096,
        archive_backend='cpu', cpu_read_cache_bytes=1<<30, async_archive_retirement=True,
        async_refresh=True, maintenance_max_steps=8, async_restore=True, session_freeze_prompt=True)
    return RuntimeOptions(**(values|changes))


def test_only_explicit_frozen_p_mode_extends_async_restore_contract():
    actual=settings()
    assert actual.async_restore and actual.session_freeze_prompt
    assert actual.session_checkpoint_policy=='sealed'
    assert not actual.session_capture_terminal and not actual.session_capture_decode and not actual.query_history


@pytest.mark.parametrize('change',[
    dict(session_freeze_prompt=False), dict(session_capture_terminal=True),
    dict(session_capture_decode=True), dict(query_history=True),
    dict(async_refresh=False), dict(maintenance_max_steps=0),
    dict(archive_backend='nvme'), dict(repository_gc=True),
])
def test_frozen_p_cannot_bypass_restore_or_independence_requirements(change):
    with pytest.raises(ValueError):settings(**change)


def controller(frozen=True):
    return SessionLedger(2,4,lambda *a:None,lambda *a:None,lambda *a:None,freeze_prompt=frozen)


@pytest.mark.parametrize('frozen',[False,True])
def test_retained_frozen_p_avoids_recapture_and_remains_reusable_after_cancel(frozen):
    control=controller(frozen);binding,_=first_turn(control)
    for index in range(3):
        # A fresh independent question changes only the suffix after exact P.
        req=request(f'next{index}',tokens=list(range(16))+[100+index]*9)
        assert control.lookup(req,0)==16
        control.allocated(req,16)
        control.take_commands()
        control.consume(SessionNotice('restored',binding.namespace,binding.generation,
            req.request_id,16,12,retained=True))
        control.capture_before_final_prefill(req,9)
        commands=control.take_commands()
        assert len(commands.captures)==(0 if frozen else 1)
        if not frozen:
            control.consume(SessionNotice('captured',binding.namespace,binding.generation,req.request_id,16))
        # No public token need run: cancellation still retires the original P.
        control.finished(req)
        control.consume(SessionNotice('retired',binding.namespace,binding.generation,
            req.request_id,16,12,ready_records=2))
        assert control.entries[binding.namespace].phase=='idle'
        assert not control.requests


def test_changed_prefix_cannot_silently_reuse_retained_frozen_p():
    control=controller();binding,_=first_turn(control)
    req=request('new');assert control.lookup(req,0)==16
    control.allocated(req,16);control.take_commands()
    control.consume(SessionNotice('restored',binding.namespace,binding.generation,'new',16,12,retained=True))
    req.prompt_token_ids[0]=999
    with pytest.raises(ValueError,match='frozen prompt differs'):
        control.capture_before_final_prefill(req,9)
    assert not control.commands.captures and not control.requests['new'].candidate_position


def test_async_p_fences_preserve_one_model_and_reset_graph_action_between_questions(tmp_path):
    worker,binding,authority,archive,_=worker_fixture(tmp_path)
    worker.options.session_freeze_prompt=worker.options.async_restore=True
    worker.options.verify_archive_on_finish=False
    worker.pool.max_payload_bytes=1<<20
    authority.action_history=ActionHistory(options(),[10,11,12,13,14])
    authority.action_history.prefill.accept(8,4,4,[3],[0],[0],0,[[1.]])
    graph=attach(authority,mode='collect')
    graph.prepare(0,8,[GraphSample(4,(0,4),(.5,.5),'prefill')]).commit()
    checkpoint=worker.pool.capture(binding.namespace,'a',authority,list(range(20)),worker.model_adapter,replace=True)
    model=checkpoint.model;window=checkpoint.window.records
    feedback=checkpoint.action_history;rows=graph.rows;partial=graph.partial
    current='a'
    for index in range(2):
        if not index:
            # First question has accepted answer state; the second cancellation
            # happens immediately after restore and must not lose the same P.
            assert authority.append(*payload(8,4),history_begin=8,local_rope_begin=8)
            drain(authority)
            graph.prepare(8,12,[GraphSample(8,(0,8),(.75,.25),'output')]).commit()
            authority.action_history.scores={0:100.}
            authority.action_history.prefill.values.clear()
        assert worker.detach(current,authority,{'request_id':current})
        del worker.runtime.authorities[current]
        wait_retirement(worker,current);worker.notices()
        rid=f'question{index}'
        state=worker.runner.req_states
        state.req_id_to_index[rid]=2
        worker.runner._kvmem_request_generations[rid]=index+3
        state.prefill_len.np[2]=20
        worker.receive(SessionCommands([SessionBinding(binding.namespace,1,rid,tuple(range(20)),8,8)],[]))
        assert worker.pool.claim(binding.namespace,rid,list(range(20))) is checkpoint
        plan=worker.pool.prepare_claim_window(checkpoint,wait=False)
        assert plan is not None
        prepared=worker.model_adapter.prepare_restore(model,rid)
        worker.model_adapter.submit_restore(prepared)
        fence=Event();assembly=NS(uploaded_bytes=128,host_prepare_seconds=.001,query=fence.query)
        future=Future();future.set_result((assembly,None,fence))
        worker.runner._kvmem_sessions=worker
        worker.runner._kvmem_probe_output=tmp_path
        worker.runner._kvmem_restore_allocations={rid:object()}
        command=RestoreCommand(rid,-index-1,1,8,8,((1,2),))
        job=RefreshJob(command,stage='restore_copy',dispatched=True,future=future,
            authority=authority,checkpoint=checkpoint,model_bytes=model.bytes,
            restore_context=RestoreContext(checkpoint,plan,NS(close=lambda:True),prepared,None))
        manager=AsyncRefresh.__new__(AsyncRefresh)
        manager.runner=worker.runner;manager.options=NS(async_restore=True)
        manager.jobs={rid:job};manager.emit=lambda *a,**kw:None
        manager.maintenance=NS(observe=lambda *a:None,release=lambda *a:None)
        authority._refresh_background=True
        assert not step_restore(manager,job)
        assert rid in worker.imports and checkpoint.phase is CheckpointPhase.CLAIMED
        worker.before_attention(NS(req_ids=['peer']))
        with pytest.raises(RuntimeError,match='Unready checkpoint'):
            worker.before_attention(NS(req_ids=[rid]))
        fence.done=True
        assert step_restore(manager,job) and not worker.imports
        assert checkpoint.phase is CheckpointPhase.LIVE and checkpoint.request_id==rid
        assert checkpoint.model is model and tuple(authority.memory._store.working_records())==window
        assert authority.action_history.snapshot()==feedback
        assert graph.rows==rows and graph.partial==partial and graph.accepted_through==8
        assert all(torch.equal(value,model.recurrent[name])
                   for name,value in worker.model_adapter._recurrent_views(rid).items())
        assert worker.notices()[0].retained
        current=rid
        del plan
    assert worker.detach(current,authority,{'request_id':current})
    del worker.runtime.authorities[current]
    wait_retirement(worker,current);worker.notices()
    worker.evict([SessionEviction(binding.namespace,1)]);worker.notices()
    assert worker.close() and archive.close() and graph.budget.used==0
