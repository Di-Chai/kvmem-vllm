from types import SimpleNamespace
import math

import pytest

from vllm_kvmem import _native
from vllm_kvmem.action_attention import ActionHistory,window_segments
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory
from test_core import append,Event


def state(scope='non-thinking',interval=1,decay=.5):
    return ActionHistory(SimpleNamespace(action_scope=scope,action_interval=interval,
                                        action_decay=decay),[10,11,12,13,14])


def test_original_parser_scope_and_speculative_rollback():
    p=_native.ActionParser('tool-call',2,[10,11,12,13],False)
    # Thinking, a fake tool call, think-end, then a real tool call.
    tokens=[12,10,20,13,10,21,22,23,11]
    assert p.prepare(tokens)==(1<<5)|(1<<7)
    assert p.commit(6)==1<<5
    assert p.commit(6)==0
    # Rejected 22/23/close do not advance the parser/interval.
    assert p.state==(False,True,1)
    assert p.prepare([25,26,11])==2
    assert p.commit(3)==2
    assert p.state==(False,False,3)


def test_non_thinking_scope_prompt_state_and_terminal_discard():
    h=state();h.begin('a',[14,12])
    assert h.parser.prepare([20,13,21,22])==12
    h.accept(3,[2,3],[0,32],[[.25,.75],[1.,0.]])
    assert h.samples==1 and h.mass=={0:.25,32:.75}
    assert h.parser.state==(False,False,1)
    # Native parser commits are idempotent, including evidence updates.
    h.accept(3,[2,3],[0,32],[[.25,.75],[1.,0.]])
    assert h.samples==1


def test_stateless_action_scope_uses_exact_native_prompt_and_initializes_once():
    import torch
    from vllm_kvmem.action_attention import ActionRuntime
    runtime=ActionRuntime.__new__(ActionRuntime)
    runtime.runner=SimpleNamespace(_kvmem_sessions=SimpleNamespace(bindings={}),
        req_states=SimpleNamespace(req_id_to_index={'a':0},prompt_len=SimpleNamespace(np=[2]),
            all_token_ids=SimpleNamespace(gpu=torch.tensor([[14,12,13,20,20]]))))
    h=state();runtime.begin_response(h,'a')
    # Prompt ends inside thinking. The generated think-end must not be
    # mistaken for prompt state, and subsequent calls cannot reset the parser.
    assert h.parser.prepare([21,13,22])==4
    h.accept(3,[2],[0],[[1.]])
    runtime.runner.req_states=None
    runtime.begin_response(h,'a')
    assert h.samples==1 and h.parser.state==(False,False,1)


def test_scores_decay_once_per_eligible_response_and_restore_isolated():
    stores=[]
    authority=SimpleNamespace(records={0:SimpleNamespace(history_begin=0),
                                       1:SimpleNamespace(history_begin=32)},
                              memory=SimpleNamespace(set_action_scores=stores.append))
    a,b=state(),state()
    for h in (a,b):h.begin('same',[14,13])
    a.parser.prepare([20]);a.accept(1,[0],[0,32],[[.2,.8]])
    assert a.finish(authority)['samples']==1
    saved=a.snapshot()
    a.begin('next',[14,13]);a.parser.prepare([20]);a.accept(1,[0],[0,32],[[1.,0.]])
    a.finish(authority)
    assert a.scores=={0:1.1,32:.4} and b.scores=={}
    a.restore(saved,authority)
    assert a.scores=={0:.2,32:.8} and a.rounds==1
    a.begin('empty',[14,12]);a.parser.prepare([20]);a.accept(1,[],[],[])
    assert a.finish(authority)['samples']==0
    assert a.scores=={0:.2,32:.8}


@pytest.mark.parametrize('bad',[-1.,math.inf,math.nan])
def test_invalid_mass_cannot_enter_history(bad):
    h=state();h.begin('a',[]);h.parser.prepare([20])
    with pytest.raises(RuntimeError,match='mass'):
        h.accept(1,[0],[0],[[bad]])


def test_dense_frame_preserves_sparse_logical_ids_and_partial_tail():
    records=[(7,1,0,4,0),(3,9,100,4,4),(1,6,124,2,8)]
    starts,counts,keys=window_segments(records,126,10,3,4)
    assert starts==[0,4,8,10,12] and counts==[4,4,2,2,1]
    assert keys==[0,100,124,124,128]
    with pytest.raises(ValueError,match='gap'):
        window_segments([(1,1,100,4,1)],104,5,1,4)


@pytest.mark.parametrize('combine',['quota','score'])
def test_native_fusion_changes_selected_pages_and_preserves_required(combine):
    cfg=KVMemConfig(block_tokens=4,select_budget=12,gen_budget=12,
        repository_blocks=16,sink_blocks=1,recent_blocks=0,
        action_attention=True,action_fraction=.9 if combine=='score' else .5,action_combine=combine)
    m=RequestMemory('a',cfg);spans=append(m,24)
    m.set_action_scores({20:1.})
    plan=m.select({s.handle.slot:float(6-i) for i,s in enumerate(spans)})
    ids=[r.block_id for r in plan.remaps]
    assert ids==[spans[i].handle.slot for i in (0,1,5)]
    m.attach_completion(Event(True));assert m.commit_selection()
    pressure=m.select({s.handle.slot:float(i) for i,s in enumerate(spans)},action=False)
    assert [r.block_id for r in pressure.remaps]==[spans[i].handle.slot for i in (0,4,5)]
    m.attach_completion(Event(True));assert m.commit_selection();del pressure
    # Resetting scores (checkpoint branch restore) removes future feedback.
    m.set_action_scores({})
    plan=m.select({s.handle.slot:float(6-i) for i,s in enumerate(spans)})
    assert [r.block_id for r in plan.remaps]==[spans[i].handle.slot for i in (0,1,2)]
    m.attach_completion(Event(True));assert m.commit_selection();del plan
    assert m.close()


def test_checkpoint_restores_action_scores_with_exact_prefix(tmp_path):
    from test_prompt_checkpoints import saved,ready
    pool,old,a,archive,adapter,_=saved(tmp_path)
    a.action_history=state()
    a.action_history.scores={0:.75,4:.25};a.action_history.rounds=2
    checkpoint=pool.capture(old.namespace,'a',a,list(range(11)),adapter,replace=True)
    assert checkpoint.action_history==({0:.75,4:.25},2)
    a.action_history.scores={0:0,4:1.}
    pool.retire(checkpoint,Event(True));ready(pool,checkpoint)
    pool.claim(checkpoint.namespace,'new',list(range(11)))
    plan=pool.prepare_claim_window(checkpoint)
    assert a.action_history.scores=={0:.75,4:.25}
    a.memory.attach_completion(Event(True));assert a.memory.commit_selection()
    pool.finish_claim(checkpoint,Event(True));del plan
    assert a.close();assert archive.close()


def test_original_layer_policy_excludes_mtp_and_validates_standard_layers():
    assert _native.resolve_action_layers('spread3',[3,7,11,15])==[3,7,15]
    assert _native.resolve_action_layers('last',[3,7,11,15])==[15]
    with pytest.raises(ValueError):_native.resolve_action_layers('2',[3,7,11,15])


@pytest.mark.parametrize('edited',[False,True])
def test_page_boundary_p_replay_keeps_m_evidence_only_for_verified_continuation(tmp_path,edited):
    from test_context_continuation import pm
    pool,p,m,a,archive=pm(tmp_path)
    a.action_history=state()
    p.action_history=({0:1.},1)
    m.action_history=({0:.2,4:.3,8:.5},2)
    tokens=list(range(20))
    if edited:tokens[9]=999  # Same P, different accepted M suffix.
    assert pool.claim(p.namespace,'next',tokens,position=8) is p
    plan=pool.prepare_claim_window(p)
    h=a.action_history
    assert h.snapshot(8)==({0:1.},1)
    assert h.scores==({0:1.} if edited else {0:.2,4:.3,8:.5})
    assert h.rounds==(1 if edited else 2)
    if not edited:assert h.snapshot(12)==({0:.2,4:.3,8:.5},2)
    a.memory.attach_completion(Event(True));assert a.memory.commit_selection()
    pool.finish_claim(p,Event(True));del plan
    assert a.close() and archive.close()


@pytest.mark.parametrize('key,value',[('action_interval',0),('action_interval',True),
    ('action_decay',math.nan),('action_decay',1.1),('action_scope','all'),('action_layers','1,1')])
def test_invalid_runtime_options_fail_before_loading_model(key,value):
    from vllm_kvmem.options import RuntimeOptions
    with pytest.raises(ValueError):RuntimeOptions(**{key:value})
