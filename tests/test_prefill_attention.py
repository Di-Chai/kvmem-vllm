from types import SimpleNamespace
import math
import pytest

from vllm_kvmem.prefill_attention import PrefillHistory, select_pressure, selection_report
from vllm_kvmem.action_attention import ActionHistory
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory
from test_core import append, Event


def options(**kwargs):
    return SimpleNamespace(memory=KVMemConfig(block_tokens=4,select_budget=12,gen_budget=12,
        repository_blocks=32,sink_blocks=1,recent_blocks=0,action_attention=True,action_fraction=.5),
        **dict(dict(prefill_attention=True,action_pressure=True,prefill_attention_interval=8,
        prefill_attention_stride=4,prefill_attention_half_life=8,
        action_scope='non-thinking',action_interval=1,action_decay=.5),**kwargs))


def test_sparse_sampling_is_anchored_at_chunk_end_and_forced_near_capacity():
    h=PrefillHistory(options(prefill_attention_interval=100))
    assert h.rows(0,10,0)==[1,5,9]
    h.end=10
    assert h.rows(10,4,10)==[]
    assert h.rows(18,4,18)==[3]
    assert h.rows(18,4,18,replay=True)==[]
    assert h.rows(0,8,0)==[]
    # An atomic explicit Query can be larger than the ordinary prefill chunk.
    assert h.rows(10,300,0)==list(range(3,300,4))


def test_query_template_suffix_is_excluded_even_in_a_later_prefill_batch():
    from vllm_kvmem.prefill_attention import exclude_query_replay
    h=ActionHistory(options(),[10,11,12,13,14])
    runner=SimpleNamespace(req_states=SimpleNamespace(req_id_to_index={'r':0},
        prompt_len=SimpleNamespace(np=[113])))
    exclude_query_replay(runner,SimpleNamespace(action_history=h),'r',96,108,True)
    assert h.prefill_replay==(96,113)


def test_mass_exposure_causality_duplicate_page_identity_and_decay():
    h=PrefillHistory(options())
    # A partial selected page and appended suffix share key 100. Exposure
    # counts it once, whereas a newly visible key gets only its causal rows.
    assert h.accept(8,8,8,[1,5],[0,2,4],[100,100,200],0,[[1.,0,0],[.2,.3,.5]])
    assert h.values=={100:[.75,1.],200:[.25,.5]}
    assert h.scores()=={100:.75,200:.5}
    assert h.accept(16,8,8,[7],[0,4],[100,300],0,[[.5,.5]])
    assert h.values=={100:[.875,1.5],300:[.5,1.]}
    assert 200 not in h.values  # State is bounded to the observed window.
    assert h.samples==3 and h.chunks==2
    snapshot=h.snapshot()
    assert not h.accept(16,8,8,[7],[0],[100],0,[[1.]])
    assert not h.accept(24,8,4,[7],[0],[100],0,[[1.]])
    assert h.snapshot()==snapshot


def test_rewind_invalidates_only_future_input_feedback_and_snapshots_do_not_alias():
    h=PrefillHistory(options())
    h.accept(8,8,8,[7],[0],[0],0,[[1.]])
    saved=h.snapshot();h.values[0][0]=.1
    h.restore(saved,8);assert h.scores()=={0:1.}
    h.restore(saved,7);assert h.values=={} and h.end==h.samples==0


def test_cold_semantic_prefill_quota_and_output_precedence_use_native_selector():
    cfg=options().memory;m=RequestMemory('a',cfg);spans=append(m,24)
    scores={s.handle.slot:float(6-i) for i,s in enumerate(spans)}
    def pick(prefill):
        plan=m.select(scores,prefill_scores=prefill)
        ids=[r.block_id for r in plan.remaps]
        m.attach_completion(Event(True));assert m.commit_selection()
        return ids
    assert pick({20:1.})==[spans[i].handle.slot for i in (0,1,5)]
    m.set_action_scores({16:1.})
    assert pick({20:1000.})==[spans[i].handle.slot for i in (0,1,4)]
    assert m.close()


def test_pressure_uses_resident_pages_not_cold_high_scores_and_preserves_sink():
    o=options();m=RequestMemory('a',o.memory);spans=append(m,24)
    plan=m.select({s.handle.slot:float(i) for i,s in enumerate(spans)},action=False)
    m.attach_completion(Event(True));assert m.commit_selection();del plan
    append(m,8,local=12)
    h=ActionHistory(o,[10,11,12,13,14]);h.prefill.values={4:[999.,1.],16:[1.,1.]}
    a=SimpleNamespace(memory=m,history_end=m.history_end,action_history=h)
    plan=select_pressure(a)
    selected=[r.block_id for r in plan.remaps]
    assert spans[0].handle.slot in selected and spans[4].handle.slot in selected
    assert spans[1].handle.slot not in selected
    m.attach_completion(Event(True));assert m.commit_selection();del plan
    assert m.close()


def test_generation_reserve_uses_original_pressure_ranking_at_smaller_budget():
    o=options();m=RequestMemory('a',o.memory);spans=append(m,24)
    h=ActionHistory(o,[10,11,12,13,14]);h.prefill.values={16:[1.,1.]}
    a=SimpleNamespace(memory=m,history_end=m.history_end,action_history=h)
    plan=select_pressure(a,max_tokens=8)
    assert plan.total_window_tokens==8
    assert [r.block_id for r in plan.remaps]==[spans[i].handle.slot for i in (0,4)]
    m.attach_completion(Event(True));assert m.commit_selection();del plan
    # The reduction is scoped to this selection, not the next Query's budget.
    plan=m.select({s.handle.slot:float(i) for i,s in enumerate(spans)})
    assert plan.total_window_tokens==12
    m.attach_completion(Event(True));assert m.commit_selection();del plan
    assert m.close()


def test_prefill_scores_survive_checkpoint_and_future_p_replay_does_not_publish_them(tmp_path):
    from test_prompt_checkpoints import saved,ready
    pool,old,a,archive,adapter,_=saved(tmp_path)
    a.action_history=ActionHistory(options(),[10,11,12,13,14])
    a.action_history.prefill.accept(8,8,8,[7],[0],[0],0,[[1.]])
    checkpoint=pool.capture(old.namespace,'a',a,list(range(11)),adapter,replace=True)
    saved=checkpoint.action_history
    a.action_history.prefill.values.clear()
    pool.retire(checkpoint,Event(True));ready(pool,checkpoint)
    pool.claim(checkpoint.namespace,'new',list(range(11)))
    plan=pool.prepare_claim_window(checkpoint)
    assert a.action_history.prefill.scores()=={0:1.}
    assert a.action_history.snapshot()==saved
    m=a.memory;m.attach_completion(Event(True));assert m.commit_selection()
    pool.finish_claim(checkpoint,Event(True));del plan
    assert a.close() and archive.close()


@pytest.mark.parametrize('key,value',[('prefill_attention_interval',0),('prefill_attention_stride',-1),
    ('prefill_attention_half_life',math.nan),('prefill_attention_half_life',0),('action_pressure',1)])
def test_prefill_option_validation(key,value):
    from vllm_kvmem.options import RuntimeOptions
    with pytest.raises(ValueError):RuntimeOptions(**{key:value})


@pytest.mark.parametrize('changes', [dict(request_planning=False), dict(session_capacity=0),
    dict(session_capture_terminal=True), dict(session_capture_decode=True), dict(query_history=True)])
def test_frozen_prompt_rejects_cross_question_state(changes):
    from vllm_kvmem.options import RuntimeOptions
    settings = dict(session_freeze_prompt=True, request_planning=True, session_capacity=3,
                    session_checkpoint_bytes=4096)
    settings.update(changes)
    with pytest.raises(ValueError, match='Frozen prompt'):
        RuntimeOptions(**settings)
