from types import SimpleNamespace as NS
import pytest

from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.session_control import SessionLedger, _Entry
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool, CheckpointPhase, token_bytes
from vllm_kvmem.config import KVMemConfig
from test_authority import setup, payload, drain
from test_core import Event


def warm(query=(48033,49997), window=34816):
    tokens=list(range(52096))
    ledger=SessionLedger(8,128,lambda *a:None,lambda *a:None,lambda *a:None,
        select_budget=32768,gen_budget=16384,checkpoint_policy='pm')
    entry=_Entry(('owner','session'),1,None,phase='idle',history=50048,window=window,
        tokens=token_bytes(tokens,50048),query=query,mandatory=((48033,49997,'query'),))
    ledger.entries[entry.namespace]=entry
    req=NS(session_id='session',cache_salt='owner',request_id='next',prompt_token_ids=tokens,
        num_prompt_tokens=len(tokens),sampling_params=NS(max_tokens=129),
        kv_transfer_params={'kvmem_request_plan':{}})
    return ledger,entry,req


@pytest.mark.parametrize('repeat_query',[False,True])
def test_stable_tool_restores_exact_selection_with_actual_headroom(repeat_query):
    ledger,entry,req=warm()
    if repeat_query:
        req.kv_transfer_params['kvmem_request_plan']['kvmem_query_span']=list(entry.query)
    assert ledger.lookup(req,0)==50048
    p=RequestPlan.parse(req.kv_transfer_params['kvmem_request_plan'],len(req.prompt_token_ids),128,16384)
    assert entry.window==34816 and not entry.pressure_restore
    assert p.query is None and p.prefill_window=='keep_selected'
    assert p.mandatory==entry.mandatory


def test_pressure_due_to_actual_headroom_keeps_frame_until_real_pressure_boundary():
    ledger,entry,req=warm(window=48000)
    assert ledger.lookup(req,0)==50048
    p=RequestPlan.parse(req.kv_transfer_params['kvmem_request_plan'],52096,128,16384)
    assert p.prefill_window=='pressure' and entry.window==48000 and not entry.pressure_restore


def test_new_pin_already_in_exact_frame_preserves_the_warm_epoch():
    ledger,entry,req=warm()
    ledger.memory_config=KVMemConfig(select_budget=32768,gen_budget=16384)
    entry.active_records=tuple((b,128) for b in range(0,32768,128))
    req.kv_transfer_params['kvmem_request_plan']={'kvmem_mandatory_spans':[{'begin':2048,'end':2304,'kind':'file'}]}
    assert ledger.lookup(req,0)==50048 and not ledger.take_commands().evictions


def test_branch_safe_p_leaves_one_complete_template_tail_block():
    p=RequestPlan.parse({},32768,128,16384)
    assert p.prefill_checkpoint(32768)==32512
    assert p.admit(16384,16416,32768,0,49152)==16128


def test_force_old_query_chooses_exact_p_prefix_instead_of_cold_rebuild():
    ledger,entry,req=warm()
    entry.prompt_history,entry.prompt_window,entry.prompt_tokens=48000,32768,token_bytes(req.prompt_token_ids,48000)
    req.kv_transfer_params['kvmem_request_plan']={'kvmem_query_span':list(entry.query),'kvmem_reselect':'force'}
    assert ledger.lookup(req,0)==48000
    assert ledger.take_commands().evictions==[] and entry.window==32768


def test_retroactive_hard_pin_rebuilds_before_using_a_view_that_can_omit_it():
    ledger,entry,req=warm()
    req.kv_transfer_params['kvmem_request_plan']={'kvmem_mandatory_spans':[{'begin':2048,'end':2304,'kind':'file'}]}
    assert ledger.lookup(req,0) is None and entry.phase=='evicting'
    assert ledger.take_commands().evictions


def test_live_header_and_complete_query_share_original_replay_boundary():
    p=RequestPlan.parse({'kvmem_query_span':[49152,50013],'kvmem_replay_span':[49149,50022]},51000,128,16384)
    assert p.replay_begin==49024 and p.replay_end==50022
    assert p.admit(49024,2000,51000,16384,49152)==998
    with pytest.raises(ValueError,match='contain'):
        RequestPlan.parse({'kvmem_query_span':[49152,50013],'kvmem_replay_span':[49160,50022]},51000,128,16384)


def test_original_mandatory_capacity_fit_keeps_edges_and_retrieval_reserve():
    c=KVMemConfig(select_budget=32768,gen_budget=16384)
    p=RequestPlan.parse({'kvmem_mandatory_spans':[{'begin':1024,'end':48000,'kind':'file'}]},50006,128,16384)
    fitted=p.mandatory_fit(48000,c)
    assert fitted['capacity_fitted'] and fitted['retrieval_reserve_blocks']==32
    assert len(fitted['hard_blocks'])==224
    assert set(range(8)).issubset(fitted['hard_blocks'])
    assert {8,9,10,11,371,372,373,374}.issubset(fitted['hard_blocks'])
    assert fitted['soft_retrievable_blocks']>0


def test_historical_hard_pin_overrides_high_scoring_newer_pages(tmp_path):
    a,archive=setup(tmp_path)
    assert a.append(*payload(0,16),history_begin=0,local_rope_begin=0)
    drain(a)
    p=RequestPlan.parse({'kvmem_mandatory_spans':[{'begin':4,'end':8,'kind':'file'}]},20,4,8)
    required=set(p.mandatory_fit(16,a.memory.config)['hard_blocks'])
    handles=[r.handle for r in a.records.values() if r.history_begin//4 in required]
    plan=a.memory.select({r.handle.slot:float(r.history_begin) for r in a.records.values()},tuple(h.slot for h in handles))
    a.memory.attach_completion(Event(True));assert a.memory.commit_selection()
    assert {r[2] for r in a.memory._store.working_records()}=={0,4}
    del plan
    assert a.close() and archive.close()


class Model:
    def capture(self,request_id,position,*,max_bytes,**kwargs):
        if max_bytes<40:
            raise BufferError('model byte budget')
        return NS(bytes=40,position=position)


def pm(tmp_path):
    a,archive=setup(tmp_path)
    assert a.append(*payload(0,8),history_begin=0,local_rope_begin=0)
    drain(a)
    pool=PromptCheckpointPool(2,4096,retain_prompt=True)
    p=pool.capture(('owner','session'),'a',a,list(range(20)),Model())
    assert a.append(*payload(8,4),history_begin=8,local_rope_begin=8)
    drain(a)
    m=pool.capture(p.namespace,'a',a,list(range(20)),Model(),replace=True,terminal=True)
    pool.retire(m,Event(True));assert pool.idle(m)
    return pool,p,m,a,archive


def test_m_keeps_complete_kv_and_promotes_p_after_restore(tmp_path):
    pool,p,m,a,archive=pm(tmp_path)
    assert a.history_end==12 and pool.prompts[p.namespace] is p
    total=pool.payload_bytes
    assert pool.claim(m.namespace,'next',list(range(20)),position=12) is m
    plan=pool.prepare_claim_window(m)
    assert plan.total_window_tokens==12  # Restore beyond K without pressure selection.
    a.memory.attach_completion(Event(True));assert a.memory.commit_selection()
    assert pool.finish_claim(m,Event(True)) is a
    assert pool.entries[p.namespace] is p and p.phase is CheckpointPhase.LIVE and p.request_id=='next'
    assert pool.payload_bytes<total
    pool.drop_live(p);del plan
    assert a.close() and archive.close() and pool.payload_bytes==0


def test_force_p_rewinds_and_reclaims_m_suffix_without_pin_deadlock(tmp_path):
    pool,p,m,a,archive=pm(tmp_path)
    assert pool.claim(p.namespace,'next',list(range(20)),position=8) is p
    plan=pool.prepare_claim_window(p)
    assert a.history_end==8 and len(a.records)==2
    a.memory.attach_completion(Event(True));assert a.memory.commit_selection()
    assert pool.finish_claim(p,Event(True)) is a and not pool.prompts and not pool.entries
    del plan
    assert a.close() and archive.close() and pool.payload_bytes==0


def test_pm_eviction_releases_both_prefixes_and_failed_claim_preserves_m(tmp_path):
    pool,p,m,a,archive=pm(tmp_path)
    assert pool.claim(m.namespace,'bad',[7]*20,position=8) is None
    assert pool.entries[m.namespace] is m and pool.prompts[p.namespace] is p
    assert pool.evict(m)
    assert not pool.entries and not pool.prompts and pool.payload_bytes==0 and archive.close()


def test_terminal_quota_failure_preserves_live_p_atomically(tmp_path):
    a,archive=setup(tmp_path)
    assert a.append(*payload(0,8),history_begin=0,local_rope_begin=0)
    drain(a)
    pool=PromptCheckpointPool(1,4096,retain_prompt=True)
    p=pool.capture(('o','s'),'a',a,list(range(20)),Model())
    pool.max_payload_bytes=pool.payload_bytes+40
    assert a.append(*payload(8,4),history_begin=8,local_rope_begin=8)
    drain(a)
    with pytest.raises(BufferError):pool.capture(p.namespace,'a',a,list(range(20)),Model(),replace=True,terminal=True)
    assert pool.entries[p.namespace] is p and not pool.prompts and p.position==8
    pool.drop_live(p)
    assert a.close() and archive.close()
