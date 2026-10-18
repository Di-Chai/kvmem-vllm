import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('external_native_capacity',
    Path(__file__).parents[1]/'scripts/analyze_external_native_capacity.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def geometry(pool=50,state=15):
    return dict(block_tokens=10,total_pool_pages=pool+1,usable_pool_pages=pool,
                recurrent_pages_per_request_lower_bound=state)


def trace(rid,length=100,begin=1.,end=9.):
    return dict(request_id=rid,session=rid,begin=begin,end=end,input_tokens=length)


def histories(rows):
    return dict(verified={r['request_id']:dict(input_tokens=r['input_tokens']) for r in rows},unknown={})


def step(at=5.,n=2,p=0,ms=100):
    return dict(sequence=int(at*100),started_at=at,decode_requests=n,prefill_requests=p,
                prefill_tokens=p*2048,step_ms=ms)


def analyze(rows,steps,geom=None,known=None):
    return m.capacity_steps(rows,steps,known or histories(rows),geom or geometry(),dict(begin=0.,end=10.))


def test_N_shortest_never_all_HTTP_and_15_is_not_18():
    rows=[trace('a'),trace('b'),trace('long',10000)]
    result=analyze(rows,[step()])
    proof=result['steps'][0]
    assert proof['native_attention_pages_lower_bound']==20
    assert proof['native_total_pages_lower_bound']==50
    assert not proof['exceeds_native_pool']  # Equal to usable pool fits this lower bound.
    assert proof['membership']=='unknown_N_shortest_superset'
    with pytest.raises(ValueError,match='18 is a planning maximum'):
        analyze(rows,[step()],geometry(state=18))


def test_counts_and_event_weighting_include_mixed_decode_subset_and_boundary_estimate():
    rows=[trace('a',200,begin=-1.,end=12.),trace('b',200,begin=-1.,end=12.)]
    out=analyze(rows,[step(),step(at=9.95,p=1,ms=200),step(at=10.)])
    pure,mixed=out['groups']['pure'],out['groups']['mixed']
    assert pure['exceeding_steps']==mixed['exceeding_steps']==1
    assert mixed['cuda_event_span_seconds']==.2
    assert mixed['exceeding_clipped_event_estimate_seconds']==pytest.approx(.05)
    assert mixed['actual_gpu_overlap_seconds'] is None
    assert mixed['exceeding_cohort_gpu_overlap_bounds_seconds']==[0.,.2]
    assert out['steps'][1]['native_total_pages_lower_bound']==70  # No prefill charge.


def test_missing_history_in_any_possible_request_makes_bound_unknown_not_larger():
    rows=[trace('short',1),trace('a',200),trace('b',200)]
    known=histories(rows);del known['verified']['short']
    out=analyze(rows,[step()],known=known)
    assert out['groups']['pure']['unknown_steps']==1
    assert out['groups']['pure']['exceeding_steps']==0
    assert out['steps'][0]['reason']=='unverified_full_history_in_possible_set'


@pytest.mark.parametrize('change,reason',[
    (lambda rs:rs.pop(),'insufficient_HTTP_superset'),
    (lambda rs:rs[1].update(session='a'),'ambiguous_or_duplicate_session'),
])
def test_missing_membership_or_duplicate_sessions_are_unknown(change,reason):
    rows=[trace('a'),trace('b')];change(rows)
    assert analyze(rows,[step()])['steps'][0]['reason']==reason


def test_unknown_state_uses_attention_only_and_never_allocates15_by_default():
    out=analyze([trace('a'),trace('b')],[step()],geometry(pool=25,state=None))
    p=out['steps'][0]
    assert p['native_total_pages_lower_bound']==20
    assert p['native_state_pages_lower_bound'] is None and p['bound_kind']=='attention_only'
    assert not p['exceeds_native_pool']


def test_empty_telemetry_is_unknown_not_zero_decode():
    result=analyze([],[])
    assert result['telemetry_available'] is False
    assert result['groups']['pure']['history_step_coverage'] is None


def test_unknown_HTTP_time_extent_cannot_remove_a_short_candidate():
    rows=[trace('short',1),trace('a',200),trace('b',200)]
    rows[0]['end']=None
    result=analyze(rows,[step()])
    assert result['steps'][0]['status']=='unknown'
    assert result['steps'][0]['reason']=='unknown_HTTP_time_extent'


def fixture_and_rows():
    plan=dict(initial_tokens=[1,2,3],turns=[dict(tool_tokens=[],output_tokens=2),
        dict(tool_tokens=[4,5],output_tokens=1)])
    fixture=dict(eos=99,agent_plans=[copy.deepcopy(plan) for _ in range(12)])
    def row(turn,tokens,ids):
        return dict(request_id=f'r{turn}',response_id=f'cmpl-r{turn}',agent=0,session='a',epoch=0,
            turn=turn,kind='warmup' if turn==0 else 'workload',begin=float(turn),end=float(turn)+.5,
            status='completed',input_tokens=len(tokens),input_sha256=m.digest(tokens),output_ids=ids,
            output_sha256=m.digest(ids),usage=dict(prompt_tokens=len(tokens),completion_tokens=len(ids)))
    rows=[row(0,[1,2,3],[7,8]),row(1,[1,2,3,7,8,4,5],[9])]
    pubs=[dict(request_id=r['response_id']+'-0-1234abcd',history_end=r['input_tokens']+len(r['output_ids'])-1,
               checkpoint_tokens=r['input_tokens']+len(r['output_ids'])-1,checkpoint_kind='M') for r in rows]
    return fixture,rows,pubs


def test_full_history_reconstructs_own_outputs_original_tools_and_terminal_M():
    fixture,rows,pubs=fixture_and_rows()
    result=m.trace_histories(fixture,rows,pubs)
    assert not result['unknown'] and result['verified']['r1']['input_tokens']==7
    rows[1]['input_sha256']=m.digest([4,5])
    result=m.trace_histories(fixture,rows,pubs)
    assert result['unknown']['r1']=='full_input_history_mismatch'


def test_missing_CPU_history_is_unknown_even_if_input_is_correct():
    fixture,rows,pubs=fixture_and_rows()
    assert m.trace_histories(fixture,rows,pubs[:1])['unknown']['r1']=='missing_full_CPU_history_or_terminal_M'
    pubs[1]['checkpoint_kind']='P'
    assert 'r1' in m.trace_histories(fixture,rows,pubs)['unknown']


def test_missing_prior_turn_or_natural_stopping_cannot_silently_qualify():
    fixture,rows,_=fixture_and_rows()
    assert m.trace_histories(fixture,rows[1:])['unknown']['r1']=='missing_previous_history'
    rows[0]['generation_mode']='natural'
    result=m.trace_histories(fixture,rows)
    assert result['unknown']['r0']=='not_fixed_output'
    assert 'r1' in result['unknown']


def controls():
    workload=dict(concurrency=12,warmup_turns=2,epoch_turns=16,duration_seconds=300,mode='tools')
    source={f:'same' for f in m.PROTOCOL_FILES};source[m.PROTOCOL_FILES[0]]=next(iter(m.FIXED_WORKLOAD_HASHES))
    manifest=dict(fixture_sha256='fixture',workload=workload,native_source={'version':'0.30.0'},source_sha256=source)
    profile=dict(model='model',max_model_len=262144)
    launch={key:'same' for key in ('--tokenizer','--hf-config-path','--dtype','--kv-cache-dtype','--load-format')}
    launch.update(model='model',**{'--max-model-len':'262144','--max-num-batched-tokens':'16444',
        '--generation-config':'vllm','--speculative-config':'{"method":"mtp","num_speculative_tokens":4}',
        '--compilation-config':'{}','--kernel-config':'{}','--reasoning-config':'{}',
        '--gpu-memory-utilization':'.81','--max-num-seqs':'12'})
    k=dict(arm='kvmem-ready',manifest=manifest,profile=profile,effective=launch,
        window=dict(begin=0.,end=300.,agents=12,warmup_turns=2))
    n=copy.deepcopy(k);n['arm']='native';n['effective'].update({'--gpu-memory-utilization':'.915','--max-num-seqs':'8'})
    return k,n


def test_identical_fixed12_accepts_different_deployment_fraction_and_slots():
    assert m.compare_controls(*controls())['actual_batch_tokens']==16444


@pytest.mark.parametrize('change,match',[
    (lambda n:n['manifest'].update(fixture_sha256='other'),'fixture'),
    (lambda n:n['manifest']['workload'].update(concurrency=8),'fixed12'),
    (lambda n:n['manifest']['workload'].update(generation_mode='natural'),'fixed12'),
    (lambda n:n['effective'].update({'--max-num-batched-tokens':'16424'}),'Effective launch'),
    (lambda n:n['effective'].update({'--speculative-config':'{"method":"mtp","num_speculative_tokens":3}'}),'Effective launch'),
    (lambda n:n['profile'].update(model='other'),'Model/sampling'),
    (lambda n:n['manifest'].update(generation_sampling={'temperature':.6}),'Sampling'),
])
def test_incompatible_comparison_rejected(change,match):
    k,n=controls();change(n)
    with pytest.raises(ValueError,match=match):m.compare_controls(k,n)


def test_native_cache_info_must_be_actual_stable_single_pool():
    def sample(pool):return dict(raw_metrics='vllm:cache_config_info{block_size="1616",num_gpu_blocks="'+str(pool)+
        '",is_attention_free="False",sliding_window="None"} 1\n')
    assert m.cache_geometry([sample(1166)])['total_pool_pages']==1166
    with pytest.raises(ValueError,match='changing'):m.cache_geometry([sample(1166),sample(1193)])
    with pytest.raises(ValueError,match='Missing'):m.cache_geometry([])


def state_contract_inputs(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    interface=tmp_path/'v1/kv_cache_interface.py';interface.parent.mkdir();interface.write_text('# spec')
    pool=tmp_path/'v1/core/block_pool.py';pool.parent.mkdir();pool.write_text('# null page')
    manager=pool.with_name('single_type_kv_cache_manager.py');manager.write_text('# actual state lifetime')
    for constant,path in [('MODEL_CONFIG_SHA',model/'config.json'),('NATIVE_SPEC_SHA',interface),
                          ('NATIVE_POOL_SHA',pool),('NATIVE_MANAGER_SHA',manager)]:
        monkeypatch.setattr(m,constant,m.sha(path))
    groups=[dict(recurrent=True,speculative_blocks=4) for _ in range(3)]+[
        dict(recurrent=False,block_tokens=1616,layers=[str(i) for i in range(17)])]
    capture=tmp_path/'K/server/capture';capture.mkdir(parents=True)
    (capture/'cache-geometry.json').write_text(json.dumps(dict(groups=groups)))
    k=dict(root=tmp_path,arm='K')
    n=dict(profile=dict(model=str(model)),effective={'--speculative-config':'{"method":"mtp","num_speculative_tokens":4}',
        '--kv-cache-dtype':'fp8_e4m3','--max-model-len':'262144'},
        manifest=dict(native_source=dict(version='0.30.0',files={str(interface):m.NATIVE_SPEC_SHA})))
    g=dict(block_tokens=1616,total_pool_pages=1166,labels=dict(cache_dtype='fp8_e4m3',mamba_cache_mode='align',
        kv_cache_max_concurrency=str(1166/181)))
    return k,n,g,pool,manager


def test_state_lower_bound_needs_model_groups_speculation_and_lifetime_evidence(tmp_path,monkeypatch):
    k,n,g,pool,manager=state_contract_inputs(tmp_path,monkeypatch)
    result=m.resolve_state_contract(k,n,copy.deepcopy(g))
    assert result['recurrent_pages_per_request_lower_bound']==15
    assert result['native_declared_state_planning_maximum']==18
    assert result['null_pages']==1 and result['usable_pool_pages']==1165
    manager.write_text('# modified manager')
    result=m.resolve_state_contract(k,n,copy.deepcopy(g))
    assert result['recurrent_pages_per_request_lower_bound'] is None
    assert 'unverified_instantaneous_state_lifetime' in result['state_unknown_reasons']


def test_unknown_model_never_gets_15_and_unknown_null_keeps_native_full_pool(tmp_path,monkeypatch):
    k,n,g,pool,_=state_contract_inputs(tmp_path,monkeypatch)
    (Path(n['profile']['model'])/'config.json').write_text('{"other":"model"}')
    pool.unlink()
    result=m.resolve_state_contract(k,n,g)
    assert result['recurrent_pages_per_request_lower_bound'] is None
    assert result['state_unknown_reasons']==['unreviewed_model_config']
    assert result['null_pages'] is None and result['usable_pool_pages']==1166
