"""Completed-event prefill math and frozen pressure-only execution controls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_query_stage_efficiency_driver import prepared_inputs, native_inputs, prefill_state_inputs


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_pressure_stage_efficiency.py'
    spec=importlib.util.spec_from_file_location('pressure_stage_driver_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_freeze_keeps_query_fixed_and_bundles_its_driver(prepared_inputs,tmp_path):
    _,args=prepared_inputs;d=module();root=tmp_path/'pressure'
    (tmp_path/'pyproject.toml').write_text('[project]\nname="fixture"\n')
    p=d.prepare(root,**args)
    assert p['workers']==1 and not p['formal_resume'] and not p['scheduled_monitor']
    assert not p['measurement_instrumentation_added']
    assert [t['variant'] for t in p['trials']]==['baseline','candidate','candidate','baseline']
    for t in p['trials']:
        assert all(t['environment'][k]==v for k,v in d.QUERY_ENV.items())
        assert d.base.read(root/t['name']/'profile.json')==d.base.read(root/'01-baseline/profile.json')
    assert p['files'][d.HELPER_NAME]==d.base.sha(root/d.HELPER_NAME)
    result=subprocess.run([sys.executable,'-B',str(root/'launch.py'),'verify','--root',str(root)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    (root/d.HELPER_NAME).write_text('tampered\n')
    with pytest.raises(ValueError,match='Frozen input/control changed'):d.verify(root)


@pytest.mark.parametrize('flags',[
    {'KVMEM_SELECT_BUDGET':'16384'}, {'KVMEM_LAYERWISE_QUERY_SCORE_BACKEND':'legacy'},
    {'KVMEM_LAYERWISE_ROLLING_DECODE':'0'}, {'KVMEM_LAYERWISE_PRESSURE_UNKNOWN':'1'},
    {'KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING':'yes'},
])
def test_only_reviewed_pressure_execution_flags_may_vary(flags):
    with pytest.raises(ValueError,match='whitelisted'):module().variant_order({'bad':flags},('bad',))


def test_new_candidate_flags_and_repeated_order_are_explicit():
    d=module();flags={'KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING':'1',
                     'KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION':'1',
                     'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER':'1'}
    trials=d.variant_order({'a':{},'b':flags},('a','b','b','a'))
    assert [t['name'] for t in trials]==['01-a','02-b','03-b','04-a']
    assert trials[1]['environment']=={**d.QUERY_ENV,**flags}


def test_batched_acceptance_is_explicit_and_prefill_binary_is_frozen(prepared_inputs,tmp_path):
    _,args=prepared_inputs;d=module();root=tmp_path/'batch-accept'
    binary,_,_=prefill_state_inputs(prepared_inputs,tmp_path)
    (tmp_path/'pyproject.toml').write_text('[project]\nname="fixture"\n')
    variants={name:{'KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT':str(value)}
              for name,value in [('baseline',0),('candidate',1)]}
    plan=d.prepare(root,**args,prefill_state_binary=binary,
        overlays=('csrc/buffered_prefill_state.cpp',),variants=variants,
        order=('baseline','candidate'))
    assert plan['prefill_state_extension']['sha256']==d.base.sha(binary)
    assert [t['environment']['KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT']
            for t in plan['trials']]==['0','1']
    result=subprocess.run([sys.executable,'-B',str(root/'launch.py'),'verify','--root',str(root)],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('workers',['1','2'])
def test_paired_gather_workers_explicit_allowed_scope(workers):
    d=module();flags={'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER':'1',
                     'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':workers}
    trials=d.variant_order({'candidate':flags},('candidate',))
    assert trials[0]['environment']=={**d.QUERY_ENV,**flags}
    assert all(trials[0]['environment'][k]==v for k,v in d.QUERY_ENV.items())


@pytest.mark.parametrize('workers',['0','3','02','two',2,None])
def test_invalid_paired_worker_value_rejected(workers):
    flags={'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER':'1',
           'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':workers}
    with pytest.raises(ValueError,match='whitelisted'):
        module().variant_order({'bad':flags},('bad',))


@pytest.mark.parametrize('paired',[None,'0'])
def test_two_workers_require_paired_gather(paired):
    flags={'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':'2'}
    if paired is not None:flags['KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER']=paired
    with pytest.raises(ValueError,match='explicit paired gather'):
        module().variant_order({'bad':flags},('bad',))


def test_explicit_one_worker_does_not_enable_gather():
    flags={'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':'1'}
    assert module().variant_order({'base':flags},('base',))[0]['environment']=={
        **module().QUERY_ENV,**flags}


def observations():
    result=dict(id='cold',prefix_reused=False,
        attention=dict(request_id='r',command_id=2,history_begin=36))
    def step(i,start,end):
        return dict(sequence=i,call_index=i,started_at=start,submitted_at=start+1,
            prefill_tokens=16,decode_requests=0,step_ms=1000.,model_ms=1000.,
            previous_sequence=i-1 if i>1 else None,previous_call_index=i-1 if i>1 else None,
            previous_end_to_model_ms=500. if i>1 else None,
            batch_geometry=[dict(request_id='r',prefill=True,query_tokens=16,logical_seq_len=end)])
    steps=[step(1,10.,20),step(2,12.,36)]
    refresh=[dict(request_id='r',command_id=1,stage='suspended',at=11.1),
             dict(request_id='r',command_id=1,stage='ready',at=11.3),
             dict(request_id='r',command_id=2,stage='suspended',at=14.)]
    return result,steps,refresh


def test_completed_event_chain_includes_gap_and_excludes_untimed_initial_tokens():
    d=module();r,s,e=observations();m=d.history_interval(r,s,e)
    assert m['executed_prefill_tokens']==32 and m['unmeasured_initial_tokens']==4
    assert m['confirmed_gpu_interval_seconds']==2.5
    assert m['host_envelope_seconds']==4.
    assert m['confirmed_gpu_interval_tokens_per_second']==12.8
    assert m['host_envelope_tokens_per_second']==8.
    assert m['pressure_timings']['suspended_to_ready']['mean_seconds']==pytest.approx(.2)


def test_readiness_polls_may_repeat_but_true_transition_timestamps_remain_unique():
    d=module();r,s,e=observations()
    baseline=d.history_interval(r,s,e)
    for stage in ('pipeline_handoff_prime','pipeline_handoff_attempt'):
        e.extend(dict(request_id='r',command_id=1,stage=stage,at=t) for t in (11.15,11.2))
    assert d.history_interval(r,s,e)==baseline
    e.append(dict(request_id='r',command_id=1,stage='suspended',at=11.2))
    with pytest.raises(ValueError,match='Repeated pressure stage'):d.history_interval(r,s,e)


@pytest.mark.parametrize('fault',['sequence','call','hole','decode','foreign','late',
    'nan','negative','missing_gap','history','query','second_cold_missing_start'])
def test_invalid_or_incomplete_history_is_not_reported_as_throughput(fault):
    d=module();r,s,e=observations()
    if fault=='sequence':s[1]['previous_sequence']=0
    elif fault=='call':s[1]['call_index']=3
    elif fault=='hole':s[1]['batch_geometry'][0]['logical_seq_len']+=1
    elif fault=='decode':s[1]['decode_requests']=1
    elif fault=='foreign':s[1]['batch_geometry'].append(dict(s[1]['batch_geometry'][0],request_id='other'))
    elif fault=='late':s[1]['submitted_at']=15.
    elif fault=='nan':s[1]['model_ms']=float('nan')
    elif fault=='negative':s[1]['previous_end_to_model_ms']=-1.
    elif fault=='missing_gap':s[1]['previous_end_to_model_ms']=None
    elif fault=='history':r['attention']['history_begin']+=16
    elif fault=='query':e.append(dict(request_id='r',command_id=2,stage='query',at=13.5))
    elif fault=='second_cold_missing_start':
        earlier=deepcopy(s[0]);earlier['batch_geometry'][0]['request_id']='earlier';s.insert(0,earlier)
    with pytest.raises(ValueError):d.history_interval(r,s,e)


def test_warm_restored_history_is_excluded_and_hidden_reprefill_is_rejected():
    d=module();r,s,e=observations();r['prefix_reused']=True
    with pytest.raises(ValueError,match='Warm prefix'):d.history_interval(r,s,e)
    m=d.history_interval(r,[],e)
    assert not m['included'] and m['executed_prefill_tokens']==0 and m['history_tokens']==36


def stored_measurement(tmp_path):
    d=module();r,s,e=observations();arm=tmp_path/'arm'
    case=arm/'run/answers/cases/cold';case.mkdir(parents=True)
    (case/'result.json').write_text(json.dumps(r))
    cap=arm/'run/server/capture';cap.mkdir(parents=True)
    for name,rows in [('model_step_latency',s),('async_refresh',e)]:
        (cap/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    counts=dict(public_prefill_tokens=36,public_batches=3)
    work=dict(counter_id='counter',completed_fence=dict(sequence=1,recorded_started_at=15.,
        recorded_at=15.1,completion_upper_bound=16.),pending_fence=None,
        submitted=counts,completed=dict(counts),all_submitted_completed=True)
    (cap/'public_work.json').write_text(json.dumps(work))
    return d,arm,cap,work


def test_final_fence_confirms_counter_coverage_but_does_not_define_phase_endpoint(tmp_path):
    d,arm,cap,work=stored_measurement(tmp_path)
    m=d.measure(arm,['cold'])
    assert m['counter_id']=='counter' and m['confirmed_gpu_interval_seconds']==2.5
    assert m['host_envelope_seconds']==4. and m['cold_cases']==1


@pytest.mark.parametrize('fault',['pending','counter','fence','unconfirmed','too_few','missing_first','cold_pending'])
def test_missing_completion_evidence_fails_closed(tmp_path,fault):
    d,arm,cap,w=stored_measurement(tmp_path)
    if fault=='pending':w['pending_fence']={'sequence':2}
    elif fault=='counter':w['counter_id']=None
    elif fault=='fence':w['completed_fence']=None
    elif fault=='unconfirmed':w['all_submitted_completed']=False
    elif fault=='too_few':w['submitted']['public_prefill_tokens']=w['completed']['public_prefill_tokens']=20
    elif fault=='missing_first':
        p=cap/'model_step_latency.jsonl';p.write_text(p.read_text().splitlines()[1]+'\n')
    elif fault=='cold_pending':
        w['completed_fence'].update(recorded_started_at=12.,recorded_at=12.1)
        w['pending_fence']=dict(sequence=2,recorded_at=13.2)
    (cap/'public_work.json').write_text(json.dumps(w))
    with pytest.raises(ValueError):d.measure(arm,['cold'])


def test_unconfirmed_cleanup_does_not_invalidate_earlier_completed_cold_history(tmp_path):
    d,arm,cap,w=stored_measurement(tmp_path)
    w['submitted']=dict(public_prefill_tokens=37,public_batches=4)
    w['not_yet_confirmed']=dict(public_prefill_tokens=1,public_batches=1)
    w['pending_fence']=dict(sequence=2,recorded_at=18.,totals=dict(w['submitted']))
    w['all_submitted_completed']=False
    (cap/'public_work.json').write_text(json.dumps(w))
    m=d.measure(arm,['cold'])
    assert m['confirmed_gpu_interval_seconds']==2.5 and m['executed_prefill_tokens']==32
    assert m['final_unconfirmed_public_work']==w['not_yet_confirmed']
    assert m['final_pending_fence']==w['pending_fence']
    assert not m['all_final_public_work_confirmed'] and m['completed_fence_covers_cold_history']
    assert m['last_cold_step_submitted_at']==13.


def test_exact_store_pair_and_isolated_binary_hashes_are_frozen(prepared_inputs,tmp_path):
    _,args=prepared_inputs;d=module()
    binary,original,_=native_inputs(prepared_inputs,tmp_path)
    runtime=args['source_arm']/'runtime-v1';source=args['overlay_source']
    for name in d.STORE_SOURCES:
        for root,text in ((runtime,'// old\n'),(source,'// new\n')):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    (args['source_arm']/'runtime-proof.json').write_text(json.dumps(d.base.tree(runtime)))
    root=tmp_path/'native-pressure'
    p=d.prepare(root,native_binary=binary,overlays=('csrc/bindings.cpp',*sorted(d.STORE_SOURCES)),**args)
    assert set(p['native_extension']['declared_source_sha256'])=={'csrc/bindings.cpp',*d.STORE_SOURCES}
    assert set(p['changed_runtime_files'])=={'csrc/bindings.cpp',*d.STORE_SOURCES}
    for name in d.STORE_SOURCES:
        assert (runtime/name).read_text()=='// old\n'
        assert (root/'01-baseline/runtime-v1'/name).read_text()=='// new\n'
    with pytest.raises(ValueError,match='both exact Store'):
        d.patch_runtime(runtime,source,(next(iter(d.STORE_SOURCES)),),tmp_path/'partial.patch')
    with pytest.raises(ValueError,match='explicit inference'):
        d.patch_runtime(runtime,source,('third_party/qw3/src/other.cpp',),tmp_path/'other.patch')
    with pytest.raises(ValueError,match='explicitly supplied binary'):
        d.freeze_native(runtime,None,list(d.STORE_SOURCES))
