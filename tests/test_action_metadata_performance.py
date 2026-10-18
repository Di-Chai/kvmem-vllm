"""CPU-only gates for the standalone, exactly qualified natural performance arm."""
import asyncio
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('action_upload_performance_test',
    Path(__file__).parents[1]/'scripts/benchmark_action_metadata_upload.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def profile():
    return dict(port=18530,max_num_seqs=8,max_model_len=262144,gpu_memory_utilization=.81,
        draft_method='mtp',draft_tokens=4,execution_mode='compiled_graph',verify_session_restore=False,
        max_num_batched_tokens=4224,query_replay_budget=16384,kvmem=dict(select_budget=65536,
            gen_budget=32768,rolling_active_window=True,archive_backend='cpu',cpu_read_cache_bytes=100_000_000_000,
            action_attention=True,prefill_attention=True,action_pressure=True))


def streams(root):
    plans = [dict(initial_tokens=[i+1,2,3,4],turns=[dict(tool_tokens=[] if t == 0 else [80+i,t])
                                                for t in range(3)]) for i in range(2)]
    fixture = dict(agent_plans=plans,eos=0)
    rows=[]
    for i,plan in enumerate(plans):
        tokens=list(plan['initial_tokens'])
        for turn,count in enumerate((1024,4096,1024)):
            if turn:tokens.extend(plan['turns'][turn]['tool_tokens'])
            output=[100+i]*count
            rows.append(dict(agent=i,turn=turn,epoch=0,begin=10*turn+i,kind='workload',session=f'u{i}',
                request_id=f'{i}-{turn}',status='completed',input_tokens=len(tokens),
                input_sha256=bench.ready.work.digest(tokens),output_ids=output,
                output_sha256=bench.ready.work.digest(output),usage=dict(completion_tokens=count)))
            tokens.extend(output)
        rows.append(dict(agent=i,turn=2,begin=30+i,kind='cleanup',session=f'u{i}',request_id=f'{i}-cleanup',
            status='completed',input_tokens=1,input_sha256=bench.ready.work.digest([0]),
            output_ids=[1],usage=dict(completion_tokens=1)))
    directory=root/'workload';directory.mkdir(parents=True)
    capture=root/'server/capture';capture.mkdir(parents=True)
    bench.cap.save(root/'fixture.json',fixture)
    bench.cap.save(directory/'traces.json',rows)
    bench.cap.save(directory/'resources-after.json',dict(at=100,session_views=[dict(namespace=[f'u{i}',f'u{i}'],
        history=1,window=1,active_records=[[0,1]]) for i in range(2)],requests={},refresh_pending={},connector_pending={},
        owned_blocks={},reservations={},cpu_history_admission=dict(active={},retiring={},waiting={})))
    bench.cap.save(capture/'archive.json',dict(archive_backend='cpu',repository_gc=False,disk_read_bytes=0,disk_write_bytes=0))
    total=sum(len(r['output_ids']) for r in rows)
    result=dict(status='passed',cleanup_completed=True,total_stream_output_tokens=total,
                native_reconciliation=dict(passed=True,preemptions=0,native_tokens=total,client_tokens=total))
    return rows,result


def qualified(root,monkeypatch):
    rows,result=streams(root)
    low=dict(tests=4,failures=0,errors=0,skipped=0,cases=['a','b','c','d'],xml_sha256='xml')
    actions=dict(requests=[{}]*6,upload_final=dict(submissions=20))
    oracle=dict(cold_restore_required=True,target_restore_records=4)
    histories=[dict(request_id=r['request_id'],history_tokens=r['input_tokens']+len(r['output_ids'])-1)
               for r in rows if r['kind']=='workload']
    result.update(low_level=low,action_upload=actions,oracle=oracle,history_proofs=histories,
                  cancellation_validated=False,low_level_lease_exception_validated=True)
    bench.cap.save(root/'results.json',result)
    bench.cap.save(root/'low-level-proof.json',low)
    bench.cap.save(root/'workload/history_publications.json',[])
    bench.cap.save(root/'status.json',dict(status='completed_pending_review'))
    hashes={'original':'source','three':'changes'}
    manifest=dict(kind=bench.QUALIFICATION_KIND,cuda_cases_expected=4,action_evidence_version=2,native_source={'source':'same'},
        candidate_source=str(root/'candidate'),runtime_sha256=hashes,installed_native_sha256='install',
        candidate_manifest_sha256='candidate',candidate_patch_sha256='patch')
    bench.cap.save(root/'manifest.json',manifest)
    model=NS(work=bench.ready.work,oracle_evidence=lambda *args:oracle,history_evidence=lambda *args:histories)
    checked=[]
    gate=NS(verify=lambda path:manifest,installed_native_proof=lambda path:checked.append('native'),GPU_TEST='oracle.py',
        candidate_proof=lambda *args:({'production_files':dict.fromkeys(bench.FILES)},hashes),
        numerical_evidence=lambda path:low,action_evidence=lambda *args:actions,module=lambda path:model)
    monkeypatch.setattr(bench,'module',lambda *args:gate)
    return manifest,result,gate,checked


def test_profile_changes_only_enabled_upload_and_port():
    original=profile()
    result=bench.performance_profile(original,18570)
    expected=copy.deepcopy(original);expected['port']=18570;expected['kvmem']['async_action_metadata']=True
    assert result==expected and 'async_action_metadata' not in original['kvmem']
    assert result['max_num_batched_tokens']==4224 and result['query_replay_budget']==16384


@pytest.mark.parametrize('key,value',[('gpu_memory_utilization',.82),('max_num_seqs',12),('draft_tokens',3)])
def test_changed_deployment_is_not_the_same_natural_control(key,value):
    original=profile();original[key]=value
    with pytest.raises(ValueError,match='original natural'):
        bench.performance_profile(original,18570)


@pytest.mark.parametrize('key',['early_capture_slot_release','native_action_metadata','action_profiling','mean_k_prewarm'])
def test_unrelated_optimization_is_forbidden(key):
    original=profile();original['kvmem'][key]=True
    with pytest.raises(ValueError,match='unrelated'):
        bench.performance_profile(original,18570)


def test_old_rolling_qualification_is_rejected_before_module_import(tmp_path,monkeypatch):
    bench.cap.save(tmp_path/'manifest.json',dict(kind='rolling_window_qualification_v1',cuda_cases_expected=4))
    monkeypatch.setattr(bench,'module',lambda *args:pytest.fail('Old rolling proof imported'))
    with pytest.raises(ValueError,match='rolling-only'):
        bench.completed_qualification(tmp_path,{}, {})


def test_old_action_validator_cannot_admit_performance(tmp_path,monkeypatch):
    bench.cap.save(tmp_path/'manifest.json',dict(kind=bench.QUALIFICATION_KIND,cuda_cases_expected=4))
    bench.cap.save(tmp_path/'status.json',dict(status='completed_pending_review'))
    monkeypatch.setattr(bench,'module',lambda *args:pytest.fail('Old Action validator imported'))
    with pytest.raises(ValueError):
        bench.completed_qualification(tmp_path,{}, {})


def test_prepared_metadata_qualification_does_not_admit_preparation(tmp_path,monkeypatch):
    q=tmp_path/'qualification';q.mkdir()
    bench.cap.save(q/'manifest.json',dict(kind=bench.QUALIFICATION_KIND,cuda_cases_expected=4,action_evidence_version=2))
    bench.cap.save(q/'status.json',dict(status='prepared'))
    monkeypatch.setattr(bench,'reference_proof',lambda *args:dict(runtime_sha256={bench.ARM:{}},native_source={}))
    with pytest.raises(ValueError,match='before even preparing'):
        bench.prepare(NS(output=tmp_path/'performance',reference=tmp_path/'reference',qualification=q,port=18570))
    assert not (tmp_path/'performance').exists()


def test_complete_new_qualification_reuses_actual_oracles_and_history(tmp_path,monkeypatch):
    manifest,result,gate,checked=qualified(tmp_path,monkeypatch)
    identity,proof=bench.completed_qualification(tmp_path,{'base':'exact'},{'source':'same'})
    assert checked==['native']
    assert identity['runtime_sha256']==manifest['runtime_sha256']
    assert proof['streams']['model_requests']==6 and proof['streams']['cleanup_requests']==2
    assert len(proof['histories'])==6 and proof['real_engine_cancellation_validated'] is False


@pytest.mark.parametrize('mutation',[None,'other_failure','review_hash','action','model','oracle','cleanup'])
def test_explicit_offline_review_only_recovers_completed_parser_import_failure(tmp_path,monkeypatch,mutation):
    manifest,result,gate,checked=qualified(tmp_path,monkeypatch)
    model=copy.deepcopy(result)
    for key in ('low_level','action_upload','cancellation_validated','low_level_lease_exception_validated'):
        model.pop(key)
    bench.cap.save(tmp_path/'model-results.json',model)
    terminal=dict(status='failed',error=bench.PARSER_IMPORT_ERROR)
    bench.cap.save(tmp_path/'status.json',terminal)
    review=tmp_path/'offline-review.json'
    evidence=dict(original_status=terminal,qualification_manifest_sha256=bench.cap.sha(tmp_path/'manifest.json'),
        original_model_result_sha256=bench.cap.sha(tmp_path/'model-results.json'),
        verifier_sha256=bench.cap.sha(bench.ROOT/bench.QUALIFIER),
        actions=result['action_upload'],low_level=result['low_level'])
    if mutation=='other_failure':terminal['error']='CUDA error';bench.cap.save(tmp_path/'status.json',terminal)
    elif mutation=='review_hash':evidence['verifier_sha256']='changed'
    elif mutation=='action':evidence['actions']={}
    elif mutation=='model':model['native_reconciliation']['native_tokens']-=1;bench.cap.save(tmp_path/'model-results.json',model)
    elif mutation=='oracle':
        model['oracle']={};bench.cap.save(tmp_path/'model-results.json',model)
        evidence['original_model_result_sha256']=bench.cap.sha(tmp_path/'model-results.json')
    elif mutation=='cleanup':bench.cap.save(tmp_path/'cleanup-required.json',{'pid':99})
    bench.cap.save(review,evidence)
    with pytest.raises(ValueError):
        bench.completed_qualification(tmp_path,{}, {'source':'same'})  # explicit review is mandatory
    if mutation:
        with pytest.raises(ValueError):
            bench.completed_qualification(tmp_path,{}, {'source':'same'},review)
    else:
        identity,proof=bench.completed_qualification(tmp_path,{}, {'source':'same'},review)
        assert proof['streams']['accepted_tokens']==12290
        assert identity['offline_review']['sha256']==bench.cap.sha(review)
        assert bench.cap.read(tmp_path/'status.json')==terminal  # original failure remains visible
        assert bench.cap.read(tmp_path/'model-results.json')==model


@pytest.mark.parametrize('bad',['skip','action','oracle','cleanup','count','unrelated_runtime','cleanup_required'])
def test_incomplete_new_qualification_never_admits_performance(tmp_path,monkeypatch,bad):
    manifest,result,gate,_=qualified(tmp_path,monkeypatch)
    if bad=='skip':result['low_level']=dict(result['low_level'],skipped=1)
    elif bad=='action':result['action_upload']={}
    elif bad=='oracle':result['oracle']={}
    elif bad=='cleanup':result['cleanup_completed']=False
    elif bad=='count':result['native_reconciliation']['native_tokens']-=1
    elif bad=='unrelated_runtime':manifest['runtime_sha256']={'other':'unqualified'}
    else:bench.cap.save(tmp_path/'cleanup-required.json',dict(pid=99))
    bench.cap.save(tmp_path/'results.json',result)
    with pytest.raises(ValueError):
        bench.completed_qualification(tmp_path,{'base':'exact'},{'source':'same'})


def test_history_uses_own_outputs_not_recorded_answers(tmp_path):
    rows,result=streams(tmp_path)
    row=next(r for r in rows if r['kind']=='workload' and r['turn']==1)
    row['input_sha256']='recorded output substituted'
    bench.cap.save(tmp_path/'workload/traces.json',rows)
    with pytest.raises(ValueError,match='own-history chain'):
        bench.qualification_streams(NS(work=bench.ready.work),tmp_path,result)


def test_performance_upload_counters_do_not_invent_formal_delta_or_idle(tmp_path):
    capture=tmp_path/bench.ARM/'server/capture';capture.mkdir(parents=True)
    bench.cap.save(capture/'action_config.json',dict(async_action_metadata=True))
    snapshot=dict(submissions=40,fallbacks={'credit':1},quarantined=False,failed_submissions=0,fence_failures=0,query_failures=0)
    (capture/'action_attention.jsonl').write_text(bench.ready.json.dumps(dict(metadata_upload_total=snapshot))+'\n')
    result=bench.upload_observations(tmp_path)
    assert result['formal_counter_delta'] is None and result['fallback_observed']
    snapshot['quarantined']=True
    (capture/'action_attention.jsonl').write_text(bench.ready.json.dumps(dict(metadata_upload_total=snapshot))+'\n')
    with pytest.raises(ValueError,match='poisoned'):
        bench.upload_observations(tmp_path)


def test_run_gate_precedes_any_wait_or_model_launch(tmp_path,monkeypatch):
    def fail(root):raise ValueError('New qualification missing')
    monkeypatch.setattr(bench,'verify',fail)
    monkeypatch.setattr(bench.ready,'run_arm',lambda *args:pytest.fail('Launched before qualification'))
    with pytest.raises(ValueError,match='qualification missing'):
        asyncio.run(bench.run(NS(output=tmp_path)))


@pytest.mark.parametrize('mutate',['error','identity'])
def test_qualification_is_rechecked_after_idle_wait(tmp_path,monkeypatch,mutate):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    calls=[]
    manifest={'metadata_qualification':{'path':'qualified'}}
    def verify(root):
        calls.append(1)
        if len(calls)>1:
            if mutate=='error':raise ValueError('Qualification changed during idle wait')
            return {'changed':True},{}
        return manifest,{}
    async def idle(*args,**kwargs):return None
    async def arm(*args):
        await bench.ready.bench.q.wait_idle(None,1)
        pytest.fail('Changed proof admitted model launch')
    monkeypatch.setattr(bench,'verify',verify)
    monkeypatch.setattr(bench.ready.bench.q,'wait_idle',idle)
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    monkeypatch.setattr(bench.fcntl,'flock',lambda *args:None)
    with pytest.raises(ValueError,match='changed during idle wait'):
        asyncio.run(bench.run(NS(output=tmp_path)))
    assert len(calls)==2 and bench.ready.bench.q.wait_idle is idle
    assert bench.cap.read(tmp_path/'status.json')['status']=='failed'


@pytest.mark.parametrize('history_proof',[
    dict(passed=False,missing_requests=['u0-turn2']),
    dict(missing_requests=['u0-turn2']),
    dict(passed=True,missing_requests=[]),
])
def test_run_requires_full_history_publication_after_saving_analysis(tmp_path,monkeypatch,history_proof):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    manifest={'metadata_qualification':{'path':'qualified'}}
    review={'arms':{bench.ARM:{'full_history_publication':history_proof}}}
    calls=[]
    async def arm(root,name,current):
        assert root==tmp_path and name==bench.ARM and current==manifest
        calls.append('drained')
    original_wait=bench.ready.bench.q.wait_idle
    monkeypatch.setattr(bench,'verify',lambda root:(manifest,{}))
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    monkeypatch.setattr(bench,'upload_observations',lambda root:{'enabled':True})
    monkeypatch.setattr(bench,'module',lambda *args:NS(review=lambda root:review))
    # Never open or acquire the shared GPU qualification lock in a CPU mock.
    monkeypatch.setattr(bench,'open',lambda *args,**kwargs:(tmp_path/'mock.lock').open('a'),raising=False)
    monkeypatch.setattr(bench.fcntl,'flock',lambda *args:None)
    if history_proof.get('passed') is True:
        asyncio.run(bench.run(NS(output=tmp_path)))
        assert bench.cap.read(tmp_path/'status.json')['status']=='completed_pending_review'
    else:
        with pytest.raises(ValueError,match='Complete CPU history/terminal M proof missing'):
            asyncio.run(bench.run(NS(output=tmp_path)))
        assert bench.cap.read(tmp_path/'status.json')['status']=='failed'
    assert calls==['drained']
    assert bench.cap.read(tmp_path/'analysis.json')==review
    assert bench.cap.read(tmp_path/'upload-observations.json')=={'enabled':True}
    assert (tmp_path/'timing.json').is_file()
    assert bench.ready.bench.q.wait_idle is original_wait
