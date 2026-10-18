"""CPU-only exact-source, lifecycle and prelaunch gates for the formal arm."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest

from test_capture_reuse_qualification import fake_workload, probe, profile

spec = importlib.util.spec_from_file_location('capture_feedback_benchmark',
    Path(__file__).parents[1]/'scripts/benchmark_capture_feedback.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def runtime_files(tmp_path):
    base, candidate = tmp_path/'base', tmp_path/'candidate'
    files = {}
    for name in bench.FILES:
        old = name.endswith(('capture_runtime.py', 'options.py'))
        if old:
            path = base/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original '+name)
        path = candidate/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('early '+name)
        files[name] = dict(base_sha256=bench.cap.sha(base/name) if old else None,
                          candidate_sha256=bench.cap.sha(path))
    return base, candidate, dict(default_enabled=False,files=files)


def test_exact_runtime_contract_excludes_unrelated_or_native_action_changes(tmp_path):
    base, candidate, manifest = runtime_files(tmp_path)
    assert bench.runtime_contract(base,candidate,manifest) == bench.cap.hashes(candidate)
    (candidate/'src/unrelated.py').write_text('unqualified optimization')
    with pytest.raises(ValueError,match='Unrelated runtime'):
        bench.runtime_contract(base,candidate,manifest)


def prepared(tmp_path, monkeypatch):
    reference, candidate_root = tmp_path/'reference', tmp_path/'early'
    (reference/'frozen/scripts').mkdir(parents=True)
    (reference/'frozen/scripts/original.py').write_text('original natural helper')
    source = profile()
    source['kvmem'].pop('early_capture_slot_release')
    source['kvmem'].pop('early_capture_feedback_bytes')
    bench.cap.save(reference/(bench.ARM+'.json'),source)
    bench.cap.save(reference/'fixture.json',dict(original='complete immutable fixture'))
    base,candidate,overlay=runtime_files(tmp_path)
    shutil.copytree(base,reference/'runtime'/bench.ARM)
    shutil.copytree(candidate,candidate_root/'runtime')
    (candidate_root/'changes.patch').write_text('reviewed four-file patch')
    overlay['patch_sha256']=bench.cap.sha(candidate_root/'changes.patch')
    bench.cap.save(candidate_root/'manifest.json',overlay)
    ref=dict(kind='ready_agents_v1',qualify=False,source_sha256=bench.cap.hashes(reference/'frozen'),
        fixture_sha256=bench.cap.sha(reference/'fixture.json'),native_source={'vllm':'qualified'},
        workload=dict(concurrency=8,warmup_turns=2,epoch_turns=16,duration_seconds=300,
            mode='tools',generation_mode='natural',generation_limit=32000),
        generation_sampling=dict(temperature=.6,ignore_eos=False),budget=dict(complete_cpu_history=True),idle_seconds=10)
    bench.cap.save(reference/'manifest.json',ref)
    byte,reuse=tmp_path/'byte',tmp_path/'reuse'
    byte.mkdir();reuse.mkdir()
    identity=dict(path=str(byte),manifest_sha256='byte-identity',runtime_sha256=bench.cap.hashes(candidate))
    reuse_identity=dict(path=str(reuse),manifest_sha256='reuse-identity',runtime_sha256=bench.cap.hashes(candidate))
    natural=NS(qualification_identity=lambda *args:identity)
    def reference_proof(path, expected_manifest=None):
        assert path==reference
        if expected_manifest:
            bench.cap.require(bench.cap.sha(reference/'manifest.json')==expected_manifest,'Reference manifest changed')
        return natural,ref
    monkeypatch.setattr(bench,'reference_proof',reference_proof)
    monkeypatch.setattr(bench,'reuse_identity',lambda *args:(None,reuse_identity))
    monkeypatch.setattr(bench,'qualification_proofs',lambda *args,**kw:dict(complete=kw['require_completed']))
    monkeypatch.setattr(bench.ready,'verify',lambda root:bench.cap.read(root/'manifest.json'))
    root=tmp_path/'performance'
    bench.prepare(NS(output=root,reference=reference,candidate=candidate_root,
        byte_qualification=byte,reuse_qualification=reuse,port=18550))
    return root,reference,ref


def test_preparation_reuses_original_helpers_fixture_workload_and_only_declared_profile_delta(tmp_path,monkeypatch):
    root,reference,original=prepared(tmp_path,monkeypatch)
    m,_=bench.verify(root,require_completed=False)
    assert m['workload']==original['workload'] and m['generation_sampling']==original['generation_sampling']
    assert (root/'fixture.json').read_bytes()==(reference/'fixture.json').read_bytes()
    assert set(m['source_sha256'])-set(original['source_sha256'])=={bench.ENTRY}
    p=bench.cap.read(root/(bench.ARM+'.json'))
    assert p.pop('port')==18550
    assert p['kvmem'].pop('early_capture_slot_release') is True
    assert p['kvmem'].pop('early_capture_feedback_bytes')==268435456
    source=bench.cap.read(reference/(bench.ARM+'.json'));source.pop('port')
    assert p==source
    assert bench.cap.read(root/'status.json')['gpu_started'] is False


@pytest.mark.parametrize('section,change,match', [
    ('workload',dict(duration_seconds=60),'Reference workload'),
    ('generation_sampling',dict(temperature=0),'Reference generation_sampling'),
    ('budget',dict(complete_cpu_history=False),'Reference budget')])
def test_performance_cannot_change_sampling_workload_or_history_budget(tmp_path,monkeypatch,section,change,match):
    root,_,_=prepared(tmp_path,monkeypatch)
    m=bench.cap.read(root/'manifest.json');m[section].update(change);bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match=match):bench.verify(root,require_completed=False)


def test_profile_may_not_enable_action_profiling_or_modify_any_other_setting(tmp_path,monkeypatch):
    root,_,_=prepared(tmp_path,monkeypatch)
    p=bench.cap.read(root/(bench.ARM+'.json'));p['kvmem']['action_profiling']=True
    bench.cap.save(root/(bench.ARM+'.json'),p)
    with pytest.raises(ValueError,match='Profile changed beyond'):bench.verify(root,require_completed=False)


def qualified_reuse(tmp_path, monkeypatch):
    result,rows,fixture=fake_workload(tmp_path,monkeypatch)
    directory,capture=tmp_path/'workload',tmp_path/'capture'
    target=tmp_path/'server/capture';target.parent.mkdir()
    shutil.move(str(capture),target)
    bench.cap.save(tmp_path/'fixture.json',fixture)
    bench.cap.save(tmp_path/'status.json',dict(status='completed_pending_review'))
    workrows=[r for r in rows if r['kind']=='workload']
    begin=min(r['begin'] for r in rows)
    old=[r['response_id']+'-0-12345678' for r in workrows[:2]]
    new=[workrows[2]['response_id']+'-0-12345678']
    def event(stage,at,generation=1,owners=old,**kw):
        return dict(stage=stage,at=begin+at,lane='bulk',slot=0,generation=generation,request_ids=owners,**kw)
    events=[event('source_submitted',0,pending_previous_generations=[]),
        event('source_copied',.1,copied_indices=[0,1]),
        event('source_released',.2,semantic_pending=2,feedback_bytes=128),
        event('source_submitted',.3,2,new,pending_previous_generations=[1]),event('semantic_complete',.4)]
    sample=dict(at=max(r['begin'] for r in rows)+1,feedback=dict(detached_packets=0,detached_owners=0,
        detached_bytes=0,packet_limit=2,owner_limit=8,byte_limit=268435456),publication_metrics={'main/bulk':dict(
            source_releases_before_semantic=1,shared_source_reuse_before_semantic_packets=1)})
    for name,values in [('capture_feedback_events',events),('publication_samples',[sample])]:
        (target/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    result.update(status='passed',native_reconciliation=dict(passed=True,preemptions=0,
        native_tokens=result['total_stream_output_tokens'],client_tokens=result['total_stream_output_tokens']))
    bench.cap.save(tmp_path/'results.json',result)
    archive=bench.cap.read(directory/'archive-after.json')
    archive.update(archive_backend='cpu',repository_gc=False,disk_read_bytes=0,disk_write_bytes=0)
    bench.cap.save(directory/'archive-after.json',archive)
    resources=dict(session_views=[dict(namespace=[r['session']]*2,history=1,window=1,active_records=[[0,1]])
        for r in rows if r['kind']=='cleanup'], requests=[],refresh_pending=[],connector_pending=[],owned_blocks={},
        reservations={},cpu_history_admission=dict(active={},retiring={},waiting={}))
    bench.cap.save(directory/'resources-after.json',resources)
    return result


def test_completed_reuse_gate_reaudits_outputs_full_history_m_cleanup_and_actual_physical_reuse(tmp_path,monkeypatch):
    result=qualified_reuse(tmp_path,monkeypatch)
    proof=bench.completed_reuse_proof(probe,tmp_path)
    assert proof['accepted_tokens']==result['total_stream_output_tokens']
    assert len(proof['output_proofs'])==len(proof['history_proofs'])==24
    assert proof['reuse']['covered'] and proof['cleanup_completed']


@pytest.mark.parametrize('failure,match', [
    ('status','has not passed'),('coverage','coverage insufficient'),('empty_reuse','No actual shared-slot'),
    ('native_count','native counters'),('historical_prefix','Incomplete accepted CPU'),('cleanup','fully drain')])
def test_insufficient_or_corrupt_reuse_never_authorizes_performance(tmp_path,monkeypatch,failure,match):
    result=qualified_reuse(tmp_path,monkeypatch)
    if failure=='status':bench.cap.save(tmp_path/'status.json',dict(status='coverage_insufficient'))
    elif failure=='coverage':
        result['status']='coverage_insufficient';bench.cap.save(tmp_path/'results.json',result)
    elif failure=='empty_reuse':(tmp_path/'server/capture/capture_feedback_events.jsonl').write_text('')
    elif failure=='native_count':
        result['native_reconciliation']['native_tokens']+=1;bench.cap.save(tmp_path/'results.json',result)
    elif failure=='historical_prefix':
        p=tmp_path/'workload/history_publications.json';rows=bench.cap.read(p);rows[0]['history_end']=1;bench.cap.save(p,rows)
    else:
        p=tmp_path/'workload/resources-after.json';r=bench.cap.read(p);r['cpu_history_admission']['retiring']={'a':1};bench.cap.save(p,r)
    with pytest.raises(ValueError,match=match):bench.completed_reuse_proof(probe,tmp_path)


def test_pending_qualification_rejects_before_idle_or_model_launch(tmp_path,monkeypatch):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    def reject(root):raise ValueError('Reuse coverage insufficient')
    monkeypatch.setattr(bench,'verify',reject)
    monkeypatch.setattr(bench.ready,'run_arm',lambda *a:pytest.fail('GPU launch before qualification'))
    with pytest.raises(ValueError,match='insufficient'):asyncio.run(bench.run(NS(output=tmp_path)))
    assert bench.cap.read(tmp_path/'status.json')['status']=='prepared'


@pytest.mark.parametrize('failure',['changed_manifest','qualification_invalid'])
def test_gate_rechecked_after_wait_before_original_arm_launch(tmp_path,monkeypatch,failure):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    checks=[]
    def verify(root):
        checks.append(root)
        if len(checks)>1:
            if failure=='changed_manifest':return {'changed':True},{}
            raise ValueError('Reuse qualification invalid after wait')
        return {},{}
    monkeypatch.setattr(bench,'verify',verify)
    async def idle(*a,**k):pass
    monkeypatch.setattr(bench.ready.bench.q,'wait_idle',idle)
    monkeypatch.setattr(bench.fcntl,'flock',lambda *a:None)
    async def arm(root,name,m):
        await bench.ready.bench.q.wait_idle(root,10)
        pytest.fail('No model launch after changed source/qualification')
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    with pytest.raises(ValueError,match='while waiting|invalid after wait'):
        asyncio.run(bench.run(NS(output=tmp_path)))
    assert len(checks)==2 and bench.ready.bench.q.wait_idle is idle


def reuse_events(at, *, slot=0, shared=True):
    owners = ['owner-a-0-12345678','owner-b-0-12345678'] if shared else ['owner-a-0-12345678']
    def event(stage,when,generation=1,**kw):
        return dict(stage=stage,at=when,lane='bulk',slot=slot,generation=generation,request_ids=owners,**kw)
    return [event('source_submitted',at-3,pending_previous_generations=[]),
        event('source_copied',at-2,copied_indices=list(range(len(owners)))),
        event('source_released',at-1,semantic_pending=len(owners),feedback_bytes=128),
        event('source_submitted',at,2,pending_previous_generations=[1]),
        event('semantic_complete',at+1)]


def window_audit(events,samples,window,rows=None):
    if rows is None:
        rows=[dict(response_id='owner-a',kind='workload'),dict(response_id='owner-b',kind='workload')]
    return bench.window_release_evidence(events,samples,window,rows)


def publication_sample(at,count,*,used=1):
    return dict(at=at,feedback=dict(detached_packets=used,detached_owners=used*2,
        detached_bytes=used*128,packet_limit=2,owner_limit=8,byte_limit=bench.FEEDBACK_BYTES),
        publication_metrics={'main/bulk':dict(copy_only_slot_wait_calls=count,
            copy_only_slot_wait_seconds=count*.01,detach_budget_deferrals=count*2),
            'worker/bulk':dict(source_releases_before_semantic=count),
            'main/decode':dict(copy_only_slot_wait_calls=100000)})


def test_window_reuse_distinguishes_warmup_from_formal_without_claiming_benefit():
    report=window_audit(reuse_events(9),[],dict(begin=10,end=310))
    assert not report['formal_shared_workload_reuse_covered']
    assert report['coverage']=='outside_formal_window_only'
    assert report['shared_reuse_observed_phases']==['warmup']
    assert report['phases']['formal']['actual_reuse_submissions']==0
    assert report['counter_deltas']['inner_sample_interval'] is None
    assert report['counter_deltas']['outer_sample_interval'] is None


def test_window_reuse_classifies_by_new_submit_with_inclusive_t0_exclusive_t1():
    events=reuse_events(10,slot=0)+reuse_events(309.9,slot=1)+reuse_events(310,slot=2)
    report=window_audit(events,[],dict(begin=10,end=310))
    assert report['formal_shared_workload_reuse_covered']
    assert report['phases']['formal']['actual_reuse_submissions']==2
    assert report['phases']['drain']['shared_packet_reuse_submissions']==1
    assert report['physical_reuse_proofs'][0]['source_released_at']<10


def test_release_without_real_shared_reuse_is_not_formal_coverage():
    events=reuse_events(100,shared=False)
    report=window_audit(events,[],dict(begin=10,end=310))
    assert report['phases']['formal']['early_source_releases']==1
    assert report['phases']['formal']['actual_reuse_submissions']==1
    assert report['phases']['formal']['shared_packet_reuse_submissions']==0
    assert not report['formal_shared_workload_reuse_covered']
    events=[e for e in reuse_events(100) if e['generation']==1]
    report=window_audit(events,[],dict(begin=10,end=310))
    assert report['phases']['formal']['actual_reuse_submissions']==0
    assert not report['formal_shared_workload_reuse_covered']


@pytest.mark.parametrize('kind',['cleanup','warmup','mixed','unknown'])
def test_formal_epoch_cleanup_and_unmatched_packets_cannot_prove_workload_reuse(kind):
    rows = [dict(response_id='owner-a',kind='cleanup' if kind=='mixed' else kind),
            dict(response_id='owner-b',kind='workload' if kind=='mixed' else kind)]
    if kind=='unknown':rows=[]
    report=window_audit(reuse_events(100),[],dict(begin=10,end=310),rows)
    assert report['formal_shared_physical_reuse_covered']
    assert not report['formal_shared_workload_reuse_covered']
    assert report['coverage']=='formal_shared_reuse_only_nonworkload_or_unknown'
    assert report['phases']['formal']['by_trace_kind'][kind]['shared_packet_reuse_submissions']==1


def test_workload_to_cleanup_reuse_does_not_establish_formal_workload_coverage():
    events=reuse_events(100)
    events[3]['request_ids']=['cleanup-0-12345678']
    rows=[dict(response_id=key,kind=kind) for key,kind in
          [('owner-a','workload'),('owner-b','workload'),('cleanup','cleanup')]]
    report=window_audit(events,[],dict(begin=10,end=310),rows)
    assert report['formal_shared_physical_reuse_covered']
    assert not report['formal_shared_workload_reuse_covered']
    assert report['phases']['formal']['by_trace_kind']['mixed']['shared_packet_reuse_submissions']==1


def test_window_credit_boundaries_and_interval_deltas_report_sampling_slack():
    samples=[publication_sample(9,2),publication_sample(11,3),
             publication_sample(309,13),publication_sample(312,17,used=0)]
    report=window_audit([],samples,dict(begin=10,end=310))
    snapshots=report['boundary_snapshots']
    assert snapshots['T0']['before']['offset_seconds']==-1
    assert snapshots['T1']['after']['offset_seconds']==2
    assert snapshots['T0']['after']['feedback']['available_bytes']==bench.FEEDBACK_BYTES-128
    assert snapshots['T1']['after']['feedback']['available_packets']==2
    inner,outer=report['counter_deltas'].values()
    assert inner['counters']['copy_only_slot_wait_calls']==10
    assert outer['counters']['copy_only_slot_wait_calls']==15
    assert inner['counters']['detach_budget_deferrals']==20
    assert inner['begin_offset_from_T0_seconds']==1 and inner['end_offset_from_T1_seconds']==-1
    assert outer['begin_offset_from_T0_seconds']==-1 and outer['end_offset_from_T1_seconds']==2
    assert outer['counters']['copy_only_slot_wait_seconds']==pytest.approx(.15)


def test_missing_boundary_samples_never_fabricate_counter_zero():
    report=window_audit([], [publication_sample(100,9)],dict(begin=10,end=310))
    assert report['boundary_snapshots']['T0']['before'] is None
    assert report['boundary_snapshots']['T1']['after'] is None
    assert report['counter_deltas']['outer_sample_interval'] is None
    assert report['counter_deltas']['inner_sample_interval'] is None


@pytest.mark.parametrize('failure',['copy_missing','semantic_finished','credits_exceeded','counter_reset'])
def test_window_audit_rejects_invalid_reuse_or_metrics(failure):
    events=reuse_events(100)
    samples=[publication_sample(10,3),publication_sample(310,4)]
    if failure=='copy_missing':events[1]['copied_indices']=[0]
    elif failure=='semantic_finished':events[-1]['at']=99.5
    elif failure=='credits_exceeded':samples[0]['feedback']['detached_bytes']=bench.FEEDBACK_BYTES+1
    else:samples[1]['publication_metrics']['main/bulk']['copy_only_slot_wait_calls']=1
    with pytest.raises(ValueError):window_audit(events,samples,dict(begin=10,end=310))
