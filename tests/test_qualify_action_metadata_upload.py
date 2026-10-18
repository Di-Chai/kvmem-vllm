"""CPU-only qualification gate tests; never import CUDA or run a model."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1]/'scripts/qualify_action_metadata_upload.py'
SPEC = importlib.util.spec_from_file_location('action_upload_qualification_test', PATH)
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()


def write_records(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def tokenizer(path):
    write(path/'tokenizer_config.json',dict(added_tokens_decoder={str(i+1):{'content':name}
        for i,name in enumerate(('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>'))}))
    return str(path)


def profile():
    return dict(execution_mode='compiled_graph',draft_method='mtp',draft_tokens=4,
        verify_session_restore=True,max_num_seqs=2,max_model_len=262144,port=18532,
        kvmem=dict(select_budget=32768,gen_budget=4096,rolling_active_window=True,
            archive_backend='cpu',active_resident_max_fraction=1e-6,
            cpu_read_cache_bytes=100_000_000_000,action_attention=True,
            prefill_attention=True,action_pressure=True))


@pytest.fixture
def prepared(tmp_path):
    ref,candidate,output = (tmp_path/name for name in ('reference','candidate','qualification'))
    for relative in ('src/vllm_kvmem/action_attention.py','src/vllm_kvmem/options.py','src/vllm_kvmem/untouched.py'):
        p=ref/'runtime'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# original '+relative)
    p=ref/'frozen/scripts/qualify_rolling_window.py';p.parent.mkdir(parents=True);p.write_text('# frozen reference')
    p=profile();p['tokenizer']=tokenizer(tmp_path/'tokenizer')
    write(ref/'profile.json',p);write(ref/'fixture.json',{'original':'unchanged fixture'})
    write(ref/'results.json',dict(status='passed',cleanup_completed=True,
        native_reconciliation=dict(passed=True,preemptions=0)))
    write(ref/'status.json',dict(status='completed_pending_review'))
    base=q.inventory(ref/'runtime')
    write(ref/'manifest.json',dict(kind='rolling_window_qualification_v1',
        source_sha256=q.inventory(ref/'frozen'),runtime_sha256=base,
        profile_sha256=q.sha(ref/'profile.json'),fixture_sha256=q.sha(ref/'fixture.json'),
        concurrency=2,output_quotas=[1024,4096,1024],native_source={'version':'locked'},
        budget={'minimum_free_bytes':8*1024**3},idle_seconds=10))
    files={}
    for relative in q.FILES:
        p=candidate/'overlay'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# candidate '+relative)
        files[relative]=dict(before_sha256=base.get(relative),after_sha256=q.sha(p))
    patch=candidate/'candidate.patch';patch.write_text('reviewed three-file patch')
    gpu_test=tmp_path/q.GPU_TEST;gpu_test.write_text('# exact four GPU cases')
    write(candidate/'manifest.json',dict(status='cpu_verified_gpu_pending',base=str(ref/'runtime'),
        cpu_validation={'isolated_candidate':{'passed':58}},production_files=files,
        patch_sha256=q.sha(patch),default=False,enable={'kvmem.async_action_metadata':True},
        limits=dict(combined_pinned_gpu_payload_bytes=q.ARENA_BYTES,max_pending_leases=q.MAX_LEASES),
        cuda_cases_expected=4,tests={'tests/'+q.GPU_TEST:q.sha(gpu_test)}))
    native=tmp_path/'native.json';write(native,dict(source_directory='/installed/vllm',files={'f':'hash'}))
    args=SimpleNamespace(output=output,reference=ref,candidate=candidate,gpu_tests=gpu_test,
                         installed_native=native,port=18564)
    q.prepare(args)
    return output,ref,candidate,args


def test_prepare_is_frozen_default_off_overlay_with_only_flag_and_port_delta(prepared):
    root,ref,_,_=prepared
    m=q.verify(root,frozen_entry=False)
    actual=q.read(root/'profile.json');expected=q.read(ref/'profile.json');expected['port']=18564
    expected['kvmem']['async_action_metadata']=True
    assert actual==expected
    assert q.read(root/'status.json')=={'status':'prepared','at':q.read(root/'status.json')['at'],'gpu_started':False}
    original=q.read(ref/'manifest.json')['runtime_sha256']
    assert {n for n,h in m['runtime_sha256'].items() if original.get(n)!=h}==set(q.FILES)
    assert (root/'frozen'/q.ENTRY).read_bytes()==PATH.read_bytes()
    assert not (root/'owner.json').exists() and not (root/'launch.json').exists()
    assert m['cancellation_scope'].endswith('No real engine request cancellation injection.')


@pytest.mark.parametrize('relative',['runtime/src/vllm_kvmem/action_attention.py','runtime/src/vllm_kvmem/untouched.py',
                                     'oracle/test_action_metadata_upload_gpu.py','provenance/candidate.patch',
                                     'frozen/tests/test_qualify_action_metadata_upload.py'])
def test_mutated_frozen_evidence_rejects_before_launch(prepared,relative):
    root,*_=prepared
    (root/relative).write_text('tampered')
    with pytest.raises(ValueError,match='changed'):q.verify(root,frozen_entry=False)


def test_original_candidate_change_is_detected(prepared):
    root,_,candidate,_=prepared
    (candidate/'overlay'/q.FILES[0]).write_text('changed after freeze')
    with pytest.raises(ValueError,match='before/after'):q.verify(root,frozen_entry=False)


def test_action_tokenizer_identity_is_frozen_before_gpu_admission(prepared):
    root,_,_,_=prepared
    path=Path(q.read(root/'profile.json')['tokenizer'])/'tokenizer_config.json'
    value=q.read(path);value['added_tokens_decoder']['99']={'content':'extra'};write(path,value)
    with pytest.raises(ValueError,match='tokenizer identity'):q.verify(root,frozen_entry=False)


def test_noncompleted_reference_cannot_be_prepared_or_run(prepared):
    root,ref,_,_=prepared
    write(ref/'status.json',{'status':'running'})
    with pytest.raises(ValueError,match='completed original'):q.verify(root,frozen_entry=False)


def test_wrong_entry_and_reprepare_fail_closed(prepared):
    root,_,_,args=prepared
    with pytest.raises(ValueError,match='frozen qualification entry'):q.verify(root)
    with pytest.raises(ValueError,match='fresh'):q.prepare(args)


@pytest.mark.parametrize('key,value',[('early_capture_slot_release',True),('native_action_metadata',True),
                                    ('action_profiling',True),('mean_k_prewarm',True)])
def test_other_optimizations_cannot_enter_qualification(key,value):
    p=profile();p['kvmem'][key]=value
    with pytest.raises(ValueError,match='Unrelated'):q.qualification_profile(p,18564)


@pytest.mark.parametrize('tests,skipped,failures',[(3,0,0),(4,1,0),(4,0,1),(5,0,0)])
def test_exact_cuda_count_and_zero_skip_required(tmp_path,tests,skipped,failures):
    p=tmp_path/'oracle.xml'
    p.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="0" skipped="{skipped}">' +
                 ''.join(f'<testcase name="case{i}"/>' for i in range(tests))+'</testsuite></testsuites>')
    with pytest.raises(ValueError,match='four exact CUDA'):q.numerical_evidence(p)


def test_xml_requires_actual_case_records(tmp_path):
    p=tmp_path/'oracle.xml';p.write_text('<testsuite tests="4" errors="0" failures="0" skipped="0"/>')
    with pytest.raises(ValueError,match='actual test case'):q.numerical_evidence(p)
    p.write_text('<testsuite tests="4" errors="0" failures="0" skipped="0">'+
                 ''.join(f'<testcase name="case{i}"/>' for i in range(4))+'</testsuite>')
    assert q.numerical_evidence(p)['tests']==4


@pytest.fixture
def action_capture(tmp_path):
    capture=tmp_path/'server/capture';capture.mkdir(parents=True)
    write(capture/'action_config.json',dict(async_action_metadata=True,prefill_attention=True,
        action_pressure=True,layers=[3,31,63],scope='non-thinking',interval=16))
    write(capture/'progress.json',dict(full_graph_decode_calls=20))
    # Use the actual host-only C++ parser, not a Python imitation of its scope rules.
    source=next((PATH.parents[1]/'src/vllm_kvmem').glob('_native*.so'))
    target=tmp_path/'runtime/src/vllm_kvmem'/source.name;target.parent.mkdir(parents=True)
    shutil.copyfile(source,target)
    config=profile();config['tokenizer']=tokenizer(tmp_path/'tokenizer')
    config['kvmem'].update(action_scope='non-thinking',action_interval=16)
    write(tmp_path/'profile.json',config)
    write(tmp_path/'manifest.json',dict(output_quotas=[1024,4096,1024],action_evidence_version=2,
        action_tokenizer_identity=q.action_tokenizer_identity(config),
        runtime_sha256={str(target.relative_to(tmp_path/'runtime')):q.sha(target)}))
    plans=[dict(initial_tokens=[3,99,5,99],turns=[dict(tool_tokens=[] if turn==0 else [5,99])
                                               for turn in range(3)]) for user in range(2)]
    write(tmp_path/'fixture.json',dict(agent_plans=plans))
    tokens=[list(p['initial_tokens']) for p in plans]
    rows=[];actions=[];prefills=[];publications=[]
    for turn in range(3):
        for user in range(2):
            if turn:tokens[user].extend(plans[user]['turns'][turn]['tool_tokens'])
            output=[99]*(1024,4096,1024)[turn]
            response=f'cmpl-rolling-{user}-{turn}';native=response+'-0-1234abcd'
            row=dict(kind='workload',turn=turn,agent=user,input_tokens=len(tokens[user]),epoch=0,
                session=f'user-{user}',status='completed',input_sha256=digest(tokens[user]),
                output_ids=output,output_sha256=digest(output),usage={'completion_tokens':len(output)},
                request_id=f'rolling-{user}-{turn}',response_id=response)
            rows.append(row)
            actions.append(dict(request_id=native,samples=(len(output)-2)//16+1,rounds=turn+1,score_sum=1.0,
                positive_pages=50,prefill_samples=(turn+1)*64,prefill_chunks=turn+1,prefill_pages=40,
                prefill_end=row['input_tokens'],metadata_upload_total=dict(submissions=len(rows)*100,
                    uploaded_bytes=len(rows)*1000,fallbacks={},quarantined=False,failed_submissions=0,
                    fence_failures=0,query_failures=0,max_combined_arena_bytes=q.ARENA_BYTES,
                    max_pending_leases=q.MAX_LEASES,arena_host_bytes=q.ARENA_BYTES//2,
                    arena_gpu_bytes=q.ARENA_BYTES//2,pending_leases=1,peak_pending_leases=2,
                    pending_payload_bytes=80,peak_pending_payload_bytes=160,completed=len(rows)*100-1)))
            prefills.append(dict(request_id=native,query_samples=64,samples=(turn+1)*64,
                chunks=turn+1,pages=40,end=row['input_tokens']))
            publications.append(dict(request_id=native,history_end=row['input_tokens']+len(output)-1,
                checkpoint_tokens=row['input_tokens']+len(output)-1,checkpoint_kind='M'))
            tokens[user].extend(output)
    write(tmp_path/'workload/history_publications.json',publications)
    for name,rs in [('action_attention.jsonl',actions),('prefill_attention.jsonl',prefills)]:
        write_records(capture/name,rs)
    return capture,rows,actions


def test_all_requests_score_and_completed_but_unpolled_lease_is_allowed(action_capture):
    capture,rows,_=action_capture
    proof=q.action_evidence(capture,rows)
    assert len(proof['requests'])==6 and proof['upload_final']['pending_leases']==1
    assert proof['version']==2 and proof['positive_output_rounds_by_user']=={'0':3,'1':3}
    assert proof['parser']['tokenizer_identity_frozen_before_run'] is True


@pytest.mark.parametrize('field,value',[('fallbacks',{'lease_limit':1}),('quarantined',True),
    ('query_failures',1),('fence_failures',1),('failed_submissions',1),('pending_payload_bytes',q.ARENA_BYTES+1),
    ('pending_leases',129),('completed',9999),('submissions',0)])
def test_model_upload_failure_or_accounting_error_blocks_qualification(action_capture,field,value):
    capture,rows,actions=action_capture
    actions[-1]['metadata_upload_total'][field]=value
    (capture/'action_attention.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actions))
    with pytest.raises(ValueError):q.action_evidence(capture,rows)


@pytest.mark.parametrize('field,value',[('samples',0),('rounds',0),('prefill_end',999999),('score_sum',float('nan'))])
def test_incomplete_action_chain_cannot_pass(action_capture,field,value):
    capture,rows,actions=action_capture
    actions[0][field]=value
    (capture/'action_attention.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actions))
    with pytest.raises(ValueError,match='scoring chain'):q.action_evidence(capture,rows)


def test_actual_graph_execution_required(action_capture):
    capture,rows,_=action_capture
    write(capture/'progress.json',dict(full_graph_decode_calls=0))
    with pytest.raises(ValueError,match='CUDA graph'):q.action_evidence(capture,rows)


def zero_last_round(capture,rows,actions):
    root=capture.parent.parent
    fixture=q.read(root/'fixture.json')
    fixture['agent_plans'][1]['turns'][2]['tool_tokens']=[5,3]
    write(root/'fixture.json',fixture)
    tokens=list(fixture['agent_plans'][1]['initial_tokens'])
    for row in sorted((r for r in rows if r['agent']==1),key=lambda r:r['turn']):
        if row['turn']:tokens.extend(fixture['agent_plans'][1]['turns'][row['turn']]['tool_tokens'])
        row['input_sha256']=digest(tokens)
        tokens.extend(row['output_ids'])
    actions[-1].update(samples=0,rounds=2)
    write_records(capture/'action_attention.jsonl',actions)


def test_zero_eligible_thinking_round_preserves_previous_scores(action_capture):
    capture,rows,actions=action_capture
    zero_last_round(capture,rows,actions)
    proof=q.action_evidence(capture,rows)
    last=proof['requests'][-1]
    assert last['samples']==last['expected']['samples']==0 and last['rounds']==2
    assert last['expected']['initial_thinking'] is True
    assert proof['positive_output_rounds_by_user']=={'0':3,'1':2}


@pytest.mark.parametrize('field,value',[('rounds',3),('score_sum',.5),('positive_pages',49)])
def test_zero_eligible_round_cannot_mutate_score_history(action_capture,field,value):
    capture,rows,actions=action_capture
    zero_last_round(capture,rows,actions)
    actions[-1][field]=value
    write_records(capture/'action_attention.jsonl',actions)
    with pytest.raises(ValueError,match='scoring chain|historical scores'):
        q.action_evidence(capture,rows)


def test_positive_eligibility_cannot_be_excused_as_thinking(action_capture):
    capture,rows,actions=action_capture
    # Historical <think> precedes a fresh im_start: begin() must reset it.
    actions[-1].update(samples=0,rounds=2)
    write_records(capture/'action_attention.jsonl',actions)
    with pytest.raises(ValueError,match='scoring chain'):q.action_evidence(capture,rows)


@pytest.mark.parametrize('mutation',['prompt_state','prior_own_output','output_hash','usage','terminal_m','native_hash'])
def test_action_oracle_rejects_wrong_history_state_or_consumed_boundary(action_capture,mutation):
    capture,rows,actions=action_capture
    root=capture.parent.parent
    if mutation=='prompt_state':
        fixture=q.read(root/'fixture.json');fixture['agent_plans'][1]['turns'][2]['tool_tokens']=[5,3]
        write(root/'fixture.json',fixture)
    elif mutation=='prior_own_output':
        rows[1]['output_ids'][0]=100;rows[1]['output_sha256']=digest(rows[1]['output_ids'])
    elif mutation=='output_hash':rows[-1]['output_sha256']='stale'
    elif mutation=='usage':rows[-1]['usage']['completion_tokens']-=1
    elif mutation=='terminal_m':
        path=root/'workload/history_publications.json';items=q.read(path)
        items[-1]['checkpoint_tokens']+=1;write(path,items)
    else:
        path=root/'manifest.json';manifest=q.read(path)
        manifest['runtime_sha256']={k:'wrong' for k in manifest['runtime_sha256']};write(path,manifest)
    with pytest.raises(ValueError,match='history|parser changed'):
        q.action_evidence(capture,rows)


def test_terminal_unconsumed_token_is_excluded_from_parser_samples(action_capture):
    capture,rows,actions=action_capture
    # The final token would be the 64th eligible sample, but lacks a consumed Q.
    rows[-1]['output_ids']=[3]+[99]*13+[4]+[99]*1009
    rows[-1]['output_sha256']=digest(rows[-1]['output_ids'])
    actions[-1]['samples']=63
    write_records(capture/'action_attention.jsonl',actions)
    proof=q.action_evidence(capture,rows)
    assert proof['requests'][-1]['expected']['consumed_tokens']==1023
    assert proof['requests'][-1]['expected']['samples']==63


def test_oracle_reuses_already_imported_native_extension(action_capture):
    capture,rows,_=action_capture
    native=importlib.import_module('vllm_kvmem._native')
    first=q.action_evidence(capture,rows)
    assert q.action_evidence(capture,rows)==first
    assert sys.modules['vllm_kvmem._native'] is native


def test_oracle_rejects_another_loaded_native_build(action_capture,monkeypatch):
    capture,rows,_=action_capture
    native=importlib.import_module('vllm_kvmem._native')
    actual=Path(native.__file__).resolve(); original=q.sha
    monkeypatch.setattr(q,'sha',lambda path:'foreign-build' if Path(path).resolve()==actual else original(path))
    with pytest.raises(ValueError,match='Loaded native Action parser differs'):
        q.action_evidence(capture,rows)


@pytest.mark.parametrize('mutation',['missing','cumulative','chunks','terminal'])
def test_each_round_needs_its_own_prefill_commit_and_increment(action_capture,mutation):
    capture,rows,actions=action_capture
    path=capture/'prefill_attention.jsonl';items=q.records(path)
    if mutation=='missing':items.pop()
    elif mutation=='cumulative':items[-1]['samples']=items[-3]['samples']
    elif mutation=='chunks':items[-1]['chunks']=items[-3]['chunks']
    else:
        actions[-1]['prefill_samples']-=1;write_records(capture/'action_attention.jsonl',actions)
    write_records(path,items)
    with pytest.raises(ValueError,match='prefill|Prefill|scoring chain'):
        q.action_evidence(capture,rows)


def test_each_user_must_exercise_actual_output_scoring(action_capture):
    capture,rows,actions=action_capture
    root=capture.parent.parent;fixture=q.read(root/'fixture.json');plan=fixture['agent_plans'][1]
    plan['initial_tokens'][-1]=3
    for turn in (1,2):plan['turns'][turn]['tool_tokens']=[5,3]
    write(root/'fixture.json',fixture)
    tokens=list(plan['initial_tokens'])
    for row in (r for r in rows if r['agent']==1):
        if row['turn']:tokens.extend(plan['turns'][row['turn']]['tool_tokens'])
        row['input_sha256']=digest(tokens);tokens.extend(row['output_ids'])
        action=next(a for a in actions if q.native_matches(row,a['request_id']))
        action.update(samples=0,rounds=0,score_sum=0,positive_pages=0)
    write_records(capture/'action_attention.jsonl',actions)
    with pytest.raises(ValueError,match='Each user'):q.action_evidence(capture,rows)


@pytest.mark.parametrize('failure',['owner_record','foreign_gpu','child_exit','timeout'])
def test_low_level_failure_always_cleans_only_its_owned_process(tmp_path,monkeypatch,failure):
    import psutil
    stopped=[]
    process=SimpleNamespace(pid=1234,returncode=1,
                            poll=lambda:1 if failure=='child_exit' else None)
    async def idle(*args):
        pass
    async def fake_thread(function,*args,**kwargs):
        # The subprocess/GPU functions below are deterministic fakes. No
        # worker-thread wakeup or real subprocess belongs in this CPU test.
        return function(*args,**kwargs)
    monkeypatch.setattr(q.asyncio,'to_thread',fake_thread)
    gpu_calls=[]
    def gpu_state():
        gpu_calls.append(True)
        return {'pids':[] if len(gpu_calls)==1 else [1234,7777]}
    def spawn(*args,**kwargs):
        assert q.read(tmp_path/'status.json')['status']=='low_level_running'
        assert kwargs['start_new_session'] is True
        return process
    monkeypatch.setattr(q.subprocess,'Popen',spawn)
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:10.0))
    original_save=q.save
    def save(path,value):
        if failure=='owner_record' and Path(path).name=='low-level-owner.json':
            raise OSError('cannot publish owned child identity')
        return original_save(path,value)
    monkeypatch.setattr(q,'save',save)
    if failure=='timeout':
        clock=iter((0.0,601.0))
        monkeypatch.setattr(q,'monotonic',lambda:next(clock))
    control=SimpleNamespace(wait_idle=idle,gpu_state=gpu_state,owned_pids=lambda p:{p.pid},
                            stop_owned=lambda p:stopped.append(p.pid))
    qual=SimpleNamespace(bench=SimpleNamespace(q=control),
                         cap=SimpleNamespace(host_memory=lambda:{'MemAvailable':1000}))
    expected=TimeoutError if failure=='timeout' else OSError if failure=='owner_record' else ValueError
    with pytest.raises(expected):
        asyncio.run(q.low_level(tmp_path,dict(idle_seconds=10,cuda_timeout_seconds=600,
            budget={'minimum_free_bytes':1}),qual,'token'))
    assert stopped==[1234]
    assert not (tmp_path/'low-level-proof.json').exists()
    if failure=='timeout':
        assert q.read(tmp_path/'low-level-timeout.json')['elapsed_seconds']==601
