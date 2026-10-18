"""CPU-only admission and provenance tests; never create a model process."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('benchmark_rolling_agents_test',
    Path(__file__).parents[1]/'scripts/benchmark_rolling_agents.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def profile():
    return dict(model_format='safetensors', draft_method='mtp', draft_tokens=4,
        execution_mode='compiled_graph', gpu_memory_utilization=.81, query_replay_budget=4096,
        tokenizer='/test/tokenizer', kvmem=dict(block_tokens=32, offline_agent_policy=True,
            prefill_batch_tokens=0, archive_backend='cpu', split_capture_dma=True,
            native_prefill_commit=True, async_checkpoint_capture=True, request_planning=True,
            async_refresh=True, async_restore=True, cpu_history_coordination=True,
            session_capture_terminal=True, action_pressure=True, session_checkpoint_policy='pm',
            session_capture_decode=False))


def fixture():
    return dict(eos=0, selection=dict(users=8, turns=16,
        manifest_order='first users without reward/timing selection',
        history_threshold=204800, allow_shorter_history_fallback=True),
        agent_plans=[dict(source_task=f'task-{i}', source_turn=10, initial_tokens=[i+1]*512,
            query_plan=dict(kvmem_query_span=[32,64]), turns=[dict(output_tokens=100000,
                tool_tokens=[] if t==0 else [i+10,t+20], tool_wait_seconds=0,
                source_turn=10+t, external_message_proof=None if t==0 else {'original':'tool'})
                for t in range(16)]) for i in range(8)])


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))


def rolling_evidence(path):
    capture=path/'server/capture'; capture.mkdir(parents=True)
    work=path/'workload'; work.mkdir()
    rows=[]; targets=[]; states=[]; refresh=[]; publications=[]
    for user in range(2):
        for turn in range(3):
            response=f'response-{user}-{turn}'; rid=response+'-0-12345678'
            length=131072+turn*1024
            rows.append(dict(agent=user,turn=turn,kind='workload',response_id=response,
                request_id=f'client-{user}-{turn}',input_tokens=length,output_ids=[1]))
            publications.append(dict(request_id=rid,history_end=length,checkpoint_tokens=length,
                                     checkpoint_kind='M'))
            if turn:
                targets.append(dict(request_id=rid,byte_equal=True,mtp_attention_layers=1,
                    working_attention_layers=17,target_kv_bytes=100,oracle_kind='fp8_immutable_rebuild'))
                states.append(dict(request_id=rid,recurrent_byte_equal=True))
            if turn==1:
                refresh.append(dict(request_id=rid,history_end=135168,selected_tokens=32768,
                    replay_tokens=0,cancelled=False,target_kv_byte_equal=True,recurrent_byte_equal=True,
                    mtp_attention_layers=1,attention_layers=17,recurrent_tensors=96,
                    target_kv_bytes=100,private_forward_steps=0,
                    oracle_kind='native_stable_rows_plus_fp8_rebuild',
                    previous_accepted_step=dict(history_begin=135166,accepted=2,scheduled=5)))
        rows.append(dict(agent=user,turn=2,kind='cleanup',input_tokens=1,output_ids=[1]))
    for name, values in [('verified_restores',targets), ('verified_state_restores',states),
            ('verified_rolling_pressure',refresh), ('model_step_latency',[
                dict(decode_requests=2,prefill_requests=0,prefill_tokens=0)])]:
        write_records(capture/(name+'.jsonl'),values)
    bench.cap.save(work/'traces.json',rows)
    bench.cap.save(work/'history_publications.json',publications)
    bench.cap.save(path/'results.json',dict(status='passed',cleanup_completed=True,
        total_stream_output_tokens=8,native_reconciliation=dict(passed=True,native_tokens=8,
                                                               client_tokens=8,preemptions=0)))
    return capture


def qualification(tmp_path, rolling, completed=False):
    path=tmp_path/('rolling' if rolling else 'pipeline'); path.mkdir()
    frozen=path/'frozen'; frozen.mkdir(); (frozen/'entry.py').write_text('qualification entry')
    runtime=path/('runtime' if rolling else 'runtime/kvmem-ready')
    runtime.mkdir(parents=True); (runtime/'runtime.py').write_text('rolling-v2' if rolling else 'pipeline')
    p=bench.profiles(profile(),18534)[bench.ARMS[2] if rolling else 'pipeline-base']
    if rolling:p=bench.qual.qualification_profile(p,18532)
    else:p.update(verify_session_restore=True)
    profile_path=path/('profile.json' if rolling else 'kvmem-ready.json')
    bench.cap.save(profile_path,p); bench.cap.save(path/'fixture.json',dict(eos=0))
    m=dict(kind='rolling_window_qualification_v1' if rolling else 'ready_agents_v1',
           source_sha256=bench.cap.hashes(frozen), fixture_sha256=bench.cap.sha(path/'fixture.json'),
           native_source={'version':'test'})
    if rolling:m.update(runtime_sha256=bench.cap.hashes(runtime), profile_sha256=bench.cap.sha(profile_path))
    else:m.update(runtime_sha256={'kvmem-ready':bench.cap.hashes(runtime)},
                  profiles_sha256={'kvmem-ready':bench.cap.sha(profile_path)})
    bench.cap.save(path/'manifest.json',m)
    bench.cap.save(path/'status.json',dict(status='completed_pending_review' if completed else 'prepared'))
    if rolling:rolling_evidence(path)
    else:
        capture=path/'kvmem-ready/server/capture'
        target=dict(request_id='response-0-12345678',byte_equal=True,mtp_attention_layers=1,
                    working_attention_layers=17,target_kv_bytes=100)
        write_records(capture/'verified_restores.jsonl',[target])
        write_records(capture/'verified_state_restores.jsonl',[dict(request_id=target['request_id'],
                                                                  recurrent_byte_equal=True)])
        bench.cap.save(capture/'scheduler_resources.json',dict(restore_admission=dict(peak_occupied=dict(resident=2))))
        tools=path/'kvmem-ready/tools'; tools.mkdir()
        bench.cap.save(tools/'traces.json',[dict(kind='workload',turn=1,response_id='response',request_id='client')])
        bench.cap.save(path/'kvmem-ready/results.json',dict(status='completed',drained=True,
            native_client_tokens_equal=True,preemptions=0,archive_final=dict(archive_backend='cpu',
                repository_gc=False,disk_read_bytes=0,disk_write_bytes=0)))
    return path, bench.cap.hashes(runtime)


def test_full_history_budget_never_uses_recorded_output_quotas_or_threshold():
    data=fixture(); before=copy.deepcopy(data)
    budget=bench.fixture_bounds(data,bench.profiles(profile(),18534)['native'])
    assert data==before
    assert budget['history_limits']==[262144]*8
    assert budget['archive_bytes']==8*(262144+64)*34816
    assert budget['required_available_bytes']==113834459136
    data['agent_plans'][0]['turns'][0]['output_tokens']=1
    assert bench.fixture_bounds(data,bench.profiles(profile(),18534)['native'])==budget


def test_profiles_keep_compiled_native_mtp_and_disable_all_oracles():
    p=bench.profiles(profile(),18534)
    assert all(v['verify_session_restore'] is False for v in p.values())
    assert all(v['max_num_seqs']==8 and v['draft_tokens']==4 for v in p.values())
    assert p['pipeline-base']['kvmem']['resident_restore_slots']==4
    assert 'rolling_active_window' not in p['pipeline-base']['kvmem']
    candidate=copy.deepcopy(p['rolling-candidate-v2'])
    assert candidate['kvmem'].pop('rolling_active_window') is True
    assert candidate==p['pipeline-base']==p['native']


@pytest.mark.parametrize('mutate,match',[
    (lambda f:f['agent_plans'][0].update(initial_tokens=[1]*230145),'generation limit'),
    (lambda f:f['agent_plans'][0]['turns'][1].update(tool_tokens=[1]*230000),'second warmup'),
    (lambda f:f['agent_plans'][0]['turns'][1].update(source_turn=99),'consecutive'),
    (lambda f:f['agent_plans'][0]['turns'][1].update(external_message_proof=None),'original external'),
    (lambda f:f['selection'].update(manifest_order='best throughput'),'first-eight'),
    (lambda f:f['agent_plans'][1].update(source_task='task-0'),'Independent'),
])
def test_incomplete_or_changed_natural_inputs_fail_without_clipping(mutate,match):
    data=fixture(); mutate(data); before=copy.deepcopy(data)
    with pytest.raises(ValueError,match=match):
        bench.fixture_bounds(data,bench.profiles(profile(),18534)['native'])
    assert data==before


@pytest.mark.parametrize('rolling',[False,True])
def test_pending_qualification_may_be_frozen_but_cannot_authorize_run(tmp_path,rolling):
    path,runtime=qualification(tmp_path,rolling)
    arm=bench.ARMS[2] if rolling else bench.ARMS[1]
    identity=bench.qualification_identity(path,arm,runtime,{'version':'test'})
    assert identity['runtime_sha256']==runtime
    with pytest.raises(ValueError,match='has not completed'):
        bench.validate_completed_qualification(path,arm)


@pytest.mark.parametrize('rolling',[False,True])
def test_qualification_rejects_unqualified_runtime_and_native_source(tmp_path,rolling):
    path,runtime=qualification(tmp_path,rolling)
    arm=bench.ARMS[2] if rolling else bench.ARMS[1]
    with pytest.raises(ValueError,match='exact qualified'):
        bench.qualification_identity(path,arm,{'other.py':'unqualified'},{'version':'test'})
    with pytest.raises(ValueError,match='native implementation'):
        bench.qualification_identity(path,arm,runtime,{'version':'other'})
    target=path/('runtime/runtime.py' if rolling else 'runtime/kvmem-ready/runtime.py')
    target.write_text('modified after correctness qualification')
    with pytest.raises(ValueError,match='runtime changed'):
        bench.qualification_identity(path,arm,runtime,{'version':'test'})


def test_completed_rolling_requires_each_session_cold_and_pressure_proofs(tmp_path):
    path,_=qualification(tmp_path,True,True)
    result=bench.validate_completed_qualification(path,bench.ARMS[2])
    assert len(result['oracle']['verified_rolling_refreshes'])==2
    assert len(result['history'])==6
    capture=path/'server/capture'
    records=bench.ready.records(capture/'verified_rolling_pressure.jsonl')
    write_records(capture/'verified_rolling_pressure.jsonl',records[:1])
    with pytest.raises(ValueError,match='Each user'):
        bench.validate_completed_qualification(path,bench.ARMS[2])


@pytest.mark.parametrize('filename,change,match',[
    ('verified_restores',dict(oracle_kind='native_terminal_window'),'cold immutable'),
    ('verified_state_restores',dict(recurrent_byte_equal=False),'recurrent restore'),
    ('verified_rolling_pressure',dict(oracle_kind='fp8_only'),'candidate-v2'),
    ('verified_rolling_pressure',dict(replay_tokens=32),'accepted-boundary'),
])
def test_rolling_gate_cannot_substitute_hot_restore_or_weaker_oracle(tmp_path,filename,change,match):
    path,_=qualification(tmp_path,True,True)
    target=path/'server/capture'/(filename+'.jsonl')
    records=bench.ready.records(target); records[0].update(change); write_records(target,records)
    with pytest.raises(ValueError,match=match):bench.validate_completed_qualification(path,bench.ARMS[2])


def test_completed_pipeline_requires_overlapping_resident_and_matched_state_bytes(tmp_path):
    path,_=qualification(tmp_path,False,True)
    assert bench.validate_completed_qualification(path,bench.ARMS[1])['peak_resident_permits']==2
    capture=path/'kvmem-ready/server/capture'
    bench.cap.save(capture/'scheduler_resources.json',dict(restore_admission=dict(peak_occupied=dict(resident=1))))
    with pytest.raises(ValueError,match='independent restore overlap'):
        bench.validate_completed_qualification(path,bench.ARMS[1])


def test_run_pending_gate_precedes_any_gpu_poll_or_launch(tmp_path,monkeypatch):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    monkeypatch.setattr(bench,'verify',lambda root:{})
    def pending(_):raise ValueError('GPU qualification has not completed')
    monkeypatch.setattr(bench,'qualification_proofs',pending)
    monkeypatch.setattr(bench.ready,'run_arm',lambda *a,**k:pytest.fail('GPU launch before qualification'))
    monkeypatch.setattr(bench.ready.bench.q,'gpu_state',lambda:pytest.fail('GPU poll before qualification'))
    with pytest.raises(ValueError,match='has not completed'):
        asyncio.run(bench.run(SimpleNamespace(output=tmp_path)))
    assert bench.cap.read(tmp_path/'status.json')=={'status':'prepared'}
    assert not (tmp_path/'qualification-proof.json').exists()


def test_original_fixture_provenance_is_rechecked_read_only(tmp_path):
    source=tmp_path/'bridge.py'; source.write_text('original source')
    tokenizer=tmp_path/'tokenizer.json'; tokenizer.write_text('original tokenizer')
    data=dict(provenance=dict(source_file_sha256={str(source):bench.cap.sha(source)},
        tokenizer_file_sha256={str(tokenizer):bench.cap.sha(tokenizer)}))
    bench.verify_provenance(data)
    source.write_text('changed source')
    with pytest.raises(ValueError,match='Original provenance changed'):bench.verify_provenance(data)


def prepared(tmp_path,monkeypatch):
    pipeline,_=qualification(tmp_path,False,True)
    rolling,_=qualification(tmp_path,True,False)
    checkout=tmp_path/'checkout'; checkout.mkdir()
    for folder in ('src','scripts','csrc'):
        (checkout/folder).mkdir(); (checkout/folder/'file.py').write_text('frozen implementation')
    config=tmp_path/'config.json'; bench.cap.save(config,profile())
    source=tmp_path/'original-source.py'; source.write_text('immutable source')
    tokenizer=tmp_path/'tokenizer.json'; tokenizer.write_text('immutable tokenizer')
    f=fixture(); f.pop('eos')
    f['provenance']=dict(source_file_sha256={str(source):bench.cap.sha(source)},
        tokenizer_file_sha256={str(tokenizer):bench.cap.sha(tokenizer)})
    fixture_path=tmp_path/'fixture.json'; bench.cap.save(fixture_path,f)
    args=SimpleNamespace(output=tmp_path/'performance',config=config,fixture=fixture_path,
        baseline=pipeline/'runtime/kvmem-ready',candidate=rolling/'runtime',
        pipeline_qualification=pipeline,rolling_qualification=rolling,port=18534)
    monkeypatch.setattr(bench,'ROOT',checkout)
    monkeypatch.setattr(bench.cap,'native_source_identity',lambda:{'version':'test'})
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a,**k:SimpleNamespace(eos_token_id=0))))
    monkeypatch.setattr(bench.ready.bench.q,'gpu_state',lambda:pytest.fail('Preparation accessed GPU'))
    bench.prepare(args)
    monkeypatch.setattr(bench.ready,'ROOT',args.output/'frozen')
    return args


def test_complete_prepare_and_frozen_verify_do_not_bypass_pending_gate(tmp_path,monkeypatch):
    args=prepared(tmp_path,monkeypatch)
    manifest=bench.verify(args.output)
    assert manifest['workload']['generation_mode']=='natural'
    assert manifest['workload']['generation_limit']==32000
    assert manifest['generation_sampling']==bench.SAMPLING
    assert manifest['budget']['history_limits']==[262144]*8
    assert all(p['verify_session_restore'] is False for p in
        [bench.cap.read(args.output/(arm+'.json')) for arm in bench.ARMS])
    monkeypatch.setattr(bench.ready,'run_arm',lambda *a:pytest.fail('Pending qualification launched model'))
    with pytest.raises(ValueError,match='has not completed'):asyncio.run(bench.run(args))
    assert bench.cap.read(args.output/'status.json')['status']=='prepared'


@pytest.mark.parametrize('what',['runtime','fixture_source','profile_controls'])
def test_frozen_natural_contract_rejects_mutation(tmp_path,monkeypatch,what):
    args=prepared(tmp_path,monkeypatch)
    bench.verify(args.output)
    if what=='runtime':
        (args.output/'runtime/rolling-candidate-v2/runtime.py').write_text('unqualified modification')
    elif what=='fixture_source':
        args.fixture.write_text(args.fixture.read_text()+'\n')
    else:
        path=args.output/'native.json'; p=bench.cap.read(path)
        p['max_num_seqs']=4; bench.cap.save(path,p)
        # Even a recomputed file fingerprint cannot silently change the
        # preregistered semantic arm configuration.
        m=bench.cap.read(args.output/'manifest.json')
        m['profiles_sha256']['native']=bench.cap.sha(path)
        bench.cap.save(args.output/'manifest.json',m)
    with pytest.raises(ValueError,match='Runtime changed|Source fixture changed|Arm controls'):
        bench.verify(args.output)
