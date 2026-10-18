"""Small CPU-only checks for independent G16 protocol and observer lifecycle."""
import asyncio
import builtins
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('capacity_chain_test',
    Path(__file__).parents[1]/'scripts/benchmark_capacity_chain.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


@pytest.fixture(autouse=True)
def isolated_queue_lock(tmp_path, monkeypatch):
    # Exercise the wrapper's locking code without touching the real benchmark lock.
    def open_local(path, *args, **kwargs):
        assert path == '/tmp/kvmem-coordination-qualification.lock'
        return builtins.open(tmp_path/'mock-gpu.lock', *args, **kwargs)
    monkeypatch.setattr(b, 'open', open_local, raising=False)


def profile():
    return dict(port=18534, max_num_seqs=8, max_model_len=262144, gpu_memory_utilization=.81,
        draft_method='mtp', draft_tokens=4, execution_mode='compiled_graph', verify_session_restore=False,
        query_replay_budget=16384, kvmem=dict(select_budget=65536, gen_budget=32768,
        archive_backend='cpu', cpu_read_cache_bytes=100_000_000_000, rolling_active_window=True))


def fake_launch(p, **kwargs):
    return dict(native_profile=dict(max_num_batched_tokens=16424, enable_prefix_caching=True,
        async_scheduling=True, kv_cache_dtype='fp8_e4m3'), argv=['original'], environment={})


def reference_profiles(root):
    for arm in b.ARMS: b.cap.save(root/(arm+'.json'), profile())


def test_only_native_fraction_and_ports_change_not_active_generation_budgets(tmp_path, monkeypatch):
    reference_profiles(tmp_path); monkeypatch.setattr(b, 'build_native_launch', fake_launch)
    before = profile(); result = b.profiles(tmp_path, 18574)
    for arm in b.ARMS:
        expected = dict(before, port=18574)
        if arm == 'native': expected['gpu_memory_utilization'] = .915
        assert result[arm] == expected
    assert b.WORKLOAD == dict(concurrency=8, warmup_turns=2, epoch_turns=16,
        duration_seconds=300, mode='tools', generation_mode='natural', generation_limit=16384)
    assert 'Per-user-per-epoch' in b.CACHE_SCOPE


def test_seven_native_slots_keep_real_prefill_budget_and_eight_clients(tmp_path):
    # Exercise the real launcher: its Query-derived budget otherwise shrinks
    # from16424 to16419 when the sequence-slot count decreases.
    source = b.cap.read(Path(__file__).parents[1]/'examples/qwen-nvfp4-offline-agent-k128-g128.json')
    source.update(profile(), max_num_batched_tokens=4224, prefill_chunk_tokens=2048)
    for arm in b.ARMS: b.cap.save(tmp_path/(arm+'.json'), source)
    old = b.profiles(tmp_path, 18574)
    new = b.profiles(tmp_path, 18574, 7)
    assert new[b.ARMS[1]] == old[b.ARMS[1]]
    before = b.build_native_launch(old['native'], port=18574, async_scheduling=True, environ={})
    after = b.build_native_launch(new['native'], port=18574, async_scheduling=True, environ={})
    assert after['native_profile'] == dict(before['native_profile'], max_num_seqs=7)
    assert after['native_profile']['max_num_batched_tokens'] == 16424
    assert b.WORKLOAD['concurrency'] == new[b.ARMS[1]]['max_num_seqs'] == 8
    delta = b.profile_delta(tmp_path, 18574, 7)
    assert delta['native'] == dict(port=[18534,18574], gpu_memory_utilization=[.81,.915],
        max_num_seqs=[8,7], max_num_batched_tokens=[4224,16424], query_replay_budget=[16384,None])
    assert delta[b.ARMS[1]] == dict(port=[18534,18574])
    assert b.profile_delta(tmp_path, 18574)['native'] == dict(
        port=[18534,18574], gpu_memory_utilization=[.81,.915])


@pytest.mark.parametrize('slots', [7,8])
@pytest.mark.parametrize('batch', [2048,4096])
def test_explicit_native_batch_tuning_reaches_cli_without_changing_workload(tmp_path, slots, batch):
    source = b.cap.read(Path(__file__).parents[1]/'examples/qwen-nvfp4-offline-agent-k128-g128.json')
    source.update(profile(), max_num_batched_tokens=4224, prefill_chunk_tokens=2048)
    for arm in b.ARMS: b.cap.save(tmp_path/(arm+'.json'), source)
    old = b.profiles(tmp_path, 18574, slots)
    new = b.profiles(tmp_path, 18574, slots, batch)
    assert new[b.ARMS[1]] == old[b.ARMS[1]]
    before = b.build_native_launch(old['native'], port=18574, async_scheduling=True, environ={})
    after = b.build_native_launch(new['native'], port=18574, async_scheduling=True, environ={})
    assert after['native_profile'] == dict(before['native_profile'], max_num_batched_tokens=batch)
    i = after['argv'].index('--max-num-batched-tokens')
    assert after['argv'][i+1] == str(batch)
    assert new['native']['query_replay_budget'] is None
    assert new['native']['max_model_len'] == 262144
    assert b.WORKLOAD['concurrency'] == new[b.ARMS[1]]['max_num_seqs'] == 8
    assert b.WORKLOAD['generation_limit'] == 16384
    delta = b.profile_delta(tmp_path, 18574, slots, batch)
    assert delta['native']['max_num_batched_tokens'] == [4224,batch]
    assert delta['native']['query_replay_budget'] == [16384,None]
    assert delta[b.ARMS[1]] == dict(port=[18534,18574])


@pytest.mark.parametrize('batch', [0,2048.0,True,1024,16424])
def test_undeclared_native_batch_tuning_is_rejected(tmp_path, batch):
    with pytest.raises(ValueError, match='Explicit native prefill'):
        b.profiles(tmp_path, 18574, native_batch_tokens=batch)


def test_native_query_allowance_cannot_silently_override_requested_small_batch(tmp_path, monkeypatch):
    reference_profiles(tmp_path)
    monkeypatch.setattr(b, 'build_native_launch', fake_launch)
    with pytest.raises(ValueError, match='Actual batch2048'):
        b.profiles(tmp_path, 18574, native_batch_tokens=2048)


@pytest.mark.parametrize('slots', [6, True, 7.0])
def test_other_admission_values_cannot_silently_change_the_experiment(tmp_path, slots):
    with pytest.raises(ValueError, match='seven-slot'):
        b.profiles(tmp_path, 18574, slots)


@pytest.mark.parametrize('field', ['async_action_metadata','early_capture_slot_release','action_profiling'])
def test_unrelated_candidate_not_imported(tmp_path, monkeypatch, field):
    reference_profiles(tmp_path); monkeypatch.setattr(b, 'build_native_launch', fake_launch)
    p = profile(); p['kvmem'][field] = True; b.cap.save(tmp_path/'native.json', p)
    with pytest.raises(ValueError, match='Unrelated'): b.profiles(tmp_path, 18574)


def test_actual_token_budget_is_not_reduced_with_request_output_cap(tmp_path, monkeypatch):
    reference_profiles(tmp_path)
    def wrong(p, **kwargs):
        result = fake_launch(p); result['native_profile']['max_num_batched_tokens'] = 16384
        return result
    monkeypatch.setattr(b, 'build_native_launch', wrong)
    with pytest.raises(ValueError, match='Actual batch16424'): b.profiles(tmp_path, 18574)


def test_profile_checks_do_not_invoke_live_launch_injection(tmp_path, monkeypatch):
    reference_profiles(tmp_path); monkeypatch.setattr(b, 'build_native_launch', fake_launch)
    def no_call(*args, **kwargs): raise AssertionError('Live run-only hook invoked during verification')
    monkeypatch.setattr(b.ready.native.launch, 'build_launch', no_call)
    assert b.profiles(tmp_path, 18574)['native']['gpu_memory_utilization'] == .915


@pytest.mark.parametrize('value', [False, True])
def test_observer_mode_is_explicit_and_legacy_runs_remain_observed(value):
    assert b.observer_enabled({}) is True
    assert b.observer_enabled(dict(arms=['native'],native_observer_enabled=value)) is value


@pytest.mark.parametrize('manifest', [
    {'native_observer_enabled':0,'arms':['native']},
    {'native_observer_enabled':'false','arms':['native']},
    {'native_observer_enabled':False,'arms':list(b.ARMS)},
    {'native_observer_enabled':False,'arms':[b.ARMS[1]]}])
def test_uninstrumented_mode_cannot_silently_change_other_arms(manifest):
    with pytest.raises(ValueError):b.observer_enabled(manifest)


def test_missing_gpu_observer_gate_prevents_prepare_before_any_output(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'reference_proof', lambda *a: ({}, {}))
    monkeypatch.setattr(b, 'profiles', lambda *a: {'native':profile()})
    def pending(*args): raise ValueError('Observer GPU qualification is not complete')
    monkeypatch.setattr(b, 'observer_proof', pending)
    output = tmp_path/'uncreated'
    args = NS(output=output, reference=tmp_path, observer_qualification=tmp_path, port=18574)
    with pytest.raises(ValueError, match='not complete'): b.prepare(args)
    assert not output.exists()


def test_release_runtime_cannot_sneak_in_unqualified_optimizations():
    before={name:'old' for name in b.RELEASE_RUNTIME_FILES}
    before.update({'src/vllm_kvmem/other.py':'same','scripts/old.py':'old helper'})
    qualified={name:'new' for name in b.RELEASE_RUNTIME_FILES}
    qualified['src/vllm_kvmem/other.py']='same'
    result=b.release_runtime_map(before,qualified)
    assert result['scripts/old.py']=='old helper'
    assert all(result[name]=='new' for name in b.RELEASE_RUNTIME_FILES)
    qualified['src/vllm_kvmem/other.py']='unqualified'
    with pytest.raises(ValueError,match='only the qualified'):
        b.release_runtime_map(before,qualified)
    del qualified['src/vllm_kvmem/other.py']
    with pytest.raises(ValueError,match='file set'):
        b.release_runtime_map(before,qualified)


@pytest.mark.parametrize('archive',[
    {}, {'session_checkpoint_deferrals':1}, {'session_checkpoint_deferrals':False},
    dict(session_checkpoint_deferrals=0,completed=[dict(checkpoint_kind='P',history_end=100,
        checkpoint_tokens=80,terminal_prefix={'status':'FINISHED_STOPPED'})]),
    dict(session_checkpoint_deferrals=0,completed=[dict(checkpoint_kind='M',history_end=100,
        checkpoint_tokens=99,terminal_prefix={'status':'FINISHED_LENGTH_CAPPED'})])])
def test_release_performance_gate_rejects_missing_or_partial_history(archive):
    with pytest.raises(ValueError):b.require_terminal_history(archive)


def test_release_performance_gate_accepts_full_m_and_ordinary_close():
    b.require_terminal_history(dict(session_checkpoint_deferrals=0,completed=[
        dict(checkpoint_kind='M',history_end=100,checkpoint_tokens=100,
             terminal_prefix={'status':'FINISHED_STOPPED'}),
        dict(history_end=1,terminal_prefix={'status':'FINISHED_STOPPED'})]))


def test_released_workload_keeps_tokens_and_workload_settings_and_restores_hook(tmp_path,monkeypatch):
    calls=[]; waits=[]; sentinel=object()
    async def transport(*args,**kwargs):
        calls.append((args,kwargs))
        return sentinel
    monkeypatch.setattr(b.work,'stream_call',transport)
    async def original(client,profile,fixture,directory,values):
        assert values.duration_seconds==300 and values.generation_limit==16384
        assert values.explicit_session_release is True
        assert await b.work.stream_call(client,profile,[11,12],'user',100,None,
            kind='workload',query=True,generation_mode='natural') is sentinel
        assert await b.work.stream_call(client,profile,[9],'user',1,None,
            kind='cleanup',query=False,generation_mode='fixed',release_session=values.explicit_session_release) is sentinel
        b.cap.save(directory/'traces.json',[
            dict(kind='workload',begin=1.,explicit_session_release=False),
            dict(kind='cleanup',begin=4.,explicit_session_release=True)])
        return dict(total_stream_output_tokens=101)
    async def monitor(capture):
        await asyncio.Event().wait()
    async def released(capture,not_before):
        waits.append((capture,not_before))
        return dict(resources={},archive={'session_checkpoint_deferrals':0})
    monkeypatch.setattr(b,'release_history_monitor',monitor)
    monkeypatch.setattr(b,'module',lambda *args:NS(wait_release=released))
    settings=NS(capture_directory=tmp_path/'capture',duration_seconds=300,generation_limit=16384)
    result=asyncio.run(b.released_workload(original,'client',profile(),{'eos':9},tmp_path,
        settings,dict(path=str(tmp_path/'qualification'))))
    assert result==dict(total_stream_output_tokens=101,explicit_session_release_verified=True)
    assert calls[0][0][2]==[11,12] and 'release_session' not in calls[0][1]
    assert calls[1][0][2]==[9] and calls[1][1]['release_session'] is True
    assert waits==[(tmp_path/'capture',4.)]
    assert b.work.stream_call is transport
    assert not hasattr(settings,'explicit_session_release')


def test_history_monitor_failure_cancels_work_and_restores_transport(tmp_path,monkeypatch):
    original_call=b.work.stream_call;cancelled=[]
    async def blocked(*args):
        try:await asyncio.Event().wait()
        finally:cancelled.append(True)
    async def broken(capture):
        raise ValueError('checkpoint capacity')
    monkeypatch.setattr(b,'release_history_monitor',broken)
    with pytest.raises(ValueError,match='checkpoint capacity'):
        asyncio.run(b.released_workload(blocked,None,profile(),{},tmp_path,
            NS(capture_directory=tmp_path),{'path':str(tmp_path)}))
    assert cancelled and b.work.stream_call is original_call


def test_release_orchestration_cannot_replace_qualified_transport_helpers(tmp_path):
    old=tmp_path/'old.py';new=tmp_path/'new.py'
    old.write_text('def stream_call():\n return 1\nasync def workload():\n return 2\n')
    new.write_text('def stream_call():\n return 1\nasync def workload():\n return 3\n')
    assert b.release_transport_hash(old,new)==b.cap.sha(new)
    new.write_text('def stream_call():\n return 99\nasync def workload():\n return 3\n')
    with pytest.raises(ValueError,match='transport or helpers changed'):
        b.release_transport_hash(old,new)


def test_pending_release_qualification_is_rejected(tmp_path,monkeypatch):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'))
    monkeypatch.setattr(b,'module',lambda *args:NS(verify=lambda root:{}))
    with pytest.raises(ValueError,match='has not passed'):
        b.release_proof(tmp_path,{},profile())


def streams(context_epoch=False):
    fixture = dict(eos=0, agent_plans=[]); rows=[]; events=[]
    for user in range(8):
        initial = [user+1]*245730 if context_epoch and user == 0 else [user+1,2,3]
        plan = dict(initial_tokens=initial, source_task=f't{user}', source_turn=10,
            turns=[dict(tool_tokens=[] if t==0 else ([99]*100 if t==2 else [90,91]), source_turn=10+t)
                   for t in range(16)])
        fixture['agent_plans'].append(plan)
        serial = 0
        def row(tokens, output, kind, epoch, turn):
            nonlocal serial
            begin=serial*10+user/10; serial+=1
            r=dict(agent=user, session=f'u{user}', request_id=f'u{user}-{serial}', begin=begin,end=begin+1,
                epoch=epoch,turn=turn,kind=kind,status='completed',input_tokens=len(tokens),
                input_sha256=b.work.digest(tokens),output_ids=list(output),output_sha256=b.work.digest(output),
                usage=dict(prompt_tokens=len(tokens),completion_tokens=len(output)),
                generation_mode='fixed' if kind=='cleanup' else 'natural',
                declared_generation_ceiling=1 if kind=='cleanup' else 16384,
                finish_reason='stop',source_metadata=None if kind=='cleanup' else dict(source_task=f't{user}',source_turn=10+turn))
            rows.append(r)
        row([0],[1],'cleanup',0,0)
        tokens=list(initial)
        for turn in range(2):
            if turn:tokens.extend(plan['turns'][turn]['tool_tokens'])
            output = [7]*4 if context_epoch and user == 0 else ([] if user == 0 else [7])
            row(tokens,output,'warmup',0,turn); tokens.extend(output)
        if context_epoch and user == 0:
            events.append(dict(event='epoch_ended',agent=user,epoch=0,completed_turns=2,reason='context_capacity',
                retained_history_tokens=len(tokens),next_prompt_tokens=len(tokens)+100,generation_limit=16384,max_model_len=262144))
            row([0],[1],'cleanup',0,2);tokens=list(initial)
            row(tokens,[8],'workload',1,0)
            row([0],[1],'cleanup',1,1)
        else:row([0],[1],'cleanup',0,2)
    return fixture,rows,events


def test_own_output_history_includes_zero_eos_and_two_complete_warmups():
    fixture,rows,events=streams()
    proof=b.own_history_proof(fixture,rows,events)
    assert proof['passed'] and proof['requests']==32 and proof['warmup_requests']==16


@pytest.mark.parametrize('change', ['history','cap','finish','warmup','task'])
def test_missing_history_reduced_cap_or_replaced_source_never_passes(change):
    fixture,rows,events=streams(); row=next(r for r in rows if r['agent']==1 and r['turn']==1 and r['kind']=='warmup')
    if change=='history':row['input_sha256']=b.work.digest(fixture['agent_plans'][1]['initial_tokens'])
    if change=='cap':row['declared_generation_ceiling']=8192
    if change=='finish':row['finish_reason']='length'
    if change=='warmup':row['kind']='workload'
    if change=='task':row['source_metadata']['source_task']='replacement'
    with pytest.raises(ValueError):b.own_history_proof(fixture,rows,events)


def test_context_epoch_keeps_full_history_and_restarts_only_after_guard():
    fixture,rows,events=streams(context_epoch=True)
    assert b.own_history_proof(fixture,rows,events)['passed']
    events[0]['reason']='declared_turn_limit'
    with pytest.raises(ValueError,match='epoch-end'):b.own_history_proof(fixture,rows,events)


def test_early_reset_cannot_be_disguised_as_a_context_epoch():
    fixture,rows,events=streams(context_epoch=True)
    # Keep all hashes intact, but make the not-yet-issued next full tool fit.
    fixture['agent_plans'][0]['turns'][2]['tool_tokens']=[]
    with pytest.raises(ValueError,match='Premature'):b.own_history_proof(fixture,rows,events)


def test_actual_native_pool_and_negative_all_phase_headroom_are_preserved(tmp_path,monkeypatch):
    d=tmp_path/'native';d.mkdir()
    gpu=[dict(memory_mib=90_000,wall_at=1),dict(memory_mib=96_000,wall_at=2)]
    native=[dict(values={'vllm:num_preemptions_total':0},raw_metrics='vllm:cache_config_info{block_size="1616",num_gpu_blocks="1177"} 1')]
    monkeypatch.setattr(b.ready,'records',lambda p:gpu if p.name=='gpu_samples.jsonl' else native)
    result=b.deployment_verdict(tmp_path,'native',dict(device_memory_identity=dict(total_mib=97887,reserved_mib=638)))
    assert result['accepted'] is False and result['minimum_sampled_free_mib']==1249
    assert '1177' in result['actual_cache_config_info'][0]  # Never the prior1192 constant.
    assert result['actual_configured_pool']==dict(block_size=1616,num_gpu_blocks=1177)
    assert result['measured_null_or_reserved_pages'] is None


@pytest.mark.parametrize('bad',['missing','changing'])
def test_unmeasured_or_changed_native_pool_cannot_support_capacity_claim(tmp_path,monkeypatch,bad):
    gpu=[dict(memory_mib=80_000,wall_at=1)]
    bodies=['vllm:cache_config_info{block_size="1616",num_gpu_blocks="1177"} 1',
        'vllm:cache_config_info{block_size="1616",num_gpu_blocks="1178"} 1']
    if bad=='missing':bodies=['vllm:cache_config_info{block_size="1616"} 1']
    samples=[dict(values={'vllm:num_preemptions_total':0},raw_metrics=line) for line in bodies]
    monkeypatch.setattr(b.ready,'records',lambda p:gpu if p.name=='gpu_samples.jsonl' else samples)
    with pytest.raises(ValueError,match='geometry'):
        b.deployment_verdict(tmp_path,'native',dict(device_memory_identity=dict(total_mib=97887,reserved_mib=638)))


def test_postseal_work_invalidates_observer_even_if_other_result_claims_pass(tmp_path):
    d=tmp_path/'native';(d/'observation/worker-1').mkdir(parents=True)
    (d/'observation/worker-1/events.jsonl').write_text('')
    snapshot=dict(complete_coverage=True,explicit_seal=True,post_close_dispatches=0)
    for name in ('observer-before.json','observer-seal.json','observer-final.json'):
        b.cap.save(d/name,snapshot)
    b.cap.save(d/'observation/worker-1/summary.json',dict(snapshot,post_close_dispatches=1))
    with pytest.raises(ValueError,match='after seal'):b.observer_postflight(tmp_path,NS())


@pytest.mark.parametrize('mismatch',[False,True])
def test_native_seal_only_follows_normal_workload_cleanup_and_fresh_counts(tmp_path,monkeypatch,mismatch):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'))
    manifest={};calls=[]
    monkeypatch.setattr(b,'verify',lambda *a:manifest)
    async def old_work(*a):calls.append('original_workload_including_all_EOS_cleanup');return {'total_stream_output_tokens':10}
    monkeypatch.setattr(b.work,'workload',old_work)
    async def wait(*a,**kw):pass
    monkeypatch.setattr(b.ready.bench.q,'wait_idle',wait)
    async def rpc(client,url,method):calls.append(method);return {}
    gate=NS(rpc=rpc)
    monkeypatch.setattr(b,'helper',lambda *a:gate)
    async def quiescent(**kw):
        calls.append('fresh_zero_running_waiting_and_count')
        return {'native_metrics':{'values':{'vllm:generation_tokens_total':9 if mismatch else 10}}}
    async def run_arm(root,arm,m,**kwargs):
        (root/arm).mkdir()
        await b.work.workload(None,{'native_vllm':True,'host':'127.0.0.1','port':18574},{},root/arm/'tools',NS(native_quiescent=quiescent))
        raise RuntimeError('stop fake arm before postflight')
    monkeypatch.setattr(b.ready,'run_arm',run_arm)
    with pytest.raises((RuntimeError,ValueError)):
        asyncio.run(b.run(NS(output=tmp_path)))
    if mismatch:
        assert calls==['native_executed_batch_snapshot','original_workload_including_all_EOS_cleanup','fresh_zero_running_waiting_and_count']
    else:
        assert calls==['native_executed_batch_snapshot','original_workload_including_all_EOS_cleanup',
            'fresh_zero_running_waiting_and_count','native_executed_batch_seal','native_executed_batch_snapshot']
    assert b.work.workload is old_work and b.ready.bench.q.wait_idle is wait


def test_uninstrumented_native_uses_original_launch_and_no_observer_rpc(tmp_path,monkeypatch):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'));calls=[]
    manifest=dict(arms=['native'],native_observer_enabled=False)
    monkeypatch.setattr(b,'verify',lambda *a:manifest)
    launch={'argv':['native-only'],'environment':{'VLLM_PLUGINS':''}}
    monkeypatch.setattr(b.ready.native.launch,'build_launch',lambda *a,**k:launch)
    def forbidden(*args,**kwargs):raise AssertionError('Disabled observer must not run')
    monkeypatch.setattr(b,'helper',lambda *a:NS(rpc=forbidden,inject_observer=forbidden))
    async def original(*args):calls.append(args);return {'original':True}
    monkeypatch.setattr(b.work,'workload',original)
    async def arm(root,arm,m,**kwargs):
        assert b.ready.native.launch.build_launch({}) is launch
        assert await b.work.workload(None,{'native_vllm':True},{},root,NS()) == {'original':True}
        raise RuntimeError('stop fake arm before postflight')
    monkeypatch.setattr(b.ready,'run_arm',arm)
    with pytest.raises(RuntimeError,match='stop fake arm'):
        asyncio.run(b.run(NS(output=tmp_path)))
    assert len(calls)==1 and b.work.workload is original


def test_changed_identity_after_idle_never_reaches_launch(tmp_path,monkeypatch):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'));checks=[]
    def verify(*a):checks.append(1);return {} if len(checks)<3 else {'changed':True}
    monkeypatch.setattr(b,'verify',verify);monkeypatch.setattr(b,'helper',lambda *a:NS())
    async def idle(*a,**kw):pass
    monkeypatch.setattr(b.ready.bench.q,'wait_idle',idle)
    async def arm(*a,**kwargs):
        await b.ready.bench.q.wait_idle(tmp_path,10)
        raise AssertionError('launch must not occur after changed identity')
    monkeypatch.setattr(b.ready,'run_arm',arm)
    with pytest.raises(ValueError,match='while waiting'):asyncio.run(b.run(NS(output=tmp_path)))
    assert b.ready.bench.q.wait_idle is idle


def batch_fixture():
    fixture=dict(agent_plans=[dict(initial_tokens=[1]*100) for _ in range(8)])
    rows=[dict(request_id=f'r{i}',response_id=f'cmpl-{i}',agent=i,epoch=0,turn=2,kind='workload',
        begin=1.,end=9.,input_tokens=150) for i in range(8)]
    batch=dict(execution_id=1,dispatch_monotonic=5.,req_ids=[f'cmpl-{i}-0-abcdef01' for i in range(8)],
        is_prefilling=[False]*8,prompt_lengths=[150]*8)
    return fixture,rows,batch


def test_native_formal_capacity_excludes_cleanup_prefill_and_warmup():
    fixture,rows,batch=batch_fixture(); window=dict(begin=3.,end=8.)
    def run():
        events=[dict(kind='dispatch',batch=batch),dict(kind='completed',execution_id=1,executed=True)]
        return b.formal_native_membership(events,rows,window,fixture)
    assert run()['pure_eight_long_workload_decode_steps']==1
    rows[0]['kind']='cleanup'
    result=run()
    assert result['pure_eight_long_workload_decode_steps']==0
    assert result['mixed_workload_decode_subset_histogram']=={'7':1}
    rows[0]['kind']='workload';batch['is_prefilling'][0]=True
    assert run()['mixed_workload_decode_subset_histogram']=={'7':1}
    batch['dispatch_monotonic']=2.
    assert run()['actual_membership']==[]


def test_native_only_successful_executions_count_as_actual_batch():
    fixture,rows,batch=batch_fixture()
    with pytest.raises(ValueError,match='Only completed'):
        b.formal_native_membership([dict(kind='dispatch',batch=batch)],rows,dict(begin=3.,end=8.),fixture)


def test_kvmem_eight_requires_exclusive_long_workload_for_whole_estimated_step():
    fixture,rows,_=batch_fixture();window=dict(begin=3.,end=8.)
    steps=[dict(started_at=5.,step_ms=100,decode_requests=8,prefill_requests=0,prefill_tokens=0)]
    result=b.conservative_kvmem_eight(steps,rows,window,fixture)
    assert result['conservative_long_workload_eight_steps']==1 and 'not_worker_IDs' in result['membership']
    rows.append(dict(kind='cleanup',begin=5.03,end=5.05))
    assert b.conservative_kvmem_eight(steps,rows,window,fixture)['conservative_long_workload_eight_steps']==0
    rows.pop();rows[0]['end']=5.05
    assert b.conservative_kvmem_eight(steps,rows,window,fixture)['conservative_long_workload_eight_steps']==0


@pytest.mark.parametrize('pids',[[],[99999]])
def test_post_owned_stop_checks_device_and_never_kills_unknown_pids(tmp_path,monkeypatch,pids):
    (tmp_path/'native').mkdir()
    monkeypatch.setattr(b.ready.bench.q,'gpu_state',lambda:dict(pids=pids))
    def no_stop(*args):raise AssertionError('Do not kill an unverified process')
    monkeypatch.setattr(b.ready.bench.q,'stop_owned',no_stop)
    if pids:
        with pytest.raises(ValueError,match='GPU processes remain'):b.post_stop_proof(tmp_path,'native')
        assert b.cap.read(tmp_path/'native/cleanup-required.json')['observed_gpu_pids']==pids
    else:
        assert b.post_stop_proof(tmp_path,'native')['gpu_empty'] is True
        assert not (tmp_path/'native/cleanup-required.json').exists()


@pytest.mark.parametrize('selected,observed',[(list(b.ARMS),True), (['native'],True),
    (['rolling-candidate-v2'],True), (['native'],False)])
@pytest.mark.parametrize('reason',['headroom_below4GiB','preemptions'])
def test_negative_deployment_preserves_raw_result_but_invalidates_accepted_comparison(tmp_path,monkeypatch,reason,selected,observed):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'));b.cap.save(tmp_path/'fixture.json',{})
    monkeypatch.setattr(b,'verify',lambda *a:dict(arms=selected,native_observer_enabled=observed))
    monkeypatch.setattr(b,'helper',lambda *a:NS())
    stopped=[]
    monkeypatch.setattr(b,'post_stop_proof',lambda root,arm:stopped.append(arm))
    async def cpu_inline(function,*args):return function(*args)
    monkeypatch.setattr(b.asyncio,'to_thread',cpu_inline)
    monkeypatch.setattr(b,'own_history_proof',lambda *a:dict(passed=True))
    def observer(*args):
        assert observed, 'Uninstrumented run must not load or claim observer evidence'
        return {}
    monkeypatch.setattr(b,'observer_postflight',observer)
    monkeypatch.setattr(b,'conservative_kvmem_eight',lambda *a:{})
    monkeypatch.setattr(b,'deployment_verdict',lambda root,arm,m:dict(accepted=arm!='native',reason=reason if arm=='native' else None))
    async def arm(root,name,m,**kwargs):
        tools=root/name/'tools';tools.mkdir(parents=True)
        b.cap.save(tools/'traces.json',[]);b.cap.save(tools/'window.json',dict(begin=1.,end=301.))
    monkeypatch.setattr(b.ready,'run_arm',arm)
    monkeypatch.setattr(b.ready,'records',lambda *a:[])
    analysis=dict(arms={'native':dict(output_tokens_per_second=99),
        'rolling-candidate-v2':dict(output_tokens_per_second=150,full_history_publication=dict(passed=True))})
    analysis['arms'] = {arm:analysis['arms'][arm] for arm in selected}
    monkeypatch.setattr(b,'module',lambda *a:NS(review=lambda root:analysis))
    asyncio.run(b.run(NS(output=tmp_path)))
    result=b.cap.read(tmp_path/'analysis.json');status=b.cap.read(tmp_path/'status.json')
    assert result['comparison_deployment_eligible'] is False
    if 'native' in selected: assert result['arms']['native']['output_tokens_per_second']==99
    assert result['comparison_complete'] == (selected == list(b.ARMS))
    assert ('native_executed_batch' in result) == ('native' in selected)
    if not observed:
        assert result['native_executed_batch']['available'] is False
        assert not (tmp_path/'native/observer-proof.json').exists()
    assert ('kvmem_eight_long_workload' in result) == (b.ARMS[1] in selected)
    assert status['status']=='completed_pending_review' and status['comparison_deployment_eligible'] is False
    assert stopped==selected


@pytest.mark.parametrize('samples,passed', [([99,101,99,101,102,103], True), ([99]*6, False)])
def test_host_wait_preserves_full_floor_and_requires_stable_recovery(tmp_path,monkeypatch,samples,passed):
    readings=iter(samples);clock=[0.]
    monkeypatch.setattr(b,'release_controller_heap',lambda:dict(own_heap_only=True))
    monkeypatch.setattr(b.cap,'host_memory',lambda:dict(MemAvailable=next(readings)))
    monkeypatch.setattr(b.time,'monotonic',lambda:clock[0])
    async def sleep(seconds):clock[0]+=seconds
    monkeypatch.setattr(b.asyncio,'sleep',sleep)
    m=dict(budget=dict(required_available_bytes=100),host_admission=dict(
        timeout_seconds=5,sample_seconds=1,consecutive_samples=3))
    if passed: asyncio.run(b.wait_host_headroom(tmp_path,'native',m))
    else:
        with pytest.raises(ValueError,match='full budget unchanged'):
            asyncio.run(b.wait_host_headroom(tmp_path,'native',m))
    proof=b.cap.read(tmp_path/'native-host-admission.json')
    assert proof['passed'] is passed and proof['required_available_bytes']==100
    assert len(proof['samples'])==6 and m['budget']['required_available_bytes']==100


def test_cancelled_host_wait_cannot_authorize_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'release_controller_heap',lambda:{})
    monkeypatch.setattr(b.cap,'host_memory',lambda:dict(MemAvailable=1))
    async def sleep(seconds):raise asyncio.CancelledError
    monkeypatch.setattr(b.asyncio,'sleep',sleep)
    m=dict(budget=dict(required_available_bytes=100),host_admission=b.HOST_ADMISSION)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(b.wait_host_headroom(tmp_path,'native',m))
    assert b.cap.read(tmp_path/'native-host-admission.json')['passed'] is False


def test_launch_jitter_requires_additional_margin_without_reducing_history(tmp_path,monkeypatch):
    readings=iter([101,102,103,129,130,127,128,129,130]);clock=[0.]
    monkeypatch.setattr(b,'release_controller_heap',lambda:{})
    monkeypatch.setattr(b.cap,'host_memory',lambda:dict(MemAvailable=next(readings)))
    monkeypatch.setattr(b.time,'monotonic',lambda:clock[0])
    async def sleep(seconds):clock[0]+=seconds
    monkeypatch.setattr(b.asyncio,'sleep',sleep)
    m=dict(budget=dict(required_available_bytes=100),host_admission=dict(
        timeout_seconds=10,sample_seconds=1,consecutive_samples=3,launch_margin_bytes=28))
    proof=asyncio.run(b.wait_host_headroom(tmp_path,'native',m))
    assert proof['passed'] and len(proof['samples'])==9
    assert proof['required_available_bytes']==m['budget']['required_available_bytes']==100
    assert proof['admission_floor_bytes']==128
    assert [r['consecutive_qualifying_samples'] for r in proof['samples']]==[0,0,0,1,2,0,1,2,3]


@pytest.mark.parametrize('margin',[-1,True,1.0])
def test_invalid_launch_margin_cannot_authorize_launch(tmp_path,margin):
    m=dict(budget=dict(required_available_bytes=100),host_admission=dict(
        b.HOST_ADMISSION,launch_margin_bytes=margin))
    with pytest.raises(ValueError,match='Invalid host launch margin'):
        asyncio.run(b.wait_host_headroom(tmp_path,'native',m))


@pytest.mark.parametrize('arms,flag',[(list(b.ARMS),True),(['rolling-candidate-v2'],True),
    (['native'],1),(['native'],'yes')])
def test_shared_prefix_mode_rejects_kvmem_and_non_boolean_flags(arms,flag):
    with pytest.raises(ValueError):
        b.native_cache_scope(dict(arms=arms,native_shared_prefix_cache=flag))


def test_shared_native_transport_only_omits_salt_and_preserves_original_payload():
    calls=[];sentinel=object()
    client=NS(stream=lambda *a,**kw:(calls.append((a,kw)) or sentinel),marker='rpc unchanged')
    adapter=b.SharedNativePrefixClient(client)
    payload=dict(prompt=[1,2,3],cache_salt='user0:epoch:0',request_id='turn2',
                 max_tokens=16384,seed=1000,temperature=.6,stream=True)
    original=copy.deepcopy(payload)
    assert adapter.stream('POST','http://127.0.0.1:18574/v1/completions',json=payload,timeout=99) is sentinel
    assert calls==[(('POST','http://127.0.0.1:18574/v1/completions'),
        dict(json={k:v for k,v in original.items() if k!='cache_salt'},timeout=99))]
    assert payload==original and adapter.marker=='rpc unchanged'
    assert b.native_cache_scope({})==b.CACHE_SCOPE
    assert b.native_cache_scope(dict(arms=['native'],native_shared_prefix_cache=True))==b.SHARED_CACHE_SCOPE
    for bad in [dict(payload,kvmem_release_session=True),dict(prompt=[1]),dict(payload,cache_salt='')]:
        with pytest.raises(ValueError,match='native completion'):
            adapter.stream('POST','http://127.0.0.1:18574/v1/completions',json=bad)
    with pytest.raises(ValueError,match='Unexpected endpoint'):
        adapter.stream('POST','http://127.0.0.1:18574/collective_rpc',json=payload)
    assert len(calls)==1


def test_shared_native_mode_is_applied_before_send_and_recorded_in_results(tmp_path,monkeypatch):
    b.cap.save(tmp_path/'status.json',dict(status='prepared'))
    manifest=dict(arms=['native'],native_observer_enabled=False,native_shared_prefix_cache=True)
    monkeypatch.setattr(b,'verify',lambda *a:manifest)
    monkeypatch.setattr(b,'helper',lambda *a:NS())
    sent=[]
    client=NS(stream=lambda *a,**kw:sent.append(kw['json']))
    async def original(client,profile,fixture,directory,values):
        client.stream('POST','http://127.0.0.1:18574/v1/completions',
                      json=dict(prompt=[10,20],cache_salt='original'))
        return dict(output_tokens=12,native_cache_salt_scope='old isolated wording')
    monkeypatch.setattr(b.work,'workload',original)
    async def arm(root,name,m,**kwargs):
        directory=root/name;directory.mkdir()
        result=await b.work.workload(client,dict(native_vllm=True),{},directory,NS())
        assert result['output_tokens']==12 and result['native_cache_salt_scope']==b.SHARED_CACHE_SCOPE
        assert b.cap.read(directory/'results.json')==result
        raise RuntimeError('end CPU transport check')
    monkeypatch.setattr(b.ready,'run_arm',arm)
    with pytest.raises(RuntimeError,match='end CPU transport check'):
        asyncio.run(b.run(NS(output=tmp_path)))
    assert sent==[dict(prompt=[10,20])]
    assert b.work.workload is original


@pytest.mark.parametrize('pids,cancel', [([],False),([9999],False),([],True)])
def test_final_device_check_uses_in_process_api_and_rejects_foreign_arrival(tmp_path,monkeypatch,pids,cancel):
    calls=[]
    nvml=NS(nvmlInit=lambda:calls.append('init'),
        nvmlDeviceGetHandleByIndex=lambda i:('device',i),
        nvmlDeviceGetComputeRunningProcesses=lambda device:(calls.append('device_check') or pids),
        nvmlShutdown=lambda:calls.append('shutdown'))
    monkeypatch.setitem(sys.modules,'pynvml',nvml)
    async def wait(*args):
        calls.append('host_wait')
        if cancel:raise asyncio.CancelledError
    monkeypatch.setattr(b,'wait_host_headroom',wait)
    if cancel:
        with pytest.raises(asyncio.CancelledError):asyncio.run(b.prelaunch_admission(tmp_path,'native',{}))
    elif pids:
        with pytest.raises(ValueError,match='Foreign GPU'):asyncio.run(b.prelaunch_admission(tmp_path,'native',{}))
    else:asyncio.run(b.prelaunch_admission(tmp_path,'native',{}))
    assert calls==(['init','host_wait','shutdown'] if cancel else
                   ['init','host_wait','device_check','shutdown'])


def test_native_memory_control_reaches_cli_and_preserves_kvmem_profile(tmp_path):
    source=b.cap.read(Path(__file__).parents[1]/'examples/qwen-nvfp4-offline-agent-k128-g128.json')
    source.update(profile(),max_num_batched_tokens=4224,prefill_chunk_tokens=2048)
    for arm in b.ARMS:b.cap.save(tmp_path/(arm+'.json'),source)
    before=b.profiles(tmp_path,18574,8,2048)
    after=b.profiles(tmp_path,18574,8,2048,.93)
    assert after[b.ARMS[1]]==before[b.ARMS[1]]
    assert after['native']==dict(before['native'],gpu_memory_utilization=.93)
    launch=b.build_native_launch(after['native'],port=18574,async_scheduling=True,environ={})
    assert launch['native_profile']['gpu_memory_utilization']==.93
    index=launch['argv'].index('--gpu-memory-utilization')
    assert float(launch['argv'][index+1])==.93
    delta=b.profile_delta(tmp_path,18574,8,2048,.93)
    assert delta['native']['gpu_memory_utilization']==[.81,.93]
    assert delta[b.ARMS[1]]==dict(port=[18534,18574])


@pytest.mark.parametrize('fraction',[True,1,.95,'0.93',float('nan')])
def test_undeclared_native_memory_fraction_is_rejected(tmp_path,fraction):
    with pytest.raises(ValueError,match='declared .915 or .93'):
        b.profiles(tmp_path,18574,8,2048,fraction)
