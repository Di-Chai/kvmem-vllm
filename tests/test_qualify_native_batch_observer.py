"""Small CPU-only qualification-harness tests; never import torch or vLLM."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_observer_qualification_test',ROOT/'scripts/qualify_native_batch_observer.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)


def profile():
    return dict(model_format='safetensors',model='/model',tokenizer='/model',hf_config_path='/model',
        host='127.0.0.1',served_model_name='native',draft_method='mtp',draft_tokens=4,draft_quantization=None,
        execution_mode='compiled_graph',max_model_len=262144,max_num_seqs=8,max_num_batched_tokens=16444,
        prefill_chunk_tokens=1024,gpu_memory_utilization=.9,
        linear_backend_per_quant={'nvfp4_w4a4':'auto','fp8_w8a8':'auto'})


class Tokenizer:
    eos_token_id=7
    def encode(self,text,add_special_tokens=False):return [10,11,12,13]


def event_fixture():
    rows=[];events=[];key=0
    for turn in range(4):
        ids=[];lengths=[]
        for user in range(2):
            response=f'cmpl-{user}-{turn}';ids.append(response+'-0-1234abcd');lengths.append(100+user*100+turn*10)
            rows.append(dict(request_id=f'request-{user}-{turn}',response_id=response,input_tokens=lengths[-1],
                agent=user,turn=turn,output_ids=[7]*(1 if turn==3 else h.COUNTS[turn]),
                status='completed',kind='cleanup' if turn==3 else 'workload'))
        pending=[]
        for phase in (True,False) if turn<3 else (True,):
            key+=1
            events.append(dict(kind='dispatch',batch=dict(execution_id=key,num_reqs=2,req_ids=ids,
                is_prefilling=[phase,phase],prompt_lengths=lengths,scheduled_token_upper_counts=[5,5],
                dispatch_monotonic=float(key))))
            pending.append(key)
        # Deliberately reverse two AsyncOutputs to validate optional evidence, not require it on GPU.
        for done in reversed(pending):events.append(dict(kind='completed',execution_id=done,executed=True,raw_sampled_counts=[1,1]))
    seal=dict(explicit_seal=True,complete_coverage=True,dispatches=key,completed=key,pending=0,
        failed=0,unmatched=0,dropped=0,observer_errors=0,post_close_dispatches=0,armed=False,pending_sample=False,
        writer=dict(written=len(events),dropped=0,io_errors=0,queued=0,writer_alive=False))
    initial=dict(dispatches=0,completed=0,pending=0,observer_errors=0)
    return events,rows,seal,initial


def test_injection_exact_delta_and_preserves_async_plugin_sanitization(tmp_path):
    p=h.qualification_profile(profile(),18572)
    launch=h.module('native_agent_launch.py').build_launch(p,port=p['port'],async_scheduling=True,
        environ={'PYTHONPATH':'/untrusted/kvmem/src','VLLM_SERVER_DEV_MODE':'0'})
    got=h.inject_observer(launch,p,tmp_path/'diagnostics',tmp_path/'observer.json')
    assert got['argv']==launch['argv']+['--worker-extension-cls','native_executed_batch_observer.Extension']
    assert got['environment']['VLLM_PLUGINS']==''
    assert got['environment']['PYTHONPATH']==str(tmp_path/'diagnostics')
    assert got['environment']['VLLM_SERVER_DEV_MODE']=='1'
    assert len(got['environment_overrides'])==3 and 'VLLM_SERVER_DEV_MODE' not in launch['environment']
    assert launch['argv']==got['original_argv']


@pytest.mark.parametrize('host',['0.0.0.0','192.168.1.2','localhost','::1'])
def test_stream_harness_only_uses_exact_supported_loopback(host):
    p=profile();p['host']=host
    with pytest.raises(ValueError,match='loopback'):h.qualification_profile(p,18572)


def test_source_effective_batch_budget_does_not_shrink_with_two_slots():
    p=profile();p['query_replay_budget']=16384
    got=h.qualification_profile(p,18572)
    assert got['max_num_seqs']==2 and got['max_num_batched_tokens']==16424 and 'query_replay_budget' not in got
    assert p['max_num_seqs']==8 and p['query_replay_budget']==16384


def test_duplicate_extension_or_disabled_async_is_rejected(tmp_path):
    p=h.qualification_profile(profile(),18572)
    launch=h.module('native_agent_launch.py').build_launch(p,port=p['port'],async_scheduling=True,environ={})
    launch['argv']+=['--worker-extension-cls=other.Class']
    with pytest.raises(ValueError,match='existing worker'):h.inject_observer(launch,p,tmp_path,tmp_path/'config')
    launch=h.module('native_agent_launch.py').build_launch(p,port=p['port'],async_scheduling=False,environ={})
    with pytest.raises(ValueError,match='async scheduling'):h.inject_observer(launch,p,tmp_path,tmp_path/'config')


def test_actual_membership_completion_and_optional_reorder_evidence():
    events,rows,seal,initial=event_fixture()
    proof=h.analyze_observation(events,rows,seal,initial)
    assert proof['completed_steps']==7 and proof['max_actual_batch']==2
    assert proof['optional_coverage']['two_inflight']==proof['optional_coverage']['out_of_order_completion']=='covered'
    assert proof['client_accepted_tokens']==1794 and proof['pure_decode_step_histogram']=={'2':3}
    selection=h.observation_proof(events,rows,seal,initial,max_requests=8,window={'begin':2.,'end':5.})
    assert selection['selected_completed_steps']==3 and selection['pure_decode_step_histogram']=={'2':2}
    assert not any('duration' in key or 'weighted' in key for key in selection)


@pytest.mark.parametrize('damage',['unknown_owner','lost_completion','startup_leak','post_seal','missing_seal','wrong_prompt','log_count'])
def test_observation_fails_closed_for_incomplete_or_wrong_owners(damage):
    events,rows,seal,initial=event_fixture()
    if damage=='unknown_owner':events[0]['batch']['req_ids'][0]='foreign-0-1234abcd'
    elif damage=='lost_completion':events.pop()
    elif damage=='startup_leak':initial['dispatches']=1
    elif damage=='post_seal':seal['post_close_dispatches']=1
    elif damage=='missing_seal':seal['explicit_seal']=False
    elif damage=='wrong_prompt':events[0]['batch']['prompt_lengths'][0]+=1
    elif damage=='log_count':seal['writer']['written']-=1
    with pytest.raises(ValueError):h.analyze_observation(events,rows,seal,initial)


def test_no_reorder_or_two_inflight_is_reported_not_fabricated():
    events,rows,seal,initial=event_fixture();flat=[]
    dispatch={e['batch']['execution_id']:e for e in events if e['kind']=='dispatch'}
    complete={e['execution_id']:e for e in events if e['kind']=='completed'}
    for key in sorted(dispatch):flat.extend([dispatch[key],complete[key]])
    proof=h.analyze_observation(flat,rows,seal,initial)
    assert proof['max_inflight']==1 and proof['optional_coverage']['two_inflight']=='not_observed'
    assert proof['optional_coverage']['out_of_order_completion']=='not_observed'


def test_fixture_and_mock_tools_keep_own_outputs_and_cleanup_before_return(tmp_path):
    fixture=h.small_fixture(Tokenizer());p=h.qualification_profile(profile(),18572);h.validate_fixture(fixture,p)
    work=h.module('timed_agent_workload.py');seen=[]
    async def fake_stream(client,profile,tokens,session,count,recorder,**kwargs):
        seen.append((kwargs['agent'],kwargs['turn'],list(tokens),kwargs['kind']))
        row=dict(**kwargs,input_tokens=len(tokens),output_ids=[kwargs['agent']+80]*count,status='completed')
        recorder.rows.append(row);return row
    shared=NS(work=NS(Recorder=work.Recorder,stream_call=fake_stream))
    rows=asyncio.run(h.workload(None,p,fixture,tmp_path/'work',shared))
    assert len(rows)==8 and all(r['status']=='completed' for r in rows)
    for user in range(2):
        calls=[r for r in seen if r[0]==user]
        assert [r[3] for r in calls]==['workload']*3+['cleanup']
        expected=fixture['agents'][user]['initial_tokens']+[user+80]*256+fixture['agents'][user]['tool_tokens'][1]
        assert calls[1][2]==expected and calls[-1][2]==[fixture['eos']]


def test_prepare_and_verify_do_not_launch_gpu(tmp_path,monkeypatch):
    native=tmp_path/'native';native.mkdir();files={}
    for name in h.SOURCE_FILES:
        p=native/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# mock native source\n');files[str(p)]=h.sha(p)
    identity=tmp_path/'identity.json';h.save(identity,dict(source_directory=str(native),files=files))
    source=tmp_path/'profile.json';h.save(source,profile())
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:Tokenizer())))
    monkeypatch.setattr(h.subprocess,'Popen',lambda *a,**kw:pytest.fail('Preparation attempted a subprocess'))
    args=NS(output=tmp_path/'prepared',config=source,port=18572,installed_native=identity,
        observer_manifest=ROOT/'artifacts/native-executed-batch-observer-source-r2-20261017.json')
    h.prepare(args);m=h.verify(args.output,frozen_entry=False)
    assert h.read(args.output/'status.json')['gpu_started'] is False and m['concurrency']==2
    with pytest.raises(ValueError,match='not complete'):
        h.completed_proof(args.output,h.sha(ROOT/h.OBSERVER))
    with (args.output/'diagnostics/native_executed_batch_observer.py').open('a') as stream:stream.write('# changed\n')
    with pytest.raises(ValueError,match='Frozen'):h.verify(args.output,frozen_entry=False)


def complete_traces():
    fixture=h.small_fixture(Tokenizer());rows=[]
    def digest(x):return h.hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
    for user in range(2):
        plan=fixture['agents'][user];tokens=list(plan['initial_tokens'])
        for turn in range(4):
            if 0<turn<3:tokens.extend(plan['tool_tokens'][turn])
            expected=[fixture['eos']] if turn==3 else tokens
            count=1 if turn==3 else h.COUNTS[turn];output=[70+user]*count
            rows.append(dict(agent=user,turn=turn,kind='cleanup' if turn==3 else 'workload',
                session=f'user-{user}',request_id=f'request-{user}-{turn}',response_id=f'cmpl-{user}-{turn}',
                input_tokens=len(expected),input_sha256=digest(expected),output_ids=output,
                output_sha256=digest(output),actual_output_tokens=count,declared_generation_ceiling=count,
                usage=dict(prompt_tokens=len(expected),completion_tokens=count),status='completed',
                generation_mode='fixed',finish_reason='length',
                new_input_tokens=1 if turn==3 else len(expected) if turn==0 else len(plan['tool_tokens'][turn])))
            if turn<3:tokens.extend(output)
    return fixture,rows


@pytest.mark.parametrize('damage',[None,'input_hash','output_hash','usage','quota','session','omitted_tool'])
def test_cross_round_exact_own_outputs_and_tool_history(damage):
    fixture,rows=complete_traces();p=h.qualification_profile(profile(),18572)
    if damage=='input_hash':rows[1]['input_sha256']=rows[0]['input_sha256']
    if damage=='output_hash':rows[1]['output_sha256']='wrong'
    if damage=='usage':rows[1]['usage']['completion_tokens']-=1
    if damage=='quota':rows[1]['declared_generation_ceiling']-=1
    if damage=='session':rows[1]['session']='other'
    if damage=='omitted_tool':fixture['agents'][0]['tool_tokens'][1].append(99)
    if damage:
        with pytest.raises(ValueError):h.history_proof(rows,fixture,p)
    else:assert h.history_proof(rows,fixture,p)['complete_rounds']==6


def test_completed_gate_rechecks_history_files_final_seal_and_native_identity(tmp_path,monkeypatch):
    events,_,seal,initial=event_fixture();fixture,rows=complete_traces();p=h.qualification_profile(profile(),18572)
    by_response={r['response_id']:r for r in rows}
    for e in events:
        if e['kind']=='dispatch':
            b=e['batch'];b['prompt_lengths']=[by_response[x.rsplit('-0-',1)[0]]['input_tokens'] for x in b['req_ids']]
    result=dict(status='passed',cleanup_completed=True,observation=h.analyze_observation(events,rows,seal,initial),
        cross_round_history=h.history_proof(rows,fixture,p),explicit_seal=seal,final_snapshot=seal,shutdown_summary=seal,
        native_reconciliation=dict(passed=True,native_tokens=1794,client_tokens=1794,preemptions=0))
    (tmp_path/'workload').mkdir();worker=tmp_path/'observation/worker-1';worker.mkdir(parents=True)
    for name,value in [('status.json',dict(status='completed_pending_review')),('results.json',result),
        ('observer-before.json',initial),('seal-response.json',seal),('observer-final.json',seal),
        ('workload/traces.json',rows),('observation/worker-1/summary.json',seal),('fixture.json',fixture),
        ('profile.json',p),('manifest.json',{})]:h.save(tmp_path/name,value)
    (worker/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    m=dict(diagnostic_sha256={'native_executed_batch_observer.py':'observer-sha'},
           files_sha256={'installed-native-identity.json':'native-sha'})
    monkeypatch.setattr(h,'verify',lambda *a,**kw:m)
    checked=[];monkeypatch.setattr(h.gate,'installed_native_proof',lambda root:checked.append(root))
    assert h.completed_proof(tmp_path,'observer-sha')['passed'] and checked==[tmp_path]
    rows[1]['input_sha256']='changed-history';h.save(tmp_path/'workload/traces.json',rows)
    with pytest.raises(ValueError,match='Prompt'):h.completed_proof(tmp_path,'observer-sha')
