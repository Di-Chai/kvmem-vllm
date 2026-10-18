import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import time

import pytest


spec = importlib.util.spec_from_file_location('rolling_qualification',
    Path(__file__).parents[1]/'scripts/qualify_rolling_window.py')
qual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qual)


def profile():
    return qual.qualification_profile(dict(model_format='safetensors',draft_method='mtp',
        draft_tokens=4,execution_mode='compiled_graph',served_model_name='model',kvmem=dict(
            block_tokens=32,archive_backend='cpu',cpu_read_cache_bytes=100_000_000_000,
            request_planning=True,async_refresh=True,async_restore=True,cpu_history_coordination=True,
            session_capture_terminal=True,action_pressure=True,session_checkpoint_policy='pm',
            session_capture_decode=False)),18532)


def fixture():
    return dict(eos=0,agent_plans=[dict(source_task=f'task-{i}',source_turn=10,
        initial_tokens=[i+1]*131072,query_plan=dict(kvmem_query_span=[18990,20807]),
        turns=[dict(output_tokens=6000,tool_tokens=[] if turn==0 else [10+i,20+turn],
            tool_wait_seconds=0,source_turn=10+turn,
            external_message_proof=None if turn==0 else {'source':'unchanged original tool'})
            for turn in range(3)]) for i in range(2)])


def test_fixture_keeps_long_original_ids_and_tool_messages_with_separate_output_quotas():
    data=fixture();before=copy.deepcopy(data)
    budget=qual.validate_fixture(data,profile())
    assert data == before
    assert budget['history_limits'] == [131072+6144+4]*2
    assert budget['queries'][0]['span'] == [18990,20807]
    assert budget['queries'][0]['replay_end']-budget['queries'][0]['replay_begin'] == 1831
    assert budget['required_available_bytes'] == budget['archive_bytes']+2*805306368+32*1024**3
    assert profile()['kvmem']['rolling_active_window'] is True
    assert profile()['max_num_seqs'] == 2
    assert profile()['kvmem']['gen_budget'] == max(qual.COUNTS) == 4096
    assert profile()['kvmem']['active_resident_max_fraction'] == 1e-6


def test_oversized_query_is_rejected_without_clipping():
    data=fixture();data['agent_plans'][0]['query_plan']['kvmem_query_span']=[18000,23000]
    with pytest.raises(ValueError,match='Query'):
        qual.validate_fixture(data,profile())
    assert data['agent_plans'][0]['query_plan']['kvmem_query_span'] == [18000,23000]


@pytest.mark.parametrize('mutation,match', [
    (lambda f:f.pop('eos'),'EOS'),
    (lambda f:f['agent_plans'][0].update(initial_tokens=[1]*131071),'131K'),
    (lambda f:f['agent_plans'][0]['turns'][1].update(tool_tokens=[]),'tool messages'),
    (lambda f:f['agent_plans'][0]['turns'][1].update(tool_wait_seconds=-1),'wait'),
    (lambda f:f['agent_plans'][1].update(source_task='task-0'),'independent')])
def test_invalid_fixture_fails_before_launch(mutation,match):
    data=fixture();mutation(data)
    with pytest.raises(ValueError,match=match):qual.validate_fixture(data,profile())


def test_live_checkout_cannot_execute_qualification(tmp_path):
    qual.cap.save(tmp_path/'manifest.json',dict(kind='rolling_window_qualification_v1'))
    with pytest.raises(ValueError,match='frozen entry'):qual.verify(tmp_path)


def test_manifest_verifies_exact_frozen_runtime_before_launch(tmp_path,monkeypatch):
    frozen=tmp_path/'frozen';runtime=tmp_path/'runtime'
    frozen.mkdir();runtime.mkdir()
    (frozen/'entry.py').write_text('qualified entry')
    (runtime/'entry.py').write_text('qualified runtime')
    data=fixture();p=profile()
    qual.cap.save(tmp_path/'fixture.json',data);qual.cap.save(tmp_path/'profile.json',p)
    monkeypatch.setattr(qual,'ROOT',frozen)
    monkeypatch.setattr(qual.cap,'native_source_identity',lambda:{'version':'test'})
    manifest=dict(kind='rolling_window_qualification_v1',source_sha256=qual.cap.hashes(frozen),
        runtime_sha256=qual.cap.hashes(runtime),profile_sha256=qual.cap.sha(tmp_path/'profile.json'),
        fixture_sha256=qual.cap.sha(tmp_path/'fixture.json'),native_source={'version':'test'},
        budget=qual.validate_fixture(data,p))
    qual.cap.save(tmp_path/'manifest.json',manifest)
    assert qual.verify(tmp_path)['budget'] == manifest['budget']
    (runtime/'entry.py').write_text('unqualified runtime')
    with pytest.raises(ValueError,match='runtime changed'):qual.verify(tmp_path)


def oracle_files(path):
    rows=[];targets=[];states=[];rolling=[]
    for user in range(2):
        for turn in range(3):
            response=f'response-{user}-{turn}';rid=response+'-0-12345678'
            rows.append(dict(agent=user,turn=turn,kind='workload',response_id=response,
                request_id=f'client-{user}-{turn}',input_tokens=131072+turn*1024,output_ids=[1]))
            if turn:
                targets.append(dict(request_id=rid,byte_equal=True,mtp_attention_layers=1,
                    working_attention_layers=17,target_kv_bytes=100,oracle_kind='fp8_immutable_rebuild'))
                states.append(dict(request_id=rid,recurrent_byte_equal=True))
            if turn==1:
                rolling.append(dict(request_id=rid,history_end=135168,selected_tokens=32768,
                    replay_tokens=0,cancelled=False,target_kv_byte_equal=True,recurrent_byte_equal=True,
                    mtp_attention_layers=1,attention_layers=17,recurrent_tensors=96,
                    target_kv_bytes=100,private_forward_steps=0,
                    previous_accepted_step=dict(history_begin=135166,accepted=2,scheduled=5)))
    for name,value in [('verified_restores',targets),('verified_state_restores',states),
                       ('verified_rolling_pressure',rolling),('model_step_latency',[
                           dict(decode_requests=2,prefill_requests=0,prefill_tokens=0)])]:
        (path/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in value))
    return rows,rolling


def test_byte_oracle_requires_each_resume_and_each_users_real_decode_pressure(tmp_path):
    rows,_=oracle_files(tmp_path)
    result=qual.oracle_evidence(tmp_path,rows)
    assert result['target_restore_records'] == result['recurrent_restore_records'] == 4
    assert len(result['verified_rolling_refreshes']) == 2
    assert result['cold_restore_required'] is True
    assert result['cancellation_validated'] is False
    (tmp_path/'verified_restores.jsonl').write_text('')
    with pytest.raises(ValueError,match='Each continuation'):qual.oracle_evidence(tmp_path,rows)


def test_hot_restore_cannot_substitute_for_cpu_rebuild_qualification(tmp_path):
    rows,_=oracle_files(tmp_path)
    path=tmp_path/'verified_restores.jsonl'
    target=[json.loads(line) for line in path.read_text().splitlines()]
    target[0]['oracle_kind']='native_terminal_window'
    path.write_text(''.join(json.dumps(row)+'\n' for row in target))
    with pytest.raises(ValueError,match='cold immutable FP8'):qual.oracle_evidence(tmp_path,rows)


@pytest.mark.parametrize('change', [dict(replay_tokens=1),dict(cancelled=True),
    dict(history_end=100),dict(private_forward_steps=1),dict(target_kv_byte_equal=False),
    dict(recurrent_byte_equal=False),dict(mtp_attention_layers=0),
    dict(previous_accepted_step=dict(history_begin=135165,accepted=2))])
def test_invalid_pressure_oracles_cannot_pass(tmp_path,change):
    rows,records=oracle_files(tmp_path);records[0].update(change)
    (tmp_path/'verified_rolling_pressure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='Each user'):qual.oracle_evidence(tmp_path,rows)


def test_single_token_boundary_does_not_claim_speculative_acceptance(tmp_path):
    rows,records=oracle_files(tmp_path)
    for row in records:row['previous_accepted_step']=dict(history_begin=135167,accepted=1)
    (tmp_path/'verified_rolling_pressure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='multi-token MTP'):qual.oracle_evidence(tmp_path,rows)


def test_valid_pressure_does_not_hide_a_failed_pressure_for_the_same_request(tmp_path):
    rows,records=oracle_files(tmp_path)
    records.append(dict(records[0],target_kv_byte_equal=False))
    (tmp_path/'verified_rolling_pressure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='Each user'):qual.oracle_evidence(tmp_path,rows)


def test_http_concurrency_alone_does_not_prove_a_concurrent_gpu_batch(tmp_path):
    rows,_=oracle_files(tmp_path)
    (tmp_path/'model_step_latency.jsonl').write_text(json.dumps(dict(
        decode_requests=1,prefill_requests=1,prefill_tokens=128))+'\n')
    with pytest.raises(ValueError,match='actual two-request'):qual.oracle_evidence(tmp_path,rows)


def test_history_proof_requires_entire_accepted_prefix_and_terminal_m():
    row=dict(request_id='client',response_id='response',input_tokens=131072,output_ids=[1]*1024)
    publication=dict(request_id='response-0-12345678',history_end=132095,
                     checkpoint_tokens=132095,checkpoint_kind='M')
    assert qual.history_evidence([row],[publication])[0]['history_tokens'] == 132095
    publication.update(history_end=32768,checkpoint_tokens=32768)
    with pytest.raises(ValueError,match='Incomplete'):qual.history_evidence([row],[publication])


def test_workload_appends_own_accepted_ids_original_tools_and_completes_eos_cleanup(tmp_path,monkeypatch):
    data=fixture();capture=tmp_path/'capture';capture.mkdir()
    requests=[];publications=[]
    async def call(client,p,tokens,session,count,recorder,**kwargs):
        copied=list(tokens);response=kwargs['request_id']
        row=dict(**kwargs,session=session,input_ids=copied,input_tokens=len(tokens),
                 output_ids=[900+kwargs['agent']]*count,response_id=response,
                 status='completed',begin=time.monotonic(),end=time.monotonic())
        requests.append(row);recorder.rows.append(row)
        publications.append(dict(request_id=response+'-0-12345678',history_end=len(tokens)+count-1,
            checkpoint_tokens=len(tokens)+count-1,checkpoint_kind='M'))
        qual.cap.save(capture/'archive.json',dict(completed=publications))
        return row
    async def retired(*args,**kwargs):return dict(expected_views=kwargs['expected_views'])
    monkeypatch.setattr(qual.work,'stream_call',call)
    monkeypatch.setattr(qual.work,'wait_retirement',retired)
    monkeypatch.setattr(qual,'oracle_evidence',lambda *args:dict(verified=True))
    result=asyncio.run(qual.workload(None,profile(),data,tmp_path/'workload',capture))
    assert result['cleanup_completed'] and result['total_stream_output_tokens'] == 2*6144+2
    assert len(requests)==8 and len(result['history_proofs'])==6
    for user in range(2):
        own=[r for r in requests if r['agent']==user]
        assert own[0]['input_ids']==data['agent_plans'][user]['initial_tokens']
        for turn in (1,2):
            assert own[turn]['input_ids'] == own[turn-1]['input_ids']+own[turn-1]['output_ids']+\
                data['agent_plans'][user]['turns'][turn]['tool_tokens']
        assert own[-1]['kind']=='cleanup' and own[-1]['input_ids']==[0]
