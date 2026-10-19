import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

spec=importlib.util.spec_from_file_location('qualify_structured_v2_test',
    Path(__file__).parents[1]/'scripts/qualify_graph_v2_sessions.py')
client=importlib.util.module_from_spec(spec);sys.modules[spec.name]=client;spec.loader.exec_module(client)


def test_frozen_fixture_applies_workload_and_timeout_without_http(tmp_path):
    path=tmp_path/'fixture.json'
    fixture=dict(kind='graph_v2_structured_sessions_v1',prompt_tokens=131072,
        output_tokens=257,concurrency=2,timeout=1200,initial_token_sha256=['a','b'])
    path.write_text(json.dumps(fixture))
    args=NS(fixture=path,timeout=1)
    assert client.load_fixture(args) == fixture
    assert (args.prompt_tokens,args.output_tokens,args.concurrency,args.timeout) == (131072,257,2,1200)
    fixture['output_tokens']=True
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='output_tokens'):
        client.load_fixture(args)


@pytest.mark.parametrize('quota', [None, 0, -1, True, 1.5])
def test_v2_fixture_requires_an_explicit_positive_append_quota(tmp_path, quota):
    path=tmp_path/'fixture.json'
    fixture=dict(kind='graph_v2_structured_sessions_v2',prompt_tokens=131072,
                 output_tokens=257,concurrency=2,continuation_append_tokens=quota)
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='continuation_append_tokens'):
        client.load_fixture(NS(fixture=path))


def test_v2_append_contract_is_explicit_and_v1_remains_short(tmp_path):
    path=tmp_path/'fixture.json'
    fixture=dict(kind='graph_v2_structured_sessions_v2',prompt_tokens=131072,
                 output_tokens=257,concurrency=2,continuation_append_tokens=49152)
    path.write_text(json.dumps(fixture))
    args=NS(fixture=path)
    client.load_fixture(args)
    assert args.continuation_append_tokens==49152
    fixture['kind']='graph_v2_structured_sessions_v1'
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='Unsupported'):
        client.load_fixture(NS(fixture=path))
    fixture.pop('continuation_append_tokens')
    path.write_text(json.dumps(fixture))
    client.load_fixture(args)
    assert args.continuation_append_tokens==0


@pytest.mark.parametrize('quota,context,match', [(16384,524288,'two full G'),(49152,150000,'model context')])
def test_invalid_output_pressure_geometry_fails_before_tokenizer_or_http(tmp_path,quota,context,match):
    fixture=tmp_path/'fixture.json'
    fixture.write_text(json.dumps(dict(kind='graph_v2_structured_sessions_v2',prompt_tokens=131072,
        output_tokens=257,concurrency=2,continuation_append_tokens=quota)))
    config=tmp_path/'profile.json'
    config.write_text(json.dumps(dict(max_model_len=context,kvmem=dict(gen_budget=16384))))
    protocol=tmp_path/'protocol.so'
    protocol.write_bytes(b'not loaded')
    output=tmp_path/'never_created'
    with pytest.raises(ValueError,match=match):
        client.run(NS(fixture=fixture,config=config,protocol=protocol,output=output))
    assert not output.exists()


def test_continuation_uses_actual_returned_call_ids_without_executing_model_input():
    payload=client.payload_for(1,'历史🙂')
    original=copy.deepcopy(payload)
    response=dict(content=[dict(type='tool_use',id='actual-1',name='InspectArchive',input=dict(record='primary')),
                           dict(type='tool_use',id='actual-2',name='InspectArchive',input=dict(record='secondary'))])
    following,calls=client.next_payload(payload,response)
    assert calls == ['actual-1','actual-2']
    assert payload == original
    assert following['messages'][-2] == dict(role='assistant',content=response['content'])
    assert [s['tool_use_id'] for s in following['messages'][-1]['content'][:-1]] == calls
    with pytest.raises(ValueError,match='complete real'):
        client.next_payload(payload,dict(content=[dict(type='text',text='I will use a tool.')]))


def test_long_tools_preserve_full_accepted_prefix_query_and_actual_tool_ids():
    payload=client.payload_for(1,'历史🙂')
    original=copy.deepcopy(payload)
    response=dict(content=[dict(type='tool_use',id='actual-1',name='InspectArchive',input=dict(record='primary')),
                           dict(type='tool_use',id='actual-2',name='InspectArchive',input=dict(record='secondary'))])
    state=NS(tokens=[11,12],output=[17,19])
    class Planner:
        encode=staticmethod(lambda text:list(text.encode()))
        def prepare(self, body, previous):
            suffix=self.encode(json.dumps(body['messages'][-1],ensure_ascii=False))
            return (previous.tokens+previous.output+suffix,None,None,None,None,True)
    planner=Planner()
    baseline,calls=client.next_payload(payload,response)
    short,short_calls,short_prepared=client.prepare_continuation(planner,payload,response,state)
    assert short==baseline and short_calls==calls and short_prepared==planner.prepare(baseline,state)
    following,actual,prepared=client.prepare_continuation(planner,payload,response,state,4096)
    assert actual==calls==['actual-1','actual-2'] and payload==original
    assert prepared[0][:4]==[11,12,17,19] and len(prepared[0])-4>=4096
    assert following['messages'][:-1]==baseline['messages'][:-1]
    assert following['messages'][-1]['content'][-1]==baseline['messages'][-1]['content'][-1]
    for new,old in zip(following['messages'][-1]['content'][:-1],baseline['messages'][-1]['content'][:-1]):
        assert new['content'].startswith(old['content']) and len(new['content'])>len(old['content'])
        assert {k:v for k,v in new.items() if k!='content'}=={k:v for k,v in old.items() if k!='content'}
    planner.prepare=lambda body,previous:([0],None,None,None,None,True)
    with pytest.raises(ValueError,match='accepted assistant IDs'):
        client.prepare_continuation(planner,payload,response,state,4096)
    with pytest.raises(ValueError,match='append quota'):
        client.prepare_continuation(planner,payload,response,state,True)


def test_client_transmits_actual_planner_tokens_and_metadata_unchanged(monkeypatch):
    bridge=client.module('qualified_bridge_transport_test',client.ROOT/'benchmarks/deepswe/claude_bridge.py')
    ids=[99,13,17]
    plan=dict(kvmem_locality_spans=[dict(begin=0,end=2,role='tool')],
        kvmem_locality_provenance=dict(schema_version=1,source='deepswe_protocol'))
    prepared=(ids,plan,None,None,None,True,None)
    sent=[]
    def open_request(request,**kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(id='response',usage=dict(prompt_tokens=3,completion_tokens=2),
            choices=[dict(token_ids=[7,8],text='complete',finish_reason='stop')])).encode())
    monkeypatch.setattr(client.urllib.request,'build_opener',lambda *args:NS(open=open_request))
    native,row=client.call(dict(port=8000,served_model_name='model'),bridge,'owner',prepared,257,30)
    assert sent[0]['prompt'] == ids
    assert {key:sent[0][key] for key in plan} == plan
    assert sent[0]['thinking_token_budget'] == 0 and sent[0]['min_tokens'] == 0
    assert row['output_ids'] == [7,8] and row['exact_output_prefix_reused'] is True


def evidence_fixture(tmp_path,monkeypatch):
    rows=[];graphs=[];action=[];target=[];state=[];completed=[]
    for owner in range(2):
        for turn in range(3):
            rid=f'{owner}-{turn}'
            provenance=dict(source='deepswe_protocol',exact_spans=3,token_ids_sha256=rid)
            rows.append(dict(response_id=rid,agent=owner,turn=turn,input_tokens=10,
                             output_ids=[5,6],expected_output_samples=1,metadata_check=dict(provenance=provenance)))
            event=dict(request_id=rid,phase='query',scoring_version='v2',alpha=.2,selected_tokens=8,
                locality_provenance=provenance)
            graphs.append(event)
            if turn==0:
                graphs.extend([dict(event,phase='pressure',seed_source='prefill') for _ in range(2)])
            action.append(dict(request_id=rid,samples=1,block_graph=dict(accepted_through=11)))
            completed.append(dict(request_id=rid,terminal_prefix=dict(history_end=11)))
            if turn:
                target.append(dict(request_id=rid,byte_equal=True,working_attention_layers=1,mtp_attention_layers=1))
                state.append(dict(request_id=rid,recurrent_byte_equal=True,checkpoint_kind='P' if turn==2 else 'M'))
    def write(name,data):
        (tmp_path/name).write_text(''.join(json.dumps(row)+'\n' for row in data))
    for name,data in [('block_graph.jsonl',graphs),('action_attention.jsonl',action),
                      ('verified_restores.jsonl',target),('verified_state_restores.jsonl',state)]:
        write(name,data)
    monkeypatch.setattr(client.q,'wait_terminal_publication',lambda *args:dict(completed=completed))
    profile=dict(kvmem=dict(graph_alpha=.2,select_budget=8))
    return rows,graphs,state,profile,write


def test_evidence_requires_pressure_provenance_and_an_actual_p_restore(tmp_path,monkeypatch):
    rows,graphs,state,profile,write=evidence_fixture(tmp_path,monkeypatch)
    assert len(client.check_evidence(tmp_path,rows,profile,2)) == 6
    graphs[1]['locality_provenance'] = dict(source='raw_token_ids')
    write('block_graph.jsonl',graphs)
    with pytest.raises(ValueError,match='structured locality'):
        client.check_evidence(tmp_path,rows,profile,2)
    graphs[1]['locality_provenance'] = rows[1]['metadata_check']['provenance']
    write('block_graph.jsonl',graphs)
    with pytest.raises(ValueError,match='stale or altered'):
        client.check_evidence(tmp_path,rows,profile,2)
    graphs[1]['locality_provenance'] = rows[0]['metadata_check']['provenance']
    write('block_graph.jsonl',graphs)
    for row in state: row['checkpoint_kind']='M'
    write('verified_state_restores.jsonl',state)
    with pytest.raises(ValueError,match='retained P'):
        client.check_evidence(tmp_path,rows,profile,2)


def test_output_seed_gate_requires_actual_changed_pressure_for_each_owner(tmp_path,monkeypatch):
    rows,graphs,state,profile,write=evidence_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='restored output attention'):
        client.check_evidence(tmp_path,rows,profile,2,require_output_pressure=True)
    added=[]
    for row in rows:
        if row['turn']==1:
            event=dict(request_id=row['response_id'],phase='pressure',scoring_version='v2',alpha=.2,
                selected_tokens=8,locality_provenance=row['metadata_check']['provenance'],
                seed_source='output',identity=False,added_blocks=1)
            graphs.append(event)
            added.append(event)
    write('block_graph.jsonl',graphs)
    assert len(client.check_evidence(tmp_path,rows,profile,2,require_output_pressure=True))==6
    added[-1]['identity']=True
    write('block_graph.jsonl',graphs)
    with pytest.raises(ValueError,match='nonidentity'):
        client.check_evidence(tmp_path,rows,profile,2,require_output_pressure=True)
    added[-1]['identity']=False
    added[-1]['seed_source']='prefill'
    write('block_graph.jsonl',graphs)
    with pytest.raises(ValueError,match='restored output attention'):
        client.check_evidence(tmp_path,rows,profile,2,require_output_pressure=True)
