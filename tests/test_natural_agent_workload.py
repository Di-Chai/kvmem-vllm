import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('natural_timed_workload',ROOT/'scripts/timed_agent_workload.py')
timed=importlib.util.module_from_spec(spec);spec.loader.exec_module(timed)


class Response:
    def __init__(self,payload,output,*,usage=None,done=True,finish='stop',delays=(.001,)):
        self.payload,self.output=payload,output
        self.usage=len(output) if usage is None else usage
        self.done,self.finish,self.delays=done,finish,delays
        self.exited_with=None

    async def __aenter__(self):return self
    async def __aexit__(self,*args):self.exited_with=args[0]
    def raise_for_status(self):pass
    async def aiter_lines(self):
        split=max(1,len(self.output)//2)
        chunks=[self.output[:split],self.output[split:]]
        for index,chunk in enumerate(chunks):
            await asyncio.sleep(self.delays[min(index,len(self.delays)-1)])
            yield 'data: '+json.dumps(dict(id=self.payload['request_id'],choices=[dict(
                index=0,token_ids=chunk,finish_reason=self.finish if index==1 else None)]))
        yield 'data: '+json.dumps(dict(id=self.payload['request_id'],choices=[],
            usage=dict(completion_tokens=self.usage)))
        if self.done:yield 'data: [DONE]'


class Client:
    def __init__(self,output=None,**response_options):
        self.payloads=[];self.responses=[];self.output=output;self.options=response_options
    def stream(self,method,url,*,json):
        payload=copy.deepcopy(json);self.payloads.append(payload)
        cleanup=payload['ignore_eos']
        output=([8000] if cleanup else [50+len(self.payloads),60,70,2]) if self.output is None else self.output
        options=self.options.copy()
        delays=options.get('delays')
        if callable(delays):options['delays']=delays(payload)
        response=Response(payload,output,**options);self.responses.append(response)
        return response


def streamed(tmp_path,client,*,limit=32000,mode='natural',kind='workload'):
    recorder=timed.Recorder(tmp_path,1)
    try:
        return asyncio.run(timed.stream_call(client,dict(served_model_name='model',port=1),
            [1,2,3],'user',limit,recorder,agent=0,epoch=0,turn=0,kind=kind,request_id='request',
            new_input_tokens=3,query_plan={'kvmem_query_span':[1,2]},generation_mode=mode))
    finally:recorder.stream.close()


def test_natural_payload_matches_real_bridge_and_keeps_eos_token_ids(tmp_path):
    client=Client(output=[55,2])
    row=streamed(tmp_path,client)
    bridge_spec=importlib.util.spec_from_file_location('natural_test_bridge',ROOT/'benchmarks/deepswe/claude_bridge.py')
    bridge=importlib.util.module_from_spec(bridge_spec);sys.modules[bridge_spec.name]=bridge;bridge_spec.loader.exec_module(bridge)
    expected=bridge.backend_request(model='model',tokens=[1,2,3],session='user',max_tokens=32000,
        plan={'kvmem_query_span':[1,2],'kvmem_reselect':'auto'},backend_kind='kvmem')
    payload=client.payloads[0]
    assert all(payload[key]==value for key,value in expected.items())
    assert payload['ignore_eos'] is False and payload['thinking_token_budget']==4096
    assert row['output_ids']==[55,2] and row['usage']['completion_tokens']==2
    assert row['finish_reason']=='stop' and row['actual_output_tokens']==2
    assert row['generation_mode']=='natural' and row['declared_generation_ceiling']==32000


@pytest.mark.parametrize('options,output', [
    ({'done':False},[10,2]),({'usage':3},[10,2]),({'finish':None},[10,2]),
    ({'finish':'error'},[10,2]),({},[10,11,12,2])])
def test_incomplete_or_over_limit_natural_stream_is_rejected(tmp_path,options,output):
    client=Client(output=output,**options)
    with pytest.raises(RuntimeError,match='Incomplete natural stream'):
        streamed(tmp_path,client,limit=3)
    row=json.loads((tmp_path/'request.json').read_text())
    assert row['status']=='failed' and row['declared_generation_ceiling']==3
    assert row['output_ids']==output


def test_length_capped_natural_completion_is_recorded(tmp_path):
    row=streamed(tmp_path,Client(output=[9,8,7],finish='length'),limit=3)
    assert row['finish_reason']=='length' and row['actual_output_tokens']==3


def test_cleanup_keeps_original_fixed_single_token_even_when_natural_requested(tmp_path):
    client=Client(output=[9])
    row=streamed(tmp_path,client,limit=1,kind='cleanup')
    payload=client.payloads[0]
    assert payload['max_tokens']==1 and payload['ignore_eos'] is True and payload['temperature']==0
    assert 'thinking_token_budget' not in payload and row['generation_mode']=='fixed'


def fixture():
    return dict(eos=2,agent_plans=[dict(source_task='task',source_turn=1,
        initial_tokens=list(range(100,250)),query_plan={'kvmem_query_span':[10,12]},
        turns=[dict(output_tokens=17,tool_tokens=[] if i==0 else [700+i]*(3 if i%2 else 5),
                    tool_wait_seconds=0 if i==0 else .002,source_turn=i+1) for i in range(4)])])


def args(**changes):
    values=dict(duration_seconds=.04,concurrency=1,warmup_turns=2,epoch_turns=4,
        mode='tools',run_tag='natural',generation_mode='natural',generation_limit=320)
    values.update(changes);return SimpleNamespace(**values)


def run(tmp_path,client,settings=None,*,ceiling=480):
    return asyncio.run(timed.workload(client,dict(port=1,served_model_name='model',
        max_model_len=ceiling,kvmem=dict(gen_budget=512)),fixture(),tmp_path,settings or args()))


@pytest.fixture
def retirement(monkeypatch):
    calls=[]
    async def retired(*a,**kw):calls.append(kw);return {'retired':True}
    monkeypatch.setattr(timed,'wait_retirement',retired)
    return calls


def test_finite_capacity_epoch_ends_before_next_full_prompt_without_clipping(tmp_path,retirement):
    client=Client();data=fixture();original=copy.deepcopy(data)
    result=asyncio.run(timed.workload(client,dict(port=1,served_model_name='model',max_model_len=480,
        kvmem=dict(gen_budget=512)),data,tmp_path,args()))
    rows=json.loads((tmp_path/'traces.json').read_text())
    by_id={row['request_id']:row for row in rows}
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    ended=next(e for e in events if e['event']=='epoch_ended')
    assert ended['reason']=='context_capacity' and ended['completed_turns']==2
    assert ended['retained_history_tokens']==161 and ended['next_prompt_tokens']==166
    assert ended['generation_limit']==320 and ended['max_model_len']==480
    previous={};epochs=set()
    for payload in client.payloads:
        row=by_id[payload['request_id']]
        if row['kind']=='cleanup':
            assert payload['prompt']==[2] and payload['max_tokens']==1
            continue
        epochs.add(row['epoch'])
        assert payload['max_tokens']==320 and len(payload['prompt'])+320<=480
        assert row['actual_output_tokens']==4 and row['finish_reason']=='stop'
        if row['turn']==0:
            assert payload['prompt']==data['agent_plans'][0]['initial_tokens']
        else:
            before=previous[row['epoch']]
            assert payload['prompt']==before[0]['prompt']+before[1]['output_ids']+\
                data['agent_plans'][0]['turns'][row['turn']]['tool_tokens']
        previous[row['epoch']]=payload,row
    assert len(epochs)>1 and data==original
    assert result['generation_limit']==320 and result['generation_mode']=='natural'
    assert 'not on-policy' in result['scope'] and len(retirement)==2
    for epoch in epochs-{0}:
        first=next(r for r in rows if r['epoch']==epoch and r['kind']=='workload')
        cleanup=next(r for r in rows if r['epoch']==epoch-1 and r['kind']=='cleanup' and r['turn']>0)
        assert cleanup['end']<=first['begin']


def test_unfit_second_warmup_fails_before_measurement_without_shortening_limit(tmp_path,retirement):
    client=Client()
    with pytest.raises(ValueError,match='warmup cannot complete two'):
        run(tmp_path,client,ceiling=475)
    assert not(tmp_path/'window.json').exists()
    warm=[p for p in client.payloads if p['request_id'].endswith('-warmup')]
    assert len(warm)==1 and warm[0]['max_tokens']==320
    assert not any(p['request_id'].endswith('-workload') for p in client.payloads)


def test_unused_recorded_output_quota_does_not_replace_natural_ceiling(tmp_path,retirement):
    data=fixture()
    for turn in data['agent_plans'][0]['turns']:turn['output_tokens']=6000
    client=Client()
    result=asyncio.run(timed.workload(client,dict(port=1,served_model_name='model',max_model_len=1000,
        kvmem=dict(gen_budget=512)),data,tmp_path,args(duration_seconds=.01)))
    assert result['status']=='completed'
    assert all(p['max_tokens']==320 for p in client.payloads if not p['ignore_eos'])
    saved=json.loads((tmp_path/'workload.json').read_text())
    assert saved['per_turn_output_tokens']==[[6000]*4]


def test_started_natural_stream_drains_after_t1_without_client_cancellation(tmp_path,retirement):
    client=Client(delays=lambda p:(.001,.06) if p['request_id'].endswith('-workload') else (.001,))
    result=run(tmp_path,client,args(duration_seconds=.015),ceiling=1000)
    rows=json.loads((tmp_path/'traces.json').read_text());window=json.loads((tmp_path/'window.json').read_text())
    measured=[row for row in rows if row['kind']=='workload']
    assert len(measured)==1 and measured[0]['end']>window['end']
    assert measured[0]['status']=='completed' and measured[0]['actual_output_tokens']==4
    assert all(response.exited_with is None for response in client.responses)
    assert result['drained'] and result['drain_seconds']>.03
    assert result['stream_output_tokens_by_kind']['workload']==4


@pytest.mark.parametrize('changes,ceiling,match', [
    ({'generation_mode':'unknown'},1000,'generation mode'),
    ({'generation_limit':513},1000,'gen_budget'),
    ({'generation_limit':True},1000,'positive'),
    ({'warmup_turns':1},1000,'two complete'),
    ({},469,'Initial complete'),
    ({},None,'explicit max_model_len')])
def test_invalid_natural_settings_fail_before_http(tmp_path,retirement,changes,ceiling,match):
    client=Client()
    with pytest.raises(ValueError,match=match):run(tmp_path,client,args(**changes),ceiling=ceiling)
    assert not client.payloads
