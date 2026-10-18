import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


spec = importlib.util.spec_from_file_location('offline',Path(__file__).parents[1]/'scripts/offline_workload.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


def options(**kwargs):
    return SimpleNamespace(**(dict(offline_requests=4,offline_concurrency=2,
        offline_input_lengths=[2,4],offline_output_tokens=2,offline_dataset=None,
        offline_warmup_seconds=1.,offline_sample_seconds=1.) | kwargs))


def profile():
    return dict(kvmem=dict(gen_budget=8,adaptive_prefill=False),max_num_seqs=2,
                max_model_len=100,served_model_name='test',port=1234)


def test_dataset_freezes_the_same_input_and_query_for_every_arm(tmp_path):
    fixture=dict(prompt=list(range(10)))
    summary=offline.prepare(options(),fixture,profile(),tmp_path)
    before=(tmp_path/'offline-dataset.jsonl').read_bytes()
    rows=[json.loads(l) for l in before.splitlines()]
    assert [len(r['prompt_token_ids']) for r in rows] == [2,4,2,4]
    assert rows[1]['query_span'] == [0,3]
    assert summary['input_tokens']==12 and summary['maximum_output_tokens']==8
    second=offline.prepare(options(offline_dataset=str(tmp_path/'offline-dataset.jsonl')),
                           fixture,profile(),tmp_path)
    assert second==summary and (tmp_path/'offline-dataset.jsonl').read_bytes()==before


@pytest.mark.parametrize('field,value', [('max_tokens',9),('prompt_token_ids',[]),
    ('query_span',[0,20]),('ignore_eos',1)])
def test_dataset_rejects_unexecutable_samples_before_gpu_launch(tmp_path,field,value):
    sample=dict(id='a',prompt_token_ids=[1,2,3],max_tokens=2)
    sample[field]=value
    source=tmp_path/'input.jsonl';source.write_text(json.dumps(sample)+'\n')
    with pytest.raises(ValueError):
        offline.prepare(options(offline_dataset=str(source)),{},profile(),tmp_path)


def test_offline_does_not_silently_enable_latency_control_or_undersupply_gpu(tmp_path):
    p=profile();p['kvmem']['adaptive_prefill']=True
    with pytest.raises(ValueError,match='adaptive_prefill'):
        offline.prepare(options(),{},p,tmp_path)
    with pytest.raises(ValueError,match='native active'):
        offline.prepare(options(offline_concurrency=1),{},profile(),tmp_path)


def test_completed_worker_refills_while_another_sample_is_still_running():
    async def run():
        slow=asyncio.Event();refilled=asyncio.Event();started=[];active=0;peak=0
        async def invoke(job):
            nonlocal active,peak
            active+=1;peak=max(peak,active);started.append(job['index'])
            if job['index']==0:
                await slow.wait()
            if job['index']==2:
                refilled.set()
            await asyncio.sleep(0)
            active-=1
            return dict(index=job['index'])
        state={}
        task=asyncio.create_task(offline.dispatch([dict(index=i) for i in range(6)],2,invoke,state=state))
        await asyncio.wait_for(refilled.wait(),timeout=1)
        assert not slow.is_set() and started[:3]==[0,1,2]
        slow.set();rows=await task
        assert [r['index'] for r in rows]==list(range(6)) and peak==2
        assert state['peak_in_flight']==2 and state['pending']==state['in_flight']==0
        assert state['submitted']==state['completed']==6
        assert state['started']<=state['drain_started']<=state['finished']
    asyncio.run(run())


@pytest.mark.parametrize('terminate,expected', [(True,True),(False,False)])
def test_independent_requests_have_no_session_and_incomplete_streams_are_failures(terminate,expected):
    seen=[]
    def serve(request):
        seen.append(json.loads(request.content))
        events=[dict(id='response',choices=[dict(token_ids=[8,9],text='answer')]),
                dict(id='response',choices=[],usage=dict(completion_tokens=2))]
        content=''.join('data: '+json.dumps(e)+'\n\n' for e in events)
        if terminate:content+='data: [DONE]\n\n'
        return httpx.Response(200,text=content)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            return await offline.infer(client,profile(),dict(index=0,id='sample-0',
                prompt_token_ids=[1,2,3],query_span=[0,2],max_tokens=2,ignore_eos=True))
    row=asyncio.run(run())
    assert row['success'] is expected and row['output_ids']==[8,9]
    assert row['output_text']=='answer'
    assert 'session_id' not in seen[0] and seen[0]['cache_salt'].startswith('offline-')
    assert seen[0]['kvmem_query_span']==[0,2]


def test_real_dataset_eos_may_complete_before_maximum_length():
    def serve(request):
        event=dict(id='r',choices=[dict(token_ids=[7])],usage=dict(completion_tokens=1))
        return httpx.Response(200,text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            return await offline.infer(client,profile(),dict(index=0,id='sample',
                prompt_token_ids=[1,2],query_span=None,max_tokens=8,ignore_eos=False))
    assert asyncio.run(run())['success']


def test_backlog_window_excludes_ramp_drain_and_failed_outputs_but_keeps_failures():
    from test_coordination_qualification import qualification
    rows=[dict(success=True,input_tokens=100,output_ids=list(range(10)),begin=0.,end=10.,
        timeline=[[1.,2],[4.,3],[8.,4],[10.,1]]),
        dict(success=False,input_tokens=200,output_ids=[1,2],begin=1.,end=9.,timeline=[[5.,2]])]
    state=dict(started=0.,finished=12.,drain_started=8.,peak_in_flight=2)
    result=offline.summarize(rows,state,2.,qualification.percentile)
    assert result['wall_seconds']==12 and result['output_tokens_per_second']==10/12
    assert result['failed_requests']==1 and result['successful_requests']==1
    assert result['steady_backlog_window']['output_tokens']==3
    assert result['steady_backlog_window']['output_tokens_per_second']==.5
    assert result['drain_seconds']==4
    state['drain_started']=1.
    assert offline.summarize(rows,state,2.,qualification.percentile)['steady_backlog_window'] is None


def test_offline_comparison_does_not_impose_stream_latency_gate():
    from test_coordination_qualification import qualification
    rows=[dict(pair=i,arm=arm,results={'offline':dict(output_tokens_per_second=rate,
        failed_requests=0,output_event_gap_p95_seconds=None,steady_backlog_window=None)})
        for i in range(5) for arm,rate in [('B0',100),('B3',110)]]
    assert qualification.compare(rows)['B3/offline']['passes_offline_throughput_gate']
    rows[-1]['results']['offline']['failed_requests']=1
    assert not qualification.compare(rows)['B3/offline']['passes_offline_throughput_gate']


def test_one_pair_does_not_report_a_degenerate_confidence_interval():
    from test_coordination_qualification import qualification
    rows=[dict(pair=0,arm=arm,results={'offline':dict(output_tokens_per_second=rate,
        failed_requests=0,steady_backlog_window=None)}) for arm,rate in [('B0',100),('B3',110)]]
    result=qualification.compare(rows)['B3/offline']
    assert result['throughput_ratio_median']==1.1 and result['paired_ratio_ci95'] is None
    assert not result['passes_offline_throughput_gate']


def test_new_offline_source_is_frozen_with_the_runner():
    from test_coordination_qualification import qualification
    assert 'scripts/offline_workload.py' in qualification.source_hashes()


def test_failed_baseline_cannot_become_a_throughput_win():
    from test_coordination_qualification import qualification
    rows=[dict(pair=0,arm=arm,results={'offline':dict(output_tokens_per_second=rate,
        failed_requests=1,output_event_gap_p95_seconds=None,steady_backlog_window=None)})
        for arm,rate in [('B0',0),('B3',100)]]
    assert not qualification.compare(rows)['B3/offline']['passes_offline_throughput_gate']


def test_actual_compute_occupancy_uses_only_the_backlog_window(tmp_path):
    path=tmp_path/'steps.jsonl'
    rows=[dict(started_at=t,decode_requests=d,prefill_requests=p,
               maintenance_ms=2.,model_ms=40.) for t,d,p in [(0,8,0),(2,2,1),(3,7,1),(5,8,0)]]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"unfinished":')
    result=offline.model_occupancy(path,dict(begin=2.,end=5.),8)
    assert result['samples']==2 and result['requests_per_step_mean']==5.5
    assert result['full_batch_sample_fraction']==.5 and result['decode_requests_per_step_mean']==4.5
    assert offline.model_occupancy(path,None,8)['requests_per_step_mean'] is None
    assert offline.model_occupancy(tmp_path/'absent',dict(begin=0,end=5),8)['samples']==0
