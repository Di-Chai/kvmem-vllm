from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.request_plan import RequestPlan, request_plan
from vllm_kvmem.guided_refresh import prepare_request
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.connector import RefreshCommand


def test_prefill_guidance_reserves_capacity_without_rewriting_final_query(monkeypatch):
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(
        guided_reserve=768, guided_reselect='prefill')))
    r=NS(num_prompt_tokens=262000, max_tokens=8192,
         kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[261900,261950]}})
    p=request_plan(r,32,16384)
    assert p.pressure_probe_tokens==768 and p.probe_tokens==0
    boundary=p.prefill_boundary(60000,53248)
    assert boundary+768+2*32<=53248
    assert p.action(60000,262000,60000-boundary,53248)=='pressure'
    assert p.action(p.replay_begin,262000,230000,53248)=='query'
    same=prepare_request(r,p,NS(history=0),NS(guided_query_tokens=512,guided_reselect='prefill'))
    assert same is p and not hasattr(r,'_kvmem_guided_epoch')


def job():
    ticket=NS(history_end=64, replay_tokens=2, original_block_ids=tuple(range(20)), window_tokens=32)
    command=RefreshCommand('owner',1,((0,ticket),),selection='pressure',input_tokens=(999,998),
                           asynchronous=True,guided_reason='prefill-pressure',guided_private=True)
    return RefreshJob(command)


def manager():
    obj=AsyncRefresh.__new__(AsyncRefresh)
    obj.options=NS(guided_query_tokens=8,guided_reserve=16,refresh_quantum_tokens=256,
                   guided_reselect='prefill',memory=NS(select_budget=64,gen_budget=64))
    from test_guided_policy import Tokenizer
    obj.tokenizer=Tokenizer(['recover',' evidence'])
    obj.guided_prompt=[21,22]
    obj.runner=NS(_kvmem_window_adapter=NS(offsets={}),
                  kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=16))]))
    return obj


def test_pressure_generation_never_reads_future_replay_tokens():
    obj=manager();j=job();obj.start_guided(j)
    assert j.stage=='guided_prompt' and j.guided_base==j.history==64
    calls=[]
    sequence=iter([(1,None),(2,torch.ones(1,1,1,2)),(1000,torch.ones(1,1,1,2))])
    def forward(value,tokens,begin,**kw):
        calls.append((tokens,begin));return next(sequence)
    obj._forward=forward
    submitted=[];obj._submit=lambda value:submitted.append(value)
    for _ in range(4):obj.step(j)
    assert submitted==[j]
    assert calls==[([21,22],64),([1],66),([2],67)]
    assert all(999 not in ids and 998 not in ids for ids,_ in calls)
    assert j.offset==0 and not j.query_rows and len(j.guided_rows)==2
    assert j.guided_diagnostics['guided_published_tokens']==2


def test_final_explicit_query_retains_raw_rows_and_never_generates_guidance():
    obj=manager();j=job()
    j.command=replace(j.command,selection='query',query_span=(64,66),
                      guided_reason='',guided_private=False)
    submitted=[];obj._submit=lambda value:submitted.append(value)
    obj.accept_query(j,torch.ones(1,2,1,2),2)
    assert submitted==[j] and len(j.query_rows)==1
    assert not j.guided_rows and j.guided_policy is None


def test_insufficient_private_scratch_is_explicit_fallback_before_forward():
    obj=manager();j=job();obj.options.guided_reserve=80
    submitted=[];obj._submit=lambda value:submitted.append(value)
    obj.start_guided(j)
    assert submitted==[j] and j.guided_fallback=='capacity' and j.guided_policy is None


def test_guided_pressure_native_picker_ignores_old_attention_and_graph(tmp_path,monkeypatch):
    from test_query_score_domain import fixture,finish
    from test_graph_pressure_v2 import configure
    a,archive=fixture(tmp_path,mode='attention')
    state=configure(a,policy='attention')
    state.scores={0:1000000.};state.prefill.values={0:(1000000.,1.)}
    before=state.snapshot(),dict(state.mass),state.samples,a.block_graph.binding
    a.graph_options.graph_aggregation='max'
    monkeypatch.setattr('vllm_kvmem.graph_integration.score_and_select',
                        lambda *a,**kw:pytest.fail('Guided pressure must not propagate'))
    plan=a.select_for_queries(torch.eye(2).reshape(1,2,1,2)*2**.5, guided_pressure=True)
    records=sorted(a.records.values(),key=lambda r:r.history_begin)
    assert [r.block_id for r in plan.remaps]==[records[1].handle.slot,records[2].handle.slot]
    report=a.graph_selection_report
    assert report['seed_source']=='guided_query' and not report['propagation_applied']
    assert report['phase']=='pressure' and report['transport_mask_count']==0
    assert before==(state.snapshot(),state.mass,state.samples,a.block_graph.binding)
    finish(a,archive,plan);del plan
    assert a.close() and archive.close()


def test_private_witness_detects_accidental_public_attention_commit():
    a=NS(history_end=64,action_history=NS(scores={},mass={},samples=0,
         prefill=NS(end=64,samples=8,chunks=2)),block_graph=NS(binding=('owner',1)))
    before=AsyncRefresh.pressure_witness(a)
    assert before==AsyncRefresh.pressure_witness(a)
    a.action_history.prefill.samples+=1
    assert before!=AsyncRefresh.pressure_witness(a)


def test_scheduler_marks_only_prefill_pressure_as_private():
    from test_scheduler_transactions import setup_scheduler
    scheduler,request,_=setup_scheduler()
    scheduler.guided_options=NS(guided_reselect='prefill',guided_query_tokens=512)
    request.num_computed_tokens=48
    request.num_prompt_tokens=request.num_computed_tokens+20
    scheduler.begin_refresh(request,16,replay_tokens=2,selection='pressure',
                            input_tokens=(1,2),asynchronous=True)
    command=scheduler.connector.commands[0]
    assert command.guided_reason=='prefill-pressure' and command.guided_private
    assert command.guided_boundary is False and command.query_span is None


def test_corrected_profile_changes_only_guidance_location_and_service_port():
    from test_ruler_pooling_protocol import contract
    from ruler_guided_pressure_protocol import expected_profile
    before,fixture=contract('max')
    after=expected_profile(fixture['reference_profile'])
    before.pop('port');after.pop('port')
    assert before['kvmem'].pop('guided_query_tokens')==0
    assert after['kvmem'].pop('guided_query_tokens')==512
    assert before['kvmem'].pop('guided_reselect')=='boundary'
    assert after['kvmem'].pop('guided_reselect')=='prefill'
    assert before==after


def test_diagnostic_state_verification_survives_launcher_environment_filtering(tmp_path,monkeypatch,capsys):
    from vllm_kvmem import serve
    config=tmp_path/'profile.json';config.write_text(json.dumps(dict(kvmem=dict(guided_reselect='prefill'))))
    monkeypatch.setattr(serve,'build_launch',lambda *a:dict(profile={},environment={}))
    serve.main(['--config',str(config),'--run-dir',str(tmp_path/'run'),
                '--dry-run','--verify-guided-pressure'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['environment']['KVMEM_VERIFY_GUIDED_PRESSURE']=='1'
