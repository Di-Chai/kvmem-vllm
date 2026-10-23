import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.retrieval_diagnostic_math import content_distributions


def test_page_average_can_reverse_a_strong_token_match():
    q=torch.ones(1,1,1)
    k=torch.tensor([10.,-10.,1.,1.]).reshape(4,1,1)
    scores=content_distributions(q,k,[2,2],[True,True],tile_pages=1)
    assert scores['mean'][0,1]>scores['mean'][0,0]
    assert scores['token'][0,0]>.99
    assert scores['max'][0,0]>.99


def test_global_token_softmax_gqa_mask_and_partial_page():
    torch.manual_seed(4)
    q=torch.randn(3,4,4)
    k=torch.randn(67,2,4)
    mask=[True,False,True]
    result=content_distributions(q,k,[32,32,3],mask,tile_pages=1)
    q=q.half().float().reshape(3,2,2,4)
    logits=torch.einsum('tkgd,nkd->tkgn',q,k)/2
    logits[...,32:64]=-torch.inf
    p=logits.softmax(-1)
    oracle=torch.stack([p[...,:32].sum(-1),p[...,32:64].sum(-1),p[...,64:].sum(-1)],-1).mean((1,2))
    torch.testing.assert_close(result['token'],oracle)
    maximum=torch.stack([p[...,:32].amax(-1),p[...,32:64].amax(-1),p[...,64:].amax(-1)],-1).mean((1,2))
    torch.testing.assert_close(result['token_max'],maximum)
    for name,v in result.items():
        if name=='token_max':continue
        torch.testing.assert_close(v.sum(-1),torch.ones(3))
        assert not v[:,1].any()


def test_no_attention_observation_for_pressure_or_missing_query():
    from vllm_kvmem.retrieval_diagnostics import capture_native,capture_archive
    capture_native(None,NS(command=NS(selection='pressure')),torch.ones(1))
    capture_native(None,NS(command=NS(selection='query')),None)
    capture_archive(None,NS(),None)


def test_guided_input_is_rejected_by_observer():
    from vllm_kvmem.retrieval_diagnostics import capture_native
    with pytest.raises(ValueError,match='original explicit Query'):
        capture_native(None,NS(command=NS(selection='query',query_span=(0,1)),guided_rows=[1]),torch.ones(1))


def test_launcher_retains_only_explicit_diagnostic_switch(tmp_path,monkeypatch,capsys):
    from vllm_kvmem import serve
    p=tmp_path/'profile.json';p.write_text(json.dumps({'kvmem':{'guided_query_tokens':0}}))
    monkeypatch.setattr(serve,'build_launch',lambda *a:dict(profile={},environment={}))
    serve.main(['--config',str(p),'--run-dir',str(tmp_path/'server'),'--dry-run','--retrieval-diagnostics'])
    assert json.loads(capsys.readouterr().out)['environment']['KVMEM_RETRIEVAL_DIAGNOSTICS']=='1'
    p.write_text(json.dumps({'kvmem':{'guided_query_tokens':512}}))
    with pytest.raises(ValueError,match='guided generation off'):
        serve.main(['--config',str(p),'--run-dir',str(tmp_path/'server'),'--dry-run','--retrieval-diagnostics'])


def test_single_token_probe_does_not_change_compressed_prefill_pressure():
    from dataclasses import replace
    from pathlib import Path
    from vllm_kvmem.request_plan import RequestPlan
    fixture=Path('artifacts/ruler-pooling-queue-20261022-r1/query_only/fixture.json')
    if not fixture.exists():pytest.skip('Frozen local fixture absent')
    rows=json.loads(fixture.read_text())['rows']
    rows=[r for r in rows if r['task']=='niah_multikey_2']
    def boundaries(plan,n,k):
        history=offset=0;events=[];capacity=k+16384
        for _ in range(300):
            action=plan.action(history,n,offset,capacity)
            if action=='query':return events,history
            if action:
                assert action=='pressure';events.append(history);offset=history-k
            count=plan.admit(history,2048,n,offset,capacity)
            assert count>0
            history+=count
        pytest.fail('Diagnostic prefill did not reach its Query')
    for row in rows:
        n=row['served_input_tokens']
        plan=RequestPlan.parse({'kvmem_query_span':row['served_query'],'kvmem_reselect':'force'},n,32,16384)
        short=replace(plan,output_tokens=1)
        original=replace(plan,output_tokens=8192)
        assert boundaries(short,n,36864)==boundaries(original,n,36864)
        assert boundaries(short,n,245760)[0]==[]


def test_full_prefix_observation_is_causal_and_rejects_compacted_positions():
    import importlib.util
    from pathlib import Path
    path=Path('scripts/analyze_ruler_retrieval_diagnostic.py')
    spec=importlib.util.spec_from_file_location('retrieval_analysis_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=dict(served_input_tokens=107,served_query=[80,100])
    observation=dict(native_length=100,dense_query_positions=list(range(80,100)))
    assert module.audit_full_prefix(observation,row)['unobserved_template_suffix_tokens']==7
    with pytest.raises(ValueError,match='complete original context'):
        module.audit_full_prefix(dict(observation,native_length=107),row)
    with pytest.raises(ValueError,match='complete original context'):
        module.audit_full_prefix(dict(observation,dense_query_positions=list(range(60,80))),row)
