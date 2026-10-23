import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_token_experiment import token_page_scores, select_token_queries


@pytest.mark.parametrize('chunk', [1, 2, 128])
def test_archive_token_page_scores_match_global_attention_gqa_partial_mask(chunk):
    torch.manual_seed(17)
    keys=torch.randn(3,71,2,8).to(torch.float8_e4m3fn)
    records=[NS(handle=i,history_begin=b,count=n) for i,(b,n) in enumerate([(0,32),(32,32),(64,7)])]
    def read(handles):
        return [(keys[:,records[h].history_begin:records[h].history_begin+records[h].count],None)
                for h in handles]
    a=NS(shape=(3,32,2,8),scoring_layers=2,history_end=71,read_records=read)
    q=torch.randn(2,3,4,8)
    result,stats=token_page_scores(a,records,q,[True,False,True],chunk_records=chunk)
    qq=q.half().float().reshape(2,3,2,2,8)
    z=torch.einsum('ltkgd,lnkd->ltkgn',qq,keys[:2].float())*8**-.5
    z[...,32:64]=-torch.inf
    p=z.softmax(-1)
    oracle=torch.stack([p[...,:32].sum(-1),p[...,32:64].sum(-1),p[...,64:].sum(-1)],-1)
    oracle=oracle.sum((0,1,2,3))/8
    torch.testing.assert_close(torch.tensor(result),oracle)
    assert result[1]==0 and abs(stats['score_sum']-3)<1e-6


def test_token_selection_preserves_native_transaction_and_graph_version(tmp_path,monkeypatch):
    from test_graph_lifecycle import fixture,selected_starts
    from test_core import Event
    a,archive,graph=fixture(tmp_path,mode='multi')
    a.graph_options.graph_alpha=0.;a.graph_options.pressure_policy='attention'
    monkeypatch.setattr('vllm_kvmem.query_token_experiment.token_page_scores',
        lambda *args:([.01,.10,.15,.12,.02,.60],dict(method='test')))
    plan=select_token_queries(a,torch.zeros(2,1,1,2),(),max_tokens=12)
    assert selected_starts(a,plan)==[8,12,20]
    assert a.graph_selection_report['seed_source']=='content_token_softmax_page_sum'
    graph.invalidate();a.memory.attach_completion(Event(True))
    with pytest.raises(RuntimeError,match='Stale'):a.memory.commit_selection()
    del plan
    assert a.close() and archive.close()


def test_archive_read_error_precedes_selection_and_releases_snapshot(tmp_path,monkeypatch):
    from test_graph_lifecycle import fixture
    a,archive,graph=fixture(tmp_path,mode='multi')
    a.graph_options.graph_alpha=0.;a.graph_options.pressure_policy='attention'
    before=a.memory.phase,graph.binding,graph.budget.used
    def fail(*args):raise OSError('archive failure')
    monkeypatch.setattr('vllm_kvmem.query_token_experiment.token_page_scores',fail)
    with pytest.raises(OSError,match='archive failure'):
        select_token_queries(a,torch.zeros(2,1,1,2),(),max_tokens=12)
    assert (a.memory.phase,graph.binding,graph.budget.used)==before
    assert a.close() and archive.close()


def test_token_launcher_is_explicit_and_rejects_graph_or_guided(tmp_path,monkeypatch,capsys):
    from vllm_kvmem import serve
    settings=dict(guided_query_tokens=0,graph_alpha=0.,graph_mode='multi',block_tokens=32)
    p=tmp_path/'profile.json';p.write_text('{}')
    monkeypatch.setattr(serve,'build_launch',lambda *a:dict(profile={'kvmem':settings},environment={}))
    args=['--config',str(p),'--run-dir',str(tmp_path/'server'),'--dry-run']
    serve.main(args)
    assert 'KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES' not in json.loads(capsys.readouterr().out)['environment']
    serve.main(args+['--experimental-query-token-scores','sum'])
    assert json.loads(capsys.readouterr().out)['environment']['KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES']=='sum'
    for key,value in [('graph_alpha',.2),('guided_query_tokens',512),('query_history',True)]:
        settings[key]=value
        with pytest.raises(ValueError,match='Query-only'):
            serve.main(args+['--experimental-query-token-scores','sum'])
        settings[key]=False if key=='query_history' else 0
