"""Independent token-level deletion and complete-selected-set oracles."""
import math

import pytest
import torch

from vllm_kvmem.recall_value_scoring import (REFERENCE_METHODS, AttentionMoments,
    aggregate_scores, aggregate_set_errors, block_scores, concatenate_moments,
    iter_attention_moments, moments_from_blocks, selected_set_error)


def direct_oracle(q, k, v, counts, eligible):
    """Flatten true tokens; independently rerun normalized attention after deletion."""
    tokens = [(b,t) for b,n in enumerate(counts) if eligible[b] for t in range(n)]
    kk = torch.stack([k[b,t] for b,t in tokens]).double()
    vv = torch.stack([v[b,t] for b,t in tokens]).double()
    logits = q.double() @ kk.T / math.sqrt(q.shape[-1])
    weights = logits.softmax(-1)
    out = weights @ vv
    result = {m:torch.zeros((len(q),len(counts)),dtype=torch.float64) for m in REFERENCE_METHODS}
    for block in range(len(counts)):
        member = torch.tensor([b==block for b,t in tokens])
        if not member.any():
            continue
        result['exact_mass'][:,block] = weights[:,member].sum(-1)
        result['block_value_norm'][:,block] = (weights[:,member]*vv[member].abs().sum(-1)).sum(-1)
        without = logits[:,~member].softmax(-1) @ vv[~member]
        result['block_deletion_error'][:,block] = (out-without).norm(dim=-1)
    return result, logits, vv, tokens, out


@pytest.mark.parametrize('block_batch',[1,2,4])
def test_value_scores_match_independent_removed_token_attention(block_batch):
    g=torch.Generator().manual_seed(301)
    k=torch.randn(4,5,3,generator=g);v=torch.randn(4,5,7,generator=g);q=torch.randn(6,3,generator=g)
    counts=[5,2,0,4];eligible=[True,False,True,True]
    for b,n in enumerate(counts):k[b,n:]=torch.nan;v[b,n:]=torch.nan
    expected,*_=direct_oracle(q,k,v,counts,eligible)
    parts=[moments_from_blocks(q,k[i:i+block_batch],v[i:i+block_batch],
        counts[i:i+block_batch],eligible[i:i+block_batch]) for i in range(0,4,block_batch)]
    actual=block_scores(concatenate_moments(parts))
    for method in REFERENCE_METHODS:
        torch.testing.assert_close(actual[method],expected[method],atol=5e-7,rtol=5e-6)


@pytest.mark.parametrize('gap',[30.,40.,1000.])
def test_deletion_remains_correct_when_one_minus_mass_rounds_to_zero(gap):
    q=torch.ones(1,1);k=torch.tensor([[[gap]],[[0.]]]);v=torch.tensor([[[1.]], [[-2.]]])
    moments=moments_from_blocks(q,k,v,[1,1],[True,True]);actual=block_scores(moments)
    expected,*_=direct_oracle(q,k,v,[1,1],[True,True])
    torch.testing.assert_close(actual['block_deletion_error'],expected['block_deletion_error'],atol=1e-12,rtol=1e-12)
    assert actual['block_deletion_error'][0,0]==pytest.approx(3.)
    assert torch.isfinite(actual['block_deletion_error']).all()


def test_constant_values_have_zero_deletion_error_despite_high_attention_mass():
    q=torch.tensor([[4.,-2.]]);k=torch.randn(3,4,2,generator=torch.Generator().manual_seed(2))
    v=torch.full((3,4,5),7.)
    actual=block_scores(moments_from_blocks(q,k,v,[4,4,4],[True]*3))
    torch.testing.assert_close(actual['block_deletion_error'],torch.zeros_like(actual['block_deletion_error']),atol=3e-6,rtol=0)
    assert actual['block_value_norm'].max()>1  # It must not be softmax-normalized again.


def test_whole_set_error_is_recomputed_and_single_block_errors_do_not_add():
    q=torch.zeros(1,1);k=torch.zeros(3,1,1);v=torch.tensor([[[1.]], [[-1.]], [[0.]]])
    moments=moments_from_blocks(q,k,v,[1]*3,[True]*3)
    individual=block_scores(moments)['block_deletion_error']
    assert float(individual[0,:2].sum())==pytest.approx(1.)
    assert float(selected_set_error(moments,[2]))==pytest.approx(0.)
    assert float(selected_set_error(moments,[0]))==pytest.approx(1.)
    assert float(selected_set_error(moments,[0,1,2]))==pytest.approx(0.)


def test_single_nonempty_block_deletion_and_empty_selected_set_fail_explicitly():
    moments=moments_from_blocks(torch.ones(1,1),torch.ones(2,1,1),torch.ones(2,1,1),[1,1],[True,False])
    with pytest.raises(ValueError,match='complementary'):block_scores(moments)
    with pytest.raises(ValueError,match='Selected'):selected_set_error(moments,[])
    with pytest.raises(ValueError,match='no valid tokens'):selected_set_error(moments,[1])


@pytest.mark.parametrize('query_batch,block_batch',[(1,1),(3,2)])
def test_query_layer_and_gqa_head_aggregation_and_set_errors(query_batch,block_batch):
    g=torch.Generator().manual_seed(601);q=torch.randn(2,5,6,3,generator=g)
    k=torch.randn(2,4,4,2,3,generator=g);v=torch.randn(2,4,4,2,3,generator=g)
    counts=[4,2,4,1];mask=[True,True,False,True]
    def reader(b,e,l,h):return k[l,b:e,:,h],v[l,b:e,:,h]
    def iterator():return iter_attention_moments(q,reader,counts,mask,kv_heads=2,
        query_batch=query_batch,block_batch=block_batch)
    actual,stats=aggregate_scores(iterator(),layers=2,heads=6)
    expected={m:torch.zeros(4,dtype=torch.float64) for m in REFERENCE_METHODS}
    errors=[]
    for l in range(2):
        for h in range(6):
            result,logits,vv,tokens,out=direct_oracle(q[l,:,h],k[l,:,:,h//3],v[l,:,:,h//3],counts,mask)
            for m in expected:expected[m]+=result[m].sum(0)/12
            keep=torch.tensor([b in (0,3) for b,t in tokens])
            errors.extend((logits[:,keep].softmax(-1)@vv[keep]-out).norm(dim=-1).tolist())
    for m in expected:torch.testing.assert_close(torch.tensor(actual[m],dtype=torch.float64),expected[m],atol=1e-6,rtol=5e-6)
    diag=aggregate_set_errors(iterator(),{'chosen':[0,3]},layers=2,heads=6)['chosen']
    assert diag['mean_query_layer_head_l2']==pytest.approx(sum(errors)/len(errors),rel=1e-5,abs=1e-6)
    assert diag['sum_query_mean_layer_head_l2']==pytest.approx(sum(errors)/12,rel=1e-5,abs=1e-6)
    assert stats['query_reduction']=='sum' and stats['head_layer_reduction']=='mean'


def test_value_scale_and_translation_have_expected_physical_effects():
    g=torch.Generator().manual_seed(80);q=torch.randn(2,3,generator=g);k=torch.randn(3,4,3,generator=g);v=torch.randn(3,4,3,generator=g)
    base=block_scores(moments_from_blocks(q,k,v,[4]*3,[True]*3))
    scaled=block_scores(moments_from_blocks(q,k,2.5*v,[4]*3,[True]*3))
    translated=block_scores(moments_from_blocks(q,k,v+10.,[4]*3,[True]*3))
    for m in ('block_value_norm','block_deletion_error'):
        torch.testing.assert_close(scaled[m],base[m]*2.5,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(translated['block_deletion_error'],base['block_deletion_error'],atol=3e-6,rtol=1e-5)


@pytest.mark.parametrize('query_batch,block_batch',[(1,1),(3,2),(128,128)])
def test_streaming_matches_reference_including_empty_tiles_and_gqa(query_batch,block_batch):
    from vllm_kvmem.recall_value_streaming import streaming_scores
    g=torch.Generator().manual_seed(912)
    q=torch.randn(2,7,6,3,generator=g);k=torch.randn(2,5,4,2,3,generator=g);v=torch.randn_like(k)
    counts=[0,4,3,2,1];mask=[True,True,False,True,True]
    def reader(b,e,l,h):return k[l,b:e,:,h],v[l,b:e,:,h]
    iterator=iter_attention_moments(q,reader,counts,mask,kv_heads=2,query_batch=2,block_batch=3)
    expected,_=aggregate_scores(iterator,layers=2,heads=6)
    actual,_=streaming_scores(q,reader,counts,mask,kv_heads=2,query_batch=query_batch,block_batch=block_batch)
    for m in REFERENCE_METHODS:
        torch.testing.assert_close(torch.tensor(actual[m]),torch.tensor(expected[m]),atol=3e-6,rtol=1e-5)
    from vllm_kvmem.recall_value_streaming import streaming_value_norm
    norm,_=streaming_value_norm(q,reader,counts,mask,kv_heads=2,query_batch=query_batch,block_batch=block_batch)
    torch.testing.assert_close(torch.tensor(norm),torch.tensor(expected['block_value_norm']),atol=3e-6,rtol=1e-5)


@pytest.mark.parametrize('dominant_block',[0,1,2,3])
def test_streaming_complement_survives_new_maximum_in_any_tile(dominant_block):
    from vllm_kvmem.recall_value_streaming import streaming_scores
    q=torch.ones(1,1,1,1);k=torch.zeros(1,4,1,1,1);k[:,dominant_block]=1000.
    v=torch.arange(4,dtype=torch.float32).reshape(1,4,1,1,1)
    def reader(b,e,l,h):return k[l,b:e,:,h],v[l,b:e,:,h]
    expected,*_=direct_oracle(q[0,:,0],k[0,:,:,0],v[0,:,:,0],[1]*4,[True]*4)
    actual,_=streaming_scores(q,reader,[1]*4,[True]*4,kv_heads=1,block_batch=1)
    for m in REFERENCE_METHODS:
        torch.testing.assert_close(torch.tensor(actual[m],dtype=torch.float64),expected[m].squeeze(0),atol=1e-12,rtol=1e-12)
