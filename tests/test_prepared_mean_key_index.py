from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.core import cuda_scoring as scoring


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_candidate_preparation_keeps_each_fresh_query_and_conversion_order(monkeypatch,dtype):
    torch.manual_seed(74)
    calls=[]
    def score(q,m,scale):
        calls.append((q.clone(),m.clone(),scale))
        logits=torch.einsum('lqhd,lphd->lqhp',q.float(),m.float())*scale
        return logits.softmax(-1).sum((0,1,2))/(q.shape[0]*q.shape[2])
    monkeypatch.setattr(scoring,'kernel',lambda:NS(score=score))
    monkeypatch.setattr(scoring,'QUERY_CONVERSION_BYTES',32)
    means=torch.randn(3,19,2,4,dtype=torch.bfloat16)
    mask=torch.tensor([i%3!=0 for i in range(19)])
    index=scoring.prepare_mean_key_index(means,mask)
    for layer in range(3):
        q=torch.randn(1,5,2,4,dtype=dtype)
        calls.clear();before=scoring.mean_key_scores(q,means[layer:layer+1],mask);old=list(calls)
        calls.clear();after=scoring.prepared_layer_scores(q,index,layer)
        assert torch.equal(before,after)
        assert len(old)==len(calls)
        for a,b in zip(old,calls):
            assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        assert torch.count_nonzero(after[~mask])==0
    # Index values are independent of later mask/index-source mutation.
    saved=index.means.clone();means.zero_();mask.zero_()
    assert torch.equal(index.means,saved)


def test_empty_domain_never_launches_score_kernel(monkeypatch):
    monkeypatch.setattr(scoring,'kernel',lambda:pytest.fail('empty candidate domain launched GPU scoring'))
    m=torch.randn(2,7,2,4);index=scoring.prepare_mean_key_index(m,torch.zeros(7,dtype=torch.bool))
    assert torch.equal(scoring.prepared_layer_scores(torch.ones(1,3,2,4),index,1),torch.zeros(7))


@pytest.mark.parametrize('invalid',['mask_size','mask_dtype','means_rank','query_layers','layer_index'])
def test_invalid_prepared_domain_or_layer_is_rejected(invalid):
    means=torch.ones(2,7,2,4);mask=torch.ones(7,dtype=torch.bool)
    if invalid=='mask_size':mask=mask[:-1]
    elif invalid=='mask_dtype':mask=mask.float()
    elif invalid=='means_rank':means=means[0]
    if invalid.startswith(('mask','means')):
        with pytest.raises(ValueError):scoring.prepare_mean_key_index(means,mask)
    else:
        index=scoring.prepare_mean_key_index(means,mask)
        q=torch.ones(2 if invalid=='query_layers' else 1,3,2,4)
        with pytest.raises(ValueError):scoring.prepared_layer_scores(q,index,2 if invalid=='layer_index' else 0)
