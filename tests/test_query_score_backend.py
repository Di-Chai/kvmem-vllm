"""An explicit Query kernel choice cannot change operands or global scoring."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.core import cuda_scoring as scoring


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tokens', [1, 7])
def test_explicit_backend_keeps_conversion_chunks_mask_scale_and_query(dtype, tokens, monkeypatch):
    torch.manual_seed(142)
    means = torch.randn(2, 17, 2, 4, dtype=torch.bfloat16)
    mask = torch.tensor([i not in (0, 3, 13, 16) for i in range(17)])
    index = scoring.prepare_mean_key_index(means, mask)
    queries = torch.randn(1, tokens, 2, 4, dtype=dtype)
    query_copy, mean_copy = queries.clone(), index.means.clone()
    calls = []
    def implementation(name):
        def score(q, k, scale):
            calls.append((name, q.clone(), k.clone(), scale))
            logits = torch.einsum('lthd,lbhd->lthb', q.float(), k.float())*scale
            return logits.softmax(-1).sum((0,1,2))/q.shape[2]
        return score
    monkeypatch.setattr(scoring, 'kernel', lambda:NS(score=implementation('legacy'),
                                                    score_tiled=implementation('tiled')))
    monkeypatch.setattr(scoring, 'QUERY_CONVERSION_BYTES', 32)
    before = scoring.prepared_layer_scores(queries,index,1,scale=.125)
    legacy = list(calls);calls.clear()
    after = scoring.prepared_layer_scores(queries,index,1,scale=.125,backend='tiled')
    assert torch.equal(before,after) and torch.count_nonzero(after[~mask])==0
    assert torch.equal(queries,query_copy) and torch.equal(index.means,mean_copy)
    assert len(legacy)==len(calls)
    for old,new in zip(legacy,calls):
        assert old[0]=='legacy' and new[0]=='tiled'
        assert torch.equal(old[1],new[1]) and torch.equal(old[2],new[2])
        assert old[3]==new[3]==.125
        assert new[1].dtype==new[2].dtype==torch.float16
        assert new[1].is_contiguous() and new[2].is_contiguous()
        assert new[2].shape[1]==int(mask.sum())
    # The public/global API keeps the original backend and three-argument ABI.
    calls.clear()
    scoring.mean_key_scores(queries,means[1:2],mask,scale=.125)
    assert all(c[0]=='legacy' for c in calls)


@pytest.mark.parametrize('backend', ['auto', '', None, 1])
def test_unknown_backend_is_rejected_even_for_empty_domain(backend,monkeypatch):
    monkeypatch.setattr(scoring,'kernel',lambda:pytest.fail('Invalid backend dispatched'))
    index = scoring.prepare_mean_key_index(torch.ones(1,7,2,4),torch.zeros(7,dtype=torch.bool))
    with pytest.raises(ValueError,match='legacy or tiled'):
        scoring.prepared_layer_scores(torch.ones(1,3,2,4),index,0,backend=backend)


def test_requested_tiled_backend_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(scoring,'kernel',lambda:NS(score=lambda *args:pytest.fail('Fallback')))
    index = scoring.prepare_mean_key_index(torch.ones(1,7,2,4),torch.ones(7,dtype=torch.bool))
    with pytest.raises(RuntimeError,match='updated CUDA extension'):
        scoring.prepared_layer_scores(torch.ones(1,3,2,4),index,0,backend='tiled')


def test_empty_domain_never_dispatches_tiled(monkeypatch):
    monkeypatch.setattr(scoring,'kernel',lambda:pytest.fail('Empty domain dispatched'))
    index = scoring.prepare_mean_key_index(torch.ones(1,7,2,4),torch.zeros(7,dtype=torch.bool))
    assert torch.equal(scoring.prepared_layer_scores(torch.ones(1,3,2,4),index,0,backend='tiled'),
                       torch.zeros(7))


def test_packaged_cuda_source_matches_the_explicit_native_source():
    repo = Path(__file__).resolve().parents[1]
    assert (repo/'csrc/qw3_mean_k.cu').read_bytes()==(
        repo/'src/vllm_kvmem/cuda_sources/qw3_mean_k.cu').read_bytes()
