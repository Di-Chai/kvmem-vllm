"""Candidate tiled dispatch retains the full MeanK probability definition."""
import os

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                             reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('pages', [1,3,8183,8192,8193])
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16])
def test_forced_tiled_scores_keep_mask_mass_and_current_stream(pages,dtype):
    from vllm_kvmem.cuda_scoring import prepare_mean_key_index,prepared_layer_scores
    from vllm_kvmem.retrieval import mean_key_scores as reference
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        generator = torch.Generator(device='cuda').manual_seed(481)
        means = torch.randn(2,pages+3,2,32,device='cuda',dtype=torch.bfloat16,generator=generator)
        mask = torch.ones(pages+3,device='cuda',dtype=torch.bool)
        mask[[0,pages//2+1,pages+2]]=False
        q = torch.randn(1,7,4,32,device='cuda',dtype=dtype,generator=generator)
        old_q,old_means=q.clone(),means.clone()
        index=prepare_mean_key_index(means,mask)
        actual=prepared_layer_scores(q,index,1,scale=.125,backend='tiled')
        expected=reference(q.half(),means[1:2].half(),mask,scale=.125)
        legacy=prepared_layer_scores(q,index,1,scale=.125)
    stream.synchronize()
    assert torch.isfinite(actual).all() and not actual[~mask].any()
    assert abs(actual.sum().item()-q.shape[1])<q.shape[1]*1e-5
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=2e-5)
    assert torch.equal(q,old_q) and torch.equal(means,old_means)
    # Above the legacy threshold this is literally the same tiled execution.
    if pages>8192: assert torch.equal(actual,legacy)
