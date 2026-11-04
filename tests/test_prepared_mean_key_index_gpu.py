"""Exact scorer operands/selection after immutable-domain preparation.

The legacy small-index kernel uses floating-point atomicAdd across Query
tokens/heads. Its own repeated scores need not be bit-identical for T > 1;
the one-Query and large-index paths also verify exact score bits. Full model
qualification separately requires every selected block and output token.
"""
import os
import importlib.util
from types import SimpleNamespace

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('pages',[4801,8192,8193])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('tokens',[1,61])
def test_prepared_layer_scores_preserve_native_contract(pages,dtype,tokens,monkeypatch):
    from vllm_kvmem import cuda_scoring
    from vllm_kvmem.cuda_scoring import mean_key_scores,prepare_mean_key_index,prepared_layer_scores
    spec=importlib.util.spec_from_file_location('frozen_mean_key_reference',os.environ['KVMEM_MEANK_REFERENCE_SOURCE'])
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    # The C++ scorer sources are unchanged and separately hash-bound. Run the
    # frozen old host implementation too, not just two paths sharing new code.
    native=cuda_scoring.kernel()
    calls=[]
    def score(q,k,scale):
        # Retain each actual argument until comparison: no in-place changes
        # are permitted to the immutable index or current layer Query.
        calls.append((q,k,scale))
        return native.score(q,k,scale)
    scorer=SimpleNamespace(score=score)
    reference.kernel=lambda:scorer
    monkeypatch.setattr(cuda_scoring,'kernel',lambda:scorer)
    generator=torch.Generator(device='cuda').manual_seed(181)
    means=torch.randn(3,pages+41,4,256,device='cuda',dtype=torch.bfloat16,generator=generator)
    mask=torch.ones(pages+41,device='cuda',dtype=torch.bool);mask[:41]=False
    index=prepare_mean_key_index(means,mask)
    for layer in range(3):
        calls.clear()
        q=torch.randn(1,tokens,24,256,device='cuda',dtype=dtype,generator=generator)
        before=mean_key_scores(q,means[layer:layer+1],mask)
        original=reference.mean_key_scores(q,means[layer:layer+1],mask)
        after=prepared_layer_scores(q,index,layer)
        repeated=reference.mean_key_scores(q,means[layer:layer+1],mask)
        assert len(calls)==4
        expected_q,expected_k,expected_scale=calls[1]
        for actual_q,actual_k,actual_scale in calls:
            assert actual_scale==expected_scale
            for actual,expected in ((actual_q,expected_q),(actual_k,expected_k)):
                assert actual.dtype==expected.dtype==torch.float16
                assert actual.shape==expected.shape and actual.is_contiguous()
                assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
        for candidate in (before,after,repeated):
            assert torch.isfinite(candidate).all() and not candidate[:41].any()
            if tokens==1 or pages>8192:
                assert torch.equal(original,candidate)
            else:
                # All contributions are nonnegative. Two FP32 accumulation
                # orders differ by at most 2*gamma_n/(1-gamma_n) relative to
                # the first result; n counts tokens * Query heads. This is
                # an analytical bound, not a tolerance fitted to a new kernel.
                terms=tokens*q.shape[2]
                unit_roundoff=torch.finfo(torch.float32).eps/2
                gamma=(terms-1)*unit_roundoff/(1-(terms-1)*unit_roundoff)
                torch.testing.assert_close(candidate,original,
                    rtol=2*gamma/(1-gamma),atol=0)
            # Same selectable quota after reserving the 41 mandatory blocks.
            budget=1152-41
            assert torch.equal(candidate.topk(budget).indices.sort().values,
                               original.topk(budget).indices.sort().values)
