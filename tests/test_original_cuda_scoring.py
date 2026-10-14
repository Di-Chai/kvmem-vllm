import pytest
torch=pytest.importorskip('torch')
from vllm_kvmem.retrieval import mean_key_scores as reference
from vllm_kvmem.cuda_scoring import mean_key_scores

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')
@pytest.mark.parametrize('tokens,blocks,heads,kvheads',[(1,17,8,2),(33,257,16,4),(1025,41,8,2)])
def test_original_fp16_kernel_matches_page_softmax_definition(tokens,blocks,heads,kvheads):
    g=torch.Generator(device='cuda').manual_seed(831)
    q=torch.randn(2,tokens,heads,128,device='cuda',generator=g).half()
    k=torch.randn(2,blocks,kvheads,128,device='cuda',generator=g).half()
    eligible=torch.ones(blocks,device='cuda',dtype=torch.bool)
    eligible[:3]=False;eligible[-4:]=False;eligible[8]=False
    actual=mean_key_scores(q,k,eligible)
    expected=reference(q,k,eligible)
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=2e-5)
    assert torch.equal(actual[~eligible],torch.zeros_like(actual[~eligible]))
    assert abs(actual.sum().item()-tokens)<tokens*1e-5

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')
def test_no_eligible_history_has_zero_mass():
    q=torch.zeros(1,1,4,128,device='cuda',dtype=torch.float16)
    k=torch.zeros(1,4,2,128,device='cuda',dtype=torch.float16)
    assert torch.equal(mean_key_scores(q,k,torch.zeros(4,device='cuda',dtype=torch.bool)),
        torch.zeros(4,device='cuda'))
