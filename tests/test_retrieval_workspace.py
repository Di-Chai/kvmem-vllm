"""CPU numerical and meta-device allocation-shape checks; no CUDA execution."""
import pytest
import torch

from vllm_kvmem.retrieval import mean_key_scores


def previous_scores(queries, means, eligible, *, scale=None, query_chunk=16):
    """Frozen pre-change all-layer oracle, including its FP32 reduction order."""
    layers, tokens, heads, dim = queries.shape
    kvheads = means.shape[2]
    score = torch.zeros(means.shape[1], dtype=torch.float32)
    index = means.float().permute(0, 2, 3, 1)
    scale = dim**-0.5 if scale is None else scale
    for begin in range(0, tokens, query_chunk):
        q = queries[:, begin:begin+query_chunk].float()
        q = q.reshape(layers, -1, kvheads, heads//kvheads, dim)
        logits = torch.einsum('ltkgd,lkdb->ltkgb', q, index)*scale
        mass = torch.nan_to_num(torch.softmax(logits.masked_fill(~eligible, -torch.inf), -1), nan=0.)
        score += mass.sum(dim=(0, 1, 2, 3))/(layers*heads)
    return score


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('chunk', [1, 7, 16, 64])
def test_streamed_layers_match_previous_full_block_gqa_oracle(dtype, chunk):
    rng = torch.Generator().manual_seed(471)
    q = torch.randn(3, 35, 6, 16, generator=rng).to(dtype)[..., ::2]
    k = torch.randn(3, 97, 3, 16, generator=rng).to(dtype)[..., ::2]
    mask = torch.arange(97) % 3 != 0
    assert not q.is_contiguous() and not k.is_contiguous()
    q_before, k_before, mask_before = q.clone(), k.clone(), mask.clone()
    actual = mean_key_scores(q, k, mask, scale=.37, query_chunk=chunk)
    expected = previous_scores(q, k, mask, scale=.37, query_chunk=chunk)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual.sum().item() == pytest.approx(35, rel=2e-6)
    assert not actual[~mask].any()
    assert torch.equal(q, q_before) and torch.equal(k, k_before) and torch.equal(mask, mask_before)


@pytest.mark.parametrize('mask_kind', ['none', 'single', 'holes'])
def test_full_block_softmax_mask_and_extreme_logits_remain_finite(mask_kind):
    q = torch.ones(2, 19, 4, 8)*100
    k = torch.arange(-4, 5).reshape(1, 9, 1, 1).expand(2, 9, 2, 8).float()*100
    mask = {'none': torch.zeros(9, dtype=torch.bool),
            'single': torch.arange(9) == 2,
            'holes': torch.arange(9) % 2 == 0}[mask_kind]
    actual = mean_key_scores(q, k, mask, query_chunk=7)
    expected = previous_scores(q, k, mask, query_chunk=7)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual.sum().item() == pytest.approx(0 if mask_kind == 'none' else 19, rel=2e-6)


def test_clear_margin_ranking_matches_previous_oracle():
    q = torch.ones(4, 33, 6, 8)
    strengths = torch.linspace(-1, 1, 11)
    k = strengths.reshape(1, 11, 1, 1).expand(4, 11, 2, 8)
    mask = torch.ones(11, dtype=torch.bool)
    expected = previous_scores(q, k, mask)
    actual = mean_key_scores(q, k, mask)
    assert torch.diff(expected).min() > .005
    assert torch.equal(torch.argsort(actual, descending=True),
                       torch.argsort(expected, descending=True))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


def test_full_256k_shapes_use_one_packed_layer_and_complete_block_softmax(monkeypatch):
    # Meta tensors exercise the actual loop and operator shapes without allocating
    # the 512MiB input index or touching CUDA.
    queries = torch.empty(16, 128, 24, 256, device='meta', dtype=torch.bfloat16)
    means = torch.empty(16, 8192, 4, 256, device='meta')
    mask = torch.ones(8192, device='meta', dtype=torch.bool)
    bmm, softmax = torch.bmm, torch.softmax
    products, normalizations = [], []
    def checked_bmm(q, index):
        assert q.is_contiguous() and index.is_contiguous()
        assert q.shape == (4, 96, 256) and index.shape == (4, 256, 8192)
        result = bmm(q, index)
        products.append(result.shape)
        return result
    def checked_softmax(values, dim):
        assert dim == -1 and values.shape[-1] == 8192
        normalizations.append(values.numel()*values.element_size())
        return softmax(values, dim)
    monkeypatch.setattr(torch, 'bmm', checked_bmm)
    monkeypatch.setattr(torch, 'softmax', checked_softmax)
    result = mean_key_scores(queries, means, mask)
    assert result.shape == (8192,) and result.device.type == 'meta'
    assert len(products) == len(normalizations) == 16*8
    assert max(normalizations) == 12*1024**2
    assert 4*256*8192*4 == 32*1024**2
    assert not torch.cuda.is_initialized()


def test_cpu_index_transfers_only_individual_packed_layers():
    from torch.utils._python_dispatch import TorchDispatchMode
    transfers = []
    class WatchTransfers(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten._to_copy.default and kwargs.get('device') == torch.device('meta'):
                transfers.append((tuple(args[0].shape), args[0].is_contiguous()))
            return func(*args, **kwargs)
    q = torch.empty(3, 17, 6, 8, device='meta')
    means = torch.randn(3, 97, 2, 8)
    mask = torch.ones(97, device='meta', dtype=torch.bool)
    with WatchTransfers():
        result = mean_key_scores(q, means, mask)
    assert result.shape == (97,)
    assert transfers == [((2, 8, 97), True)]*3


@pytest.mark.parametrize('backend,expected_device', [('torch_reference', 'cpu'), ('qw3_cuda', 'meta')])
def test_authority_keeps_cpu_index_only_for_reference_backend(tmp_path, monkeypatch, backend, expected_device):
    from test_historical_authority import setup, append, close
    import vllm_kvmem.retrieval as reference
    import vllm_kvmem.cuda_scoring as cuda
    a, archive, worker = setup(tmp_path)
    calls = []
    def score(queries, means, eligible):
        assert queries.device.type == eligible.device.type == 'meta'
        assert means.device.type == expected_device
        calls.append(tuple(means.shape))
        return torch.ones(means.shape[1])
    monkeypatch.setattr(reference if backend == 'torch_reference' else cuda, 'mean_key_scores', score)
    a.mean_k_backend = backend
    try:
        append(a)
        a.poll()
        a.select_for_queries(torch.empty(2, 1, 2, 2, device='meta'), observe_only=True)
        assert calls == [(2, 8, 1, 2)]
    finally:
        close(a, archive)


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_chunk_is_rejected_before_allocation(chunk):
    with pytest.raises(ValueError, match='GQA'):
        mean_key_scores(torch.ones(1,1,1,1), torch.ones(1,1,1,1),
                        torch.ones(1, dtype=torch.bool), query_chunk=chunk)


def test_foreign_non_cpu_index_or_mask_device_fails_closed():
    with pytest.raises(ValueError, match='device'):
        mean_key_scores(torch.ones(1,1,1,1), torch.empty(1,1,1,1, device='meta'),
                        torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match='device'):
        mean_key_scores(torch.ones(1,1,1,1), torch.ones(1,1,1,1),
                        torch.ones(1, device='meta', dtype=torch.bool))
