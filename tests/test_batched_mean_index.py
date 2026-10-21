"""Batching sealed content indices preserves token-order FP32 arithmetic."""
import numpy as np
import pytest
import torch

from vllm_kvmem import _native


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize('count', [1, 7, 32])
def test_batch_matches_original_page_finalizers_bitwise(dtype, count):
    generator = torch.Generator().manual_seed(871)
    jobs, expected, actual = [], [], []
    for heads, dim in [(1, 64), (4, 256), (2, 17)]:
        # Retain a padded token/layer stride, extra unscored layers and a
        # nonzero token offset, as in record staging and restored views.
        raw = (torch.randn(4, 37, heads, dim, generator=generator)*3).to(dtype)[:, 2:34]
        a = torch.empty(3, heads, dim)
        b = torch.full_like(a, 123.)
        view = raw.view(torch.uint8).numpy()
        fp8 = dtype == torch.float8_e4m3fn
        finalize = _native.finalize_mean_fp8 if fp8 else _native.finalize_mean_bf16
        finalize(view, a.numpy(), count)
        expected.append(a)
        actual.append(b)
        jobs.append((view, b.numpy(), count, fp8))
    _native.finalize_mean_batch(jobs)
    for a, b in zip(expected, actual):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        assert torch.equal((a/count).half().view(torch.int16), (b/count).half().view(torch.int16))


def test_mixed_batch_keeps_each_owner_and_all_finite_fp8_codes():
    codes = np.arange(256, dtype=np.uint8)
    codes = codes[(codes != 127) & (codes != 255)].reshape(1, 254, 1, 1)
    fp8_sum = np.zeros((1, 1, 1), dtype=np.float32)
    bf16 = torch.arange(12.).reshape(1, 3, 1, 4).bfloat16()
    bf16_sum = np.zeros((1, 1, 4), dtype=np.float32)
    _native.finalize_mean_batch([(codes, fp8_sum, 254, True),
                                (bf16.view(torch.uint8).numpy(), bf16_sum, 3, False)])
    reference = np.empty_like(fp8_sum)
    _native.finalize_mean_fp8(codes, reference, 254)
    assert np.array_equal(fp8_sum.view(np.uint32), reference.view(np.uint32))
    assert np.array_equal(bf16_sum, np.array([[[12., 15., 18., 21.]]]))


@pytest.mark.parametrize('fault', ['count', 'shape', 'readonly', 'stride', 'arity'])
def test_invalid_later_job_does_not_partially_update_earlier_owner(fault):
    raw = np.ones((2, 32, 1, 64), dtype=np.uint8)
    first = np.full((1, 1, 64), 123., dtype=np.float32)
    later = first.copy()
    bad = (raw, later, 32, True)
    if fault == 'count':
        bad = (raw, later, 33, True)
    elif fault == 'shape':
        bad = (raw, later[..., :63], 32, True)
    elif fault == 'readonly':
        later.flags.writeable = False
    elif fault == 'stride':
        bad = (raw[..., ::-1], later, 32, True)
    else:
        bad = (raw, later, 32)
    with pytest.raises((ValueError, BufferError)):
        _native.finalize_mean_batch([(raw, first, 32, True), bad])
    assert np.all(first == 123.) and np.all(later == 123.)


def test_empty_batch_is_noop():
    _native.finalize_mean_batch([])


def test_authority_batches_sealed_pages_and_skips_partial_decode(tmp_path, monkeypatch):
    from vllm_kvmem.authority import RequestAuthority, append_authority_batch
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive

    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=24,
                         repository_blocks=16, sink_blocks=1, recent_blocks=0)
    raw = torch.arange(2*24*64).reshape(2, 24, 1, 64).float().remainder(100).to(torch.float8_e4m3fn)
    values = raw.view(torch.uint8).clone()
    original = _native.finalize_mean_batch
    calls, results = [], []
    def batch(jobs):
        calls.append(len(jobs))
        return original(jobs)
    for use_batch in (True, False):
        archive = AsyncArchive(str(tmp_path/str(use_batch)), 1024, 32)
        authority = RequestAuthority(str(use_batch), config, archive, 2, 1, 64,
            raw_dtype=torch.float8_e4m3fn, scoring_layers=1, mean_k_backend='qw3_cuda')
        monkeypatch.setattr(_native, 'finalize_mean_batch', batch if use_batch else None)
        try:
            begin = 0
            for candidates, accepted in [(12, 12), (4, 1), (7, 5), (6, 6)]:
                assert append_authority_batch([(authority, raw[:, begin:begin+candidates],
                    values[:, begin:begin+candidates],
                    dict(history_begin=begin, local_rope_begin=begin, accepted=accepted))]) == 1
                begin += accepted
            authority.publish_sealed(lambda: None)
            assert authority.history_end == 24
            results.append([(r.history_begin, r.key_sum.clone(), r.mean_key.clone())
                            for r in authority.records.values()])
        finally:
            assert authority.close() and archive.close()
    assert calls == [3, 1, 2]
    for a, b in zip(*results):
        assert a[0] == b[0]
        assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))
        assert torch.equal(a[2].view(torch.int16), b[2].view(torch.int16))
