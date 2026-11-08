"""KVMEM_AUTHORITY_BATCHED_MEANS: identical records, sums and FP16 MeanK."""
import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive

L, B, H, D = 5, 32, 2, 64


def owner(monkeypatch, fast, dtype):
    monkeypatch.setenv('KVMEM_AUTHORITY_BATCHED_MEANS', '1' if fast else '0')
    cfg = KVMemConfig(block_tokens=B, select_budget=B*64, gen_budget=B*16,
                      repository_blocks=512, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive('', L*B*H*D*(dtype.itemsize+1), 512, cpu_direct_record_seal=True)
    archive.async_seal = True
    a = RequestAuthority('r', cfg, archive, L, H, D, raw_dtype=dtype, scoring_layers=L-1,
                         mean_k_backend='qw3_cuda', resident_records=512, cpu_mean_index_simd=True)
    assert a._fast_record_init == (fast and callable(getattr(_native, 'finalize_mean_batch', None)))
    return a, archive


def payload(seed, tokens, dtype):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.float8_e4m3fn:
        raw = torch.randint(0, 256, (L, tokens, H, D), dtype=torch.uint8, generator=g)
        raw[raw == 127] = 126; raw[raw == 255] = 254
        raw = raw.view(torch.float8_e4m3fn)
    else:
        raw = torch.randn((L, tokens, H, D), generator=g).bfloat16()
    return raw, torch.randint(0, 256, (L, tokens, H, D), dtype=torch.uint8, generator=g)


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
def test_fast_records_are_bit_identical(monkeypatch, dtype):
    if not callable(getattr(_native, 'finalize_mean_batch', None)):
        pytest.skip('Requires the batched native MeanK finalizer')
    results = []
    for fast in (False, True):
        a, archive = owner(monkeypatch, fast, dtype)
        history = 0
        # Full pages, a partial tail, its completion, and another full run.
        for seed, tokens in ((1, 1024), (2, 40), (3, 24 + 512), (4, 96)):
            raw, values = payload(seed, tokens, dtype)
            assert append_authority_batch([(a, raw, values, dict(
                history_begin=history, local_rope_begin=history))]) == 1
            history += tokens
        records = sorted(a.records.values(), key=lambda r: r.history_begin)
        a.publish_sealed(lambda: archive.wait_for_capacity(.05))
        payloads = a.read_records([r.handle for r in records])
        snap = []
        for r, (k, v) in zip(records, payloads):
            snap.append((r.history_begin, r.count,
                         k[:, :r.count].contiguous().view(torch.uint8).numpy().tobytes(),
                         v[:, :r.count].contiguous().numpy().tobytes(),
                         None if getattr(r, 'mean_key', None) is None else
                         (tuple(r.mean_key.shape), r.mean_key.contiguous().view(torch.int16).numpy().tobytes())))
        results.append(snap)
    assert results[0] == results[1]
    assert sum(1 for s in results[1] if s[4] is not None) >= 40


def test_invalid_flag_is_rejected(monkeypatch):
    monkeypatch.setenv('KVMEM_AUTHORITY_BATCHED_MEANS', 'yes')
    cfg = KVMemConfig(block_tokens=B, select_budget=B*64, gen_budget=B*16,
                      repository_blocks=512, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive('', L*B*H*D*2, 64, cpu_direct_record_seal=True)
    with pytest.raises(ValueError, match='BATCHED_MEANS'):
        RequestAuthority('r', cfg, archive, L, H, D, raw_dtype=torch.float8_e4m3fn,
                         scoring_layers=L-1, mean_k_backend='qw3_cuda')
