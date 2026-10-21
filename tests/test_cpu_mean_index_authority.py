"""The accepted authority sees identical page indices with SIMD enabled."""
import pytest
import torch

from vllm_kvmem.authority import MeanIndexStats, RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive


def test_shared_execution_counters_and_snapshots_remain_consistent():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    counters = MeanIndexStats()
    start = Barrier(4)
    def record_owner(_):
        start.wait()
        for _ in range(1000):
            counters.record(2, 1, True)
            observed = counters.snapshot()
            assert observed['completed_records'] == 2*observed['completed_batches']
            assert observed['vectorized_records'] == observed['fallback_records']
            assert observed['vectorized_records'] == observed['completed_batches']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record_owner, range(4)))
    assert counters.snapshot() == dict(completed_batches=4000, completed_records=8000,
                                       vectorized_records=4000, fallback_records=4000)


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
def test_partial_acceptance_and_multiple_owners_preserve_indices(tmp_path, dtype):
    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=128,
                         repository_blocks=16, sink_blocks=1, recent_blocks=0)
    generator = torch.Generator().manual_seed(571)
    raw = (torch.randn(2, 104, 2, 16, generator=generator)*3).to(dtype)
    values = torch.randint(0, 256, raw.shape, generator=generator, dtype=torch.uint8)
    outcomes = []
    for enabled in (False, True):
        archive = AsyncArchive(str(tmp_path/str(enabled)),
            2*32*2*16*(dtype.itemsize+1), 32)
        counters = MeanIndexStats()
        authorities = [RequestAuthority(str(i), config, archive, 2, 2, 16,
            raw_dtype=dtype, scoring_layers=1, mean_k_backend='qw3_cuda',
            cpu_mean_index_simd=enabled, mean_index_stats=counters) for i in range(2)]
        try:
            begin = 0
            for candidates, accepted in [(64, 64), (16, 5), (27, 27)]:
                entries = [(owner, raw[:, begin:begin+candidates],
                    values[:, begin:begin+candidates],
                    dict(history_begin=begin, local_rope_begin=begin, accepted=accepted))
                    for owner in authorities]
                assert append_authority_batch(entries) == 2
                begin += accepted
            for owner in authorities:
                owner.publish_sealed(lambda: None)
                assert owner.history_end == 96
            outcomes.append([[(r.history_begin, r.key_sum.clone(), r.mean_key.clone())
                for r in owner.records.values()] for owner in authorities])
            stats = counters.snapshot()
            assert stats['completed_batches'] == 4 and stats['completed_records'] == 6
            if not enabled:
                assert stats['vectorized_records'] == stats['fallback_records'] == 0
            elif dtype == torch.bfloat16:
                assert stats['vectorized_records'] == 0 and stats['fallback_records'] == 6
            else:
                assert stats['vectorized_records']+stats['fallback_records'] == 6
        finally:
            assert all(owner.close() for owner in authorities)
            assert archive.close()
    for old_owner, new_owner in zip(*outcomes):
        for before, after in zip(old_owner, new_owner):
            assert before[0] == after[0]
            assert torch.equal(before[1].view(torch.int32), after[1].view(torch.int32))
            assert torch.equal(before[2].view(torch.int16), after[2].view(torch.int16))
