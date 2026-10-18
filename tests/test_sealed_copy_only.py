"""Completed-page copies preserve the original ordered MeanK finalizer."""
import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem import authority as authority_module
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive


def native_job(dtype=torch.float8_e4m3fn, *, count=8, offset=0):
    raw = torch.arange(3 * 10 * 64).reshape(3, 10, 1, 64).float()
    raw = (raw.remainder(101) - 50).to(dtype)
    values = raw.view(torch.uint8)[..., :64].clone()
    target = torch.full((3, 8, 1, 64), 9., dtype=dtype)
    target_values = torch.full((3, 8, 1, 64), 13, dtype=torch.uint8)
    sums = torch.full((2, 1, 64), 17.)
    job = (raw.view(torch.uint8).numpy(), values.numpy(),
           target.view(torch.uint8).numpy(), target_values.numpy(),
           None, sums.numpy(), 1, offset, count, True)
    return job, (raw, values, target, target_values, sums)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize('offset,count', [(0, 8), (1, 7), (7, 1)])
def test_completed_copy_preserves_sum_and_all_target_mtp_bytes(dtype, offset, count):
    job, (raw, values, target, target_values, sums) = native_job(dtype, count=count, offset=offset)
    raw_before, value_before = target.view(torch.uint8).clone(), target_values.clone()
    _native.scatter_history_batch([job])
    assert torch.equal(target[:, offset:].view(torch.uint8), raw[:, 1:1+count].view(torch.uint8))
    assert torch.equal(target_values[:, offset:], values[:, 1:1+count])
    assert torch.equal(target[:, :offset].view(torch.uint8), raw_before[:, :offset])
    assert torch.equal(target_values[:, :offset], value_before[:, :offset])
    assert sums.eq(17).all()


@pytest.mark.parametrize('extra', [(), (False,)])
def test_legacy_reduction_api_keeps_none_and_explicit_delta_semantics(extra):
    job, (_, _, target, values, sums) = native_job()
    # Omitting or disabling copy_only cannot silently turn None into no work.
    before = target.view(torch.uint8).clone()
    with pytest.raises(ValueError, match='geometry'):
        _native.scatter_history_batch([job[:9] + extra])
    assert torch.equal(target.view(torch.uint8), before) and sums.eq(17).all()
    delta = np.full(sums.shape, 3., dtype=np.float32)
    legacy = (*job[:4], delta, *job[5:9], *extra)
    _native.scatter_history_batch([legacy])
    assert sums.eq(20).all()
    assert torch.equal(target.view(torch.uint8), torch.from_numpy(job[0])[:, 1:9])


@pytest.mark.parametrize('failure', [
    'flag_int', 'flag_none', 'flag_numpy_bool', 'delta', 'partial', 'extent',
    'sum_dtype', 'sum_readonly', 'raw_codec', 'value_codec', 'raw_readonly',
    'target_layer_stride', 'unsupported_width', 'descriptor_size', 'huge_offset', 'huge_count',
])
def test_bad_later_descriptor_cannot_copy_or_reduce_an_earlier_page(failure):
    first, (_, _, target, values, sums) = native_job()
    bad, _ = native_job()
    bad = list(bad)
    if failure == 'flag_int': bad[9] = 1
    elif failure == 'flag_none': bad[9] = None
    elif failure == 'flag_numpy_bool': bad[9] = np.bool_(True)
    elif failure == 'delta': bad[4] = np.ones((2, 1, 64), dtype=np.float32)
    elif failure == 'partial': bad[8] = 7
    elif failure == 'extent': bad[6] = 3
    elif failure == 'huge_offset': bad[7] = np.iinfo(np.intp).max
    elif failure == 'huge_count': bad[8] = np.iinfo(np.intp).max
    elif failure == 'sum_dtype': bad[5] = np.zeros((2, 1, 64), dtype=np.float64)
    elif failure == 'sum_readonly': bad[5].flags.writeable = False
    elif failure == 'raw_codec': bad[0] = bad[0].view(np.int8)
    elif failure == 'value_codec': bad[1] = bad[1].view(np.int8)
    elif failure == 'raw_readonly': bad[2].flags.writeable = False
    elif failure == 'target_layer_stride':
        bad[2] = np.lib.stride_tricks.as_strided(bad[2], bad[2].shape, (64, 64, 64, 1))
    elif failure == 'unsupported_width':
        bad[0] = np.zeros((3, 10, 1, 64 * 3), dtype=np.uint8)
        bad[2] = np.zeros((3, 8, 1, 64 * 3), dtype=np.uint8)
    elif failure == 'descriptor_size': bad.append(False)
    before = target.view(torch.uint8).clone()
    with pytest.raises(ValueError):
        _native.scatter_history_batch([first, tuple(bad)])
    assert torch.equal(target.view(torch.uint8), before)
    assert values.eq(13).all() and sums.eq(17).all()


def owner(tmp_path, name, dtype, backend, *, resident=8):
    cfg = KVMemConfig(block_tokens=8, select_budget=16, gen_budget=16,
                     repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = AsyncArchive(str(tmp_path / name), 3 * 8 * 64 * (dtype.itemsize + 1), 32)
    return RequestAuthority(name, cfg, archive, 3, 1, 64, raw_dtype=dtype,
        scoring_layers=2, mean_k_backend=backend, resident_records=resident), archive


@pytest.mark.parametrize('dtype,backend', [
    (torch.bfloat16, 'qw3_cuda'), (torch.float8_e4m3fn, 'qw3_cuda'),
    (torch.bfloat16, 'torch_reference'), (torch.float8_e4m3fn, 'torch_reference'),
    (torch.float16, 'torch_reference'),
])
def test_batched_seals_tails_accepted_rows_and_mtp_match_unmodified_serial_append(
        tmp_path, monkeypatch, dtype, backend):
    a, archive = owner(tmp_path, 'batch', dtype, backend)
    ref, ref_archive = owner(tmp_path, 'serial', dtype, backend)
    generator = torch.Generator().manual_seed(818)
    raw = (torch.randn(3, 24, 1, 64, generator=generator) * 12).to(dtype)
    # An unscored MTP layer is still copied and archived exactly.
    raw[2] = (torch.randn(24, 1, 64, generator=generator) * 80).to(dtype)
    values = torch.randint(0, 256, raw.shape, generator=generator, dtype=torch.uint8)
    jobs_seen = []
    scatter = _native.scatter_history_batch

    def observe(jobs):
        jobs_seen.extend((len(j), j[4] is None, j[7], j[8], j[9] if len(j) == 10 else False)
                         for j in jobs)
        return scatter(jobs)

    monkeypatch.setattr(_native, 'scatter_history_batch', observe)
    # Make the fallback explicit independent of host AVX/Torch version.
    monkeypatch.setattr(authority_module, '_FUSED_DECODE_SUM', False)
    try:
        for candidates, accepted in [(5, 3), (17, 14), (4, 1), (2, 0)]:
            begin = a.history_end
            r, v = raw[:, begin:begin+candidates], values[:, begin:begin+candidates]
            kwargs = dict(history_begin=begin, local_rope_begin=begin, accepted=accepted)
            assert ref.append(r, v, **kwargs)
            assert append_authority_batch([(a, r, v, kwargs)]) == 1
            for current in (a, ref): current.publish_sealed(lambda: None)
            assert a.history_end == ref.history_end
            for actual, expected in zip(a.records.values(), ref.records.values()):
                assert actual.count == expected.count
                assert torch.equal(actual.key_sum, expected.key_sum)
                assert (actual.mean_key is None) == (expected.mean_key is None)
                if actual.mean_key is not None: assert torch.equal(actual.mean_key, expected.mean_key)
                actual_k, actual_v = a.read_record(actual.handle)
                expected_k, expected_v = ref.read_record(expected.handle)
                assert torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
                assert torch.equal(actual_v, expected_v)
        assert a.history_end == 18 and sum(r.count for r in a.records.values()) == 18
        assert [(row[2], row[3]) for row in jobs_seen] == [(0, 3), (3, 5), (0, 8), (0, 1), (1, 1)]
        for size, no_delta, offset, count, copy_only in jobs_seen:
            expected_copy = backend == 'qw3_cuda' and offset + count == 8
            assert copy_only == expected_copy and no_delta == expected_copy
            assert size == (10 if expected_copy else 9)
    finally:
        assert a.close() and ref.close()
        assert archive.close() and ref_archive.close()


def test_sealed_copy_backpressure_commits_only_admitted_owners_and_retry_once(tmp_path, monkeypatch):
    a, archive = owner(tmp_path, 'a', torch.float8_e4m3fn, 'qw3_cuda', resident=1)
    b, other_archive = owner(tmp_path, 'b', torch.float8_e4m3fn, 'qw3_cuda', resident=1)
    raw = torch.ones(3, 8, 1, 64).to(torch.float8_e4m3fn)
    values = torch.ones(raw.shape, dtype=torch.uint8)
    write = other_archive.write
    monkeypatch.setattr(other_archive, 'write', lambda *args, **kwargs: None)
    try:
        assert b.append(raw, values, history_begin=0, local_rope_begin=0)
        rows = [(a, raw, values, dict(history_begin=0, local_rope_begin=0)),
                (b, raw, values, dict(history_begin=8, local_rope_begin=8))]
        assert append_authority_batch(rows) == 1
        assert a.history_end == 8 and b.history_end == 8
        assert a.memory.phase.name == 'READY' and b.memory.phase.name == 'READY'
        monkeypatch.setattr(other_archive, 'write', write)
        b.publish_sealed(lambda: None)
        assert append_authority_batch(rows[1:]) == 1
        assert b.history_end == 16
        for current in (a, b):
            current.publish_sealed(lambda: None)
            for record in current.records.values():
                assert record.count == 8 and record.key_sum.eq(8).all()
                assert record.mean_key.eq(1).all()
    finally:
        monkeypatch.setattr(other_archive, 'write', write)
        assert a.close() and b.close() and archive.close() and other_archive.close()
