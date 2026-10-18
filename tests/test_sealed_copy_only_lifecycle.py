"""Copy-only sealing must not extend packet or destination-export lifetimes."""
import gc
import weakref

import pytest
import torch

from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive


def setup(tmp_path, dtype):
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
        repository_blocks=8, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 2*4*64*(dtype.itemsize+1), 8)
    authority = RequestAuthority('sealed-lifetime', config, archive, 2, 1, 64,
        raw_dtype=dtype, scoring_layers=1, mean_k_backend='qw3_cuda')
    return authority, archive


def append_temporary_packet(authority, begin):
    # Two accepted rows; the other two candidate rows must remain unowned.
    raw = (torch.arange(2*4*64).reshape(2,4,1,64).float().remainder(32)+begin).to(authority.raw_dtype)
    values = torch.arange(2*4*64).reshape(2,4,1,64).to(torch.uint8)
    expected = raw[:1,:2].float().sum(1)
    refs = weakref.ref(raw), weakref.ref(values)
    assert append_authority_batch([(authority, raw, values,
        dict(history_begin=begin, local_rope_begin=begin, accepted=2))]) == 1
    return expected, refs


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_sealed_publication_releases_cached_exports_and_input_packets(tmp_path, monkeypatch, dtype):
    authority, archive = setup(tmp_path, dtype)
    original_write = archive.write
    monkeypatch.setattr(archive, 'write', lambda *a, **kw: None)
    try:
        expected, first_packet = append_temporary_packet(authority, 0)
        record = next(iter(authority.records.values()))
        exports = [weakref.ref(view) for view in record.scatter_views]
        raw_export, value_export = weakref.ref(record.raw), weakref.ref(record.values)
        second, second_packet = append_temporary_packet(authority, 2)
        assert authority.history_end == 4 and record.count == 4 and not record.submitted
        assert torch.equal(record.key_sum, expected+second)
        assert torch.equal(record.mean_key, ((expected+second)/4).half())
        gc.collect()
        assert all(ref() is None for ref in first_packet+second_packet)
        assert all(ref() is not None for ref in exports)

        monkeypatch.setattr(archive, 'write', original_write)
        assert authority.flush()
        for job, _ in authority.writes:
            job.result(timeout=5)
        authority.poll()
        assert record.archived and record.scatter_views is None
        assert record.raw is None and record.values is None
        gc.collect()
        assert all(ref() is None for ref in exports)
        # Immutable publication returns raw/V to the bounded authority pool.
        # Packet/export owners die, while these reusable buffers stay within
        # the original resident-record credit until authority close.
        assert raw_export() is not None and value_export() is not None
        assert record.staging_slot is None and not authority._busy_record_buffers
        assert authority.record_staging_bytes == authority.record_bytes
        assert archive.tier.used_slots == 1
    finally:
        monkeypatch.setattr(archive, 'write', original_write)
        assert authority.close()
        assert authority.record_staging_bytes == 0
        assert archive.tier.used_slots == 0
        assert archive.close()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_cancel_unpublished_copy_only_page_reclaims_record_and_exports(tmp_path, monkeypatch, dtype):
    authority, archive = setup(tmp_path, dtype)
    monkeypatch.setattr(archive, 'write', lambda *a, **kw: None)
    try:
        append_temporary_packet(authority, 0)
        append_temporary_packet(authority, 2)
        record = next(iter(authority.records.values()))
        assert record.count == 4 and record.mean_key is not None and not record.submitted
        refs = [weakref.ref(record), weakref.ref(record.raw), weakref.ref(record.values),
            *(weakref.ref(view) for view in record.scatter_views)]
        del record
        assert authority.close()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not authority.records and not authority.writes
        assert archive.tier.used_slots == 0
    finally:
        assert authority.close()
        assert archive.close()
