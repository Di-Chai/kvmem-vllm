"""CPU publication/ownership qualification for the opt-in one-copy seal path."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import weakref

import numpy as np
import pytest
import torch

from vllm_kvmem import _native, memory_storage
from vllm_kvmem.memory_storage import MemoryArchive, MemorySpan, MemoryTier
from vllm_kvmem.storage import AsyncArchive, make_shared_archive


@pytest.fixture
def legacy_record_packer(monkeypatch):
    # Retain failure/ownership coverage for deployments with older extensions.
    monkeypatch.setattr(_native, 'pack_record_pairs', None, raising=False)


def repository(count):
    repo = _native.Repository(count, 4)
    leases = [repo.reserve(i*4, 4).commit() for i in range(count)]
    handles = [_native.Handle(lease.handle.slot, lease.handle.generation) for lease in leases]
    return repo, leases, handles


def reclaim_all(repo, leases, handles, archive, namespace='owner'):
    for lease, handle in zip(leases, handles):
        repo.retire(handle)
        lease.reset()
        assert repo.reclaim(handle, lambda r: archive.reclaim(namespace, r)) == _native.ReclaimResult.RECLAIMED


@pytest.mark.parametrize('mode', ['default', 'false', 'direct', 'ssd'])
def test_direct_seal_publishes_the_exact_hashed_independent_record_bytes(tmp_path, monkeypatch, mode, legacy_record_packer):
    kwargs = {} if mode == 'default' else {'cpu_direct_record_seal': mode == 'direct'}
    archive = (AsyncArchive(str(tmp_path), 4, 2) if mode == 'ssd'
               else MemoryArchive(str(tmp_path), 4, 2, **kwargs))
    if mode == 'ssd':
        archive.cpu_direct_record_seal = True  # SSD cannot opt into the CPU hook.
    repo, leases, handles = repository(2)
    pack, copies, packed = _native.pack_record_views, [], []
    copy = memory_storage.copy_record_slices
    def observed_pack(views):
        result = pack(views)
        packed.append((len(views), result))
        return result
    def observed_copy(*args):
        copies.append(1)
        return copy(*args)
    monkeypatch.setattr(_native, 'pack_record_views', observed_pack)
    monkeypatch.setattr(memory_storage, 'copy_record_slices', observed_copy)
    sources = [bytearray(value) for value in (b'ab', b'cd', b'WX', b'YZ')]
    stats = archive.write_from_views('owner', leases, sources).result(5)
    assert stats['record_digests'] == tuple(hashlib.sha256(v).hexdigest() for v in (b'abcd', b'WXYZ'))
    assert [n for n, _ in packed] == ([2, 2] if mode == 'direct' else [4])
    assert len(copies) == (0 if mode in ('direct', 'ssd') else 1)
    usage = archive.io_usage()
    if mode != 'ssd':
        assert (usage['seal_batches'], usage['seal_records'], usage['seal_payload_bytes']) == (1, 2, 8)
        assert usage['seal_direct_bytes'] == (8 if mode == 'direct' else 0)
        assert usage['seal_pack_seconds'] == stats['seal_pack_seconds']
        assert usage['seal_hash_seconds'] == stats['seal_hash_seconds']
    if mode == 'direct':
        assert stats['seal_direct_records'] == 2 and stats['seal_direct_bytes'] == 8
        assert stats['seal_record_copy_seconds'] == 0
        assert all(isinstance(v, (int, float)) for k, v in stats.items() if k != 'record_digests')
        for index, entry in enumerate(archive._entries.values()):
            record = archive.tier.records[entry.allocation.slot]
            assert record is packed[index][1] and type(record) is bytes and len(record) == 4
        assert packed[0][1] is not packed[1][1]
    for source in sources:
        source[:] = b'!!'
    assert archive.read('owner', leases).result(5)[0] == b'abcdWXYZ'
    reclaim_all(repo, leases, handles, archive)
    assert archive.tier.used_slots == 0 and archive.close()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_authority_direct_seal_preserves_codec_digest_and_recycled_staging(dtype):
    from test_record_staging_pool import setup, source
    authority, archive = setup(dtype, backend='qw3_cuda', resident=1)
    archive.cpu_direct_record_seal = True
    first = None
    pointer = None
    for i in range(3):
        raw, values = source(dtype, i+1)
        assert authority.append(raw, values, history_begin=4*i, local_rope_begin=4*i)
        authority.publish_sealed(lambda: archive.wait_for_capacity(1))
        assert len(authority._record_buffers) == 1 and not authority._busy_record_buffers
        current = authority._record_buffers[0][0].data_ptr()
        pointer = current if pointer is None else pointer
        assert current == pointer
        if first is None:
            first = next(iter(authority.records.values()))
            expected = raw.view(torch.uint8).numpy().tobytes() + values.numpy().tobytes()
    assert first.digest == hashlib.sha256(expected).hexdigest()
    old_raw, old_values = authority.read_record(first.handle)
    raw, values = source(dtype, 1)
    assert torch.equal(old_raw.view(torch.uint8), raw.view(torch.uint8)) and torch.equal(old_values, values)
    assert authority.close() and authority.record_staging_bytes == 0
    assert archive.tier.used_slots == 0 and archive.close()


def test_authority_cancellation_waits_for_direct_seal_before_returning_staging(monkeypatch, legacy_record_packer):
    from test_record_staging_pool import setup, source
    authority, archive = setup(torch.float8_e4m3fn, resident=1)
    archive.cpu_direct_record_seal = True
    entered, release = Event(), Event()
    pack = _native.pack_record_views
    def delayed(views):
        entered.set()
        assert release.wait(5)
        return pack(views)
    monkeypatch.setattr(_native, 'pack_record_views', delayed)
    try:
        assert authority.append(*source(torch.float8_e4m3fn, 3), history_begin=0, local_rope_begin=0)
        assert entered.wait(5)
        record = next(iter(authority.records.values()))
        assert record.digest is None and not record.archived
        assert len(authority._busy_record_buffers) == 1
        assert not authority.append(*source(torch.float8_e4m3fn, 7), history_begin=4, local_rope_begin=4)
        assert not authority.close() and not archive.close()
        assert authority.record_staging_bytes == authority.record_bytes
        release.set()
        authority.writes[0][0].result(5)
        assert authority.close() and authority.record_staging_bytes == 0
        assert archive.tier.used_slots == 0 and archive.close()
    finally:
        release.set()


def test_inflight_direct_seal_is_atomic_drainable_and_keeps_generation_leases(tmp_path, monkeypatch, legacy_record_packer):
    archive = MemoryArchive(str(tmp_path), 4, 4, max_pending=2, io_max_bytes=16,
                            cpu_direct_record_seal=True)
    repo, leases, handles = repository(3)
    archive.write('owner', [leases[2]], b'live').result(5)
    started, release = Event(), Event()
    pack = _native.pack_record_views
    calls = []
    def delayed(views):
        calls.append(1)
        if len(calls) == 2:
            started.set()
            assert release.wait(5)
        return pack(views)
    monkeypatch.setattr(_native, 'pack_record_views', delayed)
    job = archive.write_from_views('owner', leases[:2], [b'ab', b'cd', b'WX', b'YZ'])
    try:
        assert started.wait(5)
        assert archive.io_usage()['archive_io_pending_bytes'] == 16
        assert list(archive.tier.records.values()) == [b'live']
        with pytest.raises(KeyError, match='not yet published'):
            archive.read('owner', leases[:2])
        # A completed sibling can be copied under the tier lock while packing.
        slot = archive._entries[archive.key('owner', leases[2])].allocation.slot
        output = bytearray(4)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(archive.tier.io, [MemorySpan(slot, 0, 4)], output, False).result(2)
        assert output == b'live'
        assert archive.write_from_views('other', leases[:1], [b'12', b'34']) is None
        assert len(calls) == 2 and not job.cancel() and not archive.close()
        for lease, handle in zip(leases[:2], handles[:2]):
            assert repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda r: archive.reclaim('owner', r)) == _native.ReclaimResult.READERS
        release.set()
        stats = job.result(5)
        assert stats['record_digests'] == tuple(hashlib.sha256(v).hexdigest() for v in (b'abcd', b'WXYZ'))
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        reclaim_all(repo, leases, handles, archive)
        assert archive.tier.records == {} and not archive._entries
        # Same physical slots may now hold a new generation, never old bytes.
        replacement = repo.reserve(12, 4).commit()
        assert replacement.handle.generation > handles[replacement.handle.slot].generation
        with pytest.raises(KeyError):
            archive.read('owner', [replacement])
        archive.write_from_views('owner', [replacement], [b'ne', b'w!']).result(5)
        assert archive.read('owner', [replacement]).result(5)[0] == b'new!'
        replacement.reset()
        assert archive.close()
    finally:
        release.set()


@pytest.mark.parametrize('failure', ['pack', 'invalid_record', 'hash', 'publication', 'submission'])
def test_direct_seal_failure_rolls_back_whole_batch_and_releases_credits(tmp_path, monkeypatch, failure, legacy_record_packer):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_direct_record_seal=True)
    repo, leases, handles = repository(2)
    pack = _native.pack_record_views
    sha256 = memory_storage.hashlib.sha256
    calls = []
    def failing_pack(views):
        calls.append(1)
        if len(calls) == 2:
            if failure == 'invalid_record':
                return bytearray(b'bad!')
            raise OSError('injected pack failure')
        return pack(views)
    def failing_hash(record):
        calls.append(1)
        if len(calls) == 2:
            raise OSError('injected hash failure')
        return sha256(record)
    class PartialInsertion(dict):
        def update(self, records):
            self[next(iter(records))] = next(iter(records.values()))
            raise MemoryError('injected publication failure')
    def failing_submit(*args):
        raise RuntimeError('injected submission failure')
    if failure in ('pack', 'invalid_record'):
        monkeypatch.setattr(_native, 'pack_record_views', failing_pack)
    elif failure == 'hash':
        monkeypatch.setattr(memory_storage.hashlib, 'sha256', failing_hash)
    elif failure == 'publication':
        archive.tier.records = PartialInsertion()
    else:
        monkeypatch.setattr(archive.executor, 'submit', failing_submit)
    with pytest.raises((OSError, MemoryError, RuntimeError)):
        job = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
        job.result(5)
    assert archive.tier.records == {} and archive.tier.used_slots == 0
    assert not archive._entries and len(archive._free_ids) == 2
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    assert archive.io_usage()['seal_batches'] == archive.io_usage()['seal_direct_bytes'] == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


@pytest.mark.parametrize('invalid', [False, True])
def test_completed_or_failed_job_does_not_retain_sealed_source_exports(tmp_path, monkeypatch, invalid, legacy_record_packer):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_direct_record_seal=True)
    repo, leases, handles = repository(2)
    entered, release = Event(), Event()
    pack = _native.pack_record_views
    calls = []
    def delayed(views):
        calls.append(1)
        if len(calls) == 2:
            entered.set()
            assert release.wait(5)
            if invalid:
                return b''  # Production validation raises after this frame returns.
        return pack(views)
    monkeypatch.setattr(_native, 'pack_record_views', delayed)
    def submit():
        arrays = [np.array([i, i+1], dtype=np.uint8) for i in range(4)]
        refs = [weakref.ref(a) for a in arrays]
        return archive.write_from_views('owner', leases, arrays), refs
    job, refs = submit()
    try:
        assert entered.wait(5) and all(ref() is not None for ref in refs)
        release.set()
        if invalid:
            with pytest.raises(RuntimeError, match='Invalid owned'):
                job.result(5)
        else:
            job.result(5)
        # Keep the completed Future (and its failure traceback) alive deliberately.
        assert all(ref() is None for ref in refs)
        reclaim_all(repo, leases, handles, archive)
        assert archive.close()
    finally:
        release.set()


def test_owned_publication_rejects_stale_generation_without_touching_siblings():
    tier = MemoryTier(4, 2)
    original = [tier.allocate(i) for i in range(2)]
    assert tier.release(original[1])
    replacement = tier.allocate(1)
    with pytest.raises(ValueError, match='allocation changed'):
        tier.publish_records(original, [b'abcd', b'WXYZ'])
    assert not tier.records
    tier.publish_records([replacement], [b'live'])
    with pytest.raises(ValueError, match='overwrite'):
        tier.publish_records([original[0], replacement], [b'abcd', b'WXYZ'])
    assert tier.records == {1: b'live'}


def test_direct_seal_admission_refusal_allocates_no_owned_record(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 1, io_max_bytes=8, cpu_direct_record_seal=True)
    repo, leases, handles = repository(2)
    monkeypatch.setattr(_native, 'pack_record_views', lambda *a: pytest.fail('Refused write packed sources'))
    with pytest.raises(ValueError, match='byte capacity'):
        archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
    archive.io_max_bytes = 0
    assert archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ']) is None
    assert not archive._entries and not archive.tier.records
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_shared_cpu_archive_receives_direct_seal_switch(tmp_path):
    from vllm_kvmem.options import RuntimeOptions
    options = RuntimeOptions(archive_backend='cpu', cpu_read_cache_bytes=8192,
                             async_seal=True, cpu_direct_record_seal=True)
    archive = make_shared_archive(options, tmp_path, 4096, 1)
    assert archive.cpu_direct_record_seal and archive.async_seal
    assert list(tmp_path.iterdir()) == [] and archive.close()
