"""Inline pressure reads keep the original bytes, admission and retirement rights."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
import torch

from vllm_kvmem import _native; from vllm_kvmem.core import storage
from vllm_kvmem.serving.packed_assembly import begin_read_packed_records, read_packed_records
from test_paired_archive_reads import archive_fixture
from test_native_memory_storage import eventually


@pytest.mark.parametrize('backend', ['cpu', 'native'])
def test_inline_pair_has_identical_bytes_without_io_executor_handoff(tmp_path, monkeypatch, backend):
    archive, repo, leases, records = archive_fixture(tmp_path, backend)
    regions, order = ((3,11),(47,9)), [2,0,1]
    selected = [leases[i] for i in order]
    expected = bytes(archive.read_regions('owner',selected,regions).result(5)[0])
    monkeypatch.setattr(archive.executor, 'submit', lambda *a,**k: pytest.fail('Executor handoff'))
    threads = [];gather = archive._gather_region
    def observed(*args):
        threads.append(get_ident())
        return gather(*args)
    monkeypatch.setattr(archive,'_gather_region',observed)
    destination = bytearray(60)
    try:
        job = archive.read_regions('owner', selected, regions, destination=destination, inline=True)
        assert job.done() and not job.cancel()
        actual, stats = job.result()
        assert bytes(actual) == expected and memoryview(actual).obj is destination
        assert stats['inline_cpu_region_read'] and threads == [get_ident()]*2
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        with pytest.raises(KeyError):
            archive.read_regions('foreign',selected,regions,destination=destination,inline=True)
    finally:
        for lease in leases:lease.reset()
        eventually(archive.close)


@pytest.mark.parametrize('backend', ['disk', 'cache'])
def test_inline_cannot_move_disk_fallback_into_cpu_producer(tmp_path, backend):
    archive, repo, leases, _ = archive_fixture(tmp_path, backend)
    try:
        with pytest.raises(ValueError, match='CPU authority'):
            archive.read_regions('owner',leases,((0,8),(32,8)),destination=bytearray(48),inline=True)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
    finally:assert archive.close()


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('backend', ['cpu', 'native'])
def test_inline_read_retains_admission_and_sources_and_releases_lock_while_copying(tmp_path, monkeypatch, failure, backend):
    archive, repo, leases, records = archive_fixture(tmp_path, backend)
    entered, release = Event(), Event();gather = archive._gather_region
    def blocked(dst, sources, offset, length):
        if offset == 32:
            entered.set();assert release.wait(5)
            if failure:raise OSError('second field failed')
        return gather(dst, sources, offset, length)
    monkeypatch.setattr(archive,'_gather_region',blocked)
    handle = leases[0].handle;destination = bytearray(32)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(archive.read_regions,'owner',[leases[0]],((0,16),(32,16)),
                             destination=destination,inline=True)
        try:
            assert entered.wait(5)
            assert archive.io_usage()['archive_io_pending_bytes'] == 32
            assert not archive.close()
            assert repo.retire(handle);leases[0].reset()
            assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r)) == _native.ReclaimResult.READERS
            # Publication requires the same lock; it must remain possible.
            extra = repo.reserve(12,4).commit()
            archive.write('owner',[extra],b'x'*64).result(5)
            archive.max_pending = 1
            assert archive.read_regions('owner',[leases[1]],((0,16),(32,16)),
                                        destination=bytearray(32),inline=True) is None
        finally:release.set()
        job = future.result(5)
        if failure:
            with pytest.raises(OSError,match='second field failed'):job.result()
        else:
            assert bytes(job.result()[0]) == records[0][:16]+records[0][32:48]
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    eventually(lambda: repo.reclaim(handle,lambda r:archive.reclaim('owner',r)) == _native.ReclaimResult.RECLAIMED)
    for lease in leases:lease.reset()
    extra.reset();eventually(archive.close)


def test_inline_holder_failure_precedes_admission_and_destination_writes(tmp_path, monkeypatch):
    archive,repo,leases,_ = archive_fixture(tmp_path,'cpu')
    def fail(*args):raise MemoryError('completion holder')
    monkeypatch.setattr(storage,'ArchiveJob',fail)
    destination = bytearray(b'z'*48)
    with pytest.raises(MemoryError):
        archive.read_regions('owner',leases,((0,8),(32,8)),destination=destination,inline=True)
    assert destination == b'z'*48 and archive.io_usage()['archive_io_pending_bytes'] == 0
    assert archive.close()


def test_inline_packed_path_requires_blocking_cpu_destination(tmp_path):
    from types import SimpleNamespace as NS
    from vllm_kvmem.serving.authority import record_key
    archive,repo,leases,records=archive_fixture(tmp_path,'native')
    authority=NS(archive=archive,record_bytes=64,shape=(2,4,2,2),poll=lambda:None,
        memory=NS(request_id='owner',admitted_read=lambda h:(repo.pin(h),None)),
        records={record_key(l.handle):NS(archived=True,count=4) for l in leases})
    handles=[l.handle for l in leases];regions=((3,11),(47,9))
    output=torch.empty(60,dtype=torch.uint8)
    try:
        actual,stats=read_packed_records(authority,handles,regions=regions,destination=output,inline_cpu=True)
        assert actual.data_ptr()==output.data_ptr() and stats['inline_cpu_region_read']
        assert actual.numpy().tobytes()==b''.join(r[o:o+n] for o,n in regions for r in records)
        with pytest.raises(ValueError,match='synchronous'):
            begin_read_packed_records(authority,handles,regions=regions,destination=output,wait=False,inline_cpu=True)
        with pytest.raises(ValueError,match='synchronous'):
            read_packed_records(authority,handles,regions=regions,inline_cpu=True)
        actual,stats=read_packed_records(authority,handles,region=(0,20),destination=output,inline_cpu=True)
        assert stats['inline_cpu_region_read'] and actual.numpy().tobytes()==b''.join(r[:20] for r in records)
    finally:
        for lease in leases:lease.reset()
        eventually(archive.close)
