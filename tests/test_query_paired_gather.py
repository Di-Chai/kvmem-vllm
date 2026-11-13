"""Explicit Query-only paired gathering preserves bytes and ownership rules."""
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import record_key
from vllm_kvmem.serving.packed_assembly import read_packed_records
from test_native_memory_storage import eventually
from test_paired_archive_reads import archive_fixture


def native_pair():
    entry = getattr(_native, 'gather_record_regions', None)
    if not callable(entry):
        pytest.skip('Paired gather candidate extension has not been built')
    return entry


def reference_pair(destination, sources, record_bytes, regions, begin, end):
    """Use the old native copy for plumbing tests before the candidate build."""
    cursor = copied = 0
    for offset,length in regions:
        n = len(sources)*length
        copied += _native.gather_record_region(memoryview(destination)[cursor:cursor+n],
            sources, record_bytes, offset, length, begin, end)
        cursor += n
    return dict(copied_bytes=copied, source_prepare_ns=1, copy_ns=2,
                gil_release_ns=3, gil_reacquire_ns=4, total_ns=10)


@pytest.mark.parametrize('kind', ['bytes', 'views', 'mixed'])
@pytest.mark.parametrize('begin,end', [(0,4),(1,3),(2,2)])
def test_native_pair_exact_order_ranges_and_none(kind,begin,end):
    copy = native_pair()
    records = [b'abcdefgh', b'01234567', None, b'ABCDEFGH']
    sources = [memoryview(x) if x is not None and
               (kind=='views' or kind=='mixed' and i%2) else x
               for i,x in enumerate(records)]
    regions = ((5,3),(1,4))
    output = bytearray(b'z'*28)
    expected = bytearray(output)
    cursor = 0
    for offset,length in regions:
        for i in range(begin,end):
            if records[i] is not None:
                expected[cursor+i*length:cursor+(i+1)*length] = records[i][offset:offset+length]
        cursor += len(sources)*length
    result = copy(output,sources,8,regions,begin,end)
    assert output == expected
    assert result['copied_bytes'] == sum(x is not None for x in records[begin:end])*7
    assert all(type(value) is int and value>=0 for value in result.values())
    assert result['total_ns'] == sum(result[k] for k in
        ('source_prepare_ns','copy_ns','gil_release_ns','gil_reacquire_ns'))


@pytest.mark.parametrize('problem', ['short_source','strided_source','bad_region',
    'empty_region','region_overflow','missing_region','source_range','short_destination',
    'readonly_destination','strided_destination'])
def test_native_pair_validates_every_source_and_region_before_writing(problem):
    copy = native_pair()
    output = bytearray(b'z'*24)
    target = output
    sources = [b'abcdefgh',b'ABCDEFGH']
    regions,begin,end = ((0,8),(2,4)),0,2
    if problem=='short_source': sources[-1] = b'bad'
    elif problem=='strided_source': sources[-1] = memoryview(b'0123456789abcdef')[::2]
    elif problem=='bad_region': regions = ((0,8),(6,4))
    elif problem=='empty_region': regions = ((0,8),(0,0))
    elif problem=='region_overflow': regions = ((0,8),(2**64-1,4))
    elif problem=='missing_region': regions = ((0,8),)
    elif problem=='source_range': end = 3
    elif problem=='short_destination': target = memoryview(output)[:-1]
    elif problem=='readonly_destination': target = bytes(output)
    elif problem=='strided_destination': target = memoryview(output)[::2]
    with pytest.raises((ValueError,TypeError,BufferError)):
        copy(target,sources,8,regions,begin,end)
    assert output == b'z'*24


@pytest.mark.parametrize('backend',['cpu','native'])
def test_archive_pair_is_explicit_and_preserves_full_packed_ownership(tmp_path,monkeypatch,backend):
    monkeypatch.setattr(_native,'gather_record_regions',reference_pair,raising=False)
    archive,repo,leases,records = archive_fixture(tmp_path,backend)
    authority = NS(archive=archive,record_bytes=64,shape=(2,4,2,2),poll=lambda:None,
        memory=NS(request_id='owner',admitted_read=lambda h:(repo.pin(h),None)),
        records={record_key(l.handle):NS(archived=True,count=4) for l in leases})
    order,regions = [2,0,1],((3,11),(47,9))
    selected = [leases[i].handle for i in order]
    target = torch.empty(60,dtype=torch.uint8)
    calls = []
    def observed(*args):
        calls.append(1)
        return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',observed)
    try:
        before = repo.stats['leases']
        original,stats = read_packed_records(authority,selected,destination=target,regions=regions)
        expected = original.clone()
        assert not calls and 'paired_native_gather' not in stats
        actual,stats = read_packed_records(authority,selected,destination=target,regions=regions,
                                         paired_native=True)
        assert torch.equal(actual,expected) and actual.data_ptr()==target.data_ptr()
        assert calls==[1] and stats['cache_gather_tasks']==1 and stats['paired_native_gather']
        assert stats['paired_native_copied_bytes']==60 and stats['paired_native_copy_ns']==2
        assert stats['logical_bytes']==60 and stats['cache_hit_bytes']==60
        assert archive.io_usage()['archive_io_pending_bytes']==0 and repo.stats['leases']==before
        if backend=='native': assert archive.io_usage()['cpu_payload_readers']==0
    finally:
        for lease in leases: lease.reset()
        eventually(archive.close)


@pytest.mark.parametrize('backend',['cpu','native'])
def test_actual_native_pair_accepts_archive_sources(tmp_path,backend):
    native_pair()
    archive,repo,leases,records = archive_fixture(tmp_path,backend)
    regions = ((3,11),(47,9))
    try:
        expected,_ = archive.read_regions('owner',leases,regions).result(5)
        output = bytearray(60)
        actual,stats = archive.read_regions('owner',leases,regions,
            destination=output,paired_native=True).result(5)
        assert bytes(actual)==bytes(expected) and stats['paired_native_copied_bytes']==60
        assert stats['paired_native_gather'] and archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        for lease in leases: lease.reset()
        eventually(archive.close)


@pytest.mark.parametrize('failure',[False,True])
@pytest.mark.parametrize('backend',['cpu','native'])
def test_native_pair_drains_admitted_writer_and_retains_leases(tmp_path,monkeypatch,backend,failure):
    entered,release = Event(),Event()
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        if failure: raise OSError('paired copy failure')
        return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',blocked,raising=False)
    archive,repo,leases,records = archive_fixture(tmp_path,backend)
    handle = leases[0].handle
    target = bytearray(32)
    job = archive.read_regions('owner',[leases[0]],((0,16),(32,16)),
                               destination=target,paired_native=True)
    try:
        assert entered.wait(5)
        assert archive.io_usage()['archive_io_pending_bytes']==32
        assert not job.cancel() and not archive.close()
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r))==_native.ReclaimResult.READERS
        # Copying does not retain the archive lock or prevent other publication.
        extra = repo.reserve(12,4).commit()
        archive.write('owner',[extra],b'x'*64).result(5)
    finally:
        release.set()
    if failure:
        with pytest.raises(OSError,match='paired copy failure'): job.result(5)
    else:
        actual,_ = job.result(5)
        assert bytes(actual)==records[0][:16]+records[0][32:48]
    assert archive.io_usage()['archive_io_pending_bytes']==0
    if backend=='native': assert archive.io_usage()['cpu_payload_readers']==0
    # Native payload destruction is deferred independently of metadata leases.
    eventually(lambda: repo.reclaim(handle,lambda r:archive.reclaim('owner',r))
               == _native.ReclaimResult.RECLAIMED)
    for lease in leases[1:]: lease.reset()
    extra.reset()
    eventually(archive.close)


@pytest.mark.parametrize('backend',['disk','cache'])
def test_native_pair_rejects_other_tiers_before_admission(tmp_path,monkeypatch,backend):
    monkeypatch.setattr(_native,'gather_record_regions',reference_pair,raising=False)
    archive,repo,leases,_ = archive_fixture(tmp_path,backend)
    try:
        with pytest.raises(ValueError,match='CPU authority'):
            archive.read_regions('owner',leases,((0,8),(32,8)),destination=bytearray(48),paired_native=True)
        assert archive.io_usage()['archive_io_pending_bytes']==0
    finally: assert archive.close()


def test_native_pair_missing_extension_and_failed_pool_are_not_silently_retried(tmp_path,monkeypatch):
    archive,repo,leases,_ = archive_fixture(tmp_path,'cpu')
    target = bytearray(b'z'*48)
    try:
        monkeypatch.setattr(_native,'gather_record_regions',None,raising=False)
        with pytest.raises(RuntimeError,match='updated native extension'):
            archive.read_regions('owner',leases,((0,8),(32,8)),destination=target,paired_native=True)
        assert target==b'z'*48 and archive.io_usage()['archive_io_pending_bytes']==0
        monkeypatch.setattr(_native,'gather_record_regions',reference_pair)
        archive._gather_unusable = 'prior submission failure'
        job = archive.read_regions('owner',leases,((0,8),(32,8)),destination=target,paired_native=True)
        with pytest.raises(RuntimeError,match='pool unavailable'): job.result(5)
        assert target==b'z'*48 and archive.io_usage()['archive_io_pending_bytes']==0
    finally: assert archive.close()
