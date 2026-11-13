from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch
from test_native_memory_storage import eventually

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import record_key
from vllm_kvmem.core.cached_storage import CachedArchive
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.serving.packed_assembly import read_packed_records
from vllm_kvmem.core.storage import AsyncArchive


def archive_fixture(tmp_path, backend):
    if backend == 'disk':
        archive = AsyncArchive(str(tmp_path),64,5)
    elif backend == 'cache':
        archive = CachedArchive(str(tmp_path),64,5,cache_bytes=320,namespace_bytes=320)
    else:
        archive = MemoryArchive(str(tmp_path),64,5,native_payloads=backend=='native')
    repo = _native.Repository(5,4)
    leases = [repo.reserve(i*4,4).commit() for i in range(3)]
    records = [bytes((i*71+j)%256 for j in range(64)) for i in range(3)]
    archive.write('owner',leases,b''.join(records)).result(5)
    if backend == 'cache':
        archive.read_cache.clear()
        archive.read_cache.put(archive.key('owner',leases[1]),records[1])
    return archive, repo, leases, records


@pytest.mark.parametrize('backend',['disk','cache','cpu','native'])
def test_paired_fields_preserve_exact_order_and_admit_one_complete_job(tmp_path,backend,monkeypatch):
    archive,repo,leases,records = archive_fixture(tmp_path,backend)
    calls=[];submit=archive.executor.submit
    def record_submit(*args,**kwargs):
        calls.append(1)
        return submit(*args,**kwargs)
    monkeypatch.setattr(archive.executor,'submit',record_submit)
    regions=((3,11),(47,9));order=[2,0,1];destination=bytearray(60)
    try:
        actual,stats=archive.read_regions('owner',[leases[i] for i in order],regions,
                                         destination=destination).result(5)
        assert bytes(actual)==b''.join(records[i][o:o+n] for o,n in regions for i in order)
        assert memoryview(actual).obj is destination
        assert len(calls)==1 and stats['logical_bytes']==60 and stats['region_read_count']==2
        assert stats['bytes']=={'disk':192,'cache':128,'cpu':0,'native':0}[backend]
        assert archive.io_usage()['archive_io_pending_bytes']==0
        if backend=='native':
            assert archive.io_usage()['cpu_payload_readers']==0
        with pytest.raises(KeyError):
            archive.read_regions('other',leases,regions)
    finally:
        for lease in leases:lease.reset()
        eventually(archive.close)


@pytest.mark.parametrize('regions', [((0,16),(60,8)),((0,16),(-1,8)),
    ((0,16),(32,0)),((0,16),(True,8)),((0,16),),((0,16),(20,8),(40,8))])
def test_all_region_extents_are_validated_before_any_destination_write(tmp_path,regions):
    archive,repo,leases,records=archive_fixture(tmp_path,'cpu')
    destination=bytearray(b'z'*96)
    try:
        with pytest.raises(ValueError):
            archive.read_regions('owner',leases,regions,destination=destination)
        assert destination==b'z'*96 and archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        assert archive.close()


def test_second_field_failure_retains_sources_and_admission_until_all_copying_ends(tmp_path,monkeypatch):
    archive,repo,leases,_=archive_fixture(tmp_path,'disk')
    entered,release=Event(),Event();gather=archive._gather_region;calls=[]
    def fail_second(destination,sources,offset,length):
        calls.append(offset)
        if offset==32:
            entered.set();assert release.wait(5)
            raise OSError('second field failed')
        return gather(destination,sources,offset,length)
    monkeypatch.setattr(archive,'_gather_region',fail_second)
    handle=leases[0].handle
    job=archive.read_regions('owner',[leases[0]],((0,16),(32,16)))
    try:
        assert entered.wait(5) and calls==[0,32]
        assert archive.io_usage()['archive_io_pending_bytes']==32+2*64
        assert not job.cancel() and not archive.close()
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle,lambda record:archive.reclaim('owner',record))==_native.ReclaimResult.READERS
    finally:
        release.set()
        with pytest.raises(OSError,match='second field failed'):
            job.result(5)
    assert archive.io_usage()['archive_io_pending_bytes']==0
    assert repo.reclaim(handle,lambda record:archive.reclaim('owner',record))==_native.ReclaimResult.RECLAIMED
    assert archive.close()


def test_packed_pair_uses_one_generation_admission_per_handle(tmp_path):
    archive,repo,leases,records=archive_fixture(tmp_path,'native');pins=[]
    def admitted(handle):
        pins.append(handle.slot)
        return repo.pin(handle),None
    authority=NS(archive=archive,record_bytes=64,shape=(2,4,2,2),poll=lambda:None,
        memory=NS(request_id='owner',admitted_read=admitted),
        records={record_key(l.handle):NS(archived=True,count=4) for l in leases})
    destination=torch.empty(60,dtype=torch.uint8);order=[2,0,1];regions=((3,11),(47,9))
    try:
        actual,stats=read_packed_records(authority,[leases[i].handle for i in order],
                                        regions=regions,destination=destination)
        assert actual.data_ptr()==destination.data_ptr()
        assert actual.numpy().tobytes()==b''.join(records[i][o:o+n] for o,n in regions for i in order)
        assert pins==[leases[i].handle.slot for i in order]
        with pytest.raises(ValueError,match='never both'):
            read_packed_records(authority,[leases[0].handle],region=(0,16),regions=regions)
        assert len(pins)==3
        assert archive.io_usage()['cpu_payload_readers']==0
    finally:
        for lease in leases:lease.reset()
        eventually(archive.close)
