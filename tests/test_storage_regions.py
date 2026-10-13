from threading import Event

import pytest

from vllm_kvmem import _native
from vllm_kvmem.storage import AsyncArchive
from vllm_kvmem.cached_storage import CachedArchive


@pytest.mark.parametrize('cached',[False,True])
@pytest.mark.parametrize('offset,length',[(0,64),(64,32)])
def test_region_reads_preserve_mixed_hit_order_namespace_and_exact_field(tmp_path,cached,offset,length):
    kwargs={'cache_bytes':384,'namespace_bytes':384} if cached else {}
    archive=(CachedArchive if cached else AsyncArchive)(str(tmp_path),96,4,**kwargs)
    repo=_native.Repository(4,4)
    leases=[repo.reserve(i*4,4).commit() for i in range(3)]
    records=[bytes(range(i*48,i*48+96)) for i in range(3)]
    archive.write('a',leases,b''.join(records)).result(5)
    if cached:
        archive.read_cache.clear()
        archive.read_cache.put(archive.key('a',leases[1]),records[1])
    order=[2,1,0]
    output=bytearray(3*length)
    data,stats=archive.read_region('a',[leases[i] for i in order],offset,length,destination=output).result(5)
    assert bytes(data)==b''.join(records[i][offset:offset+length] for i in order)
    assert memoryview(data).obj is output
    assert stats['bytes']==96*(2 if cached else 3)
    assert stats['cache_hit_bytes']==(length if cached else 0)
    assert stats['logical_bytes']==3*length
    assert archive.io_usage()['archive_io_pending_bytes']==0
    if cached:
        _,stats=archive.read_region('a',leases,offset,length).result(5)
        assert stats['bytes']==0 and stats['cache_hit_bytes']==3*length
    with pytest.raises(KeyError):archive.read_region('b',leases,offset,length)
    assert archive.close()


def test_region_read_retains_lease_and_charges_fallback_slab_until_failure(tmp_path,monkeypatch):
    archive=AsyncArchive(str(tmp_path),96,4,io_max_bytes=256)
    repo=_native.Repository(4,4)
    lease=repo.reserve(0,4).commit()
    archive.write('a',[lease],b'x'*96).result(5)
    started,release=Event(),Event()
    real=archive._gather_region
    def fail(*args):
        started.set()
        assert release.wait(5)
        raise OSError('region gather failure')
    monkeypatch.setattr(archive,'_gather_region',fail)
    handle=lease.handle
    job=archive.read_region('a',[lease],64,32)
    assert started.wait(5)
    assert archive.io_usage()['archive_io_pending_bytes']==224
    assert not archive.close() and not job.cancel()
    assert repo.retire(handle)
    lease.reset()
    assert repo.reclaim(handle,lambda record:archive.reclaim('a',record))==_native.ReclaimResult.READERS
    release.set()
    with pytest.raises(OSError,match='gather failure'):job.result(5)
    assert archive.io_usage()['archive_io_pending_bytes']==0
    assert repo.reclaim(handle,lambda record:archive.reclaim('a',record))==_native.ReclaimResult.RECLAIMED
    assert archive.close()


def test_native_region_gather_validates_bounds_and_exports(tmp_path):
    output=bytearray(8)
    assert _native.gather_record_region(output,[b'abcdefgh',None],8,2,4,0,2)==4
    assert output==b'cdef\0\0\0\0'
    for offset,length in ((-1,4),(6,4),(0,0)):
        with pytest.raises((ValueError,TypeError)):
            _native.gather_record_region(output,[b'abcdefgh',None],8,offset,length,0,2)
    with pytest.raises(ValueError):
        _native.gather_record_region(output,[b'too short',None],8,2,4,0,2)
