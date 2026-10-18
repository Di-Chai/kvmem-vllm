import pytest

from vllm_kvmem import _native
from vllm_kvmem.storage import AsyncArchive
from threading import Event


def test_global_nvme_byte_quota_is_checked_before_creating_arena(tmp_path):
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import make_shared_archive
    options = RuntimeOptions(memory=KVMemConfig(repository_blocks=4096),
                             nvme_max_bytes=4096 * 4096 * 2 - 1, nvme_preallocate=True)
    with pytest.raises(ValueError, match='byte quota'):
        make_shared_archive(options, tmp_path / 'not-created', 4096, 2)
    assert not (tmp_path / 'not-created').exists()
    archive = AsyncArchive(str(tmp_path / 'reserved'), 4096, 4, preallocate=True, max_bytes=16384)
    assert archive.preallocated and archive.capacity_bytes == 16384 and archive.close()


@pytest.mark.parametrize('memory_only',[False,True])
def test_namespaced_async_archive_roundtrip_reclaim_and_capacity(tmp_path,memory_only):
    archive = AsyncArchive(str(tmp_path), record_bytes=4096, capacity=2,memory_only=memory_only)
    a, b = _native.Repository(1, 16), _native.Repository(1, 16)
    la, lb = a.reserve(0, 16).commit(), b.reserve(0, 16).commit()
    assert la.handle == lb.handle  # Namespace must disambiguate identical handles.
    archive.write("a", [la], b"a" * 4096).result(timeout=5)
    archive.write("b", [lb], b"b" * 4096).result(timeout=5)
    assert archive.read("a", [la]).result(timeout=5)[0] == b"a" * 4096
    assert archive.read("b", [lb]).result(timeout=5)[0] == b"b" * 4096
    with pytest.raises(ValueError, match="overwrite"):
        archive.write("a", [la], b"c" * 4096)
    assert archive.write("c", [la], b"c" * 4096) is None
    handle = la.handle
    assert a.retire(handle)
    assert a.reclaim(handle, lambda rec: archive.reclaim("a", rec)) == _native.ReclaimResult.READERS
    la.reset()
    assert a.reclaim(handle, lambda rec: archive.reclaim("a", rec)) == _native.ReclaimResult.RECLAIMED
    new = a.reserve(16, 16).commit()
    with pytest.raises(KeyError):
        archive.read("a", [new])
    archive.write("a", [new], b"d" * 4096).result(timeout=5)
    assert archive.read("a", [new]).result(timeout=5)[0] == b"d" * 4096
    assert archive.close()


@pytest.mark.parametrize('memory_only',[False,True])
def test_cancelled_request_retains_io_leases_and_applies_backpressure(tmp_path,memory_only):
    archive = AsyncArchive(str(tmp_path), 4096, 2, workers=1, max_pending=1,memory_only=memory_only)
    real_tier = archive.tier
    started, finish = Event(), Event()
    class DelayedTier:
        def __getattr__(self, name):
            return getattr(real_tier, name)

        def io(self, *args):
            started.set()
            assert finish.wait(5)
            return real_tier.io(*args)
    archive.tier = DelayedTier()
    repo = _native.Repository(2, 16)
    a, b = repo.reserve(0, 16).commit(), repo.reserve(16, 16).commit()
    handle = a.handle
    job = archive.write("request", [a], b"x" * 4096)
    assert started.wait(5)
    assert not job.cancel()
    assert archive.write("request", [b], b"y" * 4096) is None
    assert not archive.close()
    assert repo.retire(handle)
    a.reset()
    assert repo.reclaim(handle, lambda rec: archive.reclaim("request", rec)) == _native.ReclaimResult.READERS
    finish.set()
    job.result(timeout=5)
    assert repo.reclaim(handle, lambda rec: archive.reclaim("request", rec)) == _native.ReclaimResult.RECLAIMED
    assert archive.close()


def test_cpu_authority_never_opens_nvme_and_reclaims_only_after_last_reader(tmp_path,monkeypatch):
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.storage import make_shared_archive
    def forbidden(*args,**kwargs):
        raise AssertionError('CPU-only archive attempted to construct NVMe storage')
    monkeypatch.setattr(_native,'NvmeTier',forbidden)
    archive=make_shared_archive(RuntimeOptions(cpu_read_cache_bytes=8192,archive_backend='cpu'),
        tmp_path/'never-created',4096,8)
    repo=_native.Repository(2,16)
    leases=[repo.reserve(i*16,16).commit() for i in range(2)]
    views=[memoryview(b'a'*2048),memoryview(b'b'*2048),memoryview(b'c'*2048),memoryview(b'd'*2048)]
    archive.write_from_views('cpu',leases,views).result(5)
    out,stats=archive.read_region('cpu',list(reversed(leases)),2048,2048).result(5)
    assert out==b'd'*2048+b'b'*2048
    assert stats['syscalls']==0 and stats['cache_hit_bytes']==4096
    _, full_stats=archive.read('cpu',leases).result(5)
    assert all(isinstance(value,(int,float)) for value in full_stats.values())
    assert archive.capacity_bytes==8192 and archive.io_usage()['cpu_authority_bytes']==8192
    assert not (tmp_path/'never-created').exists()
    handle=leases[0].handle; assert repo.retire(handle)
    assert repo.reclaim(handle,lambda r:archive.reclaim('cpu',r))==_native.ReclaimResult.READERS
    leases[0].reset()
    assert repo.reclaim(handle,lambda r:archive.reclaim('cpu',r))==_native.ReclaimResult.RECLAIMED
    assert archive.io_usage()['cpu_authority_bytes']==4096
    assert archive.close()
