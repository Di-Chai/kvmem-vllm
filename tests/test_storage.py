import pytest

from vllm_kvmem import _native
from vllm_kvmem.storage import AsyncArchive
from threading import Event


def test_namespaced_async_archive_roundtrip_reclaim_and_capacity(tmp_path):
    archive = AsyncArchive(str(tmp_path), record_bytes=4096, capacity=2)
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


def test_cancelled_request_retains_io_leases_and_applies_backpressure(tmp_path):
    archive = AsyncArchive(str(tmp_path), 4096, 2, workers=1, max_pending=1)
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
