from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from vllm_kvmem import _native
from vllm_kvmem.cached_storage import CachedArchive


def test_native_gather_keeps_miss_slots_and_disjoint_range_bytes():
    sources = [b'abcd', None, b'WXYZ', b'1234']
    target = bytearray(b'_' * 16)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(_native.gather_cached_records, target, sources, 4, a, b)
                for a, b in ((0, 2), (2, 4))]
        assert sum(job.result() for job in jobs) == 12
    assert target == b'abcd____WXYZ1234'


@pytest.mark.parametrize('sources,size,begin,end', [
    ([b'bad'], 4, 0, 1), ([bytearray(4)], 4, 0, 1),
    ([b'good'], 0, 0, 1), ([b'good'], 4, 1, 2),
    ([b'good'], 4, 1, 0),
])
def test_native_gather_rejects_bad_extents_before_writing(sources, size, begin, end):
    target = bytearray(b'____')
    with pytest.raises((ValueError, TypeError, BufferError)):
        _native.gather_cached_records(target, sources, size, begin, end)
    assert target == b'____'


def test_native_gather_requires_writable_contiguous_exact_destination():
    for target in (b'____', bytearray(8), memoryview(bytearray(8))[::2]):
        with pytest.raises((ValueError, TypeError, BufferError)):
            _native.gather_cached_records(target, [b'good'], 4, 0, 1)


def test_parallel_gather_drains_other_writes_before_propagating_failure(tmp_path, monkeypatch):
    size = 4 * 1024**2
    archive = CachedArchive(str(tmp_path), size, 2, cache_bytes=2*size,
                            namespace_bytes=2*size, gather_workers=2)
    started, release = threading.Event(), threading.Event()

    def gather(destination, sources, record_bytes, begin, end):
        if begin == 0:
            assert started.wait(5)
            raise OSError('injected copy failure')
        started.set()
        assert release.wait(5)
        destination[record_bytes:] = sources[1]

    monkeypatch.setattr(_native, 'gather_cached_records', gather)
    target = bytearray(2 * size)
    try:
        with ThreadPoolExecutor(max_workers=1) as outer:
            future = outer.submit(archive._gather, target, [b'a'*size, b'b'*size])
            assert started.wait(5)
            assert not future.done()  # The destination remains owned by the read.
            release.set()
            with pytest.raises(OSError, match='copy failure'):
                future.result(timeout=5)
        assert target[size:] == b'b' * size
    finally:
        release.set()
        assert archive.close()


def test_parallel_cache_gather_handles_misses_without_overwriting_tier_ranges(tmp_path):
    size = 2 * 1024**2
    archive = CachedArchive(str(tmp_path), size, 6, cache_bytes=6*size,
                            namespace_bytes=6*size, gather_workers=4)
    sources = [bytes([i]) * size if i != 2 else None for i in range(6)]
    target = bytearray(b'\xff' * (6*size))
    try:
        assert archive._gather(target, sources) == 4
        for i, source in enumerate(sources):
            assert target[i*size:(i+1)*size] == (source if source is not None else b'\xff'*size)
    finally:
        assert archive.close()
