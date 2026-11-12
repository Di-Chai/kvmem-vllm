"""Opt-in KVMEM_NATIVE_GATHER_THREADS: parallel archive region gathers are byte-identical."""
import os
import time

import numpy as np
import pytest

from vllm_kvmem import _native


def sources(count, record_bytes, seed=0, holes=()):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        if i in holes:
            out.append(None)
        else:
            out.append(memoryview(bytearray(rng.integers(0, 256, record_bytes, dtype=np.uint8).tobytes())))
    return out


def gather(src, record_bytes, offset, length, begin, end, threads, monkeypatch):
    if threads is None:
        monkeypatch.delenv('KVMEM_NATIVE_GATHER_THREADS', raising=False)
    else:
        monkeypatch.setenv('KVMEM_NATIVE_GATHER_THREADS', str(threads))
    destination = bytearray(b'\xee' * (len(src) * length))
    copied = _native.gather_record_region(destination, src, record_bytes, offset, length, begin, end)
    return bytes(destination), copied


@pytest.mark.parametrize('threads', [2, 4, 8, 64])
@pytest.mark.parametrize('begin,end', [(0, 101), (3, 77), (0, 9), (5, 6), (0, 0)])
def test_parallel_matches_serial(monkeypatch, threads, begin, end):
    record_bytes, offset, length = 4096, 1024, 512
    src = sources(101, record_bytes, holes={4, 40, 99})
    serial = gather(src, record_bytes, offset, length, begin, end, None, monkeypatch)
    parallel = gather(src, record_bytes, offset, length, begin, end, threads, monkeypatch)
    assert parallel == serial
    expected = sum(length for i in range(begin, end) if src[i] is not None)
    assert serial[1] == expected


def test_untouched_outside_range_and_holes(monkeypatch):
    record_bytes, length = 256, 64
    src = sources(40, record_bytes, seed=1, holes={20})
    data, _ = gather(src, record_bytes, 64, length, 10, 30, 4, monkeypatch)
    assert data[:10 * length] == b'\xee' * (10 * length)
    assert data[20 * length:21 * length] == b'\xee' * length
    assert data[30 * length:] == b'\xee' * (10 * length)
    assert data[11 * length:12 * length] == bytes(src[11][64:128])


@pytest.mark.parametrize('value', ['0', '65', 'x', '-1'])
def test_invalid_thread_count_rejected(monkeypatch, value):
    src = sources(16, 128)
    monkeypatch.setenv('KVMEM_NATIVE_GATHER_THREADS', value)
    with pytest.raises(ValueError):
        _native.gather_record_region(bytearray(16 * 64), src, 128, 0, 64, 0, 16)


def test_concurrent_callers_fall_back_to_serial(monkeypatch):
    """A second caller while the gather pool is busy runs serially, same bytes."""
    import threading
    monkeypatch.setenv('KVMEM_NATIVE_GATHER_THREADS', '4')
    record_bytes, length = 1 << 16, 1 << 14
    src = sources(256, record_bytes, seed=2)
    reference = bytearray(len(src) * length)
    monkeypatch.delenv('KVMEM_NATIVE_GATHER_THREADS')
    _native.gather_record_region(reference, src, record_bytes, 0, length, 0, len(src))
    monkeypatch.setenv('KVMEM_NATIVE_GATHER_THREADS', '4')
    results, errors = [], []

    def run():
        try:
            for _ in range(20):
                out = bytearray(len(src) * length)
                _native.gather_record_region(out, src, record_bytes, 0, length, 0, len(src))
                results.append(bytes(out) == bytes(reference))
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)
    workers = [threading.Thread(target=run) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors and len(results) == 80 and all(results)


def test_bandwidth_report(monkeypatch, capsys):
    """Informational: per-layer restore shape (32 KB regions of 1 MB records)."""
    record_bytes, length, count = 1 << 20, 32 << 10, 1024
    rng = np.random.default_rng(3)
    backing = rng.integers(0, 256, record_bytes * count, dtype=np.uint8)
    src = [memoryview(backing[i * record_bytes:(i + 1) * record_bytes]) for i in range(count)]
    out = bytearray(count * length)
    rates = {}
    for threads in (None, 4, 8):
        if threads is None:
            monkeypatch.delenv('KVMEM_NATIVE_GATHER_THREADS', raising=False)
        else:
            monkeypatch.setenv('KVMEM_NATIVE_GATHER_THREADS', str(threads))
        _native.gather_record_region(out, src, record_bytes, 5 * length, length, 0, count)
        started = time.perf_counter()
        for _ in range(5):
            _native.gather_record_region(out, src, record_bytes, 5 * length, length, 0, count)
        rates[threads or 1] = 5 * count * length / (time.perf_counter() - started) / 1e9
    with capsys.disabled():
        print('\ngather GB/s by threads:', {k: round(v, 1) for k, v in rates.items()},
              'cores:', os.cpu_count())
