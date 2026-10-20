"""Actual archive reads retain CPU rights before asynchronous job submission."""
from threading import Event
import hashlib
import time

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.memory_storage import MemoryArchive


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


@pytest.mark.parametrize('region', [False, True])
def test_native_archive_gathers_exact_records_and_regions(tmp_path, region):
    archive = MemoryArchive(str(tmp_path), 8, 4, native_payloads=True)
    repo = n.Repository(4, 4)
    leases = [repo.reserve(i*4, 4).commit() for i in range(3)]
    records = [b'ABCDEFGH', b'12345678', b'abcdefgh']
    try:
        archive.write('a', leases, b''.join(records)).result(5)
        order = [2, 0, 1]
        selected = [leases[i] for i in order]
        job = archive.read_region('a', selected, 2, 4) if region else archive.read('a', selected)
        output, stats = job.result(5)
        assert bytes(output) == b''.join(records[i][2:6] if region else records[i] for i in order)
        assert stats['syscalls'] == 0 and stats['cache_gather_tasks'] == 1
        assert archive.io_usage()['cpu_authority_bytes'] == 24
        assert archive.io_usage()['cpu_payload_readers'] == 0
        assert all(type(value) is n.HistoricalPayloadHandle for value in archive.tier.records.values())
    finally:
        leases.clear()
        eventually(archive.close)


@pytest.mark.parametrize('region', [False, True])
def test_queued_read_survives_retirement_then_cpu_frees_while_gpu_keeps_running(tmp_path, region):
    archive = MemoryArchive(str(tmp_path), 8, 2, workers=1, native_payloads=True)
    repo = n.Repository(2, 4)
    gpu = repo.reserve(0, 4).commit()
    handle = gpu.handle
    archive.write('a', [gpu], b'original').result(5)
    entered, release = Event(), Event()

    def block_worker():
        entered.set()
        assert release.wait(5)

    blocker = archive.executor.submit(block_worker)
    try:
        assert entered.wait(5)
        reading = archive.read_region('a', [gpu], 2, 4) if region else archive.read('a', [gpu])
        assert archive.io_usage()['cpu_payload_readers'] == 1
        assert repo.retire_permanently(handle) == n.RetireResult.COMMITTED
        assert not archive.retire_cpu_record('a', handle)
        assert archive.io_usage()['cpu_authority_bytes'] == 8
        # A fresh CPU operation cannot borrow rights from an old GPU lease.
        before = repo.stats['leases']
        with pytest.raises(KeyError, match='retired'):
            archive.read('a', [gpu])
        assert repo.stats['leases'] == before
        release.set()
        blocker.result(5)
        assert bytes(reading.result(5)[0]) == (b'igin' if region else b'original')
        eventually(lambda: archive.retire_cpu_record('a', handle))
        assert archive.io_usage()['cpu_authority_bytes'] == 0
        assert gpu.record.token_begin == 0 and repo.stats['retired'] == 1
        assert repo.ready_slots == 1  # CPU capacity returned; GPU metadata slot did not.
        assert len(archive._free_ids) == 2
        replacement = repo.reserve(4, 4).commit()
        archive.write('a', [replacement], b'replaced').result(5)
        assert bytes(archive.read('a', [replacement]).result(5)[0]) == b'replaced'
        assert archive.retire_cpu_record('a', handle)
        assert bytes(archive.read('a', [replacement]).result(5)[0]) == b'replaced'
        replacement.reset()
    finally:
        release.set()
        blocker.result(5)
        gpu.reset()
        eventually(archive.close)


def test_direct_seal_copies_raw_and_values_into_native_storage_without_python_slab(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 8, 2, native_payloads=True)
    repo = n.Repository(2, 4)
    lease = repo.reserve(0, 4).commit()
    raw, values = bytearray(b'ABCD'), bytearray(b'1234')
    monkeypatch.setattr(n, 'pack_record_views', lambda *_: pytest.fail('Unexpected Python bytes slab'))
    try:
        stats = archive.write_from_views('a', [lease], [memoryview(raw), memoryview(values)]).result(5)
        raw[:] = values[:] = b'????'
        assert bytes(archive.read('a', [lease]).result(5)[0]) == b'ABCD1234'
        assert stats['record_digests'] == (hashlib.sha256(b'ABCD1234').hexdigest(),)
    finally:
        lease.reset()
        eventually(archive.close)
