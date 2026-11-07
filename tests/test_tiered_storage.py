"""Real CPU/SSD archive transactions under residency and I/O pressure."""
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import os
from threading import Event

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.tiered_storage import TieredArchive


RECORD = 4096


def make_archive(path, *, capacity=8, resident=2, **kwargs):
    return TieredArchive(str(path), RECORD, capacity, cpu_bytes=resident*RECORD,
                         ssd_max_bytes=capacity*RECORD, segment_bytes=2*RECORD,
                         cpu_low_watermark=0, cpu_high_watermark=1, **kwargs)


def records(count):
    repo = n.Repository(count, 16)
    leases = [repo.reserve(i*16, 16).commit() for i in range(count)]
    payloads = [bytes([i+1])*RECORD for i in range(count)]
    return repo, leases, payloads


def reclaim(repo, lease, archive, namespace='a'):
    handle = lease.handle
    assert repo.retire(handle)
    lease.reset()
    return repo.reclaim(handle, lambda rec: archive.reclaim(namespace, rec))


def test_cpu_first_payloads_allocate_no_disk_until_pressure(tmp_path):
    path = tmp_path/'lazy'
    archive = make_archive(path)
    repo, leases, payloads = records(3)
    try:
        archive.write('a', leases[:2], b''.join(payloads[:2])).result(5)
        assert not path.exists()
        state = archive.io_usage()
        assert state['cpu_resident_bytes'] == 2*RECORD
        assert state['disk_write_bytes'] == state['ssd_file_bytes'] == 0
        archive.write('a', leases[2:], payloads[2]).result(5)
        state = archive.io_usage()
        assert state['cpu_resident_bytes'] == RECORD
        assert state['cpu_peak_bytes'] <= 2*RECORD
        assert state['ssd_used_bytes'] == state['disk_write_bytes'] == 2*RECORD
        assert archive.history_used_bytes() == 3*RECORD
        assert archive.capacity_bytes == 8*RECORD
        assert not archive.memory_only
        assert archive.borrow_read_sources('a', leases) is None
        assert bytes(archive.read('a', list(reversed(leases))).result(5)[0]) == b''.join(reversed(payloads))
    finally:
        assert archive.close()
    assert not list(path.glob('*.kv'))


def test_batch_larger_than_cpu_residency_spills_incrementally(tmp_path):
    archive = make_archive(tmp_path, capacity=12, resident=1)
    repo, leases, payloads = records(12)
    try:
        archive.write('a', leases, b''.join(payloads)).result(5)
        assert archive.io_usage()['cpu_peak_bytes'] == RECORD
        assert archive.io_usage()['spilled_records'] == 11
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(payloads)
        assert archive.history_used_bytes() == 12*RECORD
    finally:
        assert archive.close()


def test_read_through_uses_spare_cpu_room_and_clean_eviction_does_not_rewrite(tmp_path):
    archive = make_archive(tmp_path, resident=2)
    repo, leases, payloads = records(2)
    try:
        archive.write('a', leases, b''.join(payloads)).result(5)
        archive.spill(0).result(5)
        assert archive.io_usage()['disk_write_bytes'] == 2*RECORD
        assert bytes(archive.read('a', leases[:1]).result(5)[0]) == payloads[0]
        state = archive.io_usage()
        assert state['cpu_resident_bytes'] == RECORD and state['promoted_records'] == 1
        before = state['disk_read_bytes']
        assert bytes(archive.read('a', leases[:1]).result(5)[0]) == payloads[0]
        assert archive.io_usage()['disk_read_bytes'] == before
        archive.spill(0).result(5)
        assert archive.io_usage()['disk_write_bytes'] == 2*RECORD
    finally:
        assert archive.close()


def test_foreground_allocation_does_not_wait_for_a_spill_write(tmp_path, monkeypatch):
    archive = make_archive(tmp_path, resident=1)
    repo, leases, payloads = records(3)
    entered, proceed = Event(), Event()
    original = os.pwrite
    def delayed(*args):
        entered.set()
        assert proceed.wait(5)
        return original(*args)
    try:
        archive.write('a', leases[:1], payloads[0]).result(5)
        monkeypatch.setattr(os, 'pwrite', delayed)
        second = archive.write('a', leases[1:2], payloads[1])
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Admission touches metadata only, despite the busy disk worker.
            third = pool.submit(archive.write, 'a', leases[2:], payloads[2]).result(1)
            assert third is not None
            proceed.set()
        second.result(5)
        third.result(5)
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(payloads)
    finally:
        proceed.set()
        assert archive.wait_for_idle(5)
        assert archive.close()


@pytest.mark.parametrize('paired', [False, True])
def test_region_reads_preserve_field_order_across_cpu_and_ssd(tmp_path, paired):
    archive = make_archive(tmp_path)
    repo, leases, _ = records(3)
    payloads = [bytes([i])*2048+bytes([10+i])*2048 for i in range(3)]
    try:
        archive.write('a', leases, b''.join(payloads)).result(5)
        order = [2, 0, 1]
        selected = [leases[i] for i in order]
        regions = ((100, 300), (2200, 500)) if paired else ((123, 1000),)
        destination = bytearray(len(leases)*sum(size for _, size in regions))
        job = (archive.read_regions('a', selected, regions, destination=destination) if paired else
               archive.read_region('a', selected, *regions[0], destination=destination))
        output, stats = job.result(5)
        expected = b''.join(payloads[i][start:start+size] for start, size in regions for i in order)
        assert bytes(output) == bytes(destination) == expected
        assert memoryview(output).obj is destination
        assert stats['bytes'] == 2*RECORD
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        with pytest.raises(ValueError, match='Inline'):
            archive.read_region('a', leases[:1], 0, 4, destination=bytearray(4), inline=True)
    finally:
        assert archive.close()


def test_sealed_views_keep_original_digests_after_spill(tmp_path):
    archive = make_archive(tmp_path, resident=1)
    repo, leases, payloads = records(3)
    sources = [bytearray(part) for data in payloads for part in (data[:2048], data[2048:])]
    try:
        result = archive.write_from_views('a', leases, sources).result(5)
        assert result['record_digests'] == tuple(hashlib.sha256(data).hexdigest() for data in payloads)
        for source in sources:
            source[:] = b'x'*len(source)
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(payloads)
    finally:
        assert archive.close()


def test_namespaces_and_reused_slot_generations_remain_isolated(tmp_path):
    archive = make_archive(tmp_path, capacity=2, resident=1)
    a, b = n.Repository(1, 16), n.Repository(1, 16)
    la, lb = a.reserve(0, 16).commit(), b.reserve(0, 16).commit()
    try:
        assert la.handle == lb.handle
        archive.write('a', [la], b'a'*RECORD).result(5)
        archive.write('b', [lb], b'b'*RECORD).result(5)
        assert bytes(archive.read('a', [la]).result(5)[0]) == b'a'*RECORD
        assert bytes(archive.read('b', [lb]).result(5)[0]) == b'b'*RECORD
        assert reclaim(a, la, archive) == n.ReclaimResult.RECLAIMED
        replacement = a.reserve(16, 16).commit()
        with pytest.raises(KeyError):
            archive.read('a', [replacement])
        archive.write('a', [replacement], b'c'*RECORD).result(5)
        archive.spill(0).result(5)
        assert archive.io_usage()['ssd_used_bytes'] == 2*RECORD
        assert bytes(archive.read('a', [replacement]).result(5)[0]) == b'c'*RECORD
        assert bytes(archive.read('b', [lb]).result(5)[0]) == b'b'*RECORD
    finally:
        assert archive.close()


@pytest.mark.parametrize('failure', ['enospc', 'zero_write'])
def test_failed_spill_keeps_original_cpu_authority_and_rolls_back_new_write(tmp_path, monkeypatch, failure):
    archive = make_archive(tmp_path, resident=1)
    repo, leases, payloads = records(2)
    try:
        archive.write('a', leases[:1], payloads[0]).result(5)
        original = os.pwrite
        def fail(*args):
            if failure == 'zero_write':
                return 0
            raise OSError(errno.ENOSPC, 'injected full SSD')
        monkeypatch.setattr(os, 'pwrite', fail)
        with pytest.raises(OSError):
            archive.write('a', leases[1:], payloads[1]).result(5)
        state = archive.io_usage()
        assert state['cpu_resident_bytes'] == RECORD
        assert state['ssd_used_bytes'] == state['archive_io_pending_bytes'] == 0
        assert archive.history_used_bytes() == RECORD
        assert bytes(archive.read('a', leases[:1]).result(5)[0]) == payloads[0]
        monkeypatch.setattr(os, 'pwrite', original)
        archive.write('a', leases[1:], payloads[1]).result(5)
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(payloads)
    finally:
        assert archive.close()


def test_short_writes_complete_and_corrupt_disk_reads_fail_explicitly(tmp_path, monkeypatch):
    archive = make_archive(tmp_path, resident=1)
    repo, leases, payloads = records(2)
    original = os.pwrite
    def short(fd, data, offset):
        return original(fd, memoryview(data)[:71], offset)
    monkeypatch.setattr(os, 'pwrite', short)
    try:
        archive.write('a', leases, b''.join(payloads)).result(5)
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(payloads)
        disk_record = next(record.disk for record in archive.tier.records.values() if record.disk)
        seg = archive.tier.disk.segments[disk_record.segment]
        original(seg.fd, b'x', disk_record.slot*RECORD)
        with pytest.raises(OSError, match='checksum'):
            archive.read('a', leases[:1]).result(5)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert bytes(archive.read('a', leases[1:]).result(5)[0]) == payloads[1]
    finally:
        assert archive.close()


def test_background_spill_does_not_free_a_record_before_write_completion(tmp_path, monkeypatch):
    archive = make_archive(tmp_path)
    repo, leases, payloads = records(1)
    entered, proceed = Event(), Event()
    original = os.pwrite
    def delayed(*args):
        entered.set()
        assert proceed.wait(5)
        return original(*args)
    try:
        archive.write('a', leases, payloads[0]).result(5)
        monkeypatch.setattr(os, 'pwrite', delayed)
        spilling = archive.spill(0)
        assert entered.wait(5)
        assert archive.io_usage()['cpu_resident_bytes'] == RECORD
        assert not archive.close()
        assert not spilling.cancel()
        handle = leases[0].handle
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == n.ReclaimResult.RELEASE
        assert archive.history_used_bytes() == RECORD
        proceed.set()
        spilling.result(5)
        assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == n.ReclaimResult.RECLAIMED
        archive.spill(0).result(5)  # Drain retired extents, including empty segments.
        assert archive.io_usage()['ssd_used_bytes'] == 0
        assert not list(tmp_path.glob('*.kv'))
    finally:
        proceed.set()
        assert archive.wait_for_idle(5)
        assert archive.close()


def test_read_lease_survives_cancel_and_blocks_reclaim(tmp_path, monkeypatch):
    archive = make_archive(tmp_path, resident=1)
    repo, leases, payloads = records(2)
    entered, proceed = Event(), Event()
    try:
        archive.write('a', leases, b''.join(payloads)).result(5)
        original = archive.tier.disk.read_into
        def delayed(*args):
            entered.set()
            assert proceed.wait(5)
            return original(*args)
        monkeypatch.setattr(archive.tier.disk, 'read_into', delayed)
        reading = archive.read('a', leases[:1])
        assert entered.wait(5)
        handle = leases[0].handle
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == n.ReclaimResult.READERS
        assert not reading.cancel()
        proceed.set()
        assert bytes(reading.result(5)[0]) == payloads[0]
        assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == n.ReclaimResult.RECLAIMED
    finally:
        proceed.set()
        assert archive.wait_for_idle(5)
        assert archive.close()


def test_concurrent_namespaces_share_cpu_capacity_without_payload_aliases(tmp_path):
    archive = make_archive(tmp_path, capacity=16, resident=2, max_pending=32)
    owners = [records(2) for _ in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            writes = [pool.submit(lambda index=i: archive.write(str(index), owners[index][1],
                         bytes([index+30])*2*RECORD).result(5)) for i in range(8)]
            for write in writes:
                write.result(5)
            reads = [pool.submit(lambda index=i: bytes(archive.read(str(index), owners[index][1]).result(5)[0]))
                     for i in range(8)]
            for i, read in enumerate(reads):
                assert read.result(5) == bytes([i+30])*2*RECORD
        state = archive.io_usage()
        assert state['cpu_peak_bytes'] <= 2*RECORD
        assert state['logical_allocated_bytes'] == 16*RECORD
        assert state['ssd_reserved_bytes'] <= 16*RECORD
        assert state['archive_io_pending_bytes'] == 0
    finally:
        assert archive.close()


def test_region_fallback_and_spill_share_existing_io_admission(tmp_path):
    archive = make_archive(tmp_path, io_max_bytes=2*RECORD)
    repo, leases, payloads = records(1)
    try:
        archive.write('a', leases, payloads[0]).result(5)
        with pytest.raises(ValueError, match='batch exceeds'):
            archive.read_region('a', leases, 0, 1)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        archive.executor.shutdown()
        with pytest.raises(RuntimeError):
            archive.spill(0)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert archive.history_used_bytes() == RECORD
    finally:
        assert archive.close()


@pytest.mark.parametrize('kwargs', [dict(cpu_bytes=1), dict(ssd_max_bytes=1),
                                  dict(segment_bytes=1), dict(cpu_low_watermark=.9),
                                  dict(cpu_high_watermark=float('nan'))])
def test_invalid_geometry_is_rejected_before_disk_creation(tmp_path, kwargs):
    params = dict(cpu_bytes=RECORD, ssd_max_bytes=4*RECORD, segment_bytes=RECORD)
    params.update(kwargs)
    with pytest.raises(ValueError):
        TieredArchive(str(tmp_path/'absent'), RECORD, 4, **params)
    assert not (tmp_path/'absent').exists()
