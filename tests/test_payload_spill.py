"""Native payload arena spill/restore (session tiering, CPU TTL to SSD): same handle, same bytes."""
import gc
import os

import pytest

from vllm_kvmem import _native

COMMITTED = _native.HistoricalPayloadRetireResult.COMMITTED


def arena(capacity=4, record=4096):
    return _native.HistoricalPayloadArena(capacity, record, background=False)


def handle(slot, generation):
    return _native.HistoricalPayloadHandle(slot, generation)


def test_spill_drops_bytes_and_restore_brings_back_the_same_record():
    a, h = arena(), handle(1, 7)
    data = os.urandom(4096)
    a.publish(h, data)
    read = a.acquire(h)
    assert read and bytes(memoryview(read)) == data
    assert not a.spill(h)                           # busy: a reader holds it
    del read
    gc.collect()
    assert a.spill(h) and a.spilled(h)
    assert not a.acquire(h)                         # no CPU bytes while spilled
    s = a.stats
    assert s['spilled_slots'] == 1 and s['spilled_bytes'] == 4096 and s['used_bytes'] == 0
    assert s['used_slots'] == 1                     # the slot (identity) stays owned
    assert not a.spilled(handle(1, 6)) and not a.spill(h)
    assert a.restore(h, data) and not a.spilled(h)
    back = a.acquire(h)
    assert back and bytes(memoryview(back)) == data
    s = a.stats
    assert s['spilled_slots'] == 0 and s['used_bytes'] == 4096 and s['used_slots'] == 1


def test_restore_rejects_wrong_identity_or_size():
    a, h = arena(), handle(0, 3)
    a.publish(h, os.urandom(4096))
    assert a.spill(h)
    assert not a.restore(handle(0, 4), os.urandom(4096))     # stale generation
    with pytest.raises(ValueError):
        a.restore(h, b'short')
    assert a.spilled(h) and not a.restore(handle(2, 3), os.urandom(4096))
    live = handle(3, 1)
    a.publish(live, os.urandom(4096))
    assert not a.restore(live, os.urandom(4096))             # only a spilled record restores


def test_retiring_a_spilled_record_reclaims_it_with_balanced_accounting():
    a, h = arena(), handle(2, 5)
    a.publish(h, os.urandom(4096))
    assert a.spill(h)
    assert a.retire(handle(2, 4)) != COMMITTED               # stale generation
    assert a.retire(h) == COMMITTED
    a.advance()
    assert a.reclaimed(h)
    s = a.stats
    assert s['used_slots'] == 0 and s['used_bytes'] == 0 and s['spilled_slots'] == 0
    assert s['retired_bytes'] == 0
    a.publish(handle(2, 6), os.urandom(4096))                 # the slot is reusable
    assert a.acquire(handle(2, 6))


def test_close_retires_spilled_records():
    a = _native.HistoricalPayloadArena(4, 4096, background=False)
    h = handle(1, 1)
    a.publish(h, os.urandom(4096))
    assert a.spill(h)
    a.close()
    for _ in range(4):
        a.advance()
    s = a.stats
    assert s['spilled_slots'] == 0 and s['used_slots'] == 0


def tier(tmp_path, capacity=8, record=4096):
    from vllm_kvmem.native_memory_storage import NativeMemoryTier
    t = NativeMemoryTier(record, capacity)
    t.enable_spill(str(tmp_path), max_bytes=1 << 20, segment_bytes=1 << 16)
    return t


def publish(t, slots):
    allocs, data = [], []
    for slot in slots:
        a = t.allocate(slot)
        payload = os.urandom(t.record_bytes)
        t.publish_records([a], [payload])
        allocs.append(a)
        data.append(payload)
    return allocs, data


def test_tier_spill_frees_ram_and_reads_promote_identical_bytes(tmp_path):
    t = tier(tmp_path)
    allocs, data = publish(t, range(4))
    assert t.spill(allocs[:3]) == 3
    assert t.payloads.stats['spilled_slots'] == 3 and t.payloads.stats['used_bytes'] == 4096
    assert [a.slot for a in t.spilled_allocations(allocs)] == [0, 1, 2]
    # an ordinary read of a spilled record promotes it first
    sources = t.borrow_records(allocs[1:2])
    assert bytes(memoryview(sources[0])) == data[1]
    del sources
    assert t.promote(allocs) == 2
    assert t.payloads.stats['spilled_slots'] == 0
    for a, d in zip(allocs, data):
        (src,) = t.borrow_records([a])
        assert bytes(memoryview(src)) == d
    assert t.spill_stats['spilled'] == 3 and t.spill_stats['promoted'] == 3


def test_busy_records_stay_in_ram(tmp_path):
    t = tier(tmp_path)
    allocs, _ = publish(t, [0])
    (reader,) = t.borrow_records(allocs)
    assert t.spill(allocs) == 0 and t.spill_stats['spill_busy'] == 1
    del reader
    gc.collect()
    assert t.spill(allocs) == 1


def test_releasing_a_spilled_record_frees_its_ssd_extent(tmp_path):
    t = tier(tmp_path)
    allocs, _ = publish(t, [5])
    assert t.spill(allocs) == 1
    assert not t.release(allocs[0])               # reclamation is asynchronous
    t.payloads.advance()
    assert t.release(allocs[0])
    assert not t._spilled and t.payloads.stats['used_slots'] == 0
    assert t._spill_store.snapshot()['ssd_used_bytes'] == 0


def test_full_spill_store_keeps_the_rest_in_ram(tmp_path):
    from vllm_kvmem.native_memory_storage import NativeMemoryTier
    t = NativeMemoryTier(4096, 8)
    t.enable_spill(str(tmp_path), max_bytes=2 * 4096, segment_bytes=4096)
    allocs, data = publish(t, range(4))
    assert t.spill(allocs) == 2 and t.spill_stats['spill_full'] == 1
    for a, d in zip(allocs, data):
        (src,) = t.borrow_records([a])
        assert bytes(memoryview(src)) == d


def test_drop_cache_spill_and_promote_keep_identical_bytes(tmp_path):
    from vllm_kvmem.native_memory_storage import NativeMemoryTier
    t = NativeMemoryTier(4096, 8)
    t.enable_spill(str(tmp_path), max_bytes=1 << 20, segment_bytes=1 << 16, drop_cache=True)
    assert t._spill_store.drop_cache
    allocs, data = publish(t, range(6))
    assert t.spill(allocs) == 6
    assert t.promote(allocs[:3]) == 3
    for a, d in zip(allocs, data):
        (src,) = t.borrow_records([a])              # the rest promote on read
        assert bytes(memoryview(src)) == d


def test_flush_skips_released_extents(tmp_path):
    from vllm_kvmem.tiered_storage import _SegmentStore
    store = _SegmentStore(str(tmp_path), 4096, 1 << 20, 1 << 16, drop_cache=True)
    a, b = store.write(os.urandom(4096)), store.write(os.urandom(4096))
    assert store.release(a)
    store.flush([a, b])                             # a is stale, b is flushed
    out = bytearray(4096)
    store.read_into(b, memoryview(out))
    assert store.release(b)
    store.flush([b])                                # the segment is closed: nothing to do


def test_tiering_drop_cache_flag(monkeypatch):
    from vllm_kvmem.native_memory_storage import tiering_drop_cache
    monkeypatch.delenv('KVMEM_TIERING_DROP_CACHE', raising=False)
    assert tiering_drop_cache()
    monkeypatch.setenv('KVMEM_TIERING_DROP_CACHE', '0')
    assert not tiering_drop_cache()
    monkeypatch.setenv('KVMEM_TIERING_DROP_CACHE', 'yes')
    with pytest.raises(ValueError):
        tiering_drop_cache()


def test_spill_device_bench_uses_parallel_io(tmp_path):
    from vllm_kvmem.planner.hardware import bench_spill_device
    out = bench_spill_device(str(tmp_path), size=8 << 20, threads=4)
    assert out['spill_threads'] == 4 and out['spill_read_gbps'] > 0 and out['spill_write_gbps'] > 0
    assert not os.listdir(tmp_path)


def test_session_spill_flushes_once_for_all_parallel_chunks(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from vllm_kvmem.native_memory_storage import NativeMemoryTier
    from vllm_kvmem.session_tiering import SessionTiering
    t = NativeMemoryTier(4096, 128)
    t.enable_spill(str(tmp_path), max_bytes=1 << 20, segment_bytes=1 << 16, drop_cache=True)
    allocs, data = publish(t, range(100))
    flushes = []
    real = t._spill_store.flush
    monkeypatch.setattr(t._spill_store, 'flush', lambda locs: (flushes.append(len(locs)), real(locs)))
    tiering = SessionTiering.__new__(SessionTiering)
    tiering.tier, tiering.io = t, ThreadPoolExecutor(4)
    count, _, _ = tiering._timed(t.spill, allocs)               # 100 records -> 4+ parallel chunks
    assert count == 100 and flushes == [100]
    count, _, _ = tiering._timed(t.promote, allocs)
    assert count == 100 and flushes == [100]
    for a, d in zip(allocs, data):
        (src,) = t.borrow_records([a])
        assert bytes(memoryview(src)) == d
    tiering.io.shutdown()


def test_record_digest_is_xxh64():
    assert _native.record_digest(b'') == 0xEF46DB3751D8E999
    assert _native.record_digest(b'abc') == 0x44BC2CF5AD770999
    assert _native.record_digest(b'Nobody inspects the spammish repetition') == 0xFBCEA83C8A378BF1


def test_native_batch_detects_corruption_on_disk(tmp_path):
    t = tier(tmp_path)
    allocs, _ = publish(t, range(2))
    assert t.spill(allocs) == 2
    handle, location = t._spilled[allocs[0].slot]
    assert isinstance(location.digest, int)                  # the native (XXH64) path is in use
    seg = t._spill_store.segments[location.segment]
    os.pwrite(seg.fd, b'\xff', location.slot * t.record_bytes + 17)
    with pytest.raises(RuntimeError, match='checksum'):
        t.promote(allocs[:1])


def test_concurrent_promotions_restore_each_record_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from vllm_kvmem.native_memory_storage import NativeMemoryTier
    t = NativeMemoryTier(4096, 256)
    t.enable_spill(str(tmp_path), max_bytes=1 << 21, segment_bytes=1 << 15, drop_cache=True)
    allocs, data = publish(t, range(200))
    for _ in range(3):
        assert t.spill(allocs) == 200
        with ThreadPoolExecutor(4) as pool:
            counts = list(pool.map(t.promote, [allocs, allocs[::-1], allocs[50:], allocs[:120]]))
        assert sum(counts) == 200 and not t._spilled
        assert t.payloads.stats['spilled_slots'] == 0
        assert t._spill_store.snapshot()['ssd_used_bytes'] == 0      # every extent released once
        for a, d in zip(allocs, data):
            (src,) = t.borrow_records([a])
            assert bytes(memoryview(src)) == d
        del src                                 # a live reader keeps its record in RAM
        gc.collect()


def test_promoting_a_released_record_is_a_no_op(tmp_path):
    t = tier(tmp_path)
    allocs, data = publish(t, range(3))
    assert t.spill(allocs) == 3
    assert not t.release(allocs[1])
    t.payloads.advance()
    assert t.release(allocs[1])
    assert t.promote(allocs) == 2
    for a in (allocs[0], allocs[2]):
        (src,) = t.borrow_records([a])
        assert bytes(memoryview(src)) == data[a.slot]
    assert t._spill_store.snapshot()['ssd_used_bytes'] == 0
