"""Real accepted-token authority with native background history ownership."""
import time

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.retirement import retire_authority


class Ready:
    def query(self):
        return True


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


def setup(tmp_path, *, mode=n.HistoricalGcMode.DELETE, background=False):
    config = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=16,
        repository_blocks=16, sink_blocks=1, recent_blocks=1)
    archive = MemoryArchive(str(tmp_path), 48, 32, native_payloads=True)
    gc = n.HistoricalGcConfig()
    gc.mode, gc.background = mode, background
    gc.ready_low, gc.ready_high = 10, 12
    gc.scan_blocks, gc.delete_blocks, gc.poll_ms = 16, 4, 1
    worker = archive.enable_history_gc(gc)
    authority = RequestAuthority('a', config, archive, 2, 1, 2, resident_records=16)
    return authority, archive, worker


def payload(count=32):
    raw = torch.arange(4*count, dtype=torch.float32).reshape(2, count, 1, 2).to(torch.bfloat16)
    return raw, raw.to(torch.uint8)


def append(authority, count=32):
    data = payload(count)
    assert authority.append(*data, history_begin=0, local_rope_begin=0)
    for job, _ in authority.writes:
        job.result(5)
    return data, tuple(record.handle for record in authority.records.values())


def advance(authority, worker, count=16):
    for _ in range(count):
        worker.advance()
        authority.poll()


def close(authority, archive):
    eventually(authority.close)
    eventually(archive.close)


def test_lru_can_free_cpu_while_current_window_continues_then_normal_refresh_removes_gpu(tmp_path):
    authority, archive, worker = setup(tmp_path)
    try:
        (raw, _), handles = append(authority)
        advance(authority, worker)
        assert authority.gc_reclaimed_records == 4
        assert len(authority.records) == 4
        assert archive.io_usage()['cpu_authority_bytes'] == 4*48
        assert authority.gc.scope.stats['slots_released'] == 0
        assert len(authority.memory._store.working_records()) == 8
        assert not authority.memory.repository.pin(handles[1])
        with authority.index_snapshot() as (records, means):
            assert [record.history_begin for record in records] == [0, 20, 24, 28]
            expected = torch.stack([raw[:, r.history_begin:r.history_begin+r.count].float().mean(dim=1)
                                    for r in records], dim=1)
            assert torch.equal(means, expected)
        del records
        scores = {r.handle.slot: 1. for r in authority.records.values()}
        plan = authority.memory.select(scores)
        assert set(plan.stage_out) == {h.slot for h in handles[1:5]}
        authority.memory.attach_completion(Ready())
        assert authority.memory.commit_selection()
        # A caller retaining a Plan still retains the previous-window fence.
        del plan
        advance(authority, worker)
        assert authority.gc.scope.stats['slots_released'] == 4
        assert len(authority.memory._store.working_records()) == 4
    finally:
        close(authority, archive)


@pytest.mark.parametrize('mode', [n.HistoricalGcMode.OFF, n.HistoricalGcMode.SHADOW])
def test_off_shadow_authority_preserve_bytes_and_complete_finalization(tmp_path, mode):
    authority, archive, worker = setup(tmp_path, mode=mode)
    try:
        (raw, values), handles = append(authority)
        advance(authority, worker)
        assert authority.gc_reclaimed_records == 0 and len(authority.records) == 8
        chunks = authority.read_records(handles)
        assert torch.equal(torch.cat([chunk[0] for chunk in chunks], dim=1), raw)
        assert torch.equal(torch.cat([chunk[1] for chunk in chunks], dim=1), values)
        result = retire_authority(authority, verify=True, timeout=5)
        assert result['sealed_records_verified'] == 8
        assert archive.tier.used_slots == 0
    finally:
        close(authority, archive)


def test_cancel_partial_tail_waits_for_original_reader_then_backend_retires_all(tmp_path, monkeypatch):
    authority, archive, worker = setup(tmp_path, mode=n.HistoricalGcMode.OFF)
    try:
        _, handles = append(authority, count=5)
        authority.poll()
        reader = authority.memory.repository.pin(handles[-1])
        assert reader
        monkeypatch.setattr(authority.gc, 'stop', lambda: pytest.fail('nonblocking close waited for a policy pause'))
        assert not authority.close()
        advance(authority, worker)
        assert not authority.close()
        assert reader.record.token_begin == 4
        reader.reset()
        eventually(authority.close)
        assert not authority.records and not authority._busy_record_buffers
        assert archive.tier.used_slots == 0
    finally:
        close(authority, archive)


def test_original_queued_write_completes_during_cancellation(tmp_path):
    authority, archive, worker = setup(tmp_path, mode=n.HistoricalGcMode.OFF)
    try:
        data, handles = append(authority, count=4)
        # Native publication may complete before authority.poll observes it.
        assert authority.writes and not next(iter(authority.records.values())).archived
        eventually(authority.close)
        assert not authority.writes and archive.tier.used_slots == 0
        assert not authority.memory.repository.pin(handles[0])
    finally:
        close(authority, archive)


def test_background_authority_lru_and_verified_finalization(tmp_path):
    authority, archive, _ = setup(tmp_path, background=True)
    try:
        append(authority)
        eventually(lambda: (authority.poll(), authority.gc_reclaimed_records == 4)[1])
        assert authority.gc.scope.stats['slots_released'] == 0
        result = retire_authority(authority, verify=True, timeout=5)
        assert result['sealed_records_verified'] == 4
        assert result['gc_reclaimed_records'] == 4
        assert not archive._history_scopes
    finally:
        close(authority, archive)


def test_native_index_sum_and_mean_match_original_finalization_without_torch_owned_storage(tmp_path):
    authority, archive, worker = setup(tmp_path, mode=n.HistoricalGcMode.OFF)
    authority.mean_k_backend = 'qw3_cuda'
    try:
        (raw, _), _ = append(authority, count=8)
        authority.poll()
        for record in authority.records.values():
            expected = raw[:, record.history_begin:record.history_begin+record.count].float().sum(1)
            assert record.index_owner is not None
            assert torch.equal(record.key_sum, expected)
            assert torch.equal(record.mean_key, (expected/record.count).half())
        record = None
        assert archive.io_usage()['cpu_index_bytes'] == 2*4*6
    finally:
        close(authority, archive)


def test_finalization_counts_logical_retirement_while_old_cpu_reader_is_still_draining(tmp_path):
    authority, archive, worker = setup(tmp_path, mode=n.HistoricalGcMode.OFF)
    readers = []
    try:
        _, handles = append(authority, count=8)
        authority.poll()
        entry = archive._entries[('a', handles[1].slot, handles[1].generation)]
        readers = archive.tier.borrow_records([entry.allocation])
        authority.memory.repository.retire_permanently(handles[1])
        advance(authority, worker)
        authority.stop_gc()
        authority.poll()
        assert authority.gc_reclaimed_records == 1 and len(authority.records) == 1
        assert len(authority.gc.retired_index) == 1
        assert archive.io_usage()['cpu_authority_bytes'] == 96
        readers.clear()
        result = retire_authority(authority, verify=True, timeout=5)
        assert result['sealed_records_verified'] == result['gc_reclaimed_records'] == 1
        assert not authority.gc.retired_index and archive.tier.used_slots == 0
    finally:
        readers.clear()
        close(authority, archive)


def test_retirement_invalidates_saved_exact_frame_without_rescuing_it(tmp_path):
    authority, archive, worker = setup(tmp_path)
    checkpoint = None
    try:
        append(authority)
        authority.poll()
        checkpoint = authority.memory.checkpoint_window()
        advance(authority, worker)
        assert authority.gc_reclaimed_records == 0
        # The LRU strategy keeps its saved exact frame. Explicit irreversible
        # retirement still takes precedence over this policy observation.
        handles = [n.Handle(slot, generation) for slot, generation, *_ in checkpoint.records[1:5]]
        for handle in handles:
            authority.memory.repository.retire_permanently(handle)
        advance(authority, worker)
        assert authority.gc_reclaimed_records == 4
        assert archive.io_usage()['cpu_authority_bytes'] == 4*archive.record_bytes
        assert not authority.memory.checkpoint_valid(checkpoint.prefix)
        with pytest.raises(ValueError, match='Stale or foreign'):
            authority.memory.restore_window(checkpoint)
        assert authority.memory.phase.name == 'READY'
        assert authority.gc.scope.stats['retired'] == 0  # These were explicit retirements.
    finally:
        checkpoint = None
        close(authority, archive)


@pytest.mark.parametrize('region', [None, (0, 32), (32, 16)])
def test_admitted_assembly_can_finish_its_cpu_read_after_retirement_but_cannot_be_selected_again(tmp_path, region):
    authority, archive, worker = setup(tmp_path, mode=n.HistoricalGcMode.OFF)
    plan = None
    try:
        (raw, values), handles = append(authority, count=4)
        authority.poll()
        plan = authority.memory.select({handles[0].slot: 1.})
        authority.memory.repository.retire_permanently(handles[0])
        advance(authority, worker)
        assert not authority.memory.repository.pin(handles[0])
        assert archive.io_usage()['cpu_authority_bytes'] == 48
        if region is None:
            actual = authority.read_record(handles[0])
            assert torch.equal(actual[0], raw) and torch.equal(actual[1], values)
        else:
            lease, source = authority.memory.admitted_read(handles[0])
            expected = raw.view(torch.uint8).numpy().tobytes() + values.numpy().tobytes()
            data, _ = archive.read_region('a', [lease], *region, _sources=[source]).result(5)
            assert bytes(data) == expected[region[0]:sum(region)]
            lease.reset()
            del source
        with pytest.raises(n.HistoryAdmissionRetry):
            authority.memory._store.select([handles[0]])
        plan.release_cpu_sources()
        advance(authority, worker)
        assert archive.io_usage()['cpu_authority_bytes'] == 0
        assert authority.gc.scope.stats['slots_released'] == 0
        authority.memory.attach_completion(Ready())
        assert authority.memory.commit_selection()
    finally:
        plan = None
        close(authority, archive)
