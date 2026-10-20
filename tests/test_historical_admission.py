"""Whole-turn capacity guarantees using actual repository/archive writers."""
import time

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.historical_admission import HistoricalAdmission, HistoricalCapacitySnapshot as Snapshot
from vllm_kvmem.memory_storage import MemoryArchive


def test_running_and_retiring_reservations_never_become_gc_credit():
    ledger = HistoricalAdmission(24, 16, 4)
    a = Snapshot(7, 12, 12, 4)
    ledger.observe_idle('a', a)
    reservation = ledger.admit('a', a, 'r1', base_history=12, history_limit=44)
    assert reservation is not None and reservation.reserved_records == 16
    b = Snapshot(8, 0, 16, 0)
    assert ledger.admit('b', b, 'r2', base_history=0, history_limit=20) is None
    ledger.finish('a', 7, 'r1')
    assert ledger.reserved_records == 16
    with pytest.raises(ValueError, match='fenced settlement'):
        ledger.observe_idle('a', Snapshot(7, 12, 16, 0))
    assert ledger.admit('b', b, 'r2', base_history=0, history_limit=20) is None
    ledger.settle('a', 7, 'r1', Snapshot(7, 24, 12, 4))
    assert ledger.reserved_records == 4
    assert ledger.admit('b', b, 'r2', base_history=0, history_limit=20) is not None


def test_cpu_bytes_and_ready_metadata_slots_are_independent_limits():
    ledger = HistoricalAdmission(100, 16, 4)
    held_gpu = Snapshot(1, 32, 2, 0)
    ledger.observe_idle('a', held_gpu)
    assert ledger.admit('a', held_gpu, 'r', base_history=32, history_limit=36) is None
    released_gpu = Snapshot(1, 32, 8, 0)
    ledger.observe_idle('a', released_gpu)
    assert ledger.admit('a', released_gpu, 'r', base_history=32, history_limit=36) is not None

    cpu_full = HistoricalAdmission(12, 32, 4)
    retained = Snapshot(1, 48, 20, 9)
    cpu_full.observe_idle('a', retained)
    assert cpu_full.admit('a', retained, 'r', base_history=48, history_limit=52) is None


def test_stale_settlement_cannot_release_another_turn_or_generation():
    ledger = HistoricalAdmission(32, 16, 4)
    start = Snapshot(3, 0, 16, 0)
    assert ledger.admit('a', start, 'r', base_history=0, history_limit=16)
    with pytest.raises(ValueError, match='premature'):
        ledger.settle('a', 3, 'r')
    ledger.finish('a', 3, 'r')
    for generation, request in [(2, 'r'), (3, 'old')]:
        with pytest.raises(ValueError, match='Stale'):
            ledger.settle('a', generation, request)
    assert ledger.reserved_records == 8
    ledger.settle('a', 3, 'r', Snapshot(3, 16, 12, 4))
    with pytest.raises(ValueError, match='Stale'):
        ledger.evicted('a', 2)
    assert ledger.admit('a', ledger.snapshots['a'], 'r2', base_history=16, history_limit=20)
    with pytest.raises(ValueError, match='leased'):
        ledger.evicted('a', 3)


def test_partial_tail_and_inflight_envelope_are_reserved_before_execution():
    ledger = HistoricalAdmission(64, 32, 4, lookahead_tokens=5, pipeline_packets=2)
    snapshot = Snapshot(1, 11, 25, 3)
    # An edited branch can replay from P while old M data still occupies CPU.
    reservation = ledger.admit('a', snapshot, 'r', base_history=3, history_limit=18)
    assert reservation.incoming_records == 4
    assert reservation.extra_records == 6
    assert reservation.retained_records == 3
    assert ledger.reserved_records == 13
    ledger.finish('a', 1, 'r')
    ledger.settle('a', 1, 'r', Snapshot(1, 7, 28, 2))  # Cancelled successor returns to P.
    assert ledger.reserved_records == 2
    ledger.observe_idle('a', Snapshot(1, 7, 29, 1))
    with pytest.raises(ValueError, match='identity or consumed'):
        ledger.observe_idle('a', Snapshot(1, 7, 28, 2))
    ledger.evicted('a', 1)
    assert ledger.reserved_records == 0


def test_a_worker_cannot_report_the_same_capacity_as_ready_and_still_allocated():
    ledger = HistoricalAdmission(64, 16, 4)
    with pytest.raises(ValueError, match='Invalid historical scope capacity'):
        ledger.observe_idle('a', Snapshot(1, 8, 15, 2))
    assert ledger.reserved_records == 0


def eventually(predicate):
    deadline = time.monotonic()+5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_admitted_full_turn_finishes_with_native_gc_paused_and_no_capacity_wait(tmp_path):
    cfg = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=16,
                      repository_blocks=16, sink_blocks=1, recent_blocks=1)
    archive = MemoryArchive(str(tmp_path), 24, 16, native_payloads=True)
    options = n.HistoricalGcConfig()
    options.mode = n.HistoricalGcMode.DELETE
    options.ready_low, options.ready_high = 10, 12
    options.poll_ms = 1
    archive.enable_history_gc(options)
    authority = RequestAuthority('a', cfg, archive, 1, 1, 2, resident_records=16)
    authority.gc.scope.pause_policy(True)
    ledger = HistoricalAdmission(16, 16, 4, lookahead_tokens=4)
    start = Snapshot(1, 0, authority.memory.repository.ready_slots, archive.tier.used_slots)
    reservation = ledger.admit('a', start, 'r', base_history=0, history_limit=32)
    assert reservation is not None
    try:
        eventually(lambda: authority.gc.scope.policy_paused)
        for position in range(32):
            raw = torch.full((1, 1, 1, 2), position, dtype=torch.bfloat16)
            assert authority.append(raw, raw.to(torch.uint8), history_begin=position, local_rope_begin=position)
        for job, _ in authority.writes:
            job.result(5)
        authority.poll()
        assert authority.history_end == 32 and authority.gc.scope.stats['retired'] == 0
        assert authority.memory.repository.ready_slots == 8
        assert archive.tier.used_slots == 8
        ledger.finish('a', 1, 'r')
        assert ledger.reserved_records == reservation.reserved_records
        eventually(authority.close)
        ledger.settle('a', 1, 'r')
        assert ledger.reserved_records == archive.tier.used_slots == 0
    finally:
        eventually(authority.close)
        eventually(archive.close)
