"""KVMEM_ASSEMBLY_ISOLATE_PARTIAL: records outside the per-layer fast path get their own
assembly chunk; order, dense destination cursors and totals are unchanged."""
from types import SimpleNamespace

import pytest

from vllm_kvmem.layerwise_pressure_efficiency import assembly_groups, isolate_partial_records


def rec(begin, count=32, archived=True):
    return SimpleNamespace(history_begin=begin, count=count, archived=archived)


def flatten(groups):
    return [r for _, g in groups for r in g]


def test_isolated_records_get_their_own_chunk_with_dense_cursors():
    records = [rec(i * 32) for i in range(10)]
    records[3] = rec(96, count=20)                 # partial
    records[7] = rec(224, archived=False)          # not archived yet
    fast = lambda r: not (r.count == 32 and r.archived)
    base, total = assembly_groups(records, 4)
    split, total2 = assembly_groups(records, 4, isolate=fast)
    assert total == total2 and flatten(base) == flatten(split) == records
    cursor = 0
    for begin, group in split:
        assert begin == cursor
        cursor += sum(r.count for r in group)
        assert len(group) == 1 or all(not fast(r) for r in group)
    assert [len(g) for _, g in split] == [3, 1, 3, 1, 2]


def test_skips_and_isolation_together():
    records = [rec(i * 32) for i in range(6)]
    records[2] = rec(64, count=5)
    split, total = assembly_groups(records, 8, skip_history_begins={128},
                                   isolate=lambda r: r.count != 32)
    assert [g[0].history_begin for _, g in split] == [0, 64, 96, 160]
    assert [b for b, _ in split] == [0, 64, 69, 133] and total == 165


def test_flag(monkeypatch):
    monkeypatch.delenv('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', raising=False)
    assert isolate_partial_records() is False
    monkeypatch.setenv('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', '2')
    with pytest.raises(ValueError):
        isolate_partial_records()
