"""KVMEM_PRESSURE_RECORD_INDEX_CACHE and KVMEM_PRESSURE_SINGLE_PICK: identical results."""
import copy
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.authority import VersionedRecords
from vllm_kvmem.layerwise_pressure import record_index


class Handle:
    def __init__(self, slot):
        self.slot, self.generation = slot, 1


def rec(begin, slot):
    return NS(history_begin=begin, handle=Handle(slot), count=32)


def owner(records):
    return NS(records=records, memory=NS(request_id='r'))


def test_versioned_records_count_every_mutation():
    d = VersionedRecords()
    v = d.version
    d[1] = 'a'; d.update({2: 'b'}); d.setdefault(3, 'c'); d.pop(1); del d[2]; d |= {4: 'd'}; d.popitem(); d.clear()
    assert d.version - v == 8 and d == {}
    clone = copy.copy(VersionedRecords({5: 'e'}))
    assert type(clone) is VersionedRecords and clone == {5: 'e'}


def test_index_is_reused_only_while_unchanged(monkeypatch):
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    controller = NS()
    records = VersionedRecords({(i, 1): rec(32*i, i) for i in (3, 1, 2)})
    a = owner(records)
    first = record_index(controller, a)
    assert [r.history_begin for r in first[0]] == [32, 64, 96] and first[1] == [1, 2, 3]
    assert record_index(controller, a) is first
    records[(4, 1)] = rec(128, 4)
    second = record_index(controller, a)
    assert second is not first and second[1] == [1, 2, 3, 4]
    a.records = VersionedRecords(records)          # replaced dict: never reuse
    assert record_index(controller, a) is not second
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '0')
    assert record_index(controller, a) is None


def test_duplicate_slots_are_rejected(monkeypatch):
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    a = owner(VersionedRecords({(1, 1): rec(0, 7), (2, 1): rec(32, 7)}))
    with pytest.raises(RuntimeError, match='duplicate physical slots'):
        record_index(NS(), a)


@pytest.mark.parametrize('prefill', [{4: (2., 4.), 8: (1., 2.), 12: (5., 1.)}, {}])
def test_single_pick_select_pressure_matches_two_picks(tmp_path, monkeypatch, prefill):
    from test_graph_lifecycle import fixture, complete
    from test_graph_pressure_v2 import configure
    from vllm_kvmem.prefill_attention import select_pressure
    results = []
    for single in ('0', '1'):
        monkeypatch.setenv('KVMEM_PRESSURE_SINGLE_PICK', single)
        a, archive, graph = fixture(tmp_path / single, sink=1)
        state = configure(a, policy='attention')
        state.prefill.values = dict(prefill)
        plan = select_pressure(a, max_tokens=8)
        results.append((tuple((r.block_id, r.to_base, r.n_tokens, r.skip) for r in plan.remaps),
                        dict(a.graph_selection_report)))
        complete(a); del plan
        assert a.close() and archive.close()
    assert results[0] == results[1]


def full_index(records):
    ordered = sorted(records.values(), key=lambda r: r.history_begin)
    slots = [r.handle.slot for r in ordered]
    return ordered, slots, {s: i for i, s in enumerate(slots)}, {r.history_begin: (r, s) for r, s in zip(ordered, slots)}


@pytest.mark.parametrize('mutation', ['append', 'append_twice', 'out_of_order', 'remove', 'replace'])
def test_incremental_index_equals_full_rebuild(monkeypatch, mutation):
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    controller = NS()
    records = VersionedRecords({(i, 1): rec(32*i, i) for i in (2, 0, 1)})
    a = owner(records)
    first = record_index(controller, a)
    if mutation in ('append', 'append_twice'):
        records[(3, 1)] = rec(96, 3); records[(4, 1)] = rec(128, 4)
        if mutation == 'append_twice':
            record_index(controller, a)
            records[(5, 1)] = rec(160, 5)
    elif mutation == 'out_of_order':
        records[(9, 1)] = rec(16, 9)
    elif mutation == 'remove':
        records.pop((1, 1))
    elif mutation == 'replace':
        records[(0, 1)] = rec(0, 0)
    got = record_index(controller, a)
    assert got == full_index(records)
    assert first == full_index({k: v for k, v in records.items() if k[0] in (0, 1, 2)}) or mutation in ('remove', 'replace')
    assert record_index(controller, a) is got


def test_alignment_flag_tracks_records(monkeypatch):
    from vllm_kvmem.layerwise_pressure import record_index_aligned
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    controller = NS()
    records = VersionedRecords({(i, 1): rec(32*i, i) for i in range(4)})
    a = owner(records)
    record_index(controller, a)
    assert record_index_aligned(controller, a)
    records[(9, 1)] = rec(32*9 + 5, 9)            # unaligned append keeps the incremental path
    record_index(controller, a)
    assert not record_index_aligned(controller, a)
