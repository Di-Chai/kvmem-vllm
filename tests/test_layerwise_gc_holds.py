"""KVMEM_LAYERWISE_HISTORICAL_GC: per-layer windows held from historical GC; GC-aware prepared sources."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import layerwise_gc_holds as holds


class Scope:
    def __init__(self, ack_after=0, refuse=False):
        self.paused = False
        self.ack_after = ack_after
        self.refuse = refuse
        self.calls = []
        self.live = set()

    def pause_policy(self, paused):
        self.calls.append(('pause', paused))
        self.paused = paused

    @property
    def policy_paused(self):
        if not self.paused:
            return False
        if self.ack_after:
            self.ack_after -= 1
            return False
        return True

    def protect_current(self, handles):
        self.calls.append(('protect', tuple((h.slot, h.generation) for h in handles)))
        if self.refuse:
            return None
        hold = object()
        self.live.add(id(hold))
        return hold


def authority(scope):
    return NS(memory=NS(_history_scope=scope))


def record(slot, generation):
    return NS(handle=NS(slot=slot, generation=generation))


def test_everything_is_inert_without_the_opt_in(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    scope = Scope()
    a = authority(scope)
    assert holds.scope_of(a) is None and not holds.pause(a) and holds.wait_paused(a) == 0.
    assert holds.WindowHolds().protect('r', a, [[record(1, 2)]]) is False
    assert scope.calls == []


def test_selection_waits_for_the_pause_then_holds_the_union_of_all_layers(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    scope = Scope(ack_after=3)
    a = authority(scope)
    assert holds.pause(a)
    assert holds.wait_paused(a) >= 0. and scope.policy_paused
    windows = [[record(5, 1), record(2, 7)], [record(2, 7), record(9, 3)], [(4, 4, 0, 32, 0), (-1, -1, 64, 8, 32)]]
    keeper = holds.WindowHolds()
    assert keeper.protect('r', a, windows)
    assert scope.calls[-2] == ('protect', ((2, 7), (4, 4), (5, 1), (9, 3)))   # unique, sorted, no placeholder
    assert scope.calls[-1] == ('pause', False) and not scope.paused             # GC resumes
    first = keeper.holds['r']
    holds.wait_paused(a)
    keeper.protect('r', a, [[record(9, 3)]])
    assert keeper.holds['r'] is not first and keeper.counts['protected'] == 2   # replaced, not stacked
    keeper.release('r')
    assert 'r' not in keeper.holds


def test_a_refused_hold_raises_but_never_leaves_gc_paused(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    scope = Scope(refuse=True)
    a = authority(scope)
    holds.wait_paused(a)
    keeper = holds.WindowHolds()
    with pytest.raises(RuntimeError, match='retired during its paused selection'):
        keeper.protect('r', a, [[record(1, 1)]])
    assert not scope.paused and 'r' not in keeper.holds


def test_pause_wait_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    scope = Scope(ack_after=10**9)
    with pytest.raises(TimeoutError, match='acknowledge'):
        holds.wait_paused(authority(scope), timeout=.01)


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', 'yes')
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_HISTORICAL_GC'):
        holds.scope_of(authority(Scope()))


def test_gc_aware_prepared_sources_are_read_rights_and_refuse_retired_records(monkeypatch):
    from vllm_kvmem.memory_storage import MemoryArchive
    archive = MemoryArchive.__new__(MemoryArchive)
    archive.record_bytes = 8
    ready = NS(ready=True, allocation=NS(slot=1, generation=1))
    archive._entries = {('n', 1, 1): ready, ('n', 2, 1): NS(ready=False, allocation=None)}
    borrowed = []
    archive.tier = NS(borrow_records=lambda allocations: borrowed.extend(allocations) or [bytearray(8) for _ in allocations])
    assert len(archive._historical_region_sources([('n', 1, 1)])) == 1 and borrowed == [ready.allocation]
    with pytest.raises(KeyError):
        archive._historical_region_sources([('n', 2, 1)])        # unpublished
    with pytest.raises(KeyError):
        archive._historical_region_sources([('n', 3, 1)])        # absent (retired and dropped)
    archive.tier = NS(borrow_records=lambda allocations: [bytearray(4)])
    with pytest.raises(ValueError, match='exact immutable record bytes'):
        archive._historical_region_sources([('n', 1, 1)])


def test_prepared_reader_gate_accepts_historical_gc_only_with_the_opt_in(monkeypatch):
    from vllm_kvmem import pressure_reader
    from vllm_kvmem.historical_gc import HistoricalAuthorityGc
    gc = HistoricalAuthorityGc.__new__(HistoricalAuthorityGc)
    archive = NS(memory_only=True, _native_payload_arena=object(), prepare_region_reader=lambda *a, **k: None)
    a = NS(archive=archive, gc=gc, memory=NS(_history_scope=object()))
    built = []
    monkeypatch.setattr(pressure_reader, 'PressureReader', lambda *args: built.append(args) or 'reader')
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    assert pressure_reader.prepare_pressure_reader(a, 'plan', 'layers') is None
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    assert pressure_reader.prepare_pressure_reader(a, 'plan', 'layers') == 'reader'
    a.gc = object()                                               # repository GC: still excluded
    assert pressure_reader.prepare_pressure_reader(a, 'plan', 'layers') is None


class PolicyScope(Scope):
    def protect(self, handles):
        self.calls.append(('policy', tuple((h.slot, h.generation) for h in handles)))
        return None if self.refuse else object()


def test_pause_owners_keep_the_scope_paused_until_the_last_releases(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    scope = PolicyScope()
    a = authority(scope)
    assert holds.pause(a)                               # a selection (promise) pauses
    hold = holds.checkpoint_hold(a, [[(3, 1, 0, 32, 0), (1, 2, 32, 32, 32)]])
    assert hold is not None and ('policy', ((1, 2), (3, 1))) in scope.calls
    assert scope.paused                                 # the checkpoint did not resume the selection's pause
    holds.WindowHolds().protect('r', a, [[record(1, 2)]])
    assert not scope.paused                             # last owner released


def test_checkpoint_hold_refusal_raises_and_releases_its_pause(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    scope = PolicyScope(refuse=True)
    with pytest.raises(RuntimeError, match='retired before its capture'):
        holds.checkpoint_hold(authority(scope), [[(3, 1, 0, 32, 0)]])
    assert not scope.paused
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC')
    assert holds.checkpoint_hold(authority(PolicyScope()), [[(3, 1, 0, 32, 0)]]) is None


def test_lazy_query_reader_accepts_historical_gc_only_with_the_opt_in(monkeypatch):
    from vllm_kvmem.historical_gc import HistoricalAuthorityGc
    from vllm_kvmem.query_region_reader import QueryRegionReader
    gc = HistoricalAuthorityGc.__new__(HistoricalAuthorityGc)
    archive = NS(memory_only=True, _native_payload_arena=object(), _history_worker=object(), record_bytes=8)
    a = NS(archive=archive, gc=gc, memory=NS(_history_scope=object(), request_id='r'), history_end=64,
           shape=(1, 32, 1, 2), records={})
    monkeypatch.setattr(QueryRegionReader, 'validate', lambda self: None)
    a.poll = lambda: None
    archive._lock, archive._closed = __import__('threading').RLock(), False
    make = lambda: QueryRegionReader(a, request_generation=0, generation_getter=lambda: 0)
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    with pytest.raises(ValueError, match='without GC'):
        make()
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    assert make().historical
    a.gc = object()                                       # repository GC stays excluded
    with pytest.raises(ValueError, match='without GC'):
        make()


def test_pressure_reader_keeps_admission_rights_only_with_the_opt_in(monkeypatch):
    from vllm_kvmem.pressure_reader import PressureReader
    reader = PressureReader.__new__(PressureReader)
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    with pytest.raises(ValueError, match='bypass native payload admission'):
        reader._keep_admission_right(object())
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    right = object()
    reader._keep_admission_right(right)
    assert reader.admission_rights == [right]
    reader.reader = None
    assert reader.close() and not hasattr(reader, 'admission_rights')


def test_reads_outside_the_native_window_pin_plainly_only_with_the_opt_in(monkeypatch):
    from vllm_kvmem.memory import Phase, RequestMemory
    class Plan:
        def admitted_read(self, handle):
            if handle == 'in':
                return ('lease-in', 'right-in')
            raise ValueError('Read is outside the admitted assembly window')
    m = RequestMemory.__new__(RequestMemory)
    m.phase, m._plan, m._layerwise_materialization = Phase.ASSEMBLING, Plan(), None
    m.repository = NS(pin=lambda h: 'pin-' + h)
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    assert m.admitted_read('in') == ('lease-in', 'right-in')
    with pytest.raises(ValueError, match='outside the admitted'):
        m.admitted_read('out')
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    assert m.admitted_read('out') == ('pin-out', None)
    assert m.admitted_read('in') == ('lease-in', 'right-in')


def test_historical_admission_reports_per_owner_lease_views():
    from vllm_kvmem.historical_admission import HistoricalHistoryAdmission, HistoricalTurnReservation
    adm = HistoricalHistoryAdmission(64*1024, 1024, 64, 32)
    key = ('session', 'a', 'a')
    adm.ledger.reservations[key] = HistoricalTurnReservation(key, 1, 'r-1', 0, 100, 4, 2, 2)
    views = adm.snapshot()['lease_views']
    assert views == [dict(key=['session', 'a', 'a'], generation=1, request_id='r-1', phase='active', bytes=8*1024)]
    adm.ledger.retiring.add(key)
    assert adm.snapshot()['lease_views'][0]['phase'] == 'retiring'
    del adm.ledger.reservations[key]
    adm.ledger.retiring.discard(key)
    assert adm.snapshot()['lease_views'] == []


def test_drop_scope_releases_every_window_hold_and_pause_owner_of_that_scope(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    mine, other = Scope(), Scope()
    a, b = authority(mine), authority(other)
    keeper = holds.WindowHolds()
    holds.wait_paused(a); keeper.protect('r1', a, [[record(1, 1)]])
    holds.wait_paused(b); keeper.protect('r2', b, [[record(2, 1)]])
    holds.pause(a)                                     # a leaked selection pause
    assert holds.drop_scope(a) == dict(window_holds=['r1'], pause_owners=['select'])
    assert set(keeper.holds) == {'r2'} and not mine.paused
    assert holds.drop_scope(a) == {}


def test_drop_scope_also_releases_checkpoint_holds_of_that_scope(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    mine, other = PolicyScope(), PolicyScope()
    a, b = authority(mine), authority(other)
    held = holds.checkpoint_hold(a, [[(3, 1, 0, 32, 0)]])
    kept = holds.checkpoint_hold(b, [[(4, 1, 0, 32, 0)]])
    assert held.native is not None and kept.native is not None
    assert holds.drop_scope(a) == dict(checkpoint_holds=1)
    assert held.native is None and kept.native is not None
