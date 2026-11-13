"""Opt-in KVMEM_SESSION_FAST_TURN: per-turn session capture/restore kept native.

Every test compares the flag against the original code on equal inputs:
checkpoint contents (bitwise), restored decode histories and what they become
after decode feedback, error messages, and the records the GC scope is asked
to hold.
"""
import random
import struct
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem import _native
from vllm_kvmem.serving import layerwise_checkpoints as lc
from vllm_kvmem.serving import layerwise_gc_holds as holds
from vllm_kvmem.serving.layerwise_checkpoints import capture, restored_decode
from vllm_kvmem.serving.layerwise_decode import DecodeHistory, NativeDecodeHistory
from test_checkpoint_fast_state import NAMES, Scope, build, summary
from test_native_decode_segments import _mass_case, _pending

MODES = ('off', 'state', 'turn')
needs_map = pytest.mark.skipif(not hasattr(getattr(_native, 'DecodeMassMap', None), 'from_rows'),
                               reason='native extension lacks DecodeMassMap.from_rows')


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ('KVMEM_LAYERWISE_HISTORICAL_GC', 'KVMEM_CHECKPOINT_FAST_STATE', 'KVMEM_NATIVE_PREFILL_COMMIT',
                 'KVMEM_SESSION_FAST_TURN', 'KVMEM_NATIVE_DECODE_MASS_MAP'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1')
    lc._NATIVE.clear()
    holds._OWNERS.clear()


def mode(monkeypatch, name, mass_map=True):
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', '0' if name == 'off' else '1')
    monkeypatch.setenv('KVMEM_SESSION_FAST_TURN', '1' if name == 'turn' else '0')
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_MASS_MAP', '1' if mass_map else '0')


def bits(mass):
    items = mass.items() if hasattr(mass, 'items') else mass
    return [(key, type(value), struct.pack('d', value)) for key, value in items]


def random_mass(rnd, limit, count=None):
    count = rnd.randrange(0, 600) if count is None else count
    return {32 * rnd.randrange(0, limit // 32): rnd.random() * 10 ** rnd.randint(-9, 9) for _ in range(count)}


def state(history):
    return type(history).__name__, bits(history.mass), history.samples, history.end, history.chunks


# 1. flag
def test_flag_defaults_off_needs_fast_state_and_is_validated(monkeypatch):
    assert lc.fast_turn() is False
    monkeypatch.setenv('KVMEM_SESSION_FAST_TURN', '1')
    assert lc.fast_turn() is False                      # inactive without KVMEM_CHECKPOINT_FAST_STATE
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', '1')
    assert lc.fast_turn() is True
    monkeypatch.setenv('KVMEM_SESSION_FAST_TURN', 'yes')
    with pytest.raises(ValueError, match='KVMEM_SESSION_FAST_TURN'):
        lc.fast_turn()


# 2. native rows
@needs_map
@pytest.mark.parametrize('seed', range(25))
def test_sorted_rows_and_from_rows_equal_the_dictionary_code(seed):
    rnd = random.Random(seed)
    cls = _native.DecodeMassMap
    mass = random_mass(rnd, 32 * 4000)                  # insertion order is not sorted
    if seed % 5 == 0:
        mass[-64] = 0.0; mass[2 ** 62] = -0.0; mass[0] = 5e-324
    rows = cls.from_dict(mass).sorted_rows()
    assert type(rows) is tuple and all(type(row) is tuple and len(row) == 2 for row in rows)
    assert bits(rows) == bits(tuple(sorted(mass.items())))
    fast, reference = cls.from_rows(rows), cls.from_dict(dict(rows))
    assert len(fast) == len(reference) == len(rows)
    assert bits(fast.to_dict()) == bits(reference.to_dict()) and fast.sorted_rows() == rows
    # Decode feedback on top of either map gives the same map.
    a, a_n, b, b_n = reference, 3, fast, 3
    for step in range(4):
        _, indices, targets, m = _mass_case(rnd, np.float32 if (seed + step) % 2 else np.float64)
        a, a_n = _native.prepare_decode_attention_map(a, a_n, indices, targets, m, 16777216)
        b, b_n = _native.prepare_decode_attention_map(b, b_n, indices, targets, m, 16777216)
        assert bits(a.to_dict()) == bits(b.to_dict()) and a_n == b_n
        assert b.sorted_rows() == tuple(sorted(a.to_dict().items()))


@needs_map
def test_from_rows_declines_everything_but_plain_sorted_rows():
    from_rows = _native.DecodeMassMap.from_rows

    class Float(float):
        pass

    class Row(tuple):
        pass
    assert len(from_rows(())) == 0 and from_rows(()).sorted_rows() == ()
    assert from_rows(((0, 0.25), (64, 0.5))).to_dict() == {0: 0.25, 64: 0.5}
    for rows in ([(0, 0.25)], ((64, 0.5), (0, 0.25)), ((0, 0.25), (0, 0.5)), ((True, 0.25),),
                 ((0, 1),), ((0, Float(1.0)),), ((0, np.float64(1.0)),), ((2 ** 63, 1.0),),
                 ((-2 ** 63 - 1, 1.0),), ((0, 0.25, 1.0),), ((0,),), ([0, 0.25],), (Row((0, 0.25)),),
                 ((0.0, 0.25),), ((None, 0.25),), (None,), None, 'ab', {0: 0.25}):
        assert from_rows(rows) is None, rows


# 3. capture / restore / feedback / recapture
def session(monkeypatch, name, seed, *, mass_map=True, native=True):
    """Capture -> restore -> recapture -> decode feedback -> capture, in one mode."""
    mode(monkeypatch, name, mass_map)
    c, a, position = build('buffered', seed)
    rnd = random.Random(seed)
    masses = [random_mass(rnd, position - 32) for _ in NAMES]
    make = ((lambda mass, *rest: NativeDecodeHistory(_native.DecodeMassMap.from_dict(mass), *rest))
            if native else DecodeHistory)
    c.decode_histories['old'] = [make(dict(mass), 7 + i, position - 40, 2 + i) for i, mass in enumerate(masses)]
    first = capture(c, 'old', a, position, terminal=True)
    again = capture(c, 'old', a, position, terminal=True)          # the same live histories
    restored = restored_decode(first)
    c.decode_histories['old'] = restored
    second = capture(c, 'old', a, position, terminal=True)         # unchanged since the restore
    fed = []
    for history in restored:                                       # accepted decode feedback
        r = random.Random(seed + 99)
        for _ in range(3):
            accepted, cols = r.randrange(2, 6), 200
            logical = [32 * r.randrange(0, (position - 32) // 32) for _ in range(cols)]
            m = np.asarray([[r.random() for _ in range(cols)] for _ in range(5)], np.float32)
            m /= m.sum(1, keepdims=True)
            history = history.prepare(_pending(r, history.end, history.end + accepted, interval=2), accepted, logical, m)
        fed.append(history)
    c.decode_histories['old'] = fed
    third = capture(c, 'old', a, position, terminal=True)
    return NS(first=first, again=again, second=second, third=third, restored=restored, fed=fed,
              lazy=[getattr(history, '_mass', False) is None for history in restored],
              frames=[snapshot.__dict__.get('_map_pairs') for snapshot in (first, second, third)])


@needs_map
@pytest.mark.parametrize('seed', range(6))
def test_session_turns_are_identical_in_every_mode(monkeypatch, seed):
    runs = {name: session(monkeypatch, name, seed) for name in MODES}
    reference = runs['off']
    for name in MODES:
        run = runs[name]
        for stage in ('first', 'again', 'second', 'third'):
            assert summary(getattr(run, stage)) == summary(getattr(reference, stage)), (name, stage)
            assert ([bits(rows) for rows, *_ in getattr(run, stage).decode] ==
                    [bits(rows) for rows, *_ in getattr(reference, stage).decode])
        assert [state(h)[1:] for h in run.restored] == [state(h)[1:] for h in reference.restored]
        assert [state(h) for h in run.fed] == [state(h) for h in reference.fed]
        assert all(type(h) is NativeDecodeHistory for h in run.fed)
    turn = runs['turn']
    assert all(type(h) is NativeDecodeHistory for h in turn.restored)
    assert all(type(h) is DecodeHistory for h in reference.restored + runs['state'].restored)
    # One row tuple per history serves the capture, the restore and the recapture.
    for index in range(len(NAMES)):
        assert turn.again.decode[index][0] is turn.first.decode[index][0]
        assert turn.second.decode[index][0] is turn.first.decode[index][0]
    assert turn.lazy == [True] * len(NAMES)                        # no dictionary was built along the way
    assert all(type(h.mass) is dict for h in turn.restored)        # still available on demand
    assert [bits(h.mass) for h in turn.restored] == [bits(h.mass) for h in reference.restored]


@needs_map
@pytest.mark.parametrize('case', ['mass_map_off', 'dict_histories', 'old_extension', 'int_value'])
def test_other_shapes_take_the_original_code(monkeypatch, case):
    results = {}
    for name in MODES:
        mode(monkeypatch, name, mass_map=case != 'mass_map_off')
        c, a, position = build('buffered', 31)
        if case == 'old_extension':                               # a map without sorted_rows
            c.decode_histories['old'] = [NativeDecodeHistory(NS(to_dict=lambda: {64: 0.5, 0: 0.25}), 4, 90, 2)
                                         for _ in NAMES]
        elif case == 'int_value':                                 # not a plain float row
            c.decode_histories['old'] = [DecodeHistory({64: 1, 0: 0.25}, 4, 90, 2) for _ in NAMES]
        first = capture(c, 'old', a, position, terminal=True)
        restored = restored_decode(first)
        c.decode_histories['old'] = restored
        second = capture(c, 'old', a, position, terminal=True)
        results[name] = (summary(first), summary(second), [state(h) for h in restored])
        expect_native = name == 'turn' and case in ('dict_histories', 'old_extension')
        assert all((type(h) is NativeDecodeHistory) is expect_native for h in restored), (case, name)
    assert results['off'][:2] == results['state'][:2] == results['turn'][:2]
    assert ([s[1:] for s in results['off'][2]] == [s[1:] for s in results['state'][2]] ==
            [s[1:] for s in results['turn'][2]])


@needs_map
@pytest.mark.parametrize('fault', ['end', 'key', 'restored_key', 'restored_end'])
def test_future_decode_attention_is_rejected_identically(monkeypatch, fault):
    errors = []
    for name in MODES:
        mode(monkeypatch, name)
        c, a, position = build('buffered', 23)
        make = lambda mass, end: NativeDecodeHistory(_native.DecodeMassMap.from_dict(mass), 4, end, 2)
        c.decode_histories['old'] = [make({64: 0.5, 0: 0.25}, position - 8) for _ in NAMES]
        first = capture(c, 'old', a, position, terminal=True)
        check = position
        if fault.startswith('restored'):
            c.decode_histories['old'] = restored_decode(first)
            check = 64 if fault == 'restored_key' else position - 9
        elif fault == 'end':
            c.decode_histories['old'][2] = make({64: 0.5}, position + 1)
        else:
            c.decode_histories['old'][1] = make({0: 0.25, position: 1.0}, position - 8)
        with pytest.raises(ValueError) as error:
            capture(c, 'old', a, position, terminal=True) if check == position else (
                lc._decode_snapshot(c.decode_histories['old'], check))
        errors.append(str(error.value))
    assert set(errors) == {'Layerwise checkpoint contains future decode attention'}


# 4. GC holds
class PairScope(Scope):
    answer = None

    def protect_pairs(self, pairs):
        assert self.paused
        self.calls.append(('protect_pairs', list(pairs)))
        return object() if self.answer is None else self.answer

    def protect_current_pairs(self, pairs):
        assert self.paused
        self.calls.append(('current_pairs', list(pairs)))
        return object() if self.answer is None else self.answer


class HandleScope(Scope):
    """Records the (slot, generation) of the Handle objects it is given."""

    def protect(self, handles):
        assert self.paused
        self.calls.append(('protect', [(h.slot, h.generation) for h in handles]))
        return object()

    def protect_current(self, handles):
        assert self.paused
        self.calls.append(('current', [(h.slot, h.generation) for h in handles]))
        return object()


def gc_capture(monkeypatch, name, scope, **kwargs):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    mode(monkeypatch, name)
    c, a, position = build('buffered', 13, **kwargs)
    a.memory._history_scope = scope
    snapshot = capture(c, 'old', a, position, terminal=True)
    return c, a, snapshot, [call for call in scope.calls if isinstance(call, tuple)]


def unique(groups, drop=True):
    pairs = {(row[0], row[1]) for rows in groups for row in rows}
    return sorted(pairs - {(-1, -1)} if drop else pairs)


@pytest.mark.parametrize('diverge', [False, True])
def test_capture_holds_the_same_records_through_pairs(monkeypatch, diverge):
    seen = {}
    for name in MODES:
        scope = PairScope() if name == 'turn' else HandleScope()
        c, a, snapshot, calls = gc_capture(monkeypatch, name, scope, diverge=diverge)
        assert len(calls) == 1 and scope.paused is False and not holds._OWNERS
        # The session ledger's rows come with the hold unless layer zero diverged.
        assert ('_map_pairs' in snapshot.__dict__) is (name == 'turn' and not diverge)
        seen[name] = (calls[0][1], summary(snapshot), lc.map_pairs(snapshot))
        held = snapshot.maps + ((tuple(a.memory._store.working_records()),) if diverge else ())
        assert calls[0] == ('protect_pairs' if name == 'turn' else 'protect', unique(held))
        assert snapshot.gc_hold is not None
        if name != 'off':
            assert seen[name][2] == unique(snapshot.maps, drop=False)
    assert seen['off'][:2] == seen['state'][:2] == seen['turn'][:2] and seen['off'][2] is None


def test_capture_without_a_scope_computes_nothing_extra(monkeypatch):
    mode(monkeypatch, 'turn')
    c, a, position = build('buffered', 13)
    snapshot = capture(c, 'old', a, position, terminal=True)
    assert snapshot.gc_hold is None and '_map_pairs' not in snapshot.__dict__
    assert lc.map_pairs(snapshot) == unique(snapshot.maps, drop=False)


def test_scope_that_declines_or_lacks_pairs_gets_handles(monkeypatch):
    declined = PairScope()
    declined.answer = NotImplemented
    handles = []
    declined.protect = lambda items: handles.append([(h.slot, h.generation) for h in items]) or object()
    c, a, snapshot, calls = gc_capture(monkeypatch, 'turn', declined)
    assert [call[0] for call in calls] == ['protect_pairs'] and handles == [unique(snapshot.maps)]
    plain = HandleScope()                                  # an extension built before protect_pairs
    c, a, snapshot, calls = gc_capture(monkeypatch, 'turn', plain)
    assert calls == [('protect', unique(snapshot.maps))] and snapshot.gc_hold is not None


def test_refused_pair_hold_raises_like_the_handle_hold(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    mode(monkeypatch, 'turn')

    class Refusing(PairScope):
        def protect_pairs(self, pairs):
            return None

        def protect_current_pairs(self, pairs):
            return None
    c, a, position = build('buffered', 13)
    a.memory._history_scope = scope = Refusing()
    with pytest.raises(RuntimeError, match='retired before its capture'):
        capture(c, 'old', a, position, terminal=True)
    assert scope.paused is False and not holds._OWNERS
    holds.wait_paused(a)
    keeper = holds.WindowHolds()
    with pytest.raises(RuntimeError, match='retired during its paused selection'):
        keeper.protect('r', a, [[(1, 1, 0, 32, 0)]])
    assert scope.paused is False and 'r' not in keeper.holds


@pytest.mark.parametrize('rows', ['rows', 'records', 'empty'])
def test_window_hold_passes_the_same_records_through_pairs(monkeypatch, rows):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    record = lambda slot, generation: NS(handle=NS(slot=slot, generation=generation))
    groups = {'rows': [[(5, 1, 0, 32, 0), (2, 7, 32, 32, 32)], [(2, 7, 32, 32, 0), (-1, -1, 64, 8, 32)]],
              'records': [[record(5, 1), record(2, 7)], [record(2, 7), record(9, 3)]],
              'empty': [[(-1, -1, 0, 8, 0)]]}[rows]
    seen = []
    for name in MODES:
        mode(monkeypatch, name)
        scope = PairScope() if name == 'turn' else HandleScope()
        a = NS(memory=NS(_history_scope=scope))
        holds.wait_paused(a)
        keeper = holds.WindowHolds()
        assert keeper.protect('r', a, groups)
        calls = [call for call in scope.calls if isinstance(call, tuple)]
        assert [call[0] for call in calls] == ([] if rows == 'empty' else
                                               ['current_pairs' if name == 'turn' else 'current'])
        seen.append(([call[1] for call in calls], ('r' in keeper.holds), dict(keeper.counts)))
        assert scope.paused is False and not holds._OWNERS
    assert seen[0] == seen[1] == seen[2]
    assert seen[0][0] == ([] if rows == 'empty' else [[(2, 7), (5, 1)] if rows == 'rows' else [(2, 7), (5, 1), (9, 3)]])


# 5. the real native scope
def _real():
    try:
        from test_historical_manager import actual_archive, append, eventually, steps
    except Exception as error:      # noqa: BLE001
        pytest.skip(f'native archive unavailable: {error}')
    if not hasattr(_native.HistoricalGcScope, 'protect_pairs'):
        pytest.skip('native extension lacks protect_pairs')
    return actual_archive, append, eventually, steps


@pytest.mark.parametrize('pairs', [False, True])
def test_real_scope_current_hold_keeps_the_same_records(tmp_path, pairs):
    actual_archive, append, eventually, steps = _real()
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    oldest = [lease.handle for lease in leases[:4]]
    try:
        hold = (scope.protect_current_pairs([(h.slot, h.generation) for h in oldest]) if pairs
                else scope.protect_current(oldest))
        assert hold is not None and hold is not NotImplemented
        assert type(hold) is type(scope.protect_current(oldest))
        steps(archive, worker, scope)
        assert all(repo.pin(h) for h in oldest)
        assert scope.stats['retired'] == 4
        assert not any(repo.pin(lease.handle) for lease in leases[4:])
        del hold
    finally:
        leases.clear()
        eventually(archive.close)


@pytest.mark.parametrize('name', MODES)
def test_real_scope_checkpoint_hold_keeps_its_records_until_dropped(tmp_path, monkeypatch, name):
    actual_archive, append, eventually, steps = _real()
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    mode(monkeypatch, name)
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    oldest = [lease.handle for lease in leases[:4]]
    a = NS(memory=NS(_history_scope=scope))
    try:
        rows = [[(h.slot, h.generation, 32 * i, 32, 32 * i) for i, h in enumerate(oldest)]]
        stop = threading.Event()

        def advance():                              # this foreground worker acknowledges a pause when advanced
            while not stop.is_set():
                worker.advance()
                stop.wait(.001)
        holds.pause(a)                              # requested before the worker's first tick
        pump = threading.Thread(target=advance)
        pump.start()
        try:
            holds.wait_paused(a)
            keeper = holds.WindowHolds()
            assert keeper.protect('r', a, rows)     # the capturing request's live window hold
            hold = holds.checkpoint_hold(a, rows)
        finally:
            stop.set()
            pump.join()
        assert hold is not None and keeper.counts['held_records'] == 4
        keeper.release('r')                         # the request finishes; only the checkpoint holds
        steps(archive, worker, scope)
        assert all(repo.pin(h) for h in oldest) and scope.stats['retired'] == 4
        del hold
    finally:
        leases.clear()
        eventually(archive.close)


def test_real_scope_declines_pairs_that_are_not_plain(tmp_path):
    actual_archive, append, eventually, steps = _real()
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    handle = leases[0].handle
    good = (handle.slot, handle.generation)
    try:
        for bad in ((good,), [list(good)], [good + (0,)], [(True, handle.generation)], [(-1, 1)],
                    [(2 ** 32, 1)], [(0, 2 ** 64)], [(0, -1)], [(0.0, 1)], [handle], None):
            assert scope.protect_pairs(bad) is NotImplemented, bad
            assert scope.protect_current_pairs(bad) is NotImplemented, bad
        assert scope.protect_current_pairs([good]) is not None
        # A record that is not live is refused exactly like the Handle call.
        stale = (handle.slot, handle.generation + 1000)
        assert scope.protect_current_pairs([stale]) is None is scope.protect_current(
            [_native.Handle(*stale)])
    finally:
        leases.clear()
        eventually(archive.close)
