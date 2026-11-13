"""KVMEM_CHECKPOINT_FAST_STATE: native checkpoint rows, states and evidence.

Every test compares the flag against the original Python code on equal inputs:
captured maps and rows, evidence digests, restored states (values and order),
error messages, handle sets and GC pause ownership.
"""
import json
import random
import struct
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving import layerwise_checkpoints as lc
from vllm_kvmem.serving import layerwise_gc_holds as holds
from vllm_kvmem.serving.layerwise_checkpoints import (
    LayerwisePrefix, StateRows, capture, finish_async_restore, install_restored, prepare_restore,
    restore_pressure_sync, restored_decode,
)
from vllm_kvmem.serving.layerwise_decode import DecodeHistory
from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure
from vllm_kvmem.serving.prefill_attention import PrefillHistory
from vllm_kvmem.serving.prompt_checkpoints import token_bytes
from test_layerwise_pressure import options

NAMES = ('a', 'b', 'c')
KINDS = ('plain', 'buffered', 'deferred')


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ('KVMEM_LAYERWISE_HISTORICAL_GC', 'KVMEM_CHECKPOINT_FAST_STATE',
                 'KVMEM_NATIVE_PREFILL_COMMIT'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1')
    lc._NATIVE.clear()
    holds._OWNERS.clear()


def fast(monkeypatch, on):
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', '1' if on else '0')


def bits(rows):
    return [(key, struct.pack('dd', mass, exposure)) for key, mass, exposure in rows]


def build(kind, seed=0, *, blocks=40, diverge=False):
    """A finished-prefill request 'old' with random per-layer windows and scores."""
    rnd = random.Random(seed)
    q = NS(native_batches=False, optimized=True, names=NAMES,
           runner=NS(model_config=NS(enforce_eager=True), device='cpu'))
    c = LayerwisePrefillPressure(q)
    c.query.finished = set()
    c.buffered_history = kind != 'plain'
    c.deferred_fold = kind == 'deferred'
    tail = rnd.randint(1, 31)
    records = [NS(history_begin=i * 32, count=32, handle=NS(slot=100 + i, generation=1 + i % 3))
               for i in range(blocks)]
    records.append(NS(history_begin=blocks * 32, count=tail, handle=NS(slot=7, generation=9)))
    position = blocks * 32 + tail
    keep = blocks // 2

    def window(layer):
        chosen = sorted(rnd.sample(range(blocks), keep))
        if rnd.random() < .5:
            rnd.shuffle(chosen)
        rows, cursor = [], 0
        for index in chosen:
            record = records[index]
            # The live maps may carry stale handles; capture canonicalizes them.
            handle = ((record.handle.slot, record.handle.generation) if rnd.random() < .8 else (-1, -1))
            rows.append(handle + (record.history_begin, 32, cursor))
            cursor += 32
        rows.append((7, 9, blocks * 32, tail, cursor))
        return tuple(rows)
    maps = tuple(window(i) for i in NAMES)
    shared = window('shared') if diverge else tuple(
        (records[r[2] // 32].handle.slot, records[r[2] // 32].handle.generation) + r[2:] for r in maps[0])
    c.maps['old'] = (position, maps)
    a = NS(history_end=position, records={(r.handle.slot, r.handle.generation): r for r in records},
           memory=NS(config=NS(block_tokens=32), _store=NS(working_records=lambda: shared)))
    histories = c._history('old', options())
    scores = []
    for history in histories:
        keys = rnd.sample(range(blocks), rnd.randint(1, blocks))      # insertion order is not sorted
        values = {key * 32: [rnd.random() * 10 ** rnd.randint(-9, 9), rnd.random() * 50] for key in keys}
        history.values = values
        history.end, history.samples, history.chunks = position - rnd.randint(0, 5), rnd.randint(1, 99), rnd.randint(1, 9)
        scores.append(values)
    prefill = c.new_history(options()) if kind != 'deferred' else type(histories[0]).__mro__[1](options())
    prefill.values = dict(reversed(list(scores[0].items())))          # same scores, another order
    prefill.end, prefill.samples, prefill.chunks = histories[0].end, histories[0].samples, histories[0].chunks
    a.action_history = NS(prefill=prefill)
    c.decode_histories['old'] = [DecodeHistory({64: 0.5, 0: 0.25}, 4, position - 8, 2) for _ in NAMES]
    return c, a, position


def summary(snapshot):
    return dict(position=snapshot.position, names=snapshot.names, maps=snapshot.maps,
                rows=[bits(entry[0]) for entry in snapshot.histories],
                meta=[tuple(entry[1:]) for entry in snapshot.histories],
                decode=snapshot.decode, terminal=snapshot.terminal, bytes=snapshot.bytes,
                evidence=snapshot.evidence(), lazy=snapshot.evidence_resolver()(),
                lazy_yield=snapshot.evidence_resolver()(True))


def restored(histories):
    return [dict(items=[(key, struct.pack('dd', *value)) for key, value in history.values.items()],
                 end=history.end, samples=history.samples, chunks=history.chunks,
                 restores=getattr(history, 'successful_buffered_restores', None),
                 type=type(history).__name__) for history in histories]


def take(monkeypatch, kind, seed, on, **kwargs):
    fast(monkeypatch, on)
    c, a, position = build(kind, seed, **kwargs)
    return c, a, capture(c, 'old', a, position, terminal=True)


# 1. flag
def test_flag_defaults_off_and_is_validated(monkeypatch):
    assert lc.fast_state() is False and lc.fast_native() is None
    monkeypatch.setenv('KVMEM_CHECKPOINT_FAST_STATE', 'yes')
    with pytest.raises(ValueError, match='KVMEM_CHECKPOINT_FAST_STATE'):
        lc.fast_state()
    fast(monkeypatch, True)
    assert lc.fast_native() is not None


def test_extension_without_the_helpers_keeps_the_original_code(monkeypatch):
    fast(monkeypatch, True)
    lc._NATIVE.append(None)                 # an extension built before the helpers existed
    c, a, position = build('buffered', 3)
    snapshot = capture(c, 'old', a, position, terminal=True)
    assert all(type(entry[0]) is tuple for entry in snapshot.histories)
    lc._NATIVE.clear()
    assert summary(snapshot) == summary(take(monkeypatch, 'buffered', 3, True)[2])


# 2. capture
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('diverge', [False, True])
def test_capture_is_identical_with_and_without_the_flag(monkeypatch, kind, diverge):
    for seed in range(12):
        _, _, slow = take(monkeypatch, kind, seed, False, diverge=diverge)
        _, _, quick = take(monkeypatch, kind, seed, True, diverge=diverge)
        assert all(type(entry[0]) is tuple for entry in slow.histories)
        assert all(type(entry[0]) is (tuple if kind == 'plain' else StateRows) for entry in quick.histories)
        assert summary(slow) == summary(quick)
        assert slow == quick and quick == slow       # dataclass equality across representations
        for rows, *_ in quick.histories:
            assert [row[0] for row in rows] == sorted(row[0] for row in rows)


def test_evidence_digests_are_the_original_json_digests(monkeypatch):
    import hashlib
    _, _, quick = take(monkeypatch, 'deferred', 5, True)
    plain = tuple((tuple(entry[0]),) + tuple(entry[1:]) for entry in quick.histories)
    assert quick.evidence()['attention_sha256'] == hashlib.sha256(json.dumps(plain).encode()).hexdigest()
    assert quick.evidence()['map_sha256'] == hashlib.sha256(json.dumps(
        [lc._identity(rows) for rows in quick.maps]).encode()).hexdigest()
    assert quick.evidence()['layer_tokens'] == [sum(r[3] for r in rows) for rows in quick.maps]
    resolve = quick.evidence_resolver()
    fast(monkeypatch, False)                         # resolved later by a thread: still the same digests
    assert resolve(True) == quick.evidence()


def test_state_json_matches_json_dumps_for_edge_floats(monkeypatch):
    fast(monkeypatch, True)
    State = lc.fast_native().State
    if not hasattr(State, 'rows_json'):
        pytest.skip('toolchain without floating-point to_chars')
    edge = [0.0, -0.0, 1e16, 1e15, 9999999999999998.0, 1e17, 1e-4, 1e-5, 0.00012, 5e-324, 1.7976931348623157e308,
            float('inf'), float('-inf'), float('nan'), 1.0, 0.1, 1e22, 1e21, 123.456, 0.30000000000000004]
    rnd = random.Random(1)
    edge += [struct.unpack('d', struct.pack('Q', rnd.getrandbits(64)))[0] for _ in range(20000)]
    state = State.from_values({i * 32: (edge[i], edge[-1 - i]) for i in range(len(edge))})
    assert state.rows_json() == json.dumps(state.rows()).encode()


@pytest.mark.parametrize('kind', KINDS)
def test_snapshot_survives_later_changes_of_the_live_history(monkeypatch, kind):
    c, a, quick = take(monkeypatch, kind, 2, True)
    before = summary(quick)
    for history in c.histories['old']:
        history.values = {0: [1., 2.]}
        history.end += 32
    assert summary(quick) == before


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('fault', ['future_end', 'future_key', 'shared_scores', 'shared_meta', 'duplicate',
                                   'dense', 'not_retained', 'count', 'lengths', 'partial', 'frame'])
def test_capture_errors_are_identical(monkeypatch, kind, fault):
    messages = []
    for on in (False, True):
        fast(monkeypatch, on)
        c, a, position = build(kind, 4)
        maps = [list(rows) for rows in c.maps['old'][1]]
        if fault == 'future_end':
            c.histories['old'][2].end = position + 1
        elif fault == 'future_key':
            values = c.histories['old'][1].values; values[position + 32] = [1., 1.]
            c.histories['old'][1].values = values
        elif fault == 'shared_scores':
            values = a.action_history.prefill.values; values[next(iter(values))] = [123., 1.]
            a.action_history.prefill.values = values
        elif fault == 'shared_meta':
            a.action_history.prefill.samples += 1
        elif fault == 'duplicate':
            maps[1][1] = maps[1][0][:4] + (maps[1][1][4],)
        elif fault == 'dense':
            maps[2][3] = maps[2][3][:4] + (maps[2][3][4] + 1,)
        elif fault == 'not_retained':
            del a.records[next(iter(a.records))]
            maps = [[(s, g, 0 if i == 0 else b, n, d) for i, (s, g, b, n, d) in enumerate(rows)] for rows in maps]
        elif fault == 'count':
            maps[0][0] = maps[0][0][:3] + (31,) + maps[0][0][4:]
        elif fault == 'lengths':
            maps[1] = maps[1][:-1]
        elif fault == 'partial':
            maps[1] = maps[1][:-2] + [maps[1][-1][:4] + (maps[1][-2][4],)]
            maps[1].append(maps[1][0][:2] + (maps[0][0][2] if maps[0][0][2] not in {r[2] for r in maps[1]} else 0,
                                             32, maps[1][-1][4] + maps[1][-1][3]))
        elif fault == 'frame':
            shared = a.memory._store.working_records()[:-1]
            a.memory._store.working_records = lambda: shared
        c.maps['old'] = (position, tuple(tuple(rows) for rows in maps))
        c.records = lambda rid, authority, pos, extent, index: c.maps['old'][1][index]
        with pytest.raises(ValueError) as error:
            capture(c, 'old', a, position, terminal=True)
        messages.append(str(error.value))
    assert messages[0] == messages[1], messages


# 3. restore
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('captured', [False, True])
def test_restore_is_identical_for_every_flag_combination(monkeypatch, kind, captured):
    results = []
    for on in (False, True):
        c, a, snapshot = take(monkeypatch, kind, 6, captured)
        c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
        fast(monkeypatch, on)
        chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
        by_begin = {r.history_begin: r for r in a.records.values()}
        assert all(record is by_begin[row[2]] for rows, picked in zip(frame[1], chosen)
                   for row, record in zip(rows, picked))
        results.append((frame, restored(histories)))
        if kind != 'plain':
            assert histories[0].successful_buffered_restores == 1
        install_restored(c, 'fresh', a, frame, histories, snapshot)
        again = capture(c, 'fresh', a, snapshot.position, committed=True)
        assert summary(again)['rows'] == summary(snapshot)['rows']
    assert results[0] == results[1]


@pytest.mark.parametrize('kind', KINDS)
def test_restored_history_does_not_alias_mutable_state(monkeypatch, kind):
    c, a, snapshot = take(monkeypatch, kind, 8, True)
    c.maps.pop('old'); c.histories.pop('old')
    before = summary(snapshot)
    _, _, first = prepare_restore(c, 'one', a, snapshot, options())
    first[1].values = {0: [5., 6.]}
    first[1].end = 0
    _, _, second = prepare_restore(c, 'two', a, snapshot, options())
    assert summary(snapshot) == before
    assert bits((k, *v) for k, v in second[1].values.items()) == before['rows'][1]


@pytest.mark.parametrize('kind', KINDS)
def test_compacting_restore_is_identical(monkeypatch, kind):
    results = []
    for on in (False, True):
        c, a, snapshot = take(monkeypatch, kind, 9, on)
        c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
        c.query.emit = lambda event, **fields: None
        worker = NS(runner=NS(_kvmem_layerwise_pressure=c), options=options())
        restore_pressure_sync(worker, 'fresh', NS(authority=a, layerwise_pressure=snapshot))
        results.append(restored(c.histories['fresh']))
    assert results[0] == results[1]


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('fault', ['future_end', 'shared_scores', 'not_retained', 'frame'])
def test_restore_errors_are_identical(monkeypatch, kind, fault):
    from dataclasses import replace
    messages = []
    for on in (False, True):
        c, a, snapshot = take(monkeypatch, kind, 10, on)
        c.maps.pop('old'); c.histories.pop('old')
        if fault == 'future_end':
            entry = snapshot.histories[1]
            snapshot = replace(snapshot, histories=snapshot.histories[:1] + (
                (entry[0], snapshot.position + 1) + entry[2:],) + snapshot.histories[2:])
        elif fault == 'shared_scores':
            values = a.action_history.prefill.values; values[next(iter(values))] = [123., 1.]
            a.action_history.prefill.values = values
        elif fault == 'not_retained':
            del a.records[next(iter(a.records))]
        elif fault == 'frame':
            shared = a.memory._store.working_records()[:-1]
            a.memory._store.working_records = lambda: shared
        with pytest.raises(ValueError) as error:
            prepare_restore(c, 'fresh', a, snapshot, options())
        messages.append(str(error.value))
    assert messages[0] == messages[1], messages


@pytest.mark.parametrize('kind', KINDS)
def test_missed_retirement_prune_is_applied_like_before(monkeypatch, kind):
    import types
    from vllm_kvmem.serving.authority import RequestAuthority
    results = []
    for on in (False, True):
        for seed in range(11, 60):       # a retired block that had layer-zero scores and left every window
            c, a, snapshot = take(monkeypatch, kind, seed, on)
            used = {row[2] for rows in snapshot.maps for row in rows}
            scored = {row[0] for row in snapshot.histories[0][0]}
            victim = next((r for r in a.records.values()
                           if r.history_begin not in used and r.history_begin in scored), None)
            if victim is not None:
                break
        c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
        a.history_live = types.MethodType(RequestAuthority.history_live, a)
        del a.records[(victim.handle.slot, victim.handle.generation)]
        live = a.history_live()
        a.action_history.prefill.values = {k: v for k, v in a.action_history.prefill.values.items()
                                           if k in live or k >= a.history_end}
        _, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
        assert all(victim.history_begin not in h.values for h in histories)
        results.append((seed, restored(histories)))
    assert results[0] == results[1]


# 4. handles, pairs, token identity
def test_window_handles_and_map_pairs_match(monkeypatch):
    c, a, snapshot = take(monkeypatch, 'buffered', 12, True)
    groups = [list(rows) for rows in snapshot.maps] + [[(-1, -1, 0, 32, 0), (5, 2, 32, 32, 32)]]
    records = [[NS(handle=NS(slot=3, generation=4)), NS(handle=NS(slot=1, generation=9))]]
    fast(monkeypatch, False)
    expected = holds.window_handles(groups), holds.window_handles(records), holds.window_handles(groups + records)
    assert lc.map_pairs(snapshot) is None
    fast(monkeypatch, True)
    assert (holds.window_handles(groups), holds.window_handles(records),
            holds.window_handles(groups + records)) == expected
    assert holds.window_handles(iter(groups)) == expected[0]
    pairs = lc.map_pairs(snapshot)
    assert set(pairs) == {(row[0], row[1]) for rows in snapshot.maps for row in rows}
    assert lc.map_pairs(snapshot) is pairs               # built once per checkpoint


def test_token_bytes_matches_the_python_identity(monkeypatch):
    tokens = [0, 1, 2**32 - 1, 77, 5]
    fast(monkeypatch, False)
    expected = token_bytes(tokens, 4), token_bytes(tuple(tokens), 3)
    fast(monkeypatch, True)
    assert (token_bytes(tokens, 4), token_bytes(tuple(tokens), 3)) == expected
    for bad, count in (([1, 2**32, 3], 2), ([1, True, 3], 2), ([1, -1, 3], 2), ([1, 2], 2), ([1, 2], 0)):
        errors = []
        for on in (False, True):
            fast(monkeypatch, on)
            with pytest.raises(ValueError) as error:
                token_bytes(bad, count)
            errors.append(str(error.value))
        assert errors[0] == errors[1]


# 5. GC pause ownership
class Scope:
    def __init__(self):
        self.paused, self.calls = False, []

    def pause_policy(self, paused):
        self.calls.append(paused)
        self.paused = paused

    policy_paused = property(lambda self: self.paused)

    def protect(self, handles):
        assert self.paused
        self.calls.append(('protect', len(handles)))
        return object()

    def protect_current(self, handles):
        assert self.paused
        self.calls.append(('current', len(handles)))
        return object()


def gc_take(monkeypatch, on, kind='buffered'):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    fast(monkeypatch, on)
    c, a, position = build(kind, 13)
    a.memory._history_scope = Scope()
    return c, a, position


@pytest.mark.parametrize('on', [False, True])
def test_capture_takes_its_hold_under_the_pause_and_releases_it(monkeypatch, on):
    c, a, position = gc_take(monkeypatch, on)
    snapshot = capture(c, 'old', a, position, terminal=True)
    scope = a.memory._history_scope
    assert snapshot.gc_hold is not None
    assert scope.calls[0] is True and scope.paused is False and not holds._OWNERS
    assert [call for call in scope.calls if isinstance(call, tuple)] == [
        ('protect', len({(row[0], row[1]) for rows in snapshot.maps for row in rows}))]


def test_failed_capture_releases_the_early_pause(monkeypatch):
    c, a, position = gc_take(monkeypatch, True)
    c.histories['old'][1].end = position + 1
    with pytest.raises(ValueError, match='future attention evidence'):
        capture(c, 'old', a, position, terminal=True)
    assert a.memory._history_scope.paused is False and not holds._OWNERS


def test_early_capture_pause_keeps_another_owner_paused(monkeypatch):
    c, a, position = gc_take(monkeypatch, True)
    holds.pause(a)                                       # a selection in flight
    capture(c, 'old', a, position, terminal=True)
    assert a.memory._history_scope.paused is True
    holds.resume(a)
    assert a.memory._history_scope.paused is False and not holds._OWNERS


def resident(monkeypatch, on):
    c, a, position = gc_take(monkeypatch, on)
    snapshot = capture(c, 'old', a, position, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    c.query.emit = lambda event, **fields: None
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c, device='cpu'), options=options(),
                bindings={'fresh': NS(pressure_restore=False)})
    a.memory._history_scope.calls.clear()
    return c, a, snapshot, worker, NS(authority=a, layerwise_pressure=snapshot)


@pytest.mark.parametrize('on', [False, True])
def test_resident_restore_holds_the_window_and_releases_the_pause(monkeypatch, on):
    c, a, snapshot, worker, checkpoint = resident(monkeypatch, on)
    finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=False)
    scope = a.memory._history_scope
    assert scope.calls[0] is True and scope.calls[-1] is False and not holds._OWNERS
    assert sum(isinstance(call, tuple) for call in scope.calls) == 1
    assert 'fresh' in holds.holds_of(c).holds and 'fresh' in c.histories


def test_failed_resident_restore_releases_the_early_pause(monkeypatch):
    c, a, snapshot, worker, checkpoint = resident(monkeypatch, True)
    del a.records[next(iter(a.records))]
    used = {row[2] for rows in snapshot.maps for row in rows}
    if not any(r.history_begin not in used for r in a.records.values()):
        pytest.skip('fixture retired an unused record')
    a.records = {key: r for key, r in a.records.items() if r.history_begin not in list(used)[:1]}
    with pytest.raises(ValueError):
        finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=False)
    assert a.memory._history_scope.paused is False and not holds._OWNERS
    assert 'fresh' not in c.histories


# 6. round 2: pause requests, decode rows, idle history rows
class AckScope:
    """The native request/acknowledge protocol: every pause_policy call starts
    a new request, acknowledged by the worker's next tick."""

    def __init__(self):
        self.request = self.ack = self.polls = 0
        self.paused, self.calls = False, []

    def pause_policy(self, paused):
        self.calls.append(paused)
        self.paused = paused
        self.request += 1

    def tick(self):
        self.ack = self.request

    @property
    def policy_paused(self):
        self.polls += 1
        if self.polls > 3:                  # the worker's next tick, eventually
            self.tick()
        return self.paused and self.ack == self.request

    def protect(self, handles):
        assert self.paused and self.ack == self.request
        return object()


def ack_authority(monkeypatch, on):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    fast(monkeypatch, on)
    return NS(memory=NS(_history_scope=AckScope()))


@pytest.mark.parametrize('on', [False, True])
def test_an_acknowledged_early_pause_is_not_requested_again(monkeypatch, on):
    a = ack_authority(monkeypatch, on)
    scope = a.memory._history_scope
    assert holds.pause(a, key='checkpoint')
    scope.tick()                                         # acknowledged while rows are built
    holds.wait_paused(a, key='checkpoint')
    assert scope.calls == ([True] if on else [True, True])
    assert scope.polls == (1 if on else 4)               # no further worker tick needed
    holds.resume(a, key='checkpoint')
    assert scope.calls[-1] is False and scope.paused is False and not holds._OWNERS


@pytest.mark.parametrize('on', [False, True])
def test_pause_owners_request_and_release_the_native_pause(monkeypatch, on):
    a = ack_authority(monkeypatch, on)
    scope = a.memory._history_scope
    holds.pause(a)                                       # selection
    holds.pause(a, key='checkpoint')                     # overlapping capture
    assert scope.paused and scope.calls == ([True] if on else [True, True])
    holds.resume(a, key='checkpoint')
    assert scope.paused and scope.calls[-1] is True      # still owned by the selection
    holds.wait_paused(a)
    assert scope.paused and scope.ack == scope.request
    holds.resume(a)
    assert scope.calls[-1] is False and scope.paused is False and not holds._OWNERS
    holds.pause(a)                                       # a new pause is a new request
    assert scope.calls[-1] is True and scope.ack != scope.request
    assert holds.wait_paused(a) >= 0 and scope.ack == scope.request
    holds.resume(a)
    assert scope.calls.count(False) == 2


@pytest.mark.parametrize('kind', KINDS)
def test_recapture_of_a_restored_decode_history_is_identical(monkeypatch, kind):
    results = []
    for on in (False, True):
        c, a, first = take(monkeypatch, kind, 21, on)
        c.decode_histories['old'] = restored_decode(first)
        second = capture(c, 'old', a, first.position, terminal=True)
        assert second.decode == first.decode
        assert (second.decode[0][0] is first.decode[0][0]) is on      # rows reused, not rebuilt
        # Decode feedback replaces the history object; its rows are rebuilt.
        c.decode_histories['old'] = [DecodeHistory({**h.mass, 32: 1.5}, h.samples + 1, h.end, h.chunks + 1)
                                     for h in restored_decode(second)]
        third = capture(c, 'old', a, first.position, terminal=True)
        results.append((summary(first), summary(second), summary(third)))
        assert third.decode[0][0] == ((0, 0.25), (32, 1.5), (64, 0.5))
    assert results[0] == results[1]


def test_restored_decode_rows_are_dropped_when_the_mass_changes(monkeypatch):
    c, a, first = take(monkeypatch, 'buffered', 22, True)
    histories = restored_decode(first)
    histories[0].mass = {0: 9.0, 64: 0.5}                # replaced, same length
    histories[1].mass[96] = 2.0                          # grown in place
    c.decode_histories['old'] = histories
    second = capture(c, 'old', a, first.position, terminal=True)
    assert second.decode[0][0] == ((0, 9.0), (64, 0.5))
    assert second.decode[1][0] == ((0, 0.25), (64, 0.5), (96, 2.0))
    assert second.decode[2][0] is first.decode[2][0]


@pytest.mark.parametrize('fault', ['end', 'key', 'float_key', 'restored_key', 'restored_end'])
def test_future_decode_attention_is_rejected_identically(monkeypatch, fault):
    errors = []
    for on in (False, True):
        c, a, first = take(monkeypatch, 'buffered', 23, on)
        position = first.position
        if fault.startswith('restored'):
            c.decode_histories['old'] = restored_decode(first)
            position = 64 if fault == 'restored_key' else first.decode[0][2] - 1
        elif fault == 'end':
            c.decode_histories['old'][2].end = position + 1
        else:
            c.decode_histories['old'][1].mass[position if fault == 'key' else position + .5] = 1.0
        with pytest.raises(ValueError) as error:
            capture(c, 'old', a, first.position, terminal=True) if position == first.position else (
                lc._decode_snapshot(c.decode_histories['old'], position))
        errors.append(str(error.value))
    assert errors[0] == errors[1] == 'Layerwise checkpoint contains future decode attention'


def test_unusual_decode_keys_and_positions_take_the_original_expressions(monkeypatch):
    results = []
    for on in (False, True):
        fast(monkeypatch, on)
        history = DecodeHistory({64: 0.5, 0: 0.25}, 4, 90, 2)
        outcome = [lc._decode_snapshot([history], 96.0), lc._decode_snapshot([DecodeHistory()], 0)]
        for mass in ({'x': 1.0, 0: 2.0}, {None: 1.0}):
            with pytest.raises(TypeError) as error:
                lc._decode_snapshot([DecodeHistory(mass, 1, 0, 1)], 96)
            outcome.append(str(error.value))
        results.append(outcome)
    assert results[0] == results[1]
    assert results[0][0] == ((((0, 0.25), (64, 0.5)), 4, 90, 2),) and results[0][1] == (((), 0, 0, 0),)


def test_native_decode_history_rows_match(monkeypatch):
    from vllm_kvmem import _native
    from vllm_kvmem.serving.layerwise_decode import NativeDecodeHistory
    if not hasattr(_native, 'DecodeMassMap'):
        pytest.skip('extension without DecodeMassMap')
    mass = {64: 0.5, 0: 0.25, 320: 1e-9, 32: 7.0}
    rows = []
    for on in (False, True):
        fast(monkeypatch, on)
        history = NativeDecodeHistory(_native.DecodeMassMap.from_dict(mass), 4, 400, 2)
        rows.append(lc._decode_snapshot([history], 400))
        with pytest.raises(ValueError, match='future decode attention'):
            lc._decode_snapshot([history], 320)
    assert rows[0] == rows[1] == ((tuple(sorted(mass.items())), 4, 400, 2),)


class Frame:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def history_fixture(seed, blocks=60):
    from vllm_kvmem.serving.authority import VersionedRecords
    from vllm_kvmem.serving.session_worker import SessionWorker
    rnd = random.Random(seed)
    records = VersionedRecords()

    def add(index):
        record = NS(history_begin=index * 32, count=32, archived=rnd.random() < .5,
                    handle=NS(slot=100 + index, generation=1 + index % 3))
        records[(record.handle.slot, record.handle.generation)] = record
    for index in range(blocks):
        add(index)
    authority = NS(records=records, archived_epoch=0, _resident_count=3, history_end=blocks * 32,
                   gc=NS(pending_index=[1, 2]), archive=NS(namespace_allocated_records=lambda request: len(records)),
                   memory=NS(request_id='r', config=NS(sink_blocks=2, recent_blocks=3, block_tokens=32)))

    def window():
        keys = rnd.sample(sorted(records), min(len(records), 20)) + [(5, 5)]      # one unknown record
        return NS(records=tuple(key + (0, 32, 0) for key in keys))

    def prefix(held=True):
        maps = tuple(tuple(key + (0, 32, 0) for key in rnd.sample(sorted(records), min(len(records), 25)))
                     for _ in NAMES)
        return LayerwisePrefix(1, NAMES, maps, (), gc_hold=object() if held else None)
    namespace = ('tenant', 's')
    checkpoint = Frame(namespace=namespace, authority=authority, window=window(), layerwise_pressure=prefix())
    worker = SessionWorker.__new__(SessionWorker)
    worker.pool = NS(prompts={})
    return rnd, worker, checkpoint, authority, add, window, prefix


def history_rows(monkeypatch, worker, checkpoint):
    fast(monkeypatch, False)
    plain = worker._history_rows(checkpoint)
    fast(monkeypatch, True)
    assert worker._history_rows(checkpoint) == plain
    assert worker._history_rows(checkpoint) == plain          # from the cache
    return plain


@pytest.mark.parametrize('seed', range(6))
def test_idle_history_rows_match_through_every_kind_of_change(monkeypatch, seed):
    rnd, worker, checkpoint, authority, add, window, prefix = history_fixture(seed)
    records, namespace = authority.records, checkpoint.namespace
    seen, next_index = {history_rows(monkeypatch, worker, checkpoint)}, 60
    for step in range(120):
        op = rnd.randrange(11)
        if op == 0:                                      # archive completion
            live = [r for r in records.values() if not r.archived]
            if live:
                rnd.choice(live).archived = True
                authority.archived_epoch += 1
        elif op == 1:
            add(next_index); next_index += 1
            authority.history_end = next_index * 32
        elif op == 2 and len(records) > 30:              # retirement
            del records[rnd.choice(sorted(records))]
        elif op == 3:
            checkpoint.window = window()
        elif op == 4:
            checkpoint.layerwise_pressure = rnd.choice([None, prefix(), prefix(False)])
        elif op == 5:
            worker.pool.prompts[namespace] = Frame(window=window(), layerwise_pressure=rnd.choice([None, prefix()]))
        elif op == 6:
            worker.pool.prompts.pop(namespace, None)
        elif op == 7 and namespace in worker.pool.prompts:
            worker.pool.prompts[namespace].window = window()
        elif op == 8:
            authority._resident_count = rnd.randrange(5)
            authority.gc.pending_index = [0] * rnd.randrange(3)
        elif op == 9:
            authority.memory.config.sink_blocks = rnd.randrange(4)
            authority.memory.config.recent_blocks = rnd.randrange(4)
        else:                                            # a replaced record keeps its key
            key = rnd.choice(sorted(records))
            old = records[key]
            records[key] = NS(history_begin=old.history_begin, count=old.count, archived=not old.archived,
                              handle=old.handle)
        seen.add(history_rows(monkeypatch, worker, checkpoint))
    assert len(seen) > 10


def test_idle_history_rows_cache_does_not_keep_a_released_snapshot(monkeypatch):
    import gc
    import weakref
    rnd, worker, checkpoint, authority, add, window, prefix = history_fixture(3)
    worker.pool.prompts[checkpoint.namespace] = prompt = Frame(window=window(), layerwise_pressure=prefix())
    fast(monkeypatch, True)
    expected = worker._history_rows(checkpoint)
    dead = [weakref.ref(checkpoint.layerwise_pressure), weakref.ref(prompt.layerwise_pressure), weakref.ref(prompt)]
    checkpoint.layerwise_pressure = None                 # payload released with its policy hold
    del worker.pool.prompts[checkpoint.namespace], prompt
    gc.collect()
    assert all(ref() is None for ref in dead)
    fast(monkeypatch, False)
    plain = worker._history_rows(checkpoint)
    fast(monkeypatch, True)
    assert worker._history_rows(checkpoint) == plain != expected


def test_idle_history_rows_fall_back_for_frames_without_a_cache(monkeypatch):
    rnd, worker, checkpoint, authority, add, window, prefix = history_fixture(4)
    fast(monkeypatch, False)
    plain = worker._history_rows(checkpoint)
    fast(monkeypatch, True)
    checkpoint.window = NS(records=list(checkpoint.window.records))      # not the immutable tuple
    assert worker._history_rows(checkpoint) == plain and '_history_pairs' not in vars(checkpoint)
    del authority.archived_epoch                                          # an authority without the counter
    assert worker._history_rows(checkpoint) == plain


def test_authority_counts_archive_completions():
    import inspect
    from vllm_kvmem.serving import authority
    source = inspect.getsource(authority)
    assert source.count('.archived = True') == 1 and source.count('self.archived_epoch += 1') == 1
    assert source.index('self.archived_epoch += 1') < source.index('.archived = True')
