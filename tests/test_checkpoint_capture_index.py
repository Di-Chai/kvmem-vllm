"""Opt-in KVMEM_CHECKPOINT_CAPTURE_INDEX: per-capture O(history) Python work removed.

Session P/M captures run on the engine thread while other requests decode. The
flag keeps the {history_begin: record} index of _resolve on the authority's
record dict (extended for appended records) and keeps decode rows native
(MassRows) until something iterates them. Every test compares the flag against
the original code on equal inputs: checkpoint contents (bitwise), restored and
fed decode histories, error messages, and the index itself after arbitrary
record-dict mutations.
"""
import random

import pytest

from vllm_kvmem import _native
from vllm_kvmem import layerwise_checkpoints as lc
from vllm_kvmem.authority import VersionedRecords
from vllm_kvmem.layerwise_checkpoints import MassRows, capture, records_by_begin, restored_decode
from vllm_kvmem.layerwise_decode import NativeDecodeHistory
from test_checkpoint_fast_state import NAMES, build, summary
from test_session_fast_turn import _env, bits, mode, session, state  # noqa: F401  (autouse fixture)

needs_max_key = pytest.mark.skipif(not hasattr(getattr(_native, 'DecodeMassMap', None), 'sorted_copy'),
                                   reason='native extension lacks DecodeMassMap.max_key/sorted_copy')


@pytest.fixture(autouse=True)
def _index_env(monkeypatch):
    monkeypatch.delenv('KVMEM_CHECKPOINT_CAPTURE_INDEX', raising=False)


def index_on(monkeypatch):
    monkeypatch.setenv('KVMEM_CHECKPOINT_CAPTURE_INDEX', '1')


# 1. flag
def test_flag_defaults_off_needs_fast_turn_and_is_validated(monkeypatch):
    assert lc.capture_index() is False
    index_on(monkeypatch)
    assert lc.capture_index() is False                  # inactive without KVMEM_SESSION_FAST_TURN
    mode(monkeypatch, 'turn')
    assert lc.capture_index() is True
    monkeypatch.setenv('KVMEM_CHECKPOINT_CAPTURE_INDEX', 'yes')
    with pytest.raises(ValueError, match='KVMEM_CHECKPOINT_CAPTURE_INDEX'):
        lc.capture_index()


# 2. the begin index equals the comprehension after any mutation sequence
class Rec:
    def __init__(self, begin):
        self.history_begin = begin


def comprehension(records):
    return {record.history_begin: record for record in records.values()}


def same(index, expected):
    return index.keys() == expected.keys() and all(index[k] is expected[k] for k in expected)


@pytest.mark.parametrize('seed', range(40))
def test_begin_index_equals_the_comprehension_after_any_mutation(monkeypatch, seed):
    mode(monkeypatch, 'turn')
    index_on(monkeypatch)
    rnd = random.Random(seed)
    records = VersionedRecords()
    authority = type('A', (), {})()
    authority.records = records
    keys = [0]

    def new_key():
        keys[0] += 1
        return (keys[0], 1)

    begin = lambda: 32 * rnd.randrange(0, 60 if rnd.random() < .2 else 10 ** 6)
    for _ in range(rnd.randrange(0, 30)):
        records[new_key()] = Rec(begin())
    rebuilt = 0
    for step in range(120):
        op = rnd.choice(['append'] * 8 + ['append_dup', 'overwrite', 'delete', 'pop', 'update',
                                          'setdefault_new', 'setdefault_old', 'none', 'none'])
        live = list(records)
        if op == 'append':
            for _ in range(rnd.randrange(1, 4)):
                records[new_key()] = Rec(begin())
        elif op == 'append_dup' and live:
            records[new_key()] = Rec(records[rnd.choice(live)].history_begin)
        elif op == 'overwrite' and live:
            records[rnd.choice(live)] = Rec(begin())
        elif op == 'delete' and live:
            del records[rnd.choice(live)]
        elif op == 'pop' and live:
            records.pop(rnd.choice(live))
        elif op == 'update':
            records.update({new_key(): Rec(begin()) for _ in range(rnd.randrange(0, 3))})
        elif op == 'setdefault_new':
            records.setdefault(new_key(), Rec(begin()))
        elif op == 'setdefault_old' and live:
            records.setdefault(rnd.choice(live), Rec(begin()))
        if op in ('overwrite', 'delete', 'pop', 'update', 'setdefault_new', 'setdefault_old') and live:
            assert records.begin_index is None                 # no cached references survive a removal
        before = records.begin_index
        index = records_by_begin(authority)
        assert same(index, comprehension(records)), (seed, step, op)
        rebuilt += before is None or index is not before[1]
        assert records.begin_index[0] == records.version and records.begin_log == []
    assert rebuilt < 120                                   # appends extend the cached dict in place


def test_begin_index_off_and_plain_dicts_use_the_comprehension(monkeypatch):
    authority = type('A', (), {})()
    authority.records = VersionedRecords({(1, 1): Rec(0), (2, 1): Rec(32)})
    mode(monkeypatch, 'turn')
    assert same(records_by_begin(authority), comprehension(authority.records))
    assert authority.records.begin_index is None           # flag off: nothing cached
    index_on(monkeypatch)
    authority.records = {(1, 1): Rec(0), (2, 1): Rec(32)}  # not a VersionedRecords
    assert same(records_by_begin(authority), comprehension(authority.records))


# 3. capture / restore / feedback / recapture with lazy decode rows
@needs_max_key
@pytest.mark.parametrize('seed', range(6))
def test_session_turns_are_identical_with_the_capture_index(monkeypatch, seed):
    reference = session(monkeypatch, 'off', seed)
    index_on(monkeypatch)
    run = session(monkeypatch, 'turn', seed)
    for stage in ('first', 'again', 'second', 'third'):
        assert summary(getattr(run, stage)) == summary(getattr(reference, stage)), stage
        assert ([bits(rows) for rows, *_ in getattr(run, stage).decode] ==
                [bits(rows) for rows, *_ in getattr(reference, stage).decode]), stage
        assert getattr(run, stage).bytes == getattr(reference, stage).bytes, stage
    assert [state(h)[1:] for h in run.restored] == [state(h)[1:] for h in reference.restored]
    assert [state(h) for h in run.fed] == [state(h) for h in reference.fed]
    assert all(type(h) is NativeDecodeHistory for h in run.restored + run.fed)
    for index in range(len(NAMES)):
        rows = run.first.decode[index][0]
        assert type(rows) is MassRows
        assert run.again.decode[index][0] is rows           # the same live history: no new rows
        assert run.second.decode[index][0] is rows          # restored history shares the map
        assert run.second.decode[index][0].map is rows.map
        assert type(run.third.decode[index][0]) is MassRows


@needs_max_key
def test_capture_and_restore_build_no_row_tuples(monkeypatch):
    mode(monkeypatch, 'turn')
    index_on(monkeypatch)
    c, a, position = build('buffered', 5)
    c.decode_histories['old'] = [NativeDecodeHistory(_native.DecodeMassMap.from_dict({0: .25, 64: .5}), 4,
                                                     position - 8, 2) for _ in NAMES]
    first = capture(c, 'old', a, position, terminal=True)
    c.decode_histories['old'] = restored_decode(first)
    capture(c, 'old', a, position, terminal=True)
    assert all(rows._rows is None for rows, *_ in first.decode)
    assert first.decode[0][0] == ((0, .25), (64, .5)) and first.decode[0][0]._rows is not None


@needs_max_key
@pytest.mark.parametrize('fault', ['end', 'key', 'restored_key', 'restored_end'])
def test_future_decode_attention_is_rejected_identically(monkeypatch, fault):
    errors = []
    for flag in ('0', '1'):
        mode(monkeypatch, 'turn')
        monkeypatch.setenv('KVMEM_CHECKPOINT_CAPTURE_INDEX', flag)
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


def test_mass_rows_behave_like_the_row_tuple():
    if not hasattr(getattr(_native, 'DecodeMassMap', None), 'sorted_copy'):
        pytest.skip('native extension lacks DecodeMassMap.max_key/sorted_copy')
    rows = ((0, .25), (32, .5), (96, .25))
    lazy = MassRows(_native.DecodeMassMap.from_rows(rows))
    assert len(lazy) == 3 and bool(lazy) and lazy.max_key() == 96 and lazy._rows is None
    assert lazy == rows and lazy == MassRows(_native.DecodeMassMap.from_rows(rows))
    assert list(lazy) == list(rows) and lazy[-1] == (96, .25) and dict(lazy) == dict(rows)
    assert hash(lazy) == hash(rows)
    shuffled = _native.DecodeMassMap.from_dict({96: .25, 0: .25, 32: .5})
    assert list(shuffled.to_dict()) == [96, 0, 32]
    assert list(shuffled.sorted_copy().to_dict().items()) == list(rows)
    empty = MassRows(_native.DecodeMassMap.from_rows(()))
    assert len(empty) == 0 and not empty and empty.max_key() is None and empty == ()


def test_removed_records_are_released_immediately(monkeypatch):
    """The cached index must not keep a removed record (and its native index
    owner) alive: historical GC waits for that owner before an eviction closes."""
    import gc
    import weakref
    mode(monkeypatch, 'turn')
    index_on(monkeypatch)
    authority = type('A', (), {})()
    authority.records = VersionedRecords()
    for i in range(4):
        authority.records[(i, 1)] = Rec(32 * i)
    records_by_begin(authority)
    for remove in (lambda r: r.pop((0, 1)), lambda r: r.__delitem__((1, 1)),
                   lambda r: r.__setitem__((2, 1), Rec(9999)), lambda r: r.clear()):
        records_by_begin(authority)
        victims = [weakref.ref(v) for v in authority.records.values()]
        remove(authority.records)
        gc.collect()
        assert authority.records.begin_index is None
        alive = {id(v) for v in authority.records.values()}
        assert all(ref() is None or id(ref()) in alive for ref in victims)
