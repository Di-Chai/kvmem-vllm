"""KVMEM_PRESSURE_NATIVE_READER helpers equal the Python reader paths."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem import _native
from vllm_kvmem.core.memory_storage import gather_records_native

pytestmark = pytest.mark.skipif(not hasattr(_native, 'RecordSources'), reason='Requires the rebuilt extension')

R = 4096


def sources(n=12, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, R, dtype=np.uint8).tobytes() for _ in range(n)]


@pytest.mark.parametrize('offset,length', [(0, 64), (100, 900), (R-16, 16), (0, R)])
def test_gather_matches_python_region_copy(offset, length):
    src = sources()
    indices = [5, 0, 11, 5, 7]
    native = _native.RecordSources(src, R)
    out = np.zeros(len(indices)*length, np.uint8)
    native.gather(indices, offset, length, out)
    expected = np.zeros_like(out)
    gather_records_native(expected, [src[i] for i in indices], R, offset, length, 0, len(indices))
    assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize('bad', [[], [-1], [12], [0, 99]])
def test_invalid_indices_use_the_original_message(bad):
    native = _native.RecordSources(sources(), R)
    with pytest.raises(ValueError, match='Invalid prepared CPU region extent or source index'):
        native.check(bad)


def test_bad_extent_and_destination_are_rejected():
    native = _native.RecordSources(sources(), R)
    with pytest.raises(ValueError, match='Invalid prepared CPU region extent'):
        native.gather([0], R-8, 16, np.zeros(16, np.uint8))
    with pytest.raises(ValueError, match='different extent'):
        native.gather([0, 1], 0, 16, np.zeros(31, np.uint8))
    with pytest.raises(ValueError, match='exact immutable record bytes'):
        _native.RecordSources([b'x' * (R-1)], R)


class Rec:
    def __init__(self, begin, handle, count=32, archived=True):
        self.history_begin, self.handle, self.count, self.archived = begin, handle, count, archived


def python_indices(records, table, block):
    indices = []
    for record in records:
        entry = table.get(id(record))
        if (entry is None or entry[0] is not record or record.handle is not entry[1] or
                record.history_begin != entry[2] or record.count != block or not record.archived):
            raise ValueError('Prepared pressure source is absent, partial or changed')
        indices.append(entry[3])
    return indices


def outcome(fn, *args):
    try:
        return fn(*args)
    except ValueError as error:
        return str(error)


@pytest.mark.parametrize('change', ['none', 'absent', 'handle', 'begin', 'count', 'archived', 'alias'])
def test_record_indices_match_python(change):
    records = [Rec(32*i, object()) for i in range(6)]
    table = {id(r): (r, r.handle, r.history_begin, 10+i) for i, r in enumerate(records)}
    group = list(records[1:5])
    if change == 'absent': group.append(Rec(999, object()))
    elif change == 'handle': group[2].handle = object()
    elif change == 'begin': group[1].history_begin += 1
    elif change == 'count': group[0].count = 31
    elif change == 'archived': group[3].archived = False
    elif change == 'alias':
        other = Rec(0, records[0].handle); table[id(other)] = (records[0], other.handle, 0, 1); group.append(other)
    assert outcome(_native.prepared_record_indices, group, table, 32) == outcome(python_indices, group, table, 32)
    assert _native.prepared_contains(group, table) == all(id(r) in table for r in group)
    assert _native.prepared_record_indices((records[0], records[5]), table, 32) == [10, 15]


def test_parallel_gather_is_byte_identical(monkeypatch):
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', '4')
    src = sources(40)
    indices = list(range(39, -1, -1)) + [3, 3]
    native = _native.RecordSources(src, R)
    serial = np.zeros(len(indices) * 300, np.uint8)
    parallel = np.zeros_like(serial)
    native.gather(indices, 77, 300, serial)
    native.gather(indices, 77, 300, parallel, True)
    assert serial.tobytes() == parallel.tobytes()
