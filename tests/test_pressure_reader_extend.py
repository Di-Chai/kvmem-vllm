"""KVMEM_PRESSURE_READER_EXTEND: later lazily selected layers join the prepared reader.

The extension must yield the same bytes as a reader prepared over every layer,
hold its pins until close, and leave the reader unchanged (so the caller keeps
the original per-chunk path) whenever it cannot prepare a layer's records.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import record_key
from vllm_kvmem.core.memory import Phase
from vllm_kvmem.serving.pressure_reader import prepare_pressure_reader, reader_extend
from test_paired_archive_reads import archive_fixture

REGIONS = ((3, 11), (47, 9))


@pytest.fixture(autouse=True)
def native_flags(monkeypatch):
    if not callable(getattr(getattr(_native, 'RecordSources', None), 'extend', None)):
        pytest.skip('Requires the rebuilt native extension')
    for name in ('KVMEM_PRESSURE_READER_BATCH', 'KVMEM_PRESSURE_NATIVE_READER',
                 'KVMEM_PRESSURE_READER_ADOPT', 'KVMEM_PRESSURE_READER_EXTEND'):
        monkeypatch.setenv(name, '1')


def authority_fixture(tmp_path):
    archive, repo, leases, values = archive_fixture(tmp_path, 'cpu')
    records = [NS(handle=l.handle, history_begin=i*4, count=4, archived=True) for i, l in enumerate(leases)]
    plan, sets = object(), []

    def pin_set(handles):
        result = repo.pin_set(list(handles))
        sets.append(result)
        return result

    authority = NS(archive=archive, shape=(2, 4, 2, 2), history_end=12, poll=lambda: None,
        records={record_key(r.handle): r for r in records},
        memory=NS(phase=Phase.ASSEMBLING, _plan=plan, request_id='owner',
                  admitted_read=lambda handle: (repo.pin(handle), None), admitted_pin_set=pin_set))
    return authority, plan, records, values, sets


def expected(values, group):
    order = [r.history_begin // 4 for r in group]
    return b''.join(values[i][o:o+n] for o, n in REGIONS for i in order)


def read(reader, group):
    out = torch.empty(len(group)*20, dtype=torch.uint8)
    data, stats = reader.read(group, REGIONS, destination=out)
    assert data.data_ptr() == out.data_ptr() and stats['prepared_cpu_region_read']
    return data.numpy().tobytes()


def test_extended_layers_read_the_same_bytes_and_release_every_pin(tmp_path):
    a, plan, records, values, sets = authority_fixture(tmp_path)
    layer0, layer1, layer2 = [records[0]], [records[2], records[0]], [records[1], records[2], records[0]]
    with prepare_pressure_reader(a, plan, [layer0]) as reader:
        assert reader.native and reader.source_blocks == 1 and not reader.contains(layer1)
        assert reader.extend(layer1) and reader.contains(layer1) and reader.source_blocks == 2
        assert reader.extend(layer2) and reader.contains(layer2) and reader.source_blocks == 3
        assert reader.extend(layer2)            # nothing new: no further pins
        assert [len(s) for s in sets] == [1, 1, 1] and reader.extended_blocks == 2
        for group in (layer0, layer1, layer2, [records[1], records[1], records[0]]):
            assert read(reader, group) == expected(values, group)
    assert all(len(s) == 0 for s in sets)
    assert a.archive._prepared_region_readers == 0 and a.archive.close()


def test_extension_equals_a_reader_prepared_over_every_layer(tmp_path):
    a, plan, records, values, _ = authority_fixture(tmp_path)
    layers = [[records[1]], [records[0], records[2]], [records[2], records[1]]]
    with prepare_pressure_reader(a, plan, layers) as eager:
        reference = [read(eager, group) for group in layers]
    with prepare_pressure_reader(a, plan, layers[:1]) as lazy:
        for group in layers[1:]:
            assert lazy.extend(group)
        assert [read(lazy, group) for group in layers] == reference
    assert a.archive.close()


def test_partial_or_unsealed_records_stay_on_the_original_path(tmp_path):
    a, plan, records, values, sets = authority_fixture(tmp_path)
    records[1].count = 3
    with prepare_pressure_reader(a, plan, [[records[0]]]) as reader:
        assert reader.extend([records[1], records[2]])
        assert reader.contains([records[2]]) and not reader.contains([records[1]])
        assert reader.source_blocks == 2


@pytest.mark.parametrize('fault', ['foreign', 'unpublished', 'stale_pin', 'no_pin_set'])
def test_failed_extension_leaves_the_reader_and_pins_unchanged(tmp_path, monkeypatch, fault):
    a, plan, records, values, sets = authority_fixture(tmp_path)
    with prepare_pressure_reader(a, plan, [[records[0]]]) as reader:
        keys, blocks = reader.reader.keys, reader.source_blocks
        target = records[2]
        if fault == 'foreign':
            target = NS(**records[2].__dict__)          # not the authority's record object
        elif fault == 'unpublished':
            entries = a.archive._entries
            monkeypatch.setattr(a.archive, '_entries',
                {k: v for k, v in entries.items() if k != a.archive.key('owner', NS(handle=records[2].handle))}
                if hasattr(a.archive, 'key') else entries)
        elif fault == 'stale_pin':
            a.memory.admitted_pin_set = lambda handles: NS(valid=lambda: False, reset=lambda: sets.append('reset'))
        elif fault == 'no_pin_set':
            a.memory.admitted_pin_set = lambda handles: None
        assert reader.extend([target]) is False
        assert reader.reader.keys == keys and reader.source_blocks == blocks
        assert not reader.contains([target])
        if fault == 'stale_pin':
            assert sets[-1] == 'reset'
        if fault == 'unpublished':
            assert len(sets[-1]) == 0               # the new pins were released
        assert read(reader, [records[0]]) == expected(values, [records[0]])
    assert a.archive._prepared_region_readers == 0


def test_closed_or_non_native_reader_is_not_extended(tmp_path, monkeypatch):
    a, plan, records, values, _ = authority_fixture(tmp_path)
    reader = prepare_pressure_reader(a, plan, [[records[0]]])
    with reader:
        pass
    assert reader.extend([records[1]]) is False
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_READER', '0')
    with prepare_pressure_reader(a, plan, [[records[0]]]) as python_reader:
        assert not python_reader.native and python_reader.extend([records[1]]) is False


def test_flag_validation(monkeypatch):
    monkeypatch.setenv('KVMEM_PRESSURE_READER_EXTEND', 'yes')
    with pytest.raises(ValueError, match='READER_EXTEND'):
        reader_extend()
    monkeypatch.setenv('KVMEM_PRESSURE_READER_EXTEND', '0')
    assert reader_extend() is False


def test_native_sources_extend_validates_before_appending():
    sources = _native.RecordSources([bytes(64), bytes(range(64))], 64)
    assert sources.extend([bytes([7])*64]) == 2 and len(sources) == 3
    with pytest.raises(ValueError, match='exact immutable record bytes'):
        sources.extend([bytes(64), bytes(63)])
    assert len(sources) == 3
    out = bytearray(8)
    sources.gather([2], 0, 8, out)
    assert bytes(out) == bytes([7])*8
