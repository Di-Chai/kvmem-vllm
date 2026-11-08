"""KVMEM_PRESSURE_NATIVE_COMMIT_MAP: prepare_commit maps and errors equal the Python path."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import _native
from vllm_kvmem.pressure_commit import prepare_commit

pytestmark = pytest.mark.skipif(not hasattr(_native, 'commit_map'), reason='Requires the rebuilt extension')


def build(layers=3, n=42):
    records = [NS(handle=NS(slot=100+i, generation=7+i), history_begin=32*i, count=32)
              for i in range(n)]
    slots = [100+i for i in range(n)]
    chosen, selected = [], []
    for layer in range(layers):
        picked = [records[i] for i in range(n) if (i + layer) % 3]
        chosen.append(picked)
        selected.append([r.handle.slot for r in picked])
    job = NS(authority=NS(), command=NS(request_id='r'), history=32*n)
    return job, chosen, records, slots, selected


def outcome(monkeypatch, native, *args):
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_COMMIT_MAP', '1' if native else '0')
    try:
        commit = prepare_commit(*args)
        return ('ok', commit.maps, commit.tokens, commit.layer_zero_starts)
    except (RuntimeError, KeyError, ValueError) as error:
        return (type(error), str(error))


@pytest.mark.parametrize('fault', ['none', 'order', 'count', 'missing', 'length'])
def test_native_commit_map_matches_python(monkeypatch, fault):
    job, chosen, records, slots, selected = build()
    if fault == 'order':
        chosen[1] = list(chosen[1]); chosen[1][3], chosen[1][4] = chosen[1][4], chosen[1][3]
    elif fault == 'count':
        chosen[2] = list(chosen[2]); chosen[2][5] = NS(handle=chosen[2][5].handle, history_begin=chosen[2][5].history_begin, count=1)
    elif fault == 'missing':
        records = records[:-1]; slots = slots[:-1]
    elif fault == 'length':
        selected[0] = selected[0][:-1]
    args = (job, chosen, records, slots, selected)
    expected = outcome(monkeypatch, False, *args)
    actual = outcome(monkeypatch, True, *args)
    assert actual == expected
    if fault == 'none':
        assert expected[0] == 'ok'
