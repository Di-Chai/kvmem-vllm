"""KVMEM_PRESSURE_NATIVE_COMMIT_TABLE: native identity-table commit maps equal prepare_commit's."""
import random
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import pressure_commit


def _table_cls():
    try:
        from vllm_kvmem import _native
    except ImportError:
        pytest.skip('native extension not built')
    cls = getattr(_native, 'IdentityTable', None)
    if cls is None:
        pytest.skip('native extension lacks IdentityTable')
    return cls


def archive(seed, n=300, partial=0.05):
    rng = random.Random(seed)
    records, begin = [], 0
    for i in range(n):
        count = 32 if rng.random() > partial else rng.randrange(1, 32)
        records.append(NS(handle=NS(slot=1000 + 7 * i, generation=rng.randrange(1, 9)), history_begin=begin, count=count))
        begin += 32
    return records, [r.handle.slot for r in records], rng


def selection(records, rng, layers=5, keep=.4):
    chosen, slots = [], []
    count = max(1, int(len(records) * keep))
    for _ in range(layers):
        picks = sorted(rng.sample(range(len(records)), count))
        chosen.append([records[i] for i in picks])
        slots.append([records[i].handle.slot for i in picks])
    return chosen, slots


def job(authority=None):
    return NS(authority=authority or object(), command=NS(request_id='r'), history=12345)


def both(records, record_slots, chosen, slots, table):
    j = job()
    want = pressure_commit.prepare_commit(j, chosen, records, record_slots, slots)
    got = pressure_commit.prepare_commit(j, chosen, records, record_slots, slots, table=table)
    return want, got


@pytest.mark.parametrize('seed', range(12))
def test_table_maps_equal_the_python_maps(seed, monkeypatch):
    cls = _table_cls()
    monkeypatch.delenv('KVMEM_PRESSURE_NATIVE_COMMIT_MAP', raising=False)
    records, record_slots, rng = archive(seed, partial=0)
    table = cls()
    table.extend(records, record_slots)
    chosen, slots = selection(records, rng)
    # every layer keeps the same token count (prepare_commit requires it)
    want, got = both(records, record_slots, chosen, slots, table)
    assert got.maps == want.maps and got.tokens == want.tokens
    assert got.layer_zero_starts == want.layer_zero_starts and got.chosen is chosen


def test_extend_appends_and_matches_only_its_prefix():
    cls = _table_cls()
    records, record_slots, _ = archive(1, n=50)
    table = cls()
    table.extend(records[:30], record_slots[:30])
    assert len(table) == 30 and table.matches(records)
    table.extend(records, record_slots)
    assert len(table) == 50 and table.matches(records)
    assert not table.matches(records[:49]) and not table.matches(list(reversed(records)))
    with pytest.raises(RuntimeError, match='duplicate physical slots'):
        cls().extend(records[:2] + records[:1], record_slots[:2] + record_slots[:1])


@pytest.mark.parametrize('case', ['missing', 'order', 'extent', 'layers'])
def test_errors_match_the_python_path(case, monkeypatch):
    cls = _table_cls()
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_COMMIT_MAP', '1')
    records, record_slots, rng = archive(3, n=60, partial=0)
    table = cls()
    table.extend(records, record_slots)
    chosen, slots = selection(records, rng, layers=3)
    if case == 'missing':
        slots[1] = slots[1][:-1] + [99]
    elif case == 'order':
        chosen[2] = chosen[2][1:] + chosen[2][:1]
    elif case == 'extent':
        chosen[1], slots[1] = chosen[1][:-1], slots[1][:-1]
    elif case == 'layers':
        slots = slots[:2]
    j = job()
    with pytest.raises(Exception) as python_error:
        pressure_commit.prepare_commit(j, chosen, records, record_slots, slots)
    with pytest.raises(Exception) as table_error:
        pressure_commit.prepare_commit(j, chosen, records, record_slots, slots, table=table)
    assert type(table_error.value) is type(python_error.value)
    assert str(table_error.value) == str(python_error.value)


def test_identity_table_helper_is_opt_in_and_resets_on_a_new_index(monkeypatch):
    _table_cls()
    controller = NS()
    records, record_slots, _ = archive(4, n=40)
    monkeypatch.delenv('KVMEM_PRESSURE_NATIVE_COMMIT_TABLE', raising=False)
    assert pressure_commit.identity_table(controller, 'r', records, record_slots) is None
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_COMMIT_TABLE', '1')
    first = pressure_commit.identity_table(controller, 'r', records[:20], record_slots[:20])
    again = pressure_commit.identity_table(controller, 'r', records, record_slots)
    assert again is first and len(again) == 40                  # appended records extend it
    rebuilt = pressure_commit.identity_table(controller, 'r', records[1:], record_slots[1:])
    assert rebuilt is not first and len(rebuilt) == 39          # a different index rebuilds it
