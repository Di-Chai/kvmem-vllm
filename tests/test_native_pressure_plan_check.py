"""_native.check_pressure_plan equals the Python layer-zero plan checks (results and errors)."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import _native
from vllm_kvmem.window_assembly import _validate_destinations

pytestmark = pytest.mark.skipif(not hasattr(_native, 'check_pressure_plan'),
                                reason='Requires the rebuilt native extension')


class Layer:
    def __init__(self, capacity, rope):
        self.kernel_page_table = NS(numel=lambda: capacity)
        self.key = NS(shape=(None, 1))
        self.cos_sin = NS(shape=(rope,))


def python_check(plan, records, layers, rolling):
    by_slot = {r.handle.slot: r for r in records}
    remaps = list(plan.remaps)
    positions, cursor = {}, 0
    for record in records:
        positions[record.handle.slot] = cursor
        cursor += record.count
    if ({r.block_id for r in remaps} != set(by_slot) or
            any(r.to_base != positions[r.block_id] for r in remaps)):
        raise ValueError('Layer-zero selection differs from the native assembly plan')
    _validate_destinations(remaps, by_slot, layers)
    skip = set() if rolling else {by_slot[r.block_id].history_begin for r in remaps if r.skip}
    return (skip, sum(r.n_tokens for r in remaps if rolling or not r.skip),
            0 if rolling else sum(r.n_tokens for r in remaps if r.skip),
            sum(r.n_tokens for r in remaps if not r.skip and r.n_tokens != 32))


def native_check(plan, records, layers, rolling):
    bounds = [(l.kernel_page_table.numel() * l.key.shape[1], l.cos_sin.shape[0]) for l in layers]
    skip, rebuilt, skipped, partial = _native.check_pressure_plan(
        plan, [r.handle.slot for r in records], [r.count for r in records],
        [r.history_begin for r in records], bounds, False, rolling)
    return set(skip), rebuilt, skipped, partial


def outcome(fn, *args):
    try:
        return fn(*args)
    except ValueError as error:
        return ('error', str(error))


@pytest.mark.parametrize('rolling', [False, True])
@pytest.mark.parametrize('case', ['ok', 'missing', 'extra_count', 'small_capacity', 'small_rope', 'two_layers'])
def test_native_plan_check_matches_python(tmp_path, case, rolling):
    from test_graph_lifecycle import fixture, complete
    a, archive, graph = fixture(tmp_path, sink=1)
    plan = a.memory.select({}, (), action=False, max_tokens=8)
    by_slot = {r.handle.slot: r for r in a.records.values()}
    records = sorted((by_slot[r.block_id] for r in plan.remaps), key=lambda r: r.history_begin)
    records = [NS(handle=r.handle, count=r.count, history_begin=r.history_begin) for r in records]
    layers = [Layer(10**6, 10**6)]
    if case == 'missing':
        records = records[:-1]
    elif case == 'extra_count':
        records[0].count += 1
    elif case == 'small_capacity':
        layers = [Layer(1, 10**6)]
    elif case == 'small_rope':
        layers = [Layer(10**6, 1)]
    elif case == 'two_layers':
        layers = [Layer(10**6, 10**6), Layer(2, 1)]
    assert outcome(native_check, plan, records, layers, rolling) == outcome(python_check, plan, records, layers, rolling)
    complete(a); del plan
    assert a.close() and archive.close()
