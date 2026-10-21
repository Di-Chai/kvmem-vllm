"""Host lifecycle/budget qualification; fake streams never execute GPU work."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from vllm_kvmem.deferred_action_scoring import (
    FrozenLayerInput, FrozenOwnerInput, FrozenScoreLimits, FrozenScorePool)


class Event:
    def __init__(self):
        self.ready = False
        self.records = self.waits = 0
        self.error = None

    def record(self, stream=None):
        self.records += 1
        self.ready = True

    def query(self):
        if self.error: raise self.error
        return self.ready

    def synchronize(self):
        self.waits += 1
        if self.error: raise self.error
        assert self.ready


class Stream:
    def __init__(self): self.waits = []
    def wait_event(self, event): self.waits.append(event)


class Backend:
    def __init__(self):
        self.allocations = []
        self.tracked = []
        self.current_stream = Stream()

    def empty(self, shape, dtype, *, host=False):
        self.allocations.append((shape, dtype, host))
        return torch.empty(shape, dtype=dtype)
    def event(self): return Event()
    def stream(self): return Stream()
    def current(self): return self.current_stream
    def context(self, stream): return nullcontext()
    def track(self, tensor, stream): self.tracked.append((tensor.data_ptr(), stream))


def limits(**kw):
    return replace(FrozenScoreLimits(layers=3, groups=2, heads=4, dim=8,
        page_tokens=4, block_tokens=4, max_rows=6, max_owners=2,
        max_segments=4, max_length=16), **kw)


def inputs():
    layers = tuple(FrozenLayerInput(str(i), torch.zeros((5, 4, 2, 8)).to(torch.float8_e4m3fn),
        torch.arange(16 * 8).reshape(16, 8).float(), torch.tensor(float(i+1)),
        torch.tensor(.25), .5, torch.bfloat16, i % 2) for i in range(3))
    queries = tuple(torch.arange(8*4*8).reshape(8, 4, 8).to(torch.bfloat16) + i for i in range(3))
    owner = FrozenOwnerInput('r', queries, (5, 2), (6, 4), (0, 4), (4, 3), 7,
        (torch.tensor([3, 1, 99], dtype=torch.int32), torch.tensor([2, 0, 98], dtype=torch.int32)),
        torch.full((2, 2), -1., dtype=torch.float32))
    return layers, (owner,)


def fake_score(q, positions, key, table, trig, key_scale, starts, counts, length, **kw):
    assert kw['complete_partition']
    result = q[:, 0, 0, None].float() + table[:2].float()[None, :] * key_scale
    result._action_scoring_backend = 'fused' if kw['fused'] else 'reference'
    return result


def pool(*, cap=1<<30, scratch=1<<30, scorer=fake_score, **kw):
    backend = Backend()
    result = FrozenScorePool(limits(**kw), 'cpu', fused=True, max_bytes=cap,
                             max_scratch_bytes=scratch, _backend=backend, _scorer=scorer)
    return result, backend


def seal(pool, slot):
    slot.event.record()
    pool.seal(slot)


def test_capacity_budget_matches_all_fixed_allocations_and_has_one_scratch_reservation():
    p, b = pool()
    actual_gpu = actual_host = 0
    for shape, dtype, host in b.allocations:
        count = torch.tensor(shape).prod().item() * torch.empty((), dtype=dtype).element_size()
        if host: actual_host += count
        else: actual_gpu += count
    assert p.snapshot()['gpu_bytes'] == actual_gpu
    assert p.snapshot()['host_bytes'] == actual_host
    assert p.snapshot()['resident_bytes'] == actual_host + actual_gpu
    assert p.snapshot()['reserved_bytes'] == actual_host + actual_gpu + p.snapshot()['scratch_bytes']
    assert p.snapshot()['allocated_slots'] == 2
    bigger = limits(max_rows=256, max_owners=16, max_segments=2218,
                    max_length=65576, heads=24, dim=256)
    assert bigger.budget(fused=False)['scratch_bytes'] < 2<<30
    assert bigger.budget(fused=False)['scratch_bytes'] > 1536<<20
    assert bigger.budget(fused=True)['scratch_bytes'] < bigger.budget(fused=False)['scratch_bytes']


@pytest.mark.parametrize('cap,scratch,reason', [(1, 1<<30, 'resident_budget'), (1<<30, 1, 'scratch_budget')])
def test_budget_decline_allocates_nothing_or_changes_sampling(cap, scratch, reason):
    p, b = pool(cap=cap, scratch=scratch)
    assert not b.allocations and p.stream is None
    assert p.freeze(*inputs(), packet_lease=object()) is None
    assert p.snapshot()['disabled_reason'] == reason


@pytest.mark.parametrize('value', [0, -1, True, 3.5])
def test_limit_geometry_requires_positive_integral_capacity(value):
    with pytest.raises(ValueError): limits(max_rows=value)


def test_two_slots_reuse_exact_storage_and_drop_leases_without_gc():
    p, _ = pool()
    layers, owners = inputs()
    class Lease: pass
    leases = [Lease(), Lease()]
    weak = [weakref.ref(x) for x in leases]
    slots = [p.freeze(layers, owners, packet_lease=x) for x in leases]
    assert p.freeze(layers, owners, packet_lease=object()) is None
    assert p.snapshot()['active_slots'] == 2
    addresses = [(s.storage.q.data_ptr(), s.storage.tables.data_ptr(), s.host_flags.data_ptr()) for s in slots]
    del leases
    with pytest.raises(RuntimeError, match='not ready'): p.release(slots[0])
    for s in slots:
        seal(p, s)
        s.event.ready = False
        with pytest.raises(RuntimeError, match='still in use'): p.release(s)
        assert s.packet_lease is not None
        s.event.ready = True
        p.release(s)
    assert all(x() is None for x in weak)
    for s in slots:
        with pytest.raises(RuntimeError, match='Stale'): p.release(s)
    reused = [p.freeze(layers, owners, packet_lease=object()) for _ in range(2)]
    assert addresses == [(s.storage.q.data_ptr(), s.storage.tables.data_ptr(), s.host_flags.data_ptr()) for s in reused]
    assert all(x is not y for x, y in zip(slots, reused))
    with pytest.raises(RuntimeError, match='Stale'): p.seal(slots[0])


def test_freeze_copies_exact_rows_prefix_metadata_and_scales_before_sources_change():
    p, _ = pool()
    layers, owners = inputs()
    slot = p.freeze(layers, owners, packet_lease=object())
    expected = [q[list(owners[0].query_rows)].clone() for q in owners[0].queries]
    for q in owners[0].queries: q.fill_(99)
    for t in owners[0].tables: t.fill_(0)
    for l in layers: l.key_scale.fill_(10); l.q_scale.fill_(20)
    assert all(torch.equal(slot.storage.q[i, :2], e) for i, e in enumerate(expected))
    assert slot.storage.positions[:2].tolist() == [6, 4]
    assert slot.storage.starts[0, :2].tolist() == [0, 4]
    assert slot.storage.counts[0, :2].tolist() == [4, 3]
    assert slot.storage.tables[0, 0, :2].tolist() == [3, 1]
    assert slot.storage.scales.tolist() == [[1., .25], [2., .25], [3., .25]]
    seal(p, slot)
    deltas = p.fallback(slot)
    assert deltas == {'fused': 3}
    # Layer order and original sequential add/mean, independently reproduced.
    expected_mass = None
    for i, q in enumerate(expected):
        table = torch.tensor([3., 1.]) if i%2 == 0 else torch.tensor([2., 0.])
        mass = q[:, 0, 0, None].float() + table[None, :] * (i+1)
        expected_mass = mass if expected_mass is None else expected_mass + mass
    assert torch.equal(owners[0].masses, expected_mass / 3)
    assert p.stream.waits == [slot.event] and slot.event.waits == 0
    assert slot.storage.fallback_event.waits == 1
    assert p.snapshot()['backend_counts'] == {'fused': 3}
    p.release(slot)


def test_multiple_owners_layers_and_groups_keep_explicit_order():
    seen = []
    def scorer(q, *args, **kwargs):
        seen.append(q[0,0,0].item())
        return fake_score(q, *args, **kwargs)
    p, _ = pool(scorer=scorer)
    layers, (owner,) = inputs()
    other = replace(owner, request_id='second', query_rows=(0,), positions=(3,),
                    masses=torch.zeros(1,2))
    slot = p.freeze(layers, (other, owner), packet_lease=object())
    seal(p, slot); p.fallback(slot)
    assert seen == [0., 1., 2., 160., 161., 162.]
    assert tuple(x.request_id for x in slot.owners) == ('second','r')
    p.release(slot)


@pytest.mark.parametrize('change,reason', [
    ({'query_rows': (8, 2)}, 'query_geometry'),
    ({'query_rows': (True, 2)}, 'row_geometry'),
    ({'positions': (7, 4)}, 'row_geometry'),
    ({'starts': (0, 5)}, 'incomplete_partition'),
    ({'counts': (4, 2)}, 'incomplete_partition'),
    ({'counts': (5, 2)}, 'incomplete_partition'),
    ({'length': 17}, 'owner_geometry'),
    ({'queries': []}, 'unordered_inputs'),
    ({'masses': torch.zeros(2,3)}, 'mass_geometry'),
])
def test_unsupported_input_never_acquires_slot(change, reason):
    p, _ = pool()
    layers, (owner,) = inputs()
    assert p.freeze(layers, (replace(owner, **change),), packet_lease=object()) is None
    assert p.snapshot()['active_slots'] == 0
    assert p.snapshot()['counts'] == {'declined:'+reason: 1}


def test_non_bf16_raw_query_goes_to_old_path_without_codec_change():
    p, _ = pool()
    layers, (owner,) = inputs()
    owner = replace(owner, queries=tuple(q.float() for q in owner.queries))
    assert p.freeze(layers, (owner,), packet_lease=object()) is None


@pytest.mark.parametrize('where', ['score', 'record', 'wait', 'query'])
def test_failure_keeps_generation_and_packet_lease_permanently_quarantined(where):
    def scorer(*a, **k):
        if where == 'score': raise RuntimeError('injected score')
        return fake_score(*a, **k)
    p, _ = pool(scorer=scorer)
    lease = NS()
    slot = p.freeze(*inputs(), packet_lease=lease)
    seal(p, slot)
    def fail(*a, **k): raise RuntimeError('injected '+where)
    if where == 'record': slot.storage.fallback_event.record = fail
    elif where == 'wait': slot.storage.fallback_event.synchronize = fail
    elif where == 'query': slot.event.query = fail
    with pytest.raises(RuntimeError, match='injected'):
        p.release(slot) if where == 'query' else p.fallback(slot)
    assert slot.packet_lease is lease and p.snapshot()['active_slots'] == 1
    assert p.snapshot()['quarantined']
    slot.event.query = lambda: True
    slot.storage.fallback_event.synchronize = lambda: None
    with pytest.raises(RuntimeError, match='quarantined'): p.release(slot)
    with pytest.raises(RuntimeError, match='quarantined'): p.fallback(slot)
    with pytest.raises(RuntimeError, match='quarantined'): p.freeze(*inputs(), packet_lease=object())


def test_partial_snapshot_enqueue_failure_is_never_recycled():
    p, b = pool()
    def fail(*a): raise RuntimeError('injected source tracking failure')
    b.track = fail
    with pytest.raises(RuntimeError, match='tracking'): p.freeze(*inputs(), packet_lease=object())
    slot = p.storages[0].active
    assert slot and slot.state == 'failed' and slot.sources
    assert not slot.event.records
    with pytest.raises(RuntimeError, match='quarantined'): p.release(slot)


def test_unsealed_and_duplicate_fallback_rejected_without_events():
    p, _ = pool()
    slot = p.freeze(*inputs(), packet_lease=object())
    with pytest.raises(RuntimeError, match='sealed'): p.fallback(slot)
    assert not p.stream.waits
    seal(p, slot)
    with pytest.raises(RuntimeError, match='already sealed'): p.seal(slot)
    p.fallback(slot)
    with pytest.raises(RuntimeError, match='sealed'): p.fallback(slot)
    p.release(slot)
