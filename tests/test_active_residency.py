from types import SimpleNamespace

import pytest

from vllm_kvmem.active_residency import ActiveResidency
from vllm_kvmem.session_control import SessionNotice
from test_window_manager import make_manager


def setup(fraction=.5, *, pressure_only=False):
    manager, pool = make_manager()
    now = [10.]
    resident = ActiveResidency({0:manager}, pool, 0 if pressure_only else 1000, fraction,
                              pressure_only=pressure_only, clock=lambda:now[0])
    return resident, manager, pool, now


def save(resident, manager, rid='old', namespace=('', 'session'), generation=1):
    manager.offsets[rid] = 16
    manager.allocate_new_blocks(rid, 64, 64)
    notice = SessionNotice('retired', namespace, generation, rid, 64, 48, resident_ready=True)
    resident.consume(notice)
    manager.free(rid)  # Native scheduler free follows the durable worker ack.
    return resident.entries.get(namespace)


def test_terminal_pages_stay_exclusive_until_exact_import_adopts_them():
    resident, manager, pool, _ = setup()
    view = save(resident, manager)
    assert pool.get_num_free_blocks() == 13
    assert all(b.ref_cnt == 1 for b in view.blocks)
    entry = SimpleNamespace(namespace=view.namespace, generation=1, history=64, window=48)
    assert resident.match(entry) is view
    manager.prepare_import('next', 64, 48)
    assert manager.get_num_blocks_to_allocate('next', 80, (), 64, 0, 80) == 4
    resident.claim(view, 'next')
    assert manager.get_num_blocks_to_allocate('next', 80, (), 64, 0, 80) == 1
    manager.add_local_computed_blocks('next', (), 0, 64)
    manager.allocate_external_computed_blocks('next', 0, 64)
    manager.allocate_new_blocks('next', 80, 80)
    resident.committed(view, 'next')
    assert pool.get_num_free_blocks() == 12
    assert all(b.ref_cnt == 1 for b in view.blocks)
    assert not resident.entries and resident.reused_tokens == 48
    manager.free('next')
    assert pool.get_num_free_blocks() == 16


def test_failed_native_admission_rolls_back_without_losing_or_freeing_source_pages():
    resident, manager, pool, _ = setup()
    view = save(resident, manager)
    manager.prepare_import('next', 64, 48)
    resident.claim(view, 'next')
    resident.rollback(view, 'next')
    assert resident.entries[view.namespace] is view
    assert not manager.resident_imports and pool.get_num_free_blocks() == 13
    resident.claim(view, 'next')
    manager.add_local_computed_blocks('next', (), 0, 64)
    with pytest.raises(RuntimeError, match='allocated resident'):
        resident.rollback(view, 'next')
    manager.free('next')
    assert pool.get_num_free_blocks() == 16


@pytest.mark.parametrize('change', [{'generation':2}, {'history':48}, {'window':32}, {'pressure_restore':True}])
def test_changed_checkpoint_cannot_reuse_previous_physical_view(change):
    resident, manager, pool, _ = setup()
    view = save(resident, manager)
    entry = SimpleNamespace(**({'namespace':view.namespace, 'generation':1, 'history':64, 'window':48} | change))
    assert resident.match(entry) is None
    assert resident.mismatches == 1 and pool.get_num_free_blocks() == 16


@pytest.mark.parametrize('pressure_only', [False, True])
def test_pressure_releases_idle_views_but_never_live_imports_or_another_namespace(pressure_only):
    resident, manager, pool, now = setup(pressure_only=pressure_only)
    a = save(resident, manager, 'a', ('user-a','session'))
    b = save(resident, manager, 'b', ('user-b','session'))
    assert pool.get_num_free_blocks() == 10
    resident.make_room(13, protect=b)
    assert pool.get_num_free_blocks() == 13 and list(resident.entries) == [b.namespace]
    assert resident.pressure_evictions == 1
    manager.prepare_import('next',64,48)
    resident.claim(b,'next')
    now[0] += 2
    resident.expire()
    resident.make_room(16)
    assert all(block.ref_cnt == 1 for block in b.blocks)
    manager.add_local_computed_blocks('next', (), 0, 64)
    manager.free('next')
    assert pool.get_num_free_blocks() == 16


def test_pressure_only_survives_long_tool_wait_but_still_revokes_on_demand():
    resident, manager, pool, now = setup(pressure_only=True)
    view = save(resident, manager)
    entry = SimpleNamespace(namespace=view.namespace, generation=1, history=64, window=48)
    now[0] += 24 * 3600
    resident.expire()
    assert resident.match(entry) is view and pool.get_num_free_blocks() == 13
    assert resident.snapshot()['retention_policy'] == 'pressure' and resident.expired == 0
    resident.make_room(16)
    assert not resident.entries and resident.pressure_evictions == 1
    assert pool.get_num_free_blocks() == 16
    view = save(resident, manager)
    resident.consume(SessionNotice('evicted', view.namespace, 2, ''))
    assert resident.match(entry) is view
    resident.consume(SessionNotice('evicted', view.namespace, 1, ''))
    assert not resident.entries and pool.get_num_free_blocks() == 16


def test_ttl_and_explicit_session_eviction_release_only_the_matching_generation():
    resident, manager, pool, now = setup()
    view = save(resident, manager)
    resident.consume(SessionNotice('evicted', view.namespace, 2, ''))
    assert resident.entries
    now[0] += 1.1
    resident.expire()
    assert not resident.entries and resident.expired == 1
    assert pool.get_num_free_blocks() == 16
    view = save(resident, manager)
    resident.consume(SessionNotice('evicted', view.namespace, 1, ''))
    assert not resident.entries and pool.get_num_free_blocks() == 16


def test_pool_fraction_and_durability_gate_never_hold_unqualified_pages():
    resident, manager, pool, _ = setup(fraction=.1)
    assert save(resident,manager) is None
    assert pool.get_num_free_blocks() == 16
    resident.consume(SessionNotice('retired', ('','uncaptured'), 1, 'missing', 64, 48))
    assert not resident.entries
    with pytest.raises(ValueError, match='terminal native allocation'):
        resident.consume(SessionNotice('retired', ('','bad'), 1, 'missing', 64, 48, resident_ready=True))


def test_capacity_counts_adopted_pages_and_releases_idle_pages_for_peer_growth(monkeypatch):
    from test_capacity import make_gate, request
    gate, manager, pool = make_gate(9)
    resident = ActiveResidency({0:manager}, pool, 1000, 1., clock=lambda:10.)
    gate.active_residency = resident
    view = save(resident, manager)
    manager.prepare_import('next',64,48)
    entry = SimpleNamespace(namespace=view.namespace, generation=1, history=64, window=48)
    gate.resident_entry = lambda rid: entry if rid == 'next' else None
    original = gate.native.allocate_slots
    def allocate(req, count, reserved_blocks=0, num_external_computed_tokens=0):
        if not num_external_computed_tokens:
            return original(req, count, reserved_blocks=reserved_blocks)
        need = manager.get_num_blocks_to_allocate(req.request_id, 64+count, (), 64, 0, 64+count)
        if need + reserved_blocks > pool.get_num_free_blocks(): return None
        manager.add_local_computed_blocks(req.request_id, (), 0, 64)
        manager.allocate_external_computed_blocks(req.request_id, 0, 64)
        return manager.allocate_new_blocks(req.request_id, 64+count, 64+count)
    monkeypatch.setattr(gate.native, 'allocate_slots', allocate)
    peer = request('peer')
    assert gate.allocate_slots(peer, 16) is not None
    successor = request('next')
    gate.begin_step()
    try:
        assert gate.allocate_slots(successor, 16, num_external_computed_tokens=64) is not None
        assert gate.resident_restores == {'next'}
        assert pool.get_num_free_blocks() == gate._step_total == 3
    finally:
        gate.end_step()
    gate.free(successor); gate.free(peer)
    assert not gate.resident_restores and pool.get_num_free_blocks() == 8
