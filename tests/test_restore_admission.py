from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.restore_admission import RestoreAdmission


def entry(rid, namespace=None):
    return NS(namespace=namespace or ('tenant', rid), generation=1, history=64, window=48)


def request(rid):
    return NS(request_id=rid)


def command(permit, number=1):
    return NS(request_id=permit.request.request_id, command_id=number,
              generation=permit.generation, history_end=permit.history,
              window_tokens=permit.window, resident=permit.view is not None)


def test_cold_waiters_do_not_own_native_slots_and_quota_lasts_through_cancel():
    gate = RestoreAdmission()
    requests = [request(str(i)) for i in range(8)]
    assert gate.lookup(requests[0], entry('0'))
    gate.before_allocate(requests[0], 64)
    gate.allocated(requests[0])
    gate.dispatch(command(gate.permits['0']))
    assert all(not gate.lookup(r, entry(r.request_id)) for r in requests[1:])
    assert gate.snapshot()['occupied'] == {'resident': 0, 'assembly': 1}
    gate.cancel(requests[0])
    gate.end_step()
    assert not gate.lookup(requests[1], entry('1'))
    with pytest.raises(ValueError, match='Stale'):
        gate.complete('0', 999)
    assert '0' in gate.permits
    gate.complete('0', 1)
    assert gate.lookup(requests[1], entry('1'))
    gate.end_step()
    assert not gate.permits


def test_resident_pipeline_keeps_each_lease_until_its_own_ack():
    from test_active_residency import setup, save
    resident, manager, pool, _ = setup()
    views = [save(resident, manager, rid=f'old-{i}', namespace=('user', str(i))) for i in range(2)]
    gate = RestoreAdmission(residency=resident, resident_slots=2)
    hot = [request(f'hot-{i}') for i in range(2)]
    for r, view in zip(hot, views):
        manager.prepare_import(r.request_id, 64, 48)
        assert gate.lookup(r, entry(r.request_id, view.namespace))
        gate.allocated(r)
        gate.dispatch(command(gate.permits[r.request_id]))
    assert gate.snapshot()['occupied']['resident'] == 2
    cold = request('cold')
    assert gate.lookup(cold, entry('cold'))  # Cold scratch remains independent.
    gate.cancel(hot[0])
    gate.end_step()
    assert gate.snapshot()['occupied'] == {'resident': 2, 'assembly': 0}
    gate.complete('hot-1', 1)  # Out-of-order completion never releases its peer.
    assert set(gate.permits) == {'hot-0'}
    with pytest.raises(ValueError, match='Stale'):
        gate.complete('hot-0', 2)
    gate.complete('hot-0', 1)
    assert not gate.permits and gate.snapshot()['peak_occupied']['resident'] == 2


def test_existing_query_assembly_blocks_only_cold_and_ack_allows_next_waiter():
    from test_active_residency import setup, save
    resident, manager, pool, _ = setup()
    view = save(resident, manager)
    busy = [True]
    gate = RestoreAdmission(residency=resident, cold_busy=lambda: busy[0])
    cold, hot = request('cold'), request('hot')
    assert not gate.lookup(cold, entry('cold'))
    manager.prepare_import('hot', 64, 48)
    assert gate.lookup(hot, entry('hot', view.namespace))
    assert not resident.entries and manager.resident_imports['hot'] == view.groups[0][1]
    busy[0] = False
    assert gate.lookup(cold, entry('cold'))
    assert gate.snapshot()['occupied'] == {'resident': 1, 'assembly': 1}
    gate.end_step()
    assert resident.entries[view.namespace] is view
    assert not manager.resident_imports
    assert pool.get_num_free_blocks() == 13


def test_hot_admission_is_stable_through_ttl_and_pressure_until_rollback():
    from test_active_residency import setup, save
    resident, manager, pool, now = setup()
    view = save(resident, manager)
    manager.prepare_import('hot', 64, 48)
    gate = RestoreAdmission(residency=resident)
    hot = request('hot')
    assert gate.lookup(hot, entry('hot', view.namespace))
    now[0] += 10
    resident.expire()
    resident.make_room(16)
    assert gate.before_allocate(hot, 64).view is view
    assert all(b.ref_cnt == 1 for b in view.blocks)
    gate.cancel(hot)
    assert not gate.permits and not manager.resident_imports
    resident.expire()
    assert not resident.entries and pool.get_num_free_blocks() == 16


def test_expired_view_uses_cold_quota_before_allocating_any_pages():
    from test_active_residency import setup, save
    resident, manager, pool, now = setup()
    view = save(resident, manager)
    gate = RestoreAdmission(residency=resident)
    cold = request('cold')
    assert gate.lookup(cold, entry('cold'))
    now[0] += 2
    assert not gate.lookup(request('was-hot'), entry('was-hot', view.namespace))
    assert not resident.entries and not manager.resident_imports
    assert pool.get_num_free_blocks() == 16


def test_unallocated_permits_release_on_step_end_and_request_identity_is_checked():
    gate = RestoreAdmission()
    a = request('a')
    assert gate.lookup(a, entry('a'))
    with pytest.raises(ValueError, match='identity'):
        gate.lookup(request('a'), entry('a'))
    with pytest.raises(ValueError, match='exact'):
        gate.before_allocate(a, 32)
    gate.end_step()
    b = request('b')
    assert gate.lookup(b, entry('b'))
    gate.cancel(b)
    assert not gate.permits


def test_gate_wait_is_measured_before_worker_dispatch_without_repeated_log_spam(tmp_path):
    import json
    now = [1.]
    busy = [True]
    output = tmp_path / 'restore_admission.jsonl'
    gate = RestoreAdmission(cold_busy=lambda: busy[0], output=output, clock=lambda: now[0])
    a = request('a')
    for _ in range(5):
        assert not gate.lookup(a, entry('a'))
    assert len(output.read_text().splitlines()) == 2
    now[0] = 3.
    busy[0] = False
    assert gate.lookup(a, entry('a'))
    gate.allocated(a)
    gate.dispatch(command(gate.permits['a']))
    now[0] = 4.
    gate.complete('a', 1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    admitted = next(row for row in rows if row['stage'] == 'admitted')
    assert admitted['first_seen_at'] == admitted['first_deferred_at'] == 1.
    assert admitted['admission_wait_seconds'] == 2.
    assert rows[-1]['total_wait_seconds'] == 3.
    assert not gate.waiting


def test_repeated_native_deferral_does_not_log_every_provisional_retry(tmp_path):
    output = tmp_path / 'restore_admission.jsonl'
    gate = RestoreAdmission(output=output)
    a = request('a')
    for _ in range(5):
        assert gate.lookup(a, entry('a'))
        gate.end_step()
    assert len(output.read_text().splitlines()) == 2  # arrival, native-admission deferral
