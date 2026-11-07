"""Credits remain owned until physical allocation/publication/free completes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import gc
import random
from threading import Barrier, Event, Thread
import weakref

import pytest

from vllm_kvmem.resource_ledger import ResourceLedger


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready


READY = Fence(True)


def resources(ledger):
    return ledger.snapshot()['resources']


def test_multiresource_preparation_is_atomic_and_input_is_owned():
    ledger = ResourceLedger({'cpu': 64, 'ssd': 128, 'io': 1}, epoch='worker-1')
    demand = {'cpu': 48, 'ssd': 64, 'io': 1}
    first = ledger.prepare('a', 1, demand)
    demand['cpu'] = 0
    assert dict(first.amounts)['cpu'] == 48
    before = resources(ledger)
    assert ledger.prepare('b', 1, {'cpu': 8, 'ssd': 32, 'io': 1}) is None
    assert resources(ledger) == before
    assert ledger.snapshot()['deferrals_by_resource'] == {'io': 1}
    assert ledger.abort(first)
    second = ledger.prepare('b', 1, {'cpu': 64, 'ssd': 128, 'io': 1})
    assert second is not None
    assert second.identity.epoch == 'worker-1'
    assert second.identity.operation_id > first.identity.operation_id


def test_allocation_converts_credit_and_waits_for_both_completion_and_free():
    ledger = ResourceLedger({'cpu': 10, 'io': 2})
    grant = ledger.prepare('a', 7, {'cpu': 8, 'io': 2})
    with pytest.raises(ValueError, match='dispatched'):
        ledger.allocated(grant, {'cpu': 4})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 4, 'io': 1})
    assert resources(ledger)['cpu']['live'] == 4
    assert resources(ledger)['cpu']['promised'] == 4
    assert resources(ledger)['cpu']['available'] == 2
    ledger.allocated(grant, {'cpu': 6, 'io': 1})
    completion = Fence()
    assert not ledger.settle(grant, completion, retained={'cpu': 6})
    assert resources(ledger)['cpu']['available'] == 2
    assert resources(ledger)['io']['available'] == 0
    completion.ready = True
    assert ledger.settle(grant, completion, retained={'cpu': 6})
    assert resources(ledger)['cpu']['available'] == 4
    assert resources(ledger)['io']['available'] == 2
    free = Fence()
    assert not ledger.release(grant, free)
    assert ledger.prepare('b', 1, {'cpu': 5}) is None
    free.ready = True
    assert ledger.release(grant, free)
    assert resources(ledger)['cpu']['available'] == 10


def test_dispatch_closes_abort_path_even_if_worker_has_not_reported_allocation():
    ledger = ResourceLedger({'cpu': 100})
    grant = ledger.prepare('a', 0, {'cpu': 100})
    ledger.dispatch(grant)
    ledger.dispatch(grant)  # Retrying publication does not create new credit.
    with pytest.raises(ValueError, match='cannot be aborted'):
        ledger.abort(grant)
    with pytest.raises(ValueError, match='settled'):
        ledger.release(grant, READY)
    assert ledger.prepare('b', 1, {'cpu': 1}) is None
    # A failed submission that touched no memory still needs its gate's fence.
    assert ledger.settle(grant, READY)
    assert resources(ledger)['cpu']['available'] == 100


def test_abort_and_duplicate_settlement_do_not_mint_capacity():
    ledger = ResourceLedger({'cpu': 10})
    abandoned = ledger.prepare('a', 1, {'cpu': 10})
    assert ledger.abort(abandoned) and ledger.abort(abandoned)
    with pytest.raises(ValueError, match='dispatchable'):
        ledger.dispatch(abandoned)
    grant = ledger.prepare('b', 1, {'cpu': 10})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 10})
    assert ledger.settle(grant, READY)
    assert ledger.settle(grant, Fence())  # Already acknowledged completion.
    assert ledger.release(grant, READY) and ledger.release(grant, Fence())
    assert ledger.settle(grant, Fence())
    assert resources(ledger)['cpu']['available'] == 10
    assert ledger.snapshot()['active'] == 0
    assert ledger.snapshot()['aborts'] == 1
    with pytest.raises(ValueError, match='Conflicting'):
        ledger.settle(grant, READY, retained={})


def test_conflicting_completion_cannot_free_retained_owners():
    ledger = ResourceLedger({'cpu': 10})
    grant = ledger.prepare('a', 1, {'cpu': 10})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 10})
    assert ledger.settle(grant, READY, retained={'cpu': 8})
    assert ledger.settle(grant, READY, retained={'cpu': 8})
    with pytest.raises(ValueError, match='Conflicting'):
        ledger.settle(grant, READY, retained={'cpu': 7})
    assert resources(ledger)['cpu']['live'] == 8


def test_emergency_reserve_is_unavailable_to_ordinary_growth_and_replenished():
    ledger = ResourceLedger({'cpu': 10, 'io': 3}, emergency={'cpu': 3, 'io': 1})
    normal = ledger.prepare('a', 1, {'cpu': 7, 'io': 2})
    assert normal is not None
    assert ledger.prepare('b', 1, {'cpu': 1}) is None
    spill = ledger.prepare('a', 1, {'cpu': 3, 'io': 1}, emergency=True)
    assert spill is not None
    assert resources(ledger)['cpu']['emergency'] == 0
    ledger.dispatch(spill)
    ledger.allocated(spill, {'cpu': 2, 'io': 1})
    assert ledger.settle(spill, READY)
    assert resources(ledger)['cpu']['emergency'] == 1
    assert ledger.prepare('b', 1, {'cpu': 1}) is None
    assert ledger.release(spill, READY)
    assert resources(ledger)['cpu']['emergency'] == 3
    assert resources(ledger)['cpu']['available'] == 0
    ledger.abort(normal)
    assert resources(ledger)['cpu']['available'] == 7


def test_emergency_work_can_use_free_normal_capacity_without_overcommit():
    ledger = ResourceLedger({'cpu': 10}, emergency={'cpu': 3})
    spill = ledger.prepare('spill', 1, {'cpu': 8}, emergency=True)
    assert spill is not None
    assert resources(ledger)['cpu']['available'] == 2
    normal = ledger.prepare('a', 1, {'cpu': 2})
    assert normal is not None
    assert ledger.prepare('other', 1, {'cpu': 1}, emergency=True) is None
    ledger.abort(spill)
    assert resources(ledger)['cpu']['available'] == 5
    assert resources(ledger)['cpu']['emergency'] == 3


def test_reused_request_identity_and_late_ack_do_not_release_new_generation():
    ledger = ResourceLedger({'gpu': 8}, epoch='same-process')
    old = ledger.prepare('request', 11, {'gpu': 4})
    ledger.dispatch(old)
    ledger.allocated(old, {'gpu': 4})
    ledger.settle(old, READY)
    new = ledger.prepare('request', 12, {'gpu': 4})
    ledger.dispatch(new)
    ledger.allocated(new, {'gpu': 4})
    ledger.settle(new, READY)
    ledger.release(old, READY)
    ledger.release(old, READY)
    assert resources(ledger)['gpu']['live'] == 4
    assert resources(ledger)['gpu']['available'] == 4
    assert old.identity.generation == 11 and new.identity.generation == 12
    assert old.identity.operation_id != new.identity.operation_id


def test_foreign_or_reconstructed_grants_are_rejected_even_for_same_epoch():
    first = ResourceLedger({'cpu': 8}, epoch='epoch')
    second = ResourceLedger({'cpu': 8}, epoch='epoch')
    grant = first.prepare('a', 1, {'cpu': 8})
    with pytest.raises(ValueError, match='Foreign'):
        second.abort(grant)
    with pytest.raises(ValueError, match='recreated'):
        first.abort(replace(grant))
    with pytest.raises(FrozenInstanceError):
        grant.identity.generation = 2
    assert resources(first)['cpu']['promised'] == 8
    assert resources(second)['cpu']['available'] == 8


def test_invalid_allocation_or_settlement_never_changes_credit():
    ledger = ResourceLedger({'cpu': 8, 'ssd': 8})
    grant = ledger.prepare('a', 1, {'cpu': 8})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 4})
    before = resources(ledger)
    for amounts in ({'cpu': 9}, {'cpu': 3}, {'ssd': 1}):
        with pytest.raises(ValueError):
            ledger.allocated(grant, amounts)
        assert resources(ledger) == before
    with pytest.raises(ValueError, match='unreported'):
        ledger.settle(grant, READY, retained={'cpu': 5})
    assert resources(ledger) == before
    # A partial physical allocation failure still leaves four live bytes.
    ledger.settle(grant, READY)
    assert resources(ledger)['cpu']['live'] == 4
    assert resources(ledger)['cpu']['available'] == 4
    with pytest.raises(ValueError, match='dispatched'):
        ledger.allocated(grant, {'cpu': 5})


def test_failed_or_missing_fence_preserves_charge():
    class BrokenFence:
        def query(self):
            raise OSError('device completion failed')

    ledger = ResourceLedger({'cpu': 8})
    grant = ledger.prepare('a', 1, {'cpu': 8})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 8})
    with pytest.raises(ValueError, match='fence'):
        ledger.settle(grant, None)
    with pytest.raises(OSError):
        ledger.settle(grant, BrokenFence())
    assert resources(ledger)['cpu']['available'] == 0
    ledger.settle(grant, READY)
    with pytest.raises(OSError):
        ledger.release(grant, BrokenFence())
    assert resources(ledger)['cpu']['available'] == 0


def test_last_io_slot_has_only_one_concurrent_winner():
    ledger = ResourceLedger({'cpu': 100, 'io': 1})
    barrier = Barrier(16)

    def reserve(index):
        barrier.wait()
        return ledger.prepare(str(index), 1, {'cpu': 5, 'io': 1})

    with ThreadPoolExecutor(max_workers=16) as executor:
        grants = list(executor.map(reserve, range(16)))
    winners = [grant for grant in grants if grant is not None]
    assert len(winners) == 1
    assert resources(ledger)['cpu']['promised'] == 5
    assert resources(ledger)['io']['promised'] == 1
    assert ledger.snapshot()['deferrals'] == 15
    ledger.abort(winners[0])
    assert resources(ledger)['cpu']['available'] == 100


def test_concurrent_duplicate_settlement_and_release_are_idempotent():
    ledger = ResourceLedger({'cpu': 80, 'io': 2})
    grant = ledger.prepare('a', 1, {'cpu': 80, 'io': 2})
    ledger.dispatch(grant)
    ledger.allocated(grant, {'cpu': 60, 'io': 1})
    barrier = Barrier(16)

    def finish(_):
        barrier.wait()
        assert ledger.settle(grant, READY, retained={'cpu': 60})
        assert ledger.release(grant, READY)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(finish, range(16)))
    assert resources(ledger)['cpu']['available'] == 80
    assert resources(ledger)['io']['available'] == 2
    assert ledger.snapshot()['active'] == 0


def test_completion_query_runs_outside_ledger_lock():
    ledger = ResourceLedger({'cpu': 8})
    grant = ledger.prepare('a', 1, {'cpu': 8})
    ledger.dispatch(grant)

    class WorkerFence:
        def query(self):
            completed = Event()
            worker = Thread(target=lambda: (ledger.snapshot(), completed.set()), daemon=True)
            worker.start()
            assert completed.wait(2), 'Completion query held ledger lock'
            worker.join(timeout=2)
            return True

    assert ledger.settle(grant, WorkerFence())


def test_active_grants_are_retained_but_completed_receipts_do_not_leak():
    ledger = ResourceLedger({'cpu': 8})
    grant = ledger.prepare('a', 1, {'cpu': 8})
    weak = weakref.ref(grant)
    del grant
    gc.collect()
    assert weak() is not None  # Losing the caller's reference must not free credit.
    assert resources(ledger)['cpu']['available'] == 0
    ledger.abort(weak())
    gc.collect()
    assert weak() is None
    assert ledger.snapshot()['active'] == 0


def test_random_lifecycles_return_to_zero_after_fenced_drain():
    rng = random.Random(87191)
    ledger = ResourceLedger({'cpu': 43, 'io': 7}, emergency={'cpu': 5, 'io': 1})
    outstanding = []
    for iteration in range(1500):
        if not outstanding or rng.random() < .6:
            grant = ledger.prepare(str(iteration % 11), iteration // 11,
                {'cpu': rng.randint(1, 12), 'io': 1}, emergency=rng.random() < .1)
            if grant is not None:
                outstanding.append((grant, dict(grant.amounts), 'prepared'))
        else:
            index = rng.randrange(len(outstanding))
            grant, owned, phase = outstanding[index]
            if phase == 'prepared' and rng.random() < .25:
                ledger.abort(grant)
                outstanding.pop(index)
            elif phase == 'prepared':
                ledger.dispatch(grant)
                actual = {'cpu': rng.randint(0, owned['cpu']), 'io': 1}
                ledger.allocated(grant, actual)
                assert not ledger.settle(grant, Fence())
                ledger.settle(grant, READY, retained={'cpu': actual['cpu']})
                outstanding[index] = grant, {'cpu': actual['cpu'], 'io': 0}, 'settled'
            else:
                ledger.release(grant, READY)
                outstanding.pop(index)
        state = resources(ledger)
        for name in ('cpu', 'io'):
            expected_charge = sum(owned.get(name, 0) for _, owned, _ in outstanding)
            assert state[name]['live'] + state[name]['promised'] == expected_charge
            assert state[name]['available'] >= 0
            assert expected_charge + state[name]['emergency'] <= state[name]['hard_limit']
    for grant, _, phase in outstanding:
        if phase == 'prepared':
            ledger.abort(grant)
        else:
            ledger.release(grant, READY)
    assert ledger.snapshot()['active'] == 0
    assert resources(ledger)['cpu']['live'] == resources(ledger)['cpu']['promised'] == 0
    assert resources(ledger)['cpu']['available'] == 38


@pytest.mark.parametrize('limits,emergency', [({}, {}), ({'cpu': True}, {}),
    ({'cpu': -1}, {}), ({'': 1}, {}), ({'cpu': 1}, {'cpu': 2}),
    ({'cpu': 1}, {'typo': 0}), ({'cpu': 1}, {'cpu': True})])
def test_invalid_geometry(limits, emergency):
    with pytest.raises(ValueError):
        ResourceLedger(limits, emergency=emergency)


@pytest.mark.parametrize('amounts', [{}, {'cpu': 0}, {'cpu': -1}, {'cpu': True}, {'typo': 1}])
def test_invalid_demand_does_not_create_reservations(amounts):
    ledger = ResourceLedger({'cpu': 8})
    with pytest.raises(ValueError):
        ledger.prepare('a', 1, amounts)
    assert ledger.snapshot()['active'] == 0
