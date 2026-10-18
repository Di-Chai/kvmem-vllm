from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from vllm_kvmem.transfer_budget import TransferBudget


def test_idle_uploads_are_unrestricted_and_busy_burst_has_a_byte_bound():
    now = [10.]
    budget = TransferBudget(100, 20, clock=lambda:now[0])
    budget.acquire(1000)
    budget.set_busy(True)
    budget.acquire(20)
    assert budget.credit == 0
    now[0] += .2
    budget.acquire(19)  # Avoid using floating point rounding as a real wait.
    assert 0 <= budget.credit < 2
    assert budget.snapshot()['busy_uploaded_bytes'] == 39
    assert budget.snapshot()['uploaded_bytes'] == 1039
    budget.set_busy(False)
    budget.acquire(1000)
    assert budget.credit == 20


def test_waiting_restore_producer_is_released_as_soon_as_gpu_has_no_public_work():
    budget = TransferBudget(1, 20)
    budget.set_busy(True)
    budget.acquire(20)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(budget.acquire, 20)
        deadline = time.monotonic() + 2
        while budget.snapshot()['waits'] == 0 and time.monotonic() < deadline:
            time.sleep(.001)
        assert not future.done() and budget.snapshot()['waits'] > 0
        budget.set_busy(False)  # Engine notification does not wait for producer.
        future.result(timeout=1)
    assert budget.snapshot()['uploaded_bytes'] == 40


def test_large_copy_leaves_debt_instead_of_bypassing_the_rate_budget():
    budget = TransferBudget(100, 20, clock=lambda:10.)
    budget.set_busy(True)
    budget.acquire(30)
    assert budget.credit == -10
    budget.set_busy(False)
    budget.acquire(30)
    assert budget.credit == 20
    with pytest.raises(ValueError): budget.acquire(0)


def test_public_work_updates_transfer_pressure_without_waiting_for_any_upload():
    from test_maintenance import policy
    p = policy(maintenance_transfer_bytes_per_second=100, maintenance_transfer_burst_bytes=20)
    p.pump([], lambda _:False, busy=True)
    assert p.transfer_budget.busy
    p.pump([], lambda _:False, busy=False)
    assert not p.transfer_budget.busy
    assert p.snapshot()['transfer_budget']['rate_bytes_per_second'] == 100
