"""Irreversible retirement is independent of policy and active lease count."""
from vllm_kvmem import _native as n


def test_permanent_retirement_keeps_existing_operations_but_blocks_rescue():
    repo = n.Repository(1, 32)
    gpu = repo.reserve(0, 32).commit()
    transfer = gpu.clone()
    old = gpu.handle
    assert repo.retire_permanently(old) == n.RetireResult.COMMITTED
    assert repo.retire_permanently(old) == n.RetireResult.ALREADY_RETIRED
    assert not repo.pin(old)
    assert gpu.record.token_begin == transfer.record.token_begin == 0
    releases = []
    assert repo.reclaim(old, lambda r: releases.append(r.handle) or True) == n.ReclaimResult.READERS
    transfer.reset()
    gpu.reset()
    assert not releases
    assert repo.reclaim(old, lambda r: releases.append(r.handle) or True) == n.ReclaimResult.RECLAIMED
    assert len(releases) == 1
    fresh = repo.reserve(32, 32).commit()
    assert repo.retire_permanently(old) == n.RetireResult.STALE
    assert repo.pin(fresh.handle)


def test_partial_release_of_permanent_retirement_is_retried_without_reopening():
    repo = n.Repository(1, 32)
    active = repo.reserve(0, 32).commit()
    old = active.handle
    assert repo.retire_permanently(old) == n.RetireResult.COMMITTED
    active.reset()
    assert repo.reclaim(old, lambda _: False) == n.ReclaimResult.RELEASE
    assert not repo.pin(old) and not repo.reserve(32, 32)
    assert repo.reclaim(old, lambda _: True) == n.ReclaimResult.RECLAIMED
    assert repo.ready_slots == 1
