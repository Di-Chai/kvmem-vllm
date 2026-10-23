"""Host-only tests of the public native evidence and selection transaction APIs."""
from contextlib import contextmanager
import time

import pytest

from vllm_kvmem import _native as n


def eventually(predicate):
    end = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < end
        time.sleep(.001)


@contextmanager
def native_history(*, policy="statistical", params=None, mode=n.HistoricalGcMode.DELETE):
    arena = n.HistoricalPayloadArena(16, 16, background=False, poll_ms=1)
    config = n.HistoricalGcConfig()
    config.mode, config.background, config.poll_ms = mode, False, 1
    config.policy, config.policy_params = policy, params or {}
    config.scope_capacity = 8
    config.scan_blocks, config.delete_blocks = 8, 2
    worker = n.HistoricalGcWorker(config, arena)
    repo = n.Repository(8, 4)
    scope = worker.register_scope(repo)
    leases = []

    def append(flags=n.HISTORY_SEALED):
        index = len(leases)
        lease = repo.reserve(index * 4, 4).commit()
        payload = n.HistoricalPayloadHandle(index, index + 1)
        arena.publish(payload, bytes([index]) * 16)
        read = arena.acquire(payload)
        assert scope.publish(lease, read, flags)
        leases.append(lease)
        return lease.handle

    def advance(count=16):
        for _ in range(count):
            worker.advance()
            arena.advance()
            for event in scope.completions(8):
                assert scope.acknowledge(event)

    try:
        yield repo, scope, append, advance
    finally:
        for lease in leases:
            lease.reset()
        eventually(lambda: (worker.advance(), scope.close())[1])
        eventually(worker.close)
        eventually(arena.close)


def publish(scope, handle, revision=1, **changes):
    values = dict(eligible=True, observations=5, cold_observations=4,
                  importance=.05, model_reclaim_score=None)
    values.update(changes)
    return scope.publish_evidence(handle, revision=revision, **values)


@pytest.mark.parametrize("mode", [
    n.HistoricalGcMode.DELETE, n.HistoricalGcMode.SHADOW, n.HistoricalGcMode.OFF,
])
def test_proactive_native_policy_requires_observed_semantic_eligibility(mode):
    with native_history(mode=mode) as (repo, scope, append, advance):
        handles = [append() for _ in range(4)]
        handles.append(append(n.HISTORY_SEALED | n.HISTORY_MANDATORY))
        assert publish(scope, handles[1], eligible=False)
        assert publish(scope, handles[2], observations=1, cold_observations=1)
        assert publish(scope, handles[3])
        assert publish(scope, handles[4])
        advance(128)
        if mode == n.HistoricalGcMode.DELETE:
            assert scope.stats["retired"] == scope.stats["cpu_released"] == 1
            assert not repo.pin(handles[3])
        elif mode == n.HistoricalGcMode.SHADOW:
            assert scope.stats["shadow_decisions"] == 1
            assert scope.stats["retired"] == 0
            assert repo.pin(handles[3])
        else:
            assert scope.stats["decisions"] == scope.stats["retired"] == 0
        assert all(repo.pin(handles[i]) for i in (0, 1, 2, 4))


def test_selection_guard_is_not_the_optional_mandatory_flag():
    hold = None
    with native_history(policy="lru", params={"keep_mandatory": False}) as (repo, scope, append, advance):
        try:
            handle = append(n.HISTORY_SEALED | n.HISTORY_MANDATORY)
            scope.request_capacity(8)
            hold = scope.protect_selection([handle])
            assert hold is not None
            advance()
            assert repo.pin(handle)
            assert scope.stats["retired"] == 0
            hold = None
            advance()
            assert not repo.pin(handle)
            assert scope.protect_selection([handle]) is None
        finally:
            hold = None


def test_failed_selection_rolls_back_every_acquired_hold():
    with native_history() as (repo, scope, append, advance):
        handle = append()
        assert publish(scope, handle)
        stale = n.Handle(handle.slot, handle.generation + 1)
        assert scope.protect_selection([handle, stale]) is None
        advance()
        assert not repo.pin(handle)


def test_evidence_revisions_reject_duplicates_and_old_producer_results():
    with native_history() as (repo, scope, append, advance):
        handle = append()
        assert publish(scope, handle, revision=2, eligible=False)
        assert not publish(scope, handle, revision=1)
        assert not publish(scope, handle, revision=2)
        advance()
        assert repo.pin(handle)
        assert publish(scope, handle, revision=3)
        advance()
        assert not repo.pin(handle)
        assert not publish(scope, handle, revision=4)


def test_optional_model_threshold_does_not_interpret_unknown_as_safe():
    with native_history(params={"min_model_reclaim_score": .9}) as (repo, scope, append, advance):
        handle = append()
        assert publish(scope, handle)
        advance()
        assert repo.pin(handle)
        assert publish(scope, handle, revision=2, model_reclaim_score=.95)
        advance()
        assert not repo.pin(handle)


def test_pressure_only_mode_still_requires_semantics():
    with native_history(params={"proactive": False}) as (repo, scope, append, advance):
        eligible, protected = append(), append()
        assert publish(scope, eligible)
        assert publish(scope, protected, eligible=False)
        advance()
        assert repo.pin(eligible)
        scope.request_capacity(8)
        advance()
        assert not repo.pin(eligible)
        assert repo.pin(protected)


@pytest.mark.parametrize("changes", [
    {"revision": 0}, {"cold_observations": 6}, {"importance": float("nan")},
    {"importance": float("inf")}, {"importance": -.1}, {"importance": 1.1},
    {"model_reclaim_score": float("nan")}, {"model_reclaim_score": 1.1},
    {"model_reclaim_score": -.2},
])
def test_invalid_evidence_is_rejected_before_publication(changes):
    with native_history() as (repo, scope, append, advance):
        handle = append()
        with pytest.raises(ValueError, match="Invalid historical evidence"):
            publish(scope, handle, **changes)
        advance()
        assert repo.pin(handle)
