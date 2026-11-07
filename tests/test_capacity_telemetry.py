"""Read-only explanation of lifetime credits versus actual native pool pages."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')

from test_capacity import make_gate, request


def test_snapshot_explains_rejection_with_six_physical_pages_still_free():
    gate, manager, pool = make_gate()
    a, b, c = [request(name) for name in ('a', 'b', 'c')]
    assert gate.allocate_slots(a, 16) is not None
    assert gate.allocate_slots(b, 16) is not None
    assert gate.allocate_slots(c, 16) is None
    snapshot = gate.snapshot()
    assert snapshot['physical_total_pages'] == 8
    assert snapshot['physical_free_pages'] == 6
    assert snapshot['physical_allocated_pages'] == 2
    assert snapshot['request_owned_pages'] == 2
    assert snapshot['future_reserved_pages'] == 6
    assert snapshot['free_after_growth_reserve_pages'] == 0
    assert snapshot['requests']['a'] == dict(owned_pages=1, lifetime_quota_pages=4,
                                           future_reserved_pages=3)
    assert snapshot['groups'][0]['block_tokens'] == 16
    assert snapshot['rejections']['lifetime_budget'] == 1
    assert snapshot['deferred_or_other_pages'] == 0
    # Repeated observation must not change the allocation or credit ledger.
    assert gate.snapshot() == snapshot
    assert pool.get_num_free_blocks() == 6
    gate.free(a)
    gate.free(b)


def test_refresh_returns_pages_but_preserves_growth_credit_in_snapshot():
    gate, manager, pool = make_gate()
    a = request('a')
    gate.allocate_slots(a, 64)
    before = gate.snapshot()
    assert before['future_reserved_pages'] == 0
    ticket = manager.prepare_refresh('a', 64, 32)
    manager.commit_refresh(ticket, SimpleNamespace(query=lambda: True))
    after = gate.snapshot()
    assert after['physical_free_pages'] == before['physical_free_pages'] + 2
    assert after['future_reserved_pages'] == before['future_reserved_pages'] + 2
    assert after['free_after_growth_reserve_pages'] == before['free_after_growth_reserve_pages']
    gate.free(a)


def test_detached_free_remains_allocated_and_cannot_be_reported_as_credit():
    gate, _, pool = make_gate(blocks=5)
    a = request('a')
    gate.allocate_slots(a, 64)
    blocks = gate.pop_blocks_for_free(a)
    waiting = gate.snapshot()
    assert waiting['requests'] == {}
    assert waiting['future_reserved_pages'] == 0
    assert waiting['physical_free_pages'] == 0
    assert waiting['deferred_or_other_pages'] == 4
    pool.free_blocks(reversed(blocks))
    finished = gate.snapshot()
    assert finished['physical_free_pages'] == 4
    assert finished['deferred_or_other_pages'] == 0


def test_idle_lease_overlap_during_retirement_does_not_double_count_pages():
    gate, manager, pool = make_gate()
    a = request('a')
    gate.allocate_slots(a, 32)
    blocks = tuple(manager.req_to_blocks['a'])
    pool.touch(blocks)
    gate.active_residency = SimpleNamespace(entries={'session': SimpleNamespace(blocks=blocks)})
    during = gate.snapshot()
    assert during['physical_allocated_pages'] == 2
    assert during['request_owned_pages'] == 2
    assert during['idle_resident_pages'] == 0
    assert during['idle_request_overlap_pages'] == 2
    assert during['deferred_or_other_pages'] == 0
    gate.free(a)
    idle = gate.snapshot()
    assert idle['request_owned_pages'] == 0
    assert idle['idle_resident_pages'] == 2
    assert idle['idle_request_overlap_pages'] == 0
    assert idle['deferred_or_other_pages'] == 0
    # A claimed resident import is no longer in the idle registry, yet still
    # owns its physical pages before native request allocation adopts them.
    gate.active_residency.entries.clear()
    manager.resident_imports['new'] = blocks
    staged = gate.snapshot()
    assert staged['idle_resident_pages'] == 0
    assert staged['staged_import_pages'] == 2
    assert staged['deferred_or_other_pages'] == 0
    manager.resident_imports.clear()
    pool.free_blocks(blocks)


def test_authority_and_native_rejections_are_distinct_and_read_only():
    gate, _, _ = make_gate(blocks=30)
    gate.max_authorities = 1
    a, b = request('a'), request('b')
    gate.allocate_slots(a, 16)
    assert gate.allocate_slots(b, 16) is None
    assert gate.snapshot()['rejections'] == dict(lifetime_budget=0,
        retirement_or_active_limit=1, native_allocation=0)
    original = gate.native.allocate_slots
    gate.native.allocate_slots = lambda *args, **kwargs: None
    assert gate.allocate_slots(a, 16) is None
    assert gate.snapshot()['rejections'] == dict(lifetime_budget=0,
        retirement_or_active_limit=1, native_allocation=1)
    gate.native.allocate_slots = original
    gate.free(a)


def test_snapshot_counts_shared_backing_once_without_inventing_bytes():
    gate, manager, pool = make_gate()
    a = request('a')
    gate.allocate_slots(a, 16)
    blocks = tuple(manager.req_to_blocks['a'])
    pool.touch(blocks)
    manager.req_to_blocks['fork-view'] = list(blocks)
    snapshot = gate.snapshot()
    assert snapshot['request_owned_pages'] == 1
    assert snapshot['reserved_request_owned_pages'] == 1
    assert snapshot['physical_allocated_pages'] == 1
    assert snapshot['deferred_or_other_pages'] == 0
    assert not any('bytes' in name for name in snapshot)
    manager.free('fork-view')
    gate.free(a)
