"""Operation admission uses native page geometry without changing K/G policy."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec, MambaSpec
from vllm.v1.request import Request

from vllm_kvmem.serving.capacity import ReservedKVCacheManager
from vllm_kvmem.adapters.draft_contract import DraftContract
from vllm_kvmem.serving.incremental_capacity import IncrementalKVCacheManager
from vllm_kvmem.serving.request_swap import FencedPageSwap
from vllm_kvmem.serving.window_manager import WorkingSetManager, RefreshRequired


READY = SimpleNamespace(query=lambda: True)


class PauseController:
    def __init__(self):
        self.paused = set()
        self.pressure = []

    def is_suspended(self, request_id):
        return request_id in self.paused

    def on_pressure(self, request, required_pages, available_pages):
        self.pressure.append((request.request_id, required_pages, available_pages))


def make_gate(blocks=9, *, state=False, mtp=0, dynamic=True, controller=None):
    specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8)]
    if state or mtp:
        specs.append(MambaSpec(block_size=1024, shapes=((1024,),), dtypes=(torch.float32,),
                               num_speculative_blocks=mtp))
    config = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(index)], spec) for index, spec in enumerate(specs)])
    native = KVCacheManager(config, 1024, 1024, 16, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    managers[0] = WorkingSetManager(managers[0], 32, 32)
    native.coordinator.single_type_managers = tuple(managers)
    cfg = SimpleNamespace(speculative_config=object() if mtp else None, max_concurrent_batches=1,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
    draft = DraftContract(mtp+1, 0, 1024, 1, 'mtp') if mtp else None
    controller = controller or PauseController()
    gate = (IncrementalKVCacheManager(native, cfg, pause_controller=controller, draft_contract=draft)
            if dynamic else ReservedKVCacheManager(native, cfg, draft_contract=draft))
    return gate, managers, native.block_pool, controller


def request(name):
    return Request(name, [1] * 256, SamplingParams(max_tokens=128), None)


def test_more_requests_use_unspent_growth_without_weakening_native_page_limit():
    gate, managers, pool, controller = make_gate()
    requests = [request(str(i)) for i in range(4)]
    for req in requests:
        assert gate.can_schedule(req, 16)
        assert gate.allocate_slots(req, 16, full_sequence_must_fit=True) is not None
        req.num_computed_tokens = 16
    assert pool.get_num_free_blocks() == 4
    assert gate.snapshot()['future_reserved_pages'] == 0
    assert gate.allocate_slots(requests[0], 48) is not None
    assert pool.get_num_free_blocks() == 1
    before = tuple(managers[0].req_to_blocks[requests[1].request_id])
    assert not gate.can_schedule(requests[1], 48)
    assert gate.allocate_slots(requests[1], 48) is None
    assert tuple(managers[0].req_to_blocks[requests[1].request_id]) == before
    assert pool.get_num_free_blocks() == 1
    assert controller.pressure == [(requests[1].request_id, 3, 1)]
    for req in requests:
        gate.free(req)
    assert pool.get_num_free_blocks() == 8


def test_old_static_gate_still_reserves_two_complete_windows():
    gate, _, pool, _ = make_gate(dynamic=False)
    a, b, c = map(request, ('a', 'b', 'c'))
    assert gate.allocate_slots(a, 16, full_sequence_must_fit=True) is not None
    assert gate.allocate_slots(b, 16, full_sequence_must_fit=True) is not None
    assert gate.allocate_slots(c, 16, full_sequence_must_fit=True) is None
    assert pool.get_num_free_blocks() == 6
    assert gate.snapshot()['future_reserved_pages'] == 6
    gate.free(a)
    gate.free(b)


def test_recurrent_and_native_mtp_rows_are_part_of_each_atomic_operation():
    gate, managers, pool, _ = make_gate(blocks=13, mtp=4)
    a, b = request('a'), request('b')
    for req in (a, b):
        assert gate.allocate_slots(req, 8, num_lookahead_tokens=4) is not None
        req.num_computed_tokens = 8
        assert len(managers[0].req_to_blocks[req.request_id]) == 1
        assert len(managers[1].req_to_blocks[req.request_id]) == 5
    assert pool.get_num_free_blocks() == 0
    assert not gate.can_schedule(a, 8, num_lookahead_tokens=4)
    assert gate.allocate_slots(a, 8, num_lookahead_tokens=4) is None
    # No speculative/state row can be borrowed from b to make a appear to fit.
    assert len(managers[1].req_to_blocks['b']) == 5
    gate.free(b)
    assert gate.can_schedule(a, 8, num_lookahead_tokens=4)
    assert gate.allocate_slots(a, 8, num_lookahead_tokens=4) is not None
    assert len(managers[0].req_to_blocks['a']) == 2
    assert len(managers[1].req_to_blocks['a']) == 5
    gate.free(a)


def test_g_boundary_never_triggers_implicit_refresh_or_silent_query_clipping():
    gate, managers, _, _ = make_gate()
    req = request('a')
    assert gate.allocate_slots(req, 60) is not None
    req.num_computed_tokens = 60
    assert gate.can_schedule(req, 4)
    assert gate.allocate_slots(req, 4) is not None
    req.num_computed_tokens = 64
    with pytest.raises(RefreshRequired):
        gate.can_schedule(req, 1)
    assert managers[0].offsets == {}
    assert managers[0].capacity == 64 and managers[0].select_budget == 32
    # Only the original externally issued refresh changes the dense mapping.
    ticket = managers[0].prepare_refresh('a', 64, 32)
    assert managers[0].commit_refresh(ticket, READY)
    assert gate.allocate_slots(req, 16) is not None
    assert managers[0].offsets['a'] == 32
    gate.free(req)


def test_private_pressure_control_preserves_original_no_lookahead_frame():
    gate, managers, _, _ = make_gate(blocks=12, mtp=4)
    req = request('a')
    assert gate.allocate_slots(req, 48, num_lookahead_tokens=4) is not None
    req.num_computed_tokens = 48
    gate.prefill_refresh_control = lambda candidate, count: candidate is req and count == 16
    assert gate.can_schedule(req, 16, num_lookahead_tokens=4)
    assert gate.allocate_slots(req, 16, num_lookahead_tokens=4) is not None
    assert gate.refresh_control_allocations['a'] == (req, 48, 16)
    assert len(managers[0].req_to_blocks['a']) == 4
    # It remains a private control transaction: existing validator must reject
    # accidentally placing this frame into a public model forward.
    with pytest.raises(RuntimeError, match='escaped'):
        gate.validate_refresh_controls({'a': 16}, {})
    gate.refresh_control_allocations.clear()
    gate.free(req)


def test_guided_probe_reserve_is_not_lost_in_incremental_preview(monkeypatch):
    gate, _, pool, controller = make_gate(blocks=5)
    peer = request('peer')
    gate.allocate_slots(peer, 32)
    gate.guided_reserve = 32
    gate.guided_block_tokens = 16
    monkeypatch.setattr('vllm_kvmem.serving.request_plan.request_plan',
        lambda *args: SimpleNamespace(query=(0, 8), replay_begin=0, pressure_probe_tokens=0))
    req = request('query')
    assert not gate.can_schedule(req, 8)
    assert gate.allocate_slots(req, 8) is None
    assert controller.pressure == [('query', 3, 2)]
    assert pool.get_num_free_blocks() == 2
    gate.free(peer)


def test_explicit_native_reserved_pages_are_still_honored():
    gate, _, pool, _ = make_gate(blocks=5)
    req = request('a')
    assert gate.allocate_slots(req, 16, reserved_blocks=4) is None
    assert pool.get_num_free_blocks() == 4
    assert gate.allocate_slots(req, 16, reserved_blocks=3) is not None
    assert pool.get_num_free_blocks() == 3
    gate.free(req)


def test_restore_page_protection_composes_once_with_native_connector_credit():
    gate, _, pool, controller = make_gate(blocks=5)
    controller.reserved_pages = lambda request_id: 2
    req = request('a')
    # One physical page + two restore pages + one connector page fits exactly.
    assert gate.can_schedule(req, 8, reserved_blocks=1)
    assert gate.allocate_slots(req, 8, reserved_blocks=1) is not None
    req.num_computed_tokens = 8
    assert pool.get_num_free_blocks() == 3
    controller.reserved_pages = lambda request_id: pool.get_num_free_blocks()
    # A forward within already owned pages still progresses while all free
    # pages are earmarked for a pending restore. Growth/newcomers must wait.
    assert gate.can_schedule(req, 1)
    assert gate.allocate_slots(req, 1) is not None
    assert not gate.can_schedule(req, 9)
    assert gate.allocate_slots(req, 9) is None
    assert not gate.can_schedule(request('newcomer'), 1)
    assert pool.get_num_free_blocks() == 3
    gate.free(req)


def test_cold_operation_revokes_only_idle_pages_using_its_actual_extent():
    from vllm_kvmem.serving.active_residency import ActiveResidency, ActiveView
    gate, managers, pool, _ = make_gate(blocks=5)
    idle, peer, incoming = map(request, ('idle', 'peer', 'incoming'))
    gate.allocate_slots(idle, 32)
    blocks = tuple(managers[0].req_to_blocks['idle'])
    pool.touch(blocks)
    gate.free(idle)
    gate.allocate_slots(peer, 16)
    residency = ActiveResidency({0: managers[0]}, pool, 1000, 1.)
    residency.entries[('tenant', 'idle')] = ActiveView(
        ('tenant', 'idle'), 1, 32, 32, ((0, blocks),), float('inf'))
    gate.active_residency = residency
    before = tuple(managers[0].req_to_blocks['peer'])
    assert pool.get_num_free_blocks() == 1
    assert gate.can_schedule(incoming, 32)
    assert pool.get_num_free_blocks() == 1  # Preview never revokes a lease.
    assert gate.allocate_slots(incoming, 32) is not None
    assert residency.pressure_evictions == 1
    assert tuple(managers[0].req_to_blocks['peer']) == before
    assert pool.get_num_free_blocks() == 1
    gate.free(incoming)
    gate.free(peer)


def test_paused_request_keeps_identity_but_holds_no_future_growth_credit():
    gate, managers, pool, controller = make_gate(blocks=10, state=True)
    a, b = request('a'), request('b')
    gate.allocate_slots(a, 32)
    gate.allocate_slots(b, 16)
    a.num_computed_tokens = 32
    controller.paused.add('a')
    swaps = FencedPageSwap(gate)
    ticket = swaps.prepare(a)
    assert not swaps.checkpoint_complete(ticket, SimpleNamespace(query=lambda: False))
    assert gate.snapshot()['request_owned_pages'] == 5
    assert swaps.checkpoint_complete(ticket, READY)
    assert gate.snapshot()['request_owned_pages'] == 2
    gate.begin_step()
    try:
        assert gate._step_remaining == {'a': 0, 'b': 0}
        assert not gate.can_schedule(a, 16)
        assert gate.allocate_slots(a, 16) is None
        c = request('c')
        assert gate.allocate_slots(c, 32) is not None
        assert swaps.prepare_restore(ticket) is not None
        swaps.restore_dispatched(ticket)
        assert swaps.restore_complete(ticket, READY)
        controller.paused.remove('a')
        assert a.num_computed_tokens == 32
        assert managers[0].offsets == {}
        gate.free(c)
        assert gate.allocate_slots(a, 16) is not None
    finally:
        gate.end_step()
    gate.free(a)
    gate.free(b)
    assert pool.get_num_free_blocks() == 9


def test_connector_import_uses_exact_window_and_preserves_restore_permit():
    gate, managers, _, _ = make_gate(blocks=10, state=True)
    req = request('import')
    managers[0].prepare_import(req.request_id, 128, 32)
    events = []
    gate.restore_admission = SimpleNamespace(
        permits={}, before_allocate=lambda r, n: events.append(('before', n)) or object(),
        allocated=lambda r: events.append(('allocated', r.request_id)),
        rollback=lambda r: events.append(('rollback', r.request_id)))
    assert gate.can_schedule(req, 8, num_external_computed_tokens=128)
    assert gate.allocate_slots(req, 8, num_external_computed_tokens=128,
                               full_sequence_must_fit=True) is not None
    assert events == [('before', 128), ('allocated', 'import')]
    assert len(managers[0].req_to_blocks['import']) == 3
    assert len(managers[1].req_to_blocks['import']) == 1
    assert managers[0].commit_import('import', 128, READY)
    gate.free(req)


def test_new_mode_requires_pause_protocol_and_rejects_positional_options():
    old, _, _, _ = make_gate(dynamic=False)
    cfg = SimpleNamespace(speculative_config=None, max_concurrent_batches=1,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
    with pytest.raises(ValueError, match='pause controller'):
        IncrementalKVCacheManager(old.native, cfg, pause_controller=None)
    with pytest.raises(ValueError, match='native MTP'):
        IncrementalKVCacheManager(old.native, cfg, pause_controller=PauseController(),
                                  draft_contract=DraftContract(2, 32, 1024, 1, 'dflash'))
    cfg.scheduler_config = SimpleNamespace(async_scheduling=True)
    with pytest.raises(ValueError, match='synchronous'):
        IncrementalKVCacheManager(old.native, cfg, pause_controller=PauseController())
    gate, _, _, _ = make_gate()
    with pytest.raises(ValueError, match='named'):
        gate.allocate_slots(request('a'), 16, 0)
