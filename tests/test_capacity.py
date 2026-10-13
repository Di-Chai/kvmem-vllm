from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_kvmem.capacity import ReservedKVCacheManager
from vllm_kvmem.window_manager import WorkingSetManager


def make_gate(blocks=9):
    pool = BlockPool(blocks, False, 16)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8)
    manager = WorkingSetManager(FullAttentionManager(spec, pool, False, 0, 16), 32, 32)

    class Native:
        block_pool = pool
        max_model_len = 1024
        coordinator = SimpleNamespace(single_type_managers=(manager,))

        def allocate_slots(self, req, count, reserved_blocks=0):
            needed = manager.get_num_blocks_to_allocate(req.request_id, req.history + count,
                (), req.history, req.history, req.history + count)
            if needed > pool.get_num_free_blocks() - reserved_blocks:
                return None
            return manager.allocate_new_blocks(req.request_id, req.history + count, req.history + count)

        def free(self, req):
            manager.free(req.request_id)

        def pop_blocks_for_free(self, req):
            return manager.pop_blocks_for_free(req.request_id)

    cfg = SimpleNamespace(speculative_config=None, max_concurrent_batches=1,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode="none"))
    return ReservedKVCacheManager(Native(), cfg), manager, pool


def request(name):
    return SimpleNamespace(request_id=name, num_prompt_tokens=256, max_tokens=128, history=0)


def test_step_credits_match_owned_pages_through_growth_free_and_external_trim():
    gate, manager, pool = make_gate(13)
    a, b, c = [request(name) for name in ('a','b','c')]
    gate.begin_step()
    try:
        with pytest.raises(RuntimeError, match='already open'):
            gate.begin_step()
        for req in (a,b):
            assert gate.allocate_slots(req, 16) is not None
        assert gate._step_total == sum(gate._remaining(k,q) for k,(_,q) in gate.reservations.items())
        a.history = 16
        assert gate.allocate_slots(a, 48) is not None
        gate.free(b)
        assert gate.allocate_slots(c, 16) is not None
        assert gate._step_remaining == {k:gate._remaining(k,q) for k,(_,q) in gate.reservations.items()}
    finally:
        gate.end_step()
    # Simulate an acknowledged trim between schedules. The next snapshot must
    # restore those future-growth credits, even though the pool freed a page.
    page = manager.req_to_blocks['a'].pop()
    pool.free_blocks([page])
    gate.begin_step()
    try:
        assert gate._step_remaining == {k:gate._remaining(k,q) for k,(_,q) in gate.reservations.items()}
        gate.free(a); gate.free(c)
        assert gate._step_remaining == {} and gate._step_total == 0
    finally:
        gate.end_step()
    assert pool.get_num_free_blocks() == 12


def test_credit_step_rechecks_current_request_after_failed_native_allocation():
    gate, manager, pool = make_gate(9)
    a, b = request('a'), request('b')
    assert gate.allocate_slots(a, 32) is not None
    assert gate.allocate_slots(b, 16) is not None
    gate.begin_step()
    original = gate.native.allocate_slots
    def recycled_then_refused(req, *args, **kwargs):
        page = manager.req_to_blocks[req.request_id].pop()
        pool.free_blocks([page])
        return None
    gate.native.allocate_slots = recycled_then_refused
    try:
        assert gate.allocate_slots(a, 16) is None
        assert gate._step_remaining['a'] == gate._remaining('a', gate.reservations['a'][1])
        assert gate._step_total == sum(gate._remaining(k,q) for k,(_,q) in gate.reservations.items())
    finally:
        gate.native.allocate_slots = original
        gate.end_step()
    gate.free(a); gate.free(b)
    assert pool.get_num_free_blocks() == 8


def test_future_growth_is_reserved_and_refresh_does_not_overadmit():
    gate, manager, pool = make_gate()
    a, b, c = map(request, ('a', 'b', 'c'))
    assert gate.allocate_slots(a, 16) is not None
    assert gate.allocate_slots(b, 16) is not None
    assert pool.get_num_free_blocks() == 6  # Physical room alone would admit c.
    assert gate.allocate_slots(c, 16) is None
    for req in (a, b):
        req.history = 16
        assert gate.allocate_slots(req, 48) is not None
        req.history = 64
    assert pool.get_num_free_blocks() == 0
    ticket = manager.prepare_refresh('a', 64, 32)
    assert manager.commit_refresh(ticket, SimpleNamespace(query=lambda: True))
    assert pool.get_num_free_blocks() == 2
    assert gate.allocate_slots(c, 16) is None  # a must be able to grow again.
    assert gate.allocate_slots(a, 32) is not None
    gate.free(b)
    assert gate.allocate_slots(c, 16) is not None
    gate.free(a)
    gate.free(c)
    assert not gate.reservations and pool.get_num_free_blocks() == 8


def test_cancel_and_deferred_free_do_not_release_physical_pages_early():
    gate, _, pool = make_gate(blocks=5)
    a, b = request('a'), request('b')
    assert gate.allocate_slots(a, 64) is not None
    held = gate.pop_blocks_for_free(a)
    assert not gate.reservations and pool.get_num_free_blocks() == 0
    assert gate.allocate_slots(b, 16) is None
    pool.free_blocks(reversed(held))
    assert gate.allocate_slots(b, 16) is not None
    with pytest.raises(ValueError, match='identity'):
        gate.allocate_slots(request('b'), 16)
    gate.free(b)


def test_standalone_capacity_failure_is_reported_at_initialization():
    with pytest.raises(ValueError, match='one worst-case'):
        make_gate(blocks=4)


def test_authority_admission_waits_for_retirement_even_with_unused_gpu_pages():
    gate, _, pool = make_gate(blocks=30)
    gate.max_authorities = 1
    a, b = request('a'), request('b')
    assert gate.allocate_slots(a, 16) is not None
    assert pool.get_num_free_blocks() == 28
    # The connector still owns a's lifetime reservation during CPU retirement.
    assert gate.allocate_slots(b, 16) is None
    assert gate.authority_rejections == 1
    # Existing requests may finish while new authorities wait.
    a.history = 16
    assert gate.allocate_slots(a, 16) is not None
    gate.free(a)
    assert gate.allocate_slots(b, 16) is not None
    gate.free(b)
    assert pool.get_num_free_blocks() == 29


def test_real_hybrid_kv_manager_reserves_recurrent_state_and_window_growth():
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.request import Request

    specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8),
             MambaSpec(block_size=1024, shapes=((1024,),), dtypes=(torch.float32,))]
    config = KVCacheConfig(num_blocks=11, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)])
    native = KVCacheManager(config, 1024, 1024, 16, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    managers[0] = WorkingSetManager(managers[0], 32, 32)
    native.coordinator.single_type_managers = tuple(managers)
    cfg = SimpleNamespace(speculative_config=None, max_concurrent_batches=1,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode="none"))
    gate = ReservedKVCacheManager(native, cfg)
    a, b, c = [Request(name, [1] * 256, SamplingParams(max_tokens=128), None) for name in ('a', 'b', 'c')]
    assert gate.allocate_slots(a, 16, full_sequence_must_fit=True) is not None
    assert gate.allocate_slots(b, 16, full_sequence_must_fit=True) is not None
    assert native.block_pool.get_num_free_blocks() == 6
    assert gate.allocate_slots(c, 16, full_sequence_must_fit=True) is None
    for req in (a, b):
        req.num_computed_tokens = 16
        assert gate.allocate_slots(req, 48) is not None
    assert native.block_pool.get_num_free_blocks() == 0
    assert all(len(managers[1].req_to_blocks[r.request_id]) == 1 for r in (a, b))
    gate.free(a)
    assert gate.allocate_slots(c, 16, full_sequence_must_fit=True) is not None
    gate.free(b)
    gate.free(c)
    assert native.block_pool.get_num_free_blocks() == 10


def test_native_speculative_hybrid_reservation_covers_recycled_draft_and_gdn_rows():
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
    from vllm.v1.request import Request
    from vllm_kvmem.draft_contract import DraftContract

    specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8),
             MambaSpec(block_size=1024, shapes=((1024,),), dtypes=(torch.float32,), num_speculative_blocks=7),
             SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8, sliding_window=32)]
    config = KVCacheConfig(num_blocks=37, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)])
    native = KVCacheManager(config, 1024, 1024, 16, max_in_flight_tokens=32, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    managers[0] = WorkingSetManager(managers[0], 32, 32)
    native.coordinator.single_type_managers = tuple(managers)
    cfg = SimpleNamespace(speculative_config=object(), max_concurrent_batches=1, max_in_flight_tokens=32,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
    gate = ReservedKVCacheManager(native, cfg, draft_contract=DraftContract(8, 32, 1024, 1))
    # 4 target + 8 recurrent + 6 recycling-aware draft blocks per request.
    assert gate._demand(1024) == 18
    requests = [Request(name, [1] * 256, SamplingParams(max_tokens=128), None) for name in ('a', 'b', 'c')]
    a, b, c = requests
    for req in (a, b):
        assert gate.allocate_slots(req, 16, num_lookahead_tokens=8) is not None
    assert gate.allocate_slots(c, 16, num_lookahead_tokens=8) is None
    for req in (a, b):
        for history in (16, 32, 48):
            req.num_computed_tokens = history
            # At 48, refresh before allocating beyond the dense target limit.
            if history == 48:
                ticket = managers[0].prepare_refresh(req.request_id, history, 32)
                assert managers[0].commit_refresh(ticket, SimpleNamespace(query=lambda: True))
            assert gate.allocate_slots(req, 8, num_lookahead_tokens=8) is not None
        assert len(managers[1].req_to_blocks[req.request_id]) == 8
    gate.free(a)
    assert gate.allocate_slots(c, 16, num_lookahead_tokens=8) is not None
    gate.free(b); gate.free(c)
    assert native.block_pool.get_num_free_blocks() == 36 and not gate.reservations


def test_real_native_hybrid_connector_import_bounds_target_but_preserves_gdn_draft_history():
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec
    from vllm.v1.request import Request
    from vllm_kvmem.draft_contract import DraftContract

    specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8),
             MambaSpec(block_size=1024, shapes=((1024,),), dtypes=(torch.float32,), num_speculative_blocks=7),
             SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8, sliding_window=32)]
    config = KVCacheConfig(num_blocks=37, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)])
    native = KVCacheManager(config, 1024, 1024, 16, max_in_flight_tokens=32, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    managers[0] = WorkingSetManager(managers[0], 32, 32)
    native.coordinator.single_type_managers = tuple(managers)
    cfg = SimpleNamespace(speculative_config=object(), max_concurrent_batches=1, max_in_flight_tokens=32,
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode='none'))
    gate = ReservedKVCacheManager(native, cfg, draft_contract=DraftContract(8, 32, 1024, 1))
    a, b, c = [Request(name, [1] * 600, SamplingParams(max_tokens=128), None) for name in ('a', 'b', 'c')]
    assert gate.allocate_slots(b, 16, num_lookahead_tokens=8) is not None
    neighbours = [tuple(m.req_to_blocks['b']) for m in managers]
    managers[0].prepare_import('a', 512, 48)
    assert gate.allocate_slots(a, 8, num_lookahead_tokens=8, num_external_computed_tokens=512,
                               full_sequence_must_fit=True) is not None
    assert len(managers[0].req_to_blocks['a']) == 4
    assert len(managers[1].req_to_blocks['a']) == 8
    draft = managers[2].req_to_blocks['a']
    assert len(draft) == 33
    assert all(block.is_null for block in draft[:30])
    assert all(not block.is_null for block in draft[30:])
    assert [tuple(m.req_to_blocks['b']) for m in managers] == neighbours
    assert gate.allocate_slots(c, 16, num_lookahead_tokens=8) is None
    assert managers[0].commit_import('a', 512, SimpleNamespace(query=lambda: True))
    a.num_computed_tokens = 520
    # Resume/reselect is still governed by the dense K/G limit.
    ticket = managers[0].prepare_refresh('a', 520, 32)
    assert managers[0].commit_refresh(ticket, SimpleNamespace(query=lambda: True))
    assert gate.allocate_slots(a, 8, num_lookahead_tokens=8) is not None
    gate.free(a); gate.free(b)
    assert native.block_pool.get_num_free_blocks() == 36 and not gate.reservations
