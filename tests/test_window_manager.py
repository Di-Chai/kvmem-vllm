import pytest

pytest.importorskip("vllm")
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_kvmem.window_manager import WorkingSetManager, RefreshPending, RefreshRequired


class Event:
    done = False

    def query(self):
        return self.done


def make_manager():
    pool = BlockPool(17, False, 16)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8)
    native = FullAttentionManager(spec, pool, False, 0, 16)
    return WorkingSetManager(native, 32, 32), pool


def test_native_allocations_track_window_not_cumulative_history():
    manager, pool = make_manager()
    manager.allocate_new_blocks("a", 64, 64)
    other = manager.allocate_new_blocks("b", 32, 32)
    assert pool.get_num_free_blocks() == 10
    ticket = manager.prepare_refresh("a", 64, 32)
    event = Event()
    assert not manager.commit_refresh(ticket, event)
    assert pool.get_num_free_blocks() == 10
    with pytest.raises(RefreshPending):
        manager.allocate_new_blocks("a", 80, 80)
    with pytest.raises(RefreshPending):
        manager.free("a")
    event.done = True
    assert manager.commit_refresh(ticket, event)
    assert pool.get_num_free_blocks() == 12
    assert manager.offsets["a"] == 32
    assert manager.req_to_blocks["b"] == other
    assert manager.get_num_blocks_to_allocate("a", 80, (), 64, 64, 80) == 1
    manager.allocate_new_blocks("a", 96, 96)
    with pytest.raises(RefreshRequired):
        manager.allocate_new_blocks("a", 97, 97)
    next_ticket = manager.prepare_refresh("a", 96, 32)
    assert manager.commit_refresh(next_ticket, event)
    assert manager.offsets["a"] == 64
    manager.allocate_new_blocks("a", 128, 128)
    assert len(manager.req_to_blocks["a"]) == 4
    with pytest.raises(ValueError, match="Stale"):
        manager.commit_refresh(ticket, event)
    manager.free("a")
    manager.free("b")
    assert pool.get_num_free_blocks() == 16


def test_admission_reserves_bounded_capacity_for_long_logical_prompt():
    manager, _ = make_manager()
    assert manager.get_num_blocks_to_allocate("a", 1_000_000, (), 0, 0, 1_000_000,
                                               apply_admission_cap=True) == 4


def test_refresh_ack_preserves_pages_written_by_query_replay():
    manager, pool = make_manager()
    manager.allocate_new_blocks("a", 64, 64)
    ticket = manager.prepare_refresh("a", 48, 32, replay_tokens=16)
    assert len(ticket.retained_block_ids) == 3
    event = Event()
    assert not manager.commit_refresh(ticket, event)
    assert len(manager.req_to_blocks["a"]) == 4
    event.done = True
    assert manager.commit_refresh(ticket, event)
    assert manager.offsets["a"] == 16
    assert len(manager.req_to_blocks["a"]) == 3
    # The completed query is already resident; admission must not replace it.
    assert manager.get_num_blocks_to_allocate("a", 64, (), 64, 64, 64) == 0
    manager.free("a")
    assert pool.get_num_free_blocks() == 16


def test_refresh_query_requires_original_allocation_and_capacity():
    manager, _ = make_manager()
    manager.allocate_new_blocks("a", 48, 48)
    with pytest.raises(ValueError, match="unallocated"):
        manager.prepare_refresh("a", 48, 32, replay_tokens=16)
    with pytest.raises(ValueError, match="admitted"):
        manager.prepare_refresh("a", 48, 32, replay_tokens=17)
    assert not manager.pending
