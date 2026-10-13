import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1",
                                reason="Explicit GPU qualification only")

from vllm_kvmem.window_batch import WindowBatchAdapter


def make_batch():
    return SimpleNamespace(
        req_ids=["b", "a"], num_reqs=2, num_reqs_after_padding=4,
        num_tokens=5, num_tokens_after_padding=8,
        num_computed_tokens_np=np.array([70, 100]),
        query_start_loc=torch.tensor([0, 3, 5, 5, 5], dtype=torch.int32, device="cuda"),
        positions=torch.tensor([70, 71, 72, 100, 101, 777, 777, 777], device="cuda"),
        seq_lens=torch.tensor([73, 102, 0, 0], dtype=torch.int32, device="cuda"),
        seq_lens_cpu_upper_bound=torch.tensor([73, 102, 0, 0], dtype=torch.int32))


def test_native_paged_slots_use_dense_window_but_logical_batch_is_unchanged():
    from vllm.v1.worker.gpu.block_table import BlockTables
    tables = BlockTables([16, 16], 4, 8, [4, 1], torch.device("cuda"),
                         [4, 16], slot_mapping_enabled=[True, False])
    tables.append_block_ids(2, ([3, 5], [9]), overwrite=True)
    tables.append_block_ids(1, ([7, 8], [10]), overwrite=True)
    tables.apply_staged_writes()
    batch = make_batch()
    adapter = WindowBatchAdapter((0,), 16)
    adapter.commit("a", 100, 4)
    adapter.commit("b", 70, 6)
    window = adapter.prepare(batch)
    assert window.positions.tolist() == [6, 7, 8, 4, 5, 0, 0, 0]
    assert window.groups[0].seq_lens.tolist() == [9, 6, 0, 0]
    assert window.groups[0].cpu_upper_bound.tolist() == [9, 6, 0, 0]
    indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    slots = tables.compute_slot_mappings(indices, batch.query_start_loc[:3],
                                         window.positions, 8)
    assert slots[0].tolist() == [118, 119, 120, 52, 53, -1, -1, -1]
    assert slots[1].tolist() == [-1] * 8
    assert batch.positions.tolist() == [70, 71, 72, 100, 101, 777, 777, 777]
    assert batch.seq_lens.tolist() == [73, 102, 0, 0]
    assert batch.num_computed_tokens_np.tolist() == [70, 100]


def test_growth_reselection_can_reintroduce_more_historical_tokens():
    adapter = WindowBatchAdapter((0,), 128)
    adapter.commit("a", 100, 4)
    adapter.commit("a", 110, 40)  # offset decreases legitimately when window grows
    assert adapter.offsets["a"] == 70
    with pytest.raises(ValueError, match="history"):
        adapter.commit("a", 99, 10)
    adapter.free("a")
    adapter.commit("a", 5, 5)
    assert adapter.offsets["a"] == 0
    assert adapter.prepare(make_batch()) is None


def test_window_capacity_and_stale_logical_counters_fail_before_cache_access():
    batch = make_batch()
    adapter = WindowBatchAdapter((0,), 16)
    adapter.commit("a", 100, 4)
    with pytest.raises(ValueError, match="extent"):
        adapter.prepare(batch)  # b was never admitted into a bounded window
    adapter.commit("b", 70, 6)
    batch.num_computed_tokens_np[0] = 60
    with pytest.raises(ValueError, match="precedes"):
        adapter.prepare(batch)


def test_graph_positions_and_attention_lengths_share_stable_storage_across_refresh():
    adapter = WindowBatchAdapter((0,), 128)
    adapter.reserve_graph_buffers(8, 4, 'cuda')
    batch = make_batch()
    first = adapter.capture_window(batch)
    assert first.positions.tolist() == batch.positions.tolist()
    address = first.positions.data_ptr()
    lens_address = first.groups[0].seq_lens.data_ptr()
    adapter.commit('a', 100, 4); adapter.commit('b', 70, 6)
    second = adapter.prepare(batch)
    assert second.positions.data_ptr() == address
    assert second.groups[0].seq_lens.data_ptr() == lens_address
    assert first.positions.tolist() == [6, 7, 8, 4, 5, 0, 0, 0]
    assert first.groups[0].seq_lens.tolist() == [9, 6, 0, 0]
    assert batch.seq_lens.tolist() == [73, 102, 0, 0]
    ready = adapter.offsets_ready
    adapter.prepare(batch)
    assert adapter.offsets_ready is ready  # unchanged decode mapping needs no DMA
    adapter.commit('b', 70, 10)
    changed = adapter.prepare(batch)
    assert adapter.offsets_ready is not ready
    assert changed.positions.data_ptr() == address
    assert changed.positions.tolist() == [10, 11, 12, 4, 5, 0, 0, 0]
    assert changed.groups[0].seq_lens.tolist() == [13, 6, 0, 0]
