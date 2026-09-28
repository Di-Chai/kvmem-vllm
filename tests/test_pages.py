import pytest

from vllm_kvmem.pages import EnginePageLeases


def test_actual_vllm_pool_waits_for_correct_worker_ack():
    pytest.importorskip("vllm")
    from vllm.v1.core.block_pool import BlockPool
    pool = BlockPool(num_gpu_blocks=5, enable_caching=False, hash_block_size=16)
    manager = EnginePageLeases(pool, engine_block_tokens=1568, kernel_page_tokens=16)
    a = manager.reserve("a", 1568 * 2)
    b = manager.reserve("b", 1568 * 2)
    assert a is not None and b is not None
    assert manager.reserve("c", 1) is None
    assert len(a.kernel_pages) == 196
    assert not set(a.kernel_pages) & set(b.kernel_pages)
    with pytest.raises(ValueError, match="active"):
        manager.acknowledge_release("a", a.lease_id)
    manager.retire(a)
    assert pool.get_num_free_blocks() == 0
    with pytest.raises(ValueError, match="Cross-request"):
        manager.acknowledge_release("b", a.lease_id)
    assert manager.acknowledge_release("a", a.lease_id)
    c = manager.reserve("c", 1568 * 2)
    assert c is not None
    assert not manager.acknowledge_release("a", a.lease_id)
    assert manager.live_blocks == 4
    manager.retire(b)
    manager.retire(c)
    assert manager.acknowledge_release("b", b.lease_id)
    assert manager.acknowledge_release("c", c.lease_id)
    assert pool.get_num_free_blocks() == 4
