import gc

import pytest

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import Phase, RequestMemory


class Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


def small(request_id="a"):
    return RequestMemory(request_id, KVMemConfig(block_tokens=16, select_budget=32,
                         gen_budget=32, repository_blocks=8, sink_blocks=1, recent_blocks=0))


def append(memory, count, accepted=None, local=0):
    spans = memory.begin_append(count, local)
    memory.attach_completion(Event(True))
    assert memory.commit_append(count if accepted is None else accepted)
    return spans


def test_dma_lease_prevents_recycle_and_old_generation_cannot_pin():
    repo = _native.Repository(1, 16)
    reservation = repo.reserve(0, 16)
    dma = reservation.lease()
    published = reservation.commit()
    old = published.handle
    assert repo.retire(old)
    published.reset()
    assert repo.reclaim(old, lambda _: True) == _native.ReclaimResult.READERS
    assert not repo.reserve(16, 16)
    dma.reset()
    assert repo.reclaim(old, lambda _: False) == _native.ReclaimResult.RELEASE
    assert not repo.reserve(16, 16)
    assert repo.reclaim(old, lambda _: True) == _native.ReclaimResult.RECLAIMED
    new = repo.reserve(16, 16).commit()
    assert new.handle.slot == old.slot
    assert new.handle.generation != old.generation
    assert not repo.pin(old)


def test_speculative_prefix_and_nonblocking_completion():
    memory = small()
    memory.begin_append(32, 0)
    with pytest.raises(RuntimeError, match="fence"):
        memory.commit_append(19)
    event = Event()
    memory.attach_completion(event)
    assert not memory.commit_append(19)
    assert memory.history_end == 0
    with pytest.raises(RuntimeError):
        memory.select({})
    event.done = True
    assert memory.commit_append(19)
    assert memory.history_end == 19
    read = memory.read(0, 19)
    assert sum(s.token_count for s in read.spans) == 19
    with pytest.raises((RuntimeError, ValueError)):
        memory.read(19, 1)


def test_selection_preserves_sink_and_rebuilds_moved_fp8_k():
    memory = small()
    spans = append(memory, 64)
    plan = memory.select({s.handle.slot: float(i) for i, s in enumerate(spans)})
    assert plan.total_window_tokens == 32
    assert [r.block_id for r in plan.remaps] == [spans[0].handle.slot, spans[-1].handle.slot]
    assert plan.remaps[-1].raw_refresh
    assert plan.remaps[-1].to_base == 16
    event = Event()
    memory.attach_completion(event)
    assert not memory.commit_selection()
    assert not memory.close()
    event.done = True
    assert memory.commit_selection()
    assert memory.phase is Phase.READY


def test_request_state_isolation_and_cancel():
    a, b = small("a"), small("b")
    append(a, 32)
    b.begin_append(16, 0)
    event = Event()
    b.attach_completion(event)
    assert not b.close()
    assert a.history_end == 32
    event.done = True
    assert b.close()
    assert b.phase is Phase.CLOSED
    assert a.read(0, 32).spans[0].history_begin == 0
    assert a.close()


def test_failed_assembly_cannot_commit_or_recycle_before_cleanup_fence():
    memory = small()
    append(memory, 64)
    memory.select({})
    event = Event()
    memory.fail_transaction(event)
    assert memory.phase is Phase.FAILED
    with pytest.raises(RuntimeError, match="ASSEMBLING"):
        memory.commit_selection()
    with pytest.raises(RuntimeError, match="READY"):
        memory.begin_append(1, 32)
    assert not memory.close()
    event.done = True
    assert memory.close()
    assert memory.phase is Phase.CLOSED


def test_binding_keeps_store_alive_until_append_destructor():
    cfg = KVMemConfig(block_tokens=16, select_budget=32, gen_budget=32, repository_blocks=4,
                     sink_blocks=1, recent_blocks=0)
    repo = _native.Repository(4, 16)
    store = _native.Store(cfg.native(), repo)
    txn = store.prepare_append(16, 0)
    del store
    gc.collect()
    txn.commit(16)
    del txn
    gc.collect()
    assert repo.stats["writing"] == 0


@pytest.mark.parametrize("kwargs", [
    {"block_tokens": 0}, {"select_budget": 129}, {"repository_blocks": 1},
    {"sink_blocks": -1}, {"recent_blocks": 99999}, {"kv_dtype": "fp4"},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        KVMemConfig(**kwargs)


def test_default_bands_use_original_qw3_auto_policy_without_silently_shrinking():
    cfg = KVMemConfig()
    # Original policy rounds 1% sink and 8% recent to whole 128-token blocks.
    assert (cfg.sink_blocks, cfg.recent_blocks) == (11, 82)
    assert (cfg.native().sink_blocks, cfg.native().recent_blocks) == (11, 82)
    explicit = KVMemConfig(sink_blocks=1, recent_blocks=0)
    assert (explicit.sink_blocks, explicit.recent_blocks) == (1, 0)
    with pytest.raises(ValueError, match="exceeds"):
        KVMemConfig(block_tokens=16, select_budget=32, gen_budget=32, repository_blocks=4)
