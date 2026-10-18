"""Independent native score-frame ownership, chronology and budget boundaries."""
import gc

import pytest

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory


class Ready:
    def query(self):
        return True


def history(*, sink=1, recent=1):
    memory = RequestMemory('native-score-review', KVMemConfig(block_tokens=4,
        select_budget=16, gen_budget=16, repository_blocks=16,
        sink_blocks=sink, recent_blocks=recent))
    spans = memory.begin_append(24, 0)
    # Span.handle is a pybind reference into Append, whose own keep-alive
    # retains Store. Use independent value handles, as authority records do.
    handles = [_native.Handle(span.handle.slot, span.handle.generation) for span in spans]
    memory.attach_completion(Ready())
    assert memory.commit_append(24)
    return memory, handles


def test_native_frame_closes_leases_even_if_closed_python_object_remains():
    memory, handles = history()
    repository = memory.repository
    initial = repository.stats['leases']
    frame = memory._store.query_score_frame(handles, [4]*6, 24, [], 12)
    assert repository.stats['leases'] == initial+6
    frame.close()
    frame.close()  # Explicit release is idempotent.
    assert repository.stats['leases'] == initial
    assert memory.close()
    gc.collect()
    # A closed QueryScoreFrame must not keep Store's resident leases alive.
    assert repository.stats['leases'] == 0
    assert frame.score_mask == [False, True, True, True, True, False]


def test_unclosed_frame_owns_repository_leases_without_owning_store():
    memory, handles = history()
    repository = memory.repository
    frame = memory._store.query_score_frame(handles, [4]*6, 24, [], 12)
    assert memory.close()
    gc.collect()
    assert repository.stats['leases'] == 6
    assert repository.reclaim(handles[2], lambda _: True) == _native.ReclaimResult.READERS
    frame.close()
    assert repository.stats['leases'] == 0
    assert repository.reclaim(handles[2], lambda _: True) == _native.ReclaimResult.RECLAIMED


def test_native_frame_budget_restored_when_fixed_keep_validation_fails():
    memory, handles = history()
    try:
        memory._store.set_runtime_select_budget(16)
        before = memory.repository.stats['leases']
        with pytest.raises(RuntimeError, match='mandatory'):
            memory._store.query_score_frame(handles, [4]*6, 24,
                [handles[1].slot, handles[2].slot, handles[3].slot], 8)
        assert memory._store.runtime_select_budget == 16
        assert memory.repository.stats['leases'] == before
        frame = memory._store.query_score_frame(handles, [4]*6, 24,
            [handles[2].slot, handles[3].slot], 12)
        assert frame.score_mask == [False, True, True, True, True, False]
        # Sink + mandatory consume the budget: the recent node can compete.
        assert frame.transport_mask == [False, True, False, False, True, True]
        frame.close()
        assert memory._store.runtime_select_budget == 16
    finally:
        assert memory.close()


def test_pressure_preview_budget_does_not_apply_query_recent_constraint():
    memory, handles = history(sink=1, recent=2)
    try:
        memory._store.set_runtime_select_budget(12)
        candidates = [(handle.slot, 0.) for handle in handles]
        assert memory.preview_selection({}, action=False, pressure_scores=candidates,
                                        max_tokens=4) == (handles[0].slot,)
        assert memory._store.runtime_select_budget == 12
        with pytest.raises(ValueError, match=r'sink \+ recent'):
            memory.preview_selection({}, action=False, max_tokens=4)
        assert memory._store.runtime_select_budget == 12
    finally:
        assert memory.close()


def test_native_score_mask_uses_history_order_after_physical_slot_reuse():
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=24,
        repository_blocks=8, sink_blocks=1, recent_blocks=1)
    repository = _native.Repository(8, 4)
    # Reserve the two lowest physical slots before Store's first history page.
    placeholders = [repository.reserve(1000+4*i, 4).commit() for i in range(2)]
    store = _native.Store(config.native(), repository)
    append = store.prepare_append(24, 0)
    handles = [_native.Handle(span.handle.slot, span.handle.generation) for span in append.spans]
    append.commit(24)
    del append
    assert [handle.slot for handle in handles] == [2, 3, 4, 5, 6, 7]
    for lease in placeholders:
        handle = lease.handle
        assert repository.retire(handle)
        lease.reset()
        assert repository.reclaim(handle, lambda _: True) == _native.ReclaimResult.RECLAIMED
    append = store.prepare_append(8, 24)
    handles.extend(_native.Handle(span.handle.slot, span.handle.generation) for span in append.spans)
    append.commit(8)
    del append
    assert [handle.slot for handle in handles][-2:] == [0, 1]
    assert handles[-1].generation > handles[0].generation
    frame = store.query_score_frame(handles, [4]*8, 32, [], 8)
    assert frame.score_mask == [False, True, True, True, True, True, True, False]
    assert frame.transport_mask == frame.score_mask
    # The native selector fixes precisely historical first/last, not slot 0/7.
    store.scores(handles, [float(i) for i in range(8)])
    assert store.pick([]) == [handles[0].slot, handles[-1].slot]
    frame.close()
    del store
    gc.collect()
    assert repository.stats['leases'] == 0
