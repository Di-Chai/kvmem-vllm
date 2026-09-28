from dataclasses import replace

import pytest

from vllm_kvmem.memory import Phase
from test_core import Event, append, small


def select(memory, scores):
    plan = memory.select(scores)
    memory.attach_completion(Event(True))
    assert memory.commit_selection()
    return plan


def test_restore_preserves_original_noncontiguous_frame_and_reloads_both_kv_on_new_pages():
    memory = small()
    spans = append(memory, 64)
    handles = [s.handle for s in spans]
    select(memory, {h.slot: float(i) for i, h in enumerate(handles)})
    append(memory, 16, local=32)
    checkpoint = memory.checkpoint_window()
    assert checkpoint.position == 80 and checkpoint.window_tokens == 48
    assert [(r[2], r[4]) for r in checkpoint.records] == [(0, 0), (48, 16), (64, 32)]
    # The live frame changes and a later branch is generated.
    select(memory, {h.slot: float(i == 1) for i, h in enumerate(handles)})
    append(memory, 16, local=32)
    later = memory.checkpoint_window()
    memory.begin_rewind(checkpoint.prefix)
    memory.attach_completion(Event(True))
    assert memory.commit_rewind()
    with pytest.raises(ValueError, match='Stale'):
        memory.restore_window(later)
    plan = memory.restore_window(checkpoint)
    assert plan.total_window_tokens == 48
    assert [(r.block_id, r.to_base) for r in plan.remaps] == [(r[0], r[4]) for r in checkpoint.records]
    assert all(not r.skip and not r.working_k_resident for r in plan.remaps)
    event = Event()
    memory.attach_completion(event)
    assert not memory.commit_selection()
    event.done = True
    assert memory.commit_selection()
    restored = memory.checkpoint_window()
    assert restored.records == checkpoint.records
    assert memory.close()


def test_invalid_window_cannot_mutate_live_residency_or_cross_namespaces():
    a, b = small('a'), small('b')
    append(a, 32); append(b, 32)
    checkpoint = a.checkpoint_window()
    before = a._store.working_records()
    with pytest.raises(ValueError, match='foreign'):
        b.restore_window(checkpoint)
    bad = list(checkpoint.records)
    bad[0] = (*bad[0][:-1], 1)
    with pytest.raises(ValueError, match='dense frame'):
        a.restore_window(replace(checkpoint, records=tuple(bad)))
    assert a._store.working_records() == before and a.phase is Phase.READY
    append(a, 16, local=32)
    with pytest.raises(ValueError, match='Rewind authority'):
        a.restore_window(checkpoint)
    assert a.close() and b.close()


def test_window_checkpoint_refuses_empty_or_partial_sealed_prefix():
    memory = small()
    with pytest.raises(ValueError, match='sealed prefix'):
        memory.checkpoint_window()
    append(memory, 15)
    with pytest.raises(ValueError, match='sealed prefix'):
        memory.checkpoint_window()
    assert memory.close()
