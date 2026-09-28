"""Host branch identity and fence ownership; not persistent GPU sessions."""
import gc

import pytest

from vllm_kvmem import _native
from vllm_kvmem.memory import Phase
from test_core import Event, append, small


def test_rewind_waits_for_suffix_fence_and_invalidates_later_checkpoint_permanently():
    memory = small()
    append(memory, 16)
    prefix = memory.checkpoint()
    suffix = append(memory, 32, local=16)
    later = memory.checkpoint()
    spans = memory.begin_rewind(prefix)
    assert sum(span[2] for span in spans) == 32
    with pytest.raises(RuntimeError, match='fence'):
        memory.commit_rewind()
    fence = Event()
    memory.attach_completion(fence)
    assert not memory.commit_rewind() and memory.history_end == 48
    assert memory.checkpoint_valid(later)
    assert not memory.close()
    with pytest.raises(RuntimeError):
        memory.begin_append(1, 16)
    fence.done = True
    assert memory.commit_rewind() and memory.phase is Phase.READY
    assert memory.history_end == 16 and memory.checkpoint_valid(prefix)
    assert not memory.checkpoint_valid(later) and not later.valid
    # Returned descriptions contain value handles, not hidden rewind owners.
    for _, _, _, handle in spans:
        assert memory.repository.reclaim(handle, lambda _: True) == _native.ReclaimResult.RECLAIMED
    append(memory, 32, local=16)
    assert memory.history_end == 48 and not memory.checkpoint_valid(later)
    assert memory.close() and not memory.checkpoint_valid(prefix)


def test_checkpoint_pins_prefix_but_foreign_or_partial_boundary_cannot_rewind():
    a, b = small('a'), small('b')
    append(a, 16); append(b, 16)
    own, foreign = a.checkpoint(), b.checkpoint()
    with pytest.raises(ValueError, match='foreign'):
        a.begin_rewind(foreign)
    append(a, 1, local=16)
    partial = a.checkpoint()
    with pytest.raises(ValueError, match='sealed-record'):
        a.begin_rewind(partial)
    assert a.phase is Phase.READY and a.history_end == 17
    handle = a._store.handle(0)
    assert a.close()
    assert a.repository.reclaim(handle, lambda _: True) == _native.ReclaimResult.READERS
    del own, partial
    gc.collect()
    assert a.repository.reclaim(handle, lambda _: True) == _native.ReclaimResult.RECLAIMED
    assert b.close()


def test_failed_rewind_holds_suffix_until_cleanup_and_native_transaction_owns_store():
    memory = small()
    append(memory, 16)
    prefix = memory.checkpoint()
    append(memory, 16, local=16)
    memory.begin_rewind(prefix)
    fence = Event()
    memory.fail_transaction(fence)
    assert not memory.close() and memory.history_end == 32
    fence.done = True
    assert memory.close()
    assert not memory.checkpoint_valid(prefix)

    # A bound native transaction outlives its Python Store safely (keep_alive).
    other = small('other')
    append(other, 32)
    store, repo = other._store, other.repository
    txn = store.prepare_rewind(16)
    other._store = None
    del store, other
    gc.collect()
    assert txn.position == 16
    txn.commit()
    del txn
    gc.collect()
    assert repo.stats['capacity'] == 8
