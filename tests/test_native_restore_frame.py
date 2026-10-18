"""CPU differential qualification of the opt-in native restore adapter."""
from dataclasses import replace

import pytest

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import Phase, RequestMemory


class Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done


def append(memory, count, local=0):
    assert memory.begin_append(count, local) is not None
    memory.attach_completion(Event())
    assert memory.commit_append(count)


def install(memory, plan):
    # Test-only equivalent of RequestMemory's transaction bookkeeping. No
    # serving entry point changes until this adapter is independently accepted.
    memory._plan, memory.phase, memory._submitted = plan, Phase.ASSEMBLING, False
    return plan


def commit(memory):
    memory.attach_completion(Event())
    assert memory.commit_selection()


def select(memory, indices):
    handles = [memory._store.handle(index) for index in indices]
    plan = install(memory, memory._store.select(handles))
    commit(memory)
    return plan


def fixture(kind='sparse', current='different'):
    memory = RequestMemory('request', KVMemConfig(block_tokens=4, select_budget=12,
        gen_budget=12, repository_blocks=16, sink_blocks=0, recent_blocks=0))
    append(memory, 22 if kind == 'partial' else 24)
    chosen = [0, 3, 5] if kind != 'dense' else list(range(6))
    select(memory, chosen)
    checkpoint = memory.checkpoint_window(allow_partial=kind == 'partial')
    if current == 'different':
        select(memory, [0, 1, 2])
    elif current == 'cold':
        for slot, generation, *_ in memory._store.working_records():
            memory._store.working_resident(_native.Handle(slot, generation), False)
    return memory, checkpoint


def plan_value(plan):
    attrs = ('block_id', 'n_tokens', 'from_base', 'to_base', 'skip', 'raw_refresh', 'working_k_resident')
    return dict(stage_in=list(plan.stage_in), stage_out=list(plan.stage_out),
        remaps=[tuple(getattr(row, key) for key in attrs) for row in plan.remaps],
        **{key: getattr(plan, key) for key in ('total_window_tokens', 'selection_overlap_blocks',
            'gpu_reused_blocks', 'retained_position_stable', 'retained_position_moved')})


@pytest.mark.parametrize('kind', ['dense', 'sparse', 'partial'])
@pytest.mark.parametrize('current', ['same', 'different', 'cold'])
def test_native_restore_matches_original_frame_and_full_reload_plan(kind, current):
    old, old_checkpoint = fixture(kind, current)
    native, checkpoint = fixture(kind, current)
    expected = old.restore_window(old_checkpoint)
    actual = install(native, native._store.restore_frame(checkpoint.prefix, checkpoint.records))
    assert plan_value(actual) == plan_value(expected)
    assert all(not row.skip and not row.working_k_resident for row in actual.remaps)
    assert all(row.raw_refresh for row in actual.remaps)
    commit(old)
    commit(native)
    assert tuple(native._store.working_records()) == tuple(old._store.working_records()) == checkpoint.records
    assert native.history_end == old.history_end == checkpoint.position
    assert old.close() and native.close()


@pytest.mark.parametrize('damage', ['empty', 'duplicate', 'dense_gap', 'reverse_history',
    'count_zero', 'count_large', 'partial_middle', 'past_prefix', 'stale_generation',
    'wrong_handle', 'overflow', 'negative_position'])
def test_invalid_frame_fails_before_any_residency_change(damage):
    memory, checkpoint = fixture(current='same')
    control, control_checkpoint = fixture(current='same')
    records = [list(row) for row in checkpoint.records]
    if damage == 'empty':
        records = []
    elif damage == 'duplicate':
        records[1] = [*records[0][:4], 4]
    elif damage == 'dense_gap':
        records[-1][4] += 1
    elif damage == 'reverse_history':
        records[-1][2] = 0
    elif damage == 'count_zero':
        records[-1][3] = 0
    elif damage == 'count_large':
        records[-1][3] = 5
    elif damage == 'partial_middle':
        records[0][3] = 3
    elif damage == 'past_prefix':
        records[-1][2] = checkpoint.position
    elif damage == 'stale_generation':
        records[-1][1] += 1
    elif damage == 'wrong_handle':
        records[-1][0] = 1  # Live generation, but belongs to different history.
    elif damage == 'overflow':
        records[-1][2] = 2**64 - 1
    else:
        records[-1][4] = -1
    records = tuple(tuple(row) for row in records)
    before = tuple(memory._store.working_records()), memory.repository.stats['leases']
    with pytest.raises((ValueError, RuntimeError)):
        memory._store.restore_frame(checkpoint.prefix, records)
    assert (tuple(memory._store.working_records()), memory.repository.stats['leases']) == before
    # A normal, non-forced selection exposes accidental cold-tier mutations;
    # comparing only working_records would miss changed resident flags.
    actual = memory._store.select([memory._store.handle(row[0]) for row in checkpoint.records])
    expected = control._store.select([control._store.handle(row[0]) for row in control_checkpoint.records])
    assert plan_value(actual) == plan_value(expected)
    assert memory.close() and control.close()


def test_foreign_prefix_and_unrewound_history_are_rejected_without_change():
    memory, checkpoint = fixture(current='same')
    foreign, other = fixture(current='same')
    before = tuple(memory._store.working_records())
    with pytest.raises(ValueError, match='foreign'):
        memory._store.restore_frame(other.prefix, checkpoint.records)
    assert tuple(memory._store.working_records()) == before
    # Add a suffix while keeping the historical checkpoint pinned.
    append(memory, 4, local=12)
    grown = tuple(memory._store.working_records())
    with pytest.raises(ValueError, match='Rewind'):
        memory._store.restore_frame(checkpoint.prefix, checkpoint.records)
    assert tuple(memory._store.working_records()) == grown
    memory.begin_rewind(checkpoint.prefix)
    memory.attach_completion(Event())
    assert memory.commit_rewind()
    plan = install(memory, memory._store.restore_frame(checkpoint.prefix, checkpoint.records))
    assert plan.total_window_tokens == checkpoint.window_tokens
    commit(memory)
    assert tuple(memory._store.working_records()) == checkpoint.records
    assert memory.close() and foreign.close()


@pytest.mark.parametrize('transaction', ['append', 'rewind'])
def test_pending_native_transaction_is_rejected_before_tier_reset(transaction):
    memory, checkpoint = fixture(current='same')
    before = tuple(memory._store.working_records())
    pending = (memory._store.prepare_append(4, 12) if transaction == 'append' else
               memory._store.prepare_rewind(memory.history_end))
    with pytest.raises(RuntimeError, match='pending repository transaction'):
        memory._store.restore_frame(checkpoint.prefix, checkpoint.records)
    assert tuple(memory._store.working_records()) == before
    if transaction == 'append':
        pending.commit(0)
    else:
        pending.commit()
    plan = memory._store.select([memory._store.handle(row[0]) for row in checkpoint.records])
    assert all(row.working_k_resident and row.skip for row in plan.remaps)
    assert memory.close()


def test_restore_adapter_does_not_bypass_completion_or_cancel_leases():
    memory, checkpoint = fixture(current='different')
    plan = install(memory, memory._store.restore_frame(checkpoint.prefix, checkpoint.records))
    old = memory._store.handle(1)  # Prior window, not retained in the saved frame.
    assert memory.repository.retire(old)
    assert memory.repository.reclaim(old, lambda _: True) == _native.ReclaimResult.READERS
    fence = Event(False)
    memory.attach_completion(fence)
    assert not memory.commit_selection()
    assert not memory.close()
    assert plan.total_window_tokens == checkpoint.window_tokens
    fence.done = True
    # Cancellation may release memory's plan only after the fence. The caller's
    # local Plan is another intentional lease owner until it too is discarded.
    assert memory.close()
    assert memory.repository.reclaim(old, lambda _: True) == _native.ReclaimResult.READERS
    del plan
    assert memory.repository.reclaim(old, lambda _: True) == _native.ReclaimResult.RECLAIMED


def test_retired_checkpoint_page_is_rejected_not_rescued():
    memory, checkpoint = fixture(current='same')
    handle = _native.Handle(*checkpoint.records[-1][:2])
    assert memory.repository.retire(handle)
    with pytest.raises(ValueError, match='Stale'):
        memory._store.restore_frame(checkpoint.prefix, checkpoint.records)
    assert not memory.repository.pin(handle)
    assert memory.close()


def test_complete_frame_cannot_exceed_store_active_plus_generation_budget():
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=4,
                        repository_blocks=8, sink_blocks=0, recent_blocks=0)
    repository = _native.Repository(config.repository_blocks, config.block_tokens)
    store = _native.Store(config.native(), repository)
    # Native history may be larger than the active window. Direct preparation
    # here deliberately bypasses RequestMemory's separate append admission.
    pending = store.prepare_append(16, 0)
    pending.commit(16)
    prefix, records = store.checkpoint(16), tuple(store.working_records())
    with pytest.raises(ValueError, match='bounded extent'):
        store.restore_frame(prefix, records)
    assert tuple(store.working_records()) == records
    plan = store.select([store.handle(row[0]) for row in records])
    assert all(row.working_k_resident and row.skip for row in plan.remaps)


def test_live_prefix_alone_does_not_bypass_missing_historical_page_validation():
    memory, checkpoint = fixture(current='different')
    control, _ = fixture(current='different')
    unpinned_prefix = memory._store.checkpoint(memory.history_end, False)
    missing = _native.Handle(*checkpoint.records[-1][:2])
    assert memory.repository.retire(missing)
    assert memory._store.checkpoint_valid(unpinned_prefix)
    before = tuple(memory._store.working_records())
    with pytest.raises(RuntimeError, match='required-history-deleted'):
        memory._store.restore_frame(unpinned_prefix, checkpoint.records)
    assert tuple(memory._store.working_records()) == before
    actual = memory._store.select([memory._store.handle(i) for i in (0, 1, 2)])
    expected = control._store.select([control._store.handle(i) for i in (0, 1, 2)])
    assert plan_value(actual) == plan_value(expected)
    assert memory.close() and control.close()


@pytest.mark.parametrize('arm', ['old_python', 'native_adapter'])
def test_handmade_partial_tail_cannot_start_inside_immutable_record(arm):
    memory, checkpoint = fixture('partial', 'same')
    records = list(checkpoint.records)
    slot, generation, begin, count, dense = records[-1]
    assert begin == 20 and count == 2
    # This fabricated slice still ends at the checkpoint and reads the same
    # handle, but the selector restores the whole logical block. Reject it.
    records[-1] = (slot, generation, begin + 1, count - 1, dense)
    malformed = replace(checkpoint, records=tuple(records))
    before = tuple(memory._store.working_records()), memory.repository.stats['leases']
    with pytest.raises(ValueError, match='complete ordered dense frame'):
        if arm == 'old_python':
            memory.restore_window(malformed)
        else:
            memory._store.restore_frame(malformed.prefix, malformed.records)
    assert (tuple(memory._store.working_records()), memory.repository.stats['leases']) == before
    plan = memory._store.select([memory._store.handle(row[0]) for row in checkpoint.records])
    assert all(row.working_k_resident and row.skip for row in plan.remaps)
    assert memory.close()


def test_extended_partial_tail_keeps_earlier_aligned_checkpoint_valid():
    memory, checkpoint = fixture('partial', 'same')
    assert checkpoint.position == 22 and checkpoint.records[-1][2:4] == (20, 2)
    append(memory, 2, local=checkpoint.window_tokens)
    assert memory.history_end == 24
    assert memory._store.read(20, 4).spans[0].token_count == 4
    # A valid checkpoint may end before the current committed tail. Validate
    # its exact prefix slice; do not require count == current block.n_tokens.
    memory._validate_window_checkpoint(checkpoint)
    before = tuple(memory._store.working_records())
    for restore in (lambda: memory.restore_window(checkpoint),
                    lambda: memory._store.restore_frame(checkpoint.prefix, checkpoint.records)):
        with pytest.raises(ValueError, match='Rewind authority'):
            restore()
    assert tuple(memory._store.working_records()) == before
    assert memory.close()
