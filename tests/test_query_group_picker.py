"""Native logical-group budget and host assembly transaction qualification."""
import math

import pytest

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import Phase, RequestMemory
from test_core import Event, append


def populated(*, count=256, budget=128, gen=256, repository=16, sink=1, recent=0, owner='groups'):
    memory = RequestMemory(owner, KVMemConfig(block_tokens=32, select_budget=budget,
        gen_budget=gen, repository_blocks=repository, sink_blocks=sink, recent_blocks=recent))
    spans = append(memory, count)
    handles = [_native.Handle(s.handle.slot, s.handle.generation) for s in spans]
    return memory, handles


def finish(memory):
    memory.attach_completion(Event(True))
    assert memory.commit_selection()


def test_overlapping_groups_charge_each_block_once_and_preserve_fixed_sink():
    memory, handles = populated(budget=160)
    try:
        selected = memory.preview_group_selection([(32, 128), (96, 160)], [10., 9.])
        assert selected == tuple(h.slot for h in handles[:5])
        assert memory.phase is Phase.READY and memory._plan is None
        plan = memory.select_groups([(32, 128), (96, 160)], [10., 9.],
            expected_slots=reversed(selected))
        assert plan.total_window_tokens == 160
        assert [r.block_id for r in plan.remaps] == list(selected)
        finish(memory)
    finally:
        assert memory.close()


def test_oversized_group_is_skipped_whole_and_does_not_fill_remaining_budget_with_fragments():
    memory, handles = populated()
    try:
        selected = memory.preview_group_selection([(32, 192), (192, 256)], [100., 1.])
        assert selected == (handles[0].slot, handles[6].slot, handles[7].slot)
        assert len(selected) == 3  # One unused slot is preferable to a partial group.
    finally:
        assert memory.close()


def test_fixed_sink_recent_and_explicit_mandatory_stay_authoritative():
    memory, handles = populated(recent=1)
    try:
        selected = memory.preview_group_selection([(64, 160), (96, 128)], [100., 1.],
            (handles[2].slot,))
        assert selected == (handles[0].slot, handles[2].slot, handles[3].slot, handles[7].slot)
    finally:
        assert memory.close()


def test_partial_tail_and_unaligned_group_spans_still_charge_physical_32_token_blocks():
    memory, handles = populated(count=233)
    try:
        groups = [(200, 233), (35, 90), (100, 120)]
        scores = [10., 9., 8.]
        selected = memory.preview_group_selection(groups, scores)
        assert selected == (handles[0].slot, handles[3].slot, handles[6].slot, handles[7].slot)
        plan = memory.select_groups(groups, scores)
        assert len(plan.remaps) == 4 and plan.total_window_tokens == 105
        finish(memory)
    finally:
        assert memory.close()


def test_runtime_budget_is_scoped_and_does_not_change_the_block_picker_baseline():
    memory, handles = populated(budget=160)
    try:
        scores = {h.slot: float(i) for i, h in enumerate(handles)}
        baseline = memory.preview_selection(scores, action=False)
        memory._store.set_runtime_select_budget(128)
        selected = memory.preview_group_selection([(32, 96), (160, 224)], [10., 9.], max_tokens=96)
        assert selected == tuple(h.slot for h in handles[:3])
        assert memory._store.runtime_select_budget == 128
        memory._store.set_runtime_select_budget(0)
        assert memory.preview_selection(scores, action=False) == baseline
    finally:
        assert memory.close()


def test_mandatory_budget_failure_restores_previous_runtime_budget_and_leaves_ready():
    memory, handles = populated()
    try:
        memory._store.set_runtime_select_budget(96)
        before = memory._store.working_records()
        with pytest.raises(RuntimeError, match='mandatory.*exceeds budget'):
            memory.select_groups([(96, 128)], [1.], (handles[1].slot, handles[2].slot), max_tokens=64)
        assert memory._store.runtime_select_budget == 96
        assert memory._store.working_records() == before
        assert memory.phase is Phase.READY and memory._plan is None
    finally:
        assert memory.close()


def test_budget_smaller_than_fixed_bands_is_rejected_without_changing_existing_budget():
    memory, _ = populated(recent=1)
    try:
        memory._store.set_runtime_select_budget(96)
        with pytest.raises(ValueError):
            memory.preview_group_selection([(32, 64)], [1.], max_tokens=32)
        assert memory._store.runtime_select_budget == 96 and memory.phase is Phase.READY
    finally:
        assert memory.close()


@pytest.mark.parametrize('groups,scores', [
    ([(32, 64)], []), ([(64, 64)], [1.]), ([(-1, 32)], [1.]),
    ([(0, 257)], [1.]), ([(0., 32)], [1.]), ([(False, 32)], [1.]),
    ([(0, 32, 64)], [1.]), ([(0, 32)], [math.nan]),
    ([(0, 32)], [math.inf]), ([(0, 32)], [-math.inf]),
    ([(0, 32)], [True]), ([(0, 32)], ['1']),
])
def test_invalid_group_inputs_fail_before_any_transaction(groups, scores):
    memory, _ = populated()
    try:
        before = memory._store.working_records()
        with pytest.raises(ValueError):
            memory.select_groups(groups, scores, max_tokens=96)
        assert memory.phase is Phase.READY and memory._plan is None
        assert memory._store.runtime_select_budget == 0
        assert memory._store.working_records() == before
    finally:
        assert memory.close()


@pytest.mark.parametrize('groups,scores', [
    ([(0, 32)], []), ([(32, 32)], [1.]), ([(0, 65)], [1.]),
    ([(0, 32)], [math.nan]), ([(0, 32)], [math.inf]),
])
def test_native_binding_validates_even_when_all_history_fits(groups, scores):
    memory, _ = populated(count=64)
    try:
        with pytest.raises(ValueError):
            memory._store.pick_groups(groups, scores)
    finally:
        assert memory.close()


def test_all_fit_keeps_original_native_behavior_and_negative_finite_scores_are_valid():
    memory, handles = populated(count=64)
    try:
        assert memory.preview_group_selection([(32, 64)], [-1.]) == tuple(h.slot for h in handles)
        assert memory.preview_group_selection([], []) == tuple(h.slot for h in handles)
    finally:
        assert memory.close()


def test_original_logical_hole_invalidates_whole_group_without_splicing_live_neighbors():
    memory, handles = populated()
    try:
        memory.repository.retire_permanently(handles[2])
        assert not memory.repository.pin(handles[2])
        selected = memory.preview_group_selection([(32, 128), (160, 224)], [100., 1.])
        assert selected == (handles[0].slot, handles[5].slot, handles[6].slot)
        assert handles[1].slot not in selected and handles[3].slot not in selected
    finally:
        assert memory.close()


def test_score_and_transport_masks_do_not_act_as_loading_permissions():
    memory, handles = populated()
    frame = None
    try:
        frame = memory._store.query_score_frame(handles, [32]*8, 256, [handles[2].slot], 0)
        assert not frame.score_mask[0] and not frame.transport_mask[0]
        assert frame.score_mask[2] and not frame.transport_mask[2]
        selected = memory.preview_group_selection([(0, 96), (192, 224)], [10., 9.],
            (handles[2].slot,))
        assert selected == (handles[0].slot, handles[1].slot, handles[2].slot, handles[6].slot)
    finally:
        if frame is not None:
            frame.close()
        assert memory.close()


def test_recycled_slot_cannot_substitute_for_missing_logical_group_or_stale_generation():
    memory, handles = populated(gen=128, repository=8)
    try:
        plan = memory.select_groups([(192, 224)], [1.])
        finish(memory)
        del plan
        old = handles[2]
        memory.repository.retire_permanently(old)
        assert memory.repository.reclaim(old, lambda _: True) == _native.ReclaimResult.RECLAIMED
        new = append(memory, 32, local=64)[0].handle
        assert new.slot == old.slot and new.generation != old.generation
        assert not memory.repository.pin(old)
        before = memory._store.working_records()
        with pytest.raises(RuntimeError, match='stale generation'):
            memory._store.select([old])
        assert memory._store.working_records() == before and memory.phase is Phase.READY
        selected = memory.preview_group_selection([(64, 96), (256, 288), (192, 224)], [100., 10., 9.])
        assert selected == (handles[0].slot, handles[6].slot, new.slot)
    finally:
        assert memory.close()


class StoreProxy:
    def __init__(self, original, select):
        self.original, self.select = original, select

    def __getattr__(self, name):
        return getattr(self.original, name)


def test_admission_retirement_retries_whole_groups_without_partial_assembly(monkeypatch):
    memory, handles = populated()
    original = memory._store
    calls = []
    before = original.working_records()
    def select(selected):
        calls.append(tuple(h.slot for h in selected))
        if len(calls) == 1:
            memory.repository.retire_permanently(handles[2])
            assert memory.phase is Phase.READY and memory._plan is None
            raise _native.HistoryAdmissionRetry('retired between preview and admission')
        assert original.working_records() == before
        return original.select(selected)
    monkeypatch.setattr(memory, '_store', StoreProxy(original, select))
    try:
        groups = ((begin, end) for begin, end in [(32, 128), (160, 224)])
        plan = memory.select_groups(groups, iter([100., 1.]), max_tokens=128)
        assert len(calls) == 2 and calls[0] == tuple(h.slot for h in handles[:4])
        assert calls[1] == (handles[0].slot, handles[5].slot, handles[6].slot)
        assert memory.phase is Phase.ASSEMBLING and plan.total_window_tokens == 96
        assert original.runtime_select_budget == 0
        finish(memory)
    finally:
        assert memory.close()


def test_other_admission_failure_is_not_retried_and_leaves_no_half_transaction(monkeypatch):
    memory, _ = populated()
    original, calls = memory._store, []
    before = original.working_records()
    def select(selected):
        calls.append(selected)
        raise RuntimeError('injected admission failure')
    monkeypatch.setattr(memory, '_store', StoreProxy(original, select))
    try:
        with pytest.raises(RuntimeError, match='injected admission'):
            memory.select_groups([(32, 96)], [1.], max_tokens=96)
        assert len(calls) == 1 and original.working_records() == before
        assert memory.phase is Phase.READY and memory._plan is None
        assert original.runtime_select_budget == 0
    finally:
        assert memory.close()


@pytest.mark.parametrize('bind_expected', [False, True])
def test_real_historical_execution_admission_retries_after_permanent_retirement(tmp_path, monkeypatch,
        bind_expected):
    from test_historical_authority import setup, append as append_authority, close
    authority, archive, _ = setup(tmp_path, mode=_native.HistoricalGcMode.OFF)
    plan = None
    try:
        _, handles = append_authority(authority)
        authority.poll()
        memory = authority.memory
        preview = memory.preview_group_selection
        calls = []
        before = memory._store.working_records()
        def retire_after_first_preview(*args, **kwargs):
            selected = preview(*args, **kwargs)
            calls.append(selected)
            if len(calls) == 1:
                memory.repository.retire_permanently(handles[2])
            else:
                assert memory._store.working_records() == before
            return selected
        monkeypatch.setattr(memory, 'preview_group_selection', retire_after_first_preview)
        # The first native Store.select itself raises HistoryAdmissionRetry;
        # no mocked exception or scoring/transport permission is involved.
        expected = tuple(handles[i].slot for i in (0, 1, 2, 7))
        if bind_expected:
            with pytest.raises(RuntimeError, match='changed after preview'):
                memory.select_groups([(4, 12), (16, 24)], [100., 1.],
                    max_tokens=16, expected_slots=expected)
            assert memory.phase is Phase.READY and memory._plan is None
            assert memory._store.working_records() == before
            assert memory._store.runtime_select_budget == 0
        else:
            plan = memory.select_groups([(4, 12), (16, 24)], [100., 1.])
        assert len(calls) == 2
        assert calls[0] == tuple(handles[i].slot for i in (0, 1, 2, 7))
        assert calls[1] == tuple(handles[i].slot for i in (0, 4, 5, 7))
        assert not memory.repository.pin(handles[2])
        if not bind_expected:
            assert plan.total_window_tokens == 16
            finish(memory)
            del plan
            plan = None
    finally:
        plan = None
        close(authority, archive)


def test_nonconverging_admission_is_bounded_and_leaves_ready(monkeypatch):
    memory, _ = populated()
    original, calls = memory._store, []
    def select(selected):
        calls.append(selected)
        raise _native.HistoryAdmissionRetry('injected concurrent retirement')
    monkeypatch.setattr(memory, '_store', StoreProxy(original, select))
    try:
        with pytest.raises(RuntimeError, match='did not converge'):
            memory.select_groups([(32, 96)], [1.])
        assert len(calls) == memory.config.repository_blocks+1
        assert memory.phase is Phase.READY and memory._plan is None
    finally:
        assert memory.close()


def test_group_assembly_uses_existing_completion_fence_and_failure_cleanup():
    memory, _ = populated()
    memory.select_groups([(32, 96)], [1.])
    with pytest.raises(RuntimeError, match='READY'):
        memory.preview_group_selection([(32, 64)], [1.])
    with pytest.raises(RuntimeError, match='fence'):
        memory.commit_selection()
    event = Event()
    memory.fail_transaction(event)
    assert memory.phase is Phase.FAILED and not memory.close()
    event.done = True
    assert memory.close() and memory.phase is Phase.CLOSED


def test_group_preview_does_not_mutate_other_owner_or_per_block_scores():
    first, handles = populated(owner='first')
    second, second_handles = populated(owner='second')
    try:
        scores = {h.slot: float(i) for i, h in enumerate(handles)}
        baseline = first.preview_selection(scores, action=False)
        second.repository.retire_permanently(second_handles[2])
        assert first.preview_group_selection([(32, 128)], [100.]) == tuple(h.slot for h in handles[:4])
        assert second.preview_group_selection([(32, 128)], [100.]) == (second_handles[0].slot,)
        assert first.preview_selection(scores, action=False) == baseline
    finally:
        assert first.close() and second.close()
