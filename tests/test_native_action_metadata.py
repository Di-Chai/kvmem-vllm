"""CPU differential checks for the opt-in Action window metadata adapter."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime, window_segments
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.prefill_attention import PrefillHistory
from test_native_restore_frame import fixture
from test_options import config


def reference(memory, history, local, count):
    return window_segments(memory._store.working_records(), history, local, count, 4)


def native(memory, history, local, count):
    return memory._store.action_window_segments(history, local, count, 4)


def assert_equal(actual, expected):
    starts, counts, history = actual
    assert starts.dtype == counts.dtype == np.int32
    assert starts.ndim == counts.ndim == 1
    assert starts.flags.c_contiguous and counts.flags.c_contiguous
    assert starts.flags.owndata and counts.flags.owndata
    assert starts.tolist() == expected[0] and counts.tolist() == expected[1]
    assert type(history) is list and all(type(key) is int for key in history)
    assert history == expected[2]
    assert torch.equal(torch.tensor(starts, dtype=torch.int32), torch.tensor(expected[0], dtype=torch.int32))
    assert torch.equal(torch.tensor(counts, dtype=torch.int32), torch.tensor(expected[1], dtype=torch.int32))
    assert json.dumps(history) == json.dumps(expected[2])


@pytest.mark.parametrize('kind,history,local,count', [
    ('dense', 0, 0, 1), ('dense', 1, 1, 5), ('dense', 4, 4, 0),
    ('dense', 7, 7, 5), ('dense', 24, 24, 5), ('dense', 26, 26, 2),
    ('sparse', 24, 12, 5), ('sparse', 20, 8, 4), ('sparse', 21, 9, 7),
    ('partial', 22, 10, 3), ('partial', 21, 9, 5), ('partial', 22, 10, 0),
])
def test_native_matches_dense_sparse_partial_and_local_suffix_boundaries(kind, history, local, count):
    memory, checkpoint = fixture(kind, current='same')
    before = tuple(memory._store.working_records())
    try:
        assert_equal(native(memory, history, local, count), reference(memory, history, local, count))
        assert tuple(memory._store.working_records()) == before
        assert memory.checkpoint_valid(checkpoint.prefix)
    finally:
        assert memory.close()


def test_empty_store_and_owned_results_outlive_source():
    memory = RequestMemory('empty', KVMemConfig(block_tokens=4, select_budget=12,
        gen_budget=12, repository_blocks=16, sink_blocks=0, recent_blocks=0))
    assert_equal(native(memory, 0, 0, 0), ([], [], []))
    result = native(memory, 1, 0, 8)
    expected = reference(memory, 1, 0, 8)
    assert memory.close()
    assert_equal(result, expected)
    result[0][0] = 123  # Arrays are owned writable values, not Store views.


@pytest.mark.parametrize('history,count', [(2**63-1, 1), (2**63-4, 4), (2**63-5, 5)])
def test_final_segment_does_not_require_unused_int64_exclusive_end(history, count):
    memory = RequestMemory('limit', KVMemConfig(block_tokens=4, select_budget=12,
        gen_budget=12, repository_blocks=16, sink_blocks=0, recent_blocks=0))
    try:
        assert_equal(native(memory, history, 0, count), reference(memory, history, 0, count))
    finally:
        assert memory.close()


@pytest.mark.parametrize('history,count', [(2**63-1, 2), (2**63-4, 5)])
def test_actual_following_segment_outside_native_int64_coordinates_is_rejected(history, count):
    memory = RequestMemory('overflow', KVMemConfig(block_tokens=4, select_budget=12,
        gen_budget=12, repository_blocks=16, sink_blocks=0, recent_blocks=0))
    try:
        # Python integers can describe this non-serving domain; the native
        # interface must fail explicitly instead of wrapping a history key.
        assert reference(memory, history, 0, count)[2][-1] == 2**63
        with pytest.raises(ValueError, match='history exceeds int64'):
            native(memory, history, 0, count)
        assert memory.history_end == 0 and memory._store.working_records() == []
    finally:
        assert memory.close()


@pytest.mark.parametrize('position', [-1, 1, 2, 5])
def test_dense_gaps_and_overlaps_reject_without_mutation(position):
    memory, _ = fixture('dense', current='same')
    record = memory._store.working_records()[1]
    memory._store.set_baked(record[0], position, record[1])
    before = tuple(memory._store.working_records())
    try:
        with pytest.raises(ValueError, match='gap'):
            reference(memory, 24, 24, 1)
        with pytest.raises(ValueError, match='gap'):
            native(memory, 24, 24, 1)
        assert tuple(memory._store.working_records()) == before
    finally:
        assert memory.close()


def test_int32_window_overflow_rejects_before_large_allocation():
    memory, _ = fixture('dense', current='same')
    before = tuple(memory._store.working_records())
    try:
        with pytest.raises(ValueError, match='int32 kernel coordinates'):
            native(memory, 24, 2**31-1, 1)
        assert tuple(memory._store.working_records()) == before
    finally:
        assert memory.close()


def test_int64_min_baked_position_rejects_without_subtraction_overflow():
    memory, _ = fixture('dense', current='same')
    record = memory._store.working_records()[0]
    memory._store.set_baked(record[0], -(2**63), record[1])
    before = tuple(memory._store.working_records())
    try:
        for function in (native, reference):
            with pytest.raises(ValueError, match='gap'):
                function(memory, 24, 24, 1)
        assert tuple(memory._store.working_records()) == before
    finally:
        assert memory.close()


def test_missing_prefix_retains_original_suffix_fill_and_invalid_origin_check():
    memory, _ = fixture('sparse', current='same')
    try:
        # Preserve the reference's suffix fill when local begins beyond the
        # selected frame; do not add a new policy or silently truncate it.
        assert_equal(native(memory, 26, 14, 5), reference(memory, 26, 14, 5))
        for function in (native, reference):
            with pytest.raises(ValueError, match='suffix origin'):
                function(memory, 0, 14, 1)
    finally:
        assert memory.close()


def test_directory_filter_keeps_gpu_resident_pages_outside_selected_set():
    memory, _ = fixture('sparse', current='same')
    handle = memory._store.handle(1)  # History 4, outside selected slots 0/3/5.
    try:
        before = tuple(memory._store.working_records())
        assert len(before) == 3
        memory._store.set_baked(handle.slot, 12, handle.generation)
        memory._store.working_resident(handle, True)
        assert len(memory._store.working_records()) == 4
        assert_equal(native(memory, 24, 16, 1), reference(memory, 24, 16, 1))
        memory._store.working_resident(handle, False)
        assert tuple(memory._store.working_records()) == before
        assert_equal(native(memory, 24, 12, 1), reference(memory, 24, 12, 1))
    finally:
        assert memory.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_capture_switch_selects_only_the_requested_metadata_path(enabled):
    calls = []
    records = [(0, 1, 0, 4, 0)]
    def original():
        calls.append('python')
        if enabled:
            raise AssertionError('Native capture must avoid working_records tuples')
        return records
    def adapter(*args):
        calls.append(('native', args))
        assert enabled
        return np.array([0, 4], dtype=np.int32), np.array([4, 1], dtype=np.int32), [0, 4]
    runtime = NS(options=NS(native_action_metadata=enabled), runtime=NS(config=NS(block_tokens=4)))
    authority = NS(memory=NS(_store=NS(working_records=original, action_window_segments=adapter)))
    result = ActionRuntime._window_segments(runtime, authority, 4, 4, 1)
    if enabled:
        assert_equal(result, ([0, 4], [4, 1], [0, 4]))
        assert calls == [('native', (4, 4, 1, 4))]
    else:
        assert result == ([0, 4], [4, 1], [0, 4]) and calls == ['python']


def test_enabled_old_extension_fails_explicitly_and_default_keeps_reference():
    runtime = NS(options=NS(native_action_metadata=True), runtime=NS(config=NS(block_tokens=4)))
    authority = NS(memory=NS(_store=NS(working_records=lambda: [])))
    with pytest.raises(RuntimeError, match='rebuilt native'):
        ActionRuntime._window_segments(runtime, authority, 0, 0, 1)
    runtime.options = NS()
    assert ActionRuntime._window_segments(runtime, authority, 0, 0, 1) == ([0], [1], [0])


@pytest.mark.parametrize('native_commit', [False, True])
def test_prefill_both_commit_backends_preserve_exact_scores_and_python_keys(monkeypatch, native_commit):
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_COMMIT', str(int(native_commit)))
    memory, _ = fixture('partial', current='same')
    try:
        original = reference(memory, 22, 10, 3)
        actual = native(memory, 22, 10, 3)
        states = [PrefillHistory(NS(prefill_attention_half_life=8192)) for _ in range(2)]
        for state, (starts, _, history) in zip(states, (original, actual)):
            state.values = {0: [.25, .5], 20: [.125, .75]}
            # A partial tail and appended suffix deliberately share history 20.
            masses = np.array([[.25, .25, .25, .25, 0.], [.125, .125, .125, .125, .5]])
            assert history == [0, 12, 20, 20, 24]
            assert state.accept(25, 3, 3, [0, 2], starts, history, 10, masses)
            assert state.last_commit_backend == ('native' if native_commit else 'python')
            assert all(type(key) is int for key in state.values)
            json.dumps(state.values)
        assert states[0].snapshot() == states[1].snapshot()
        assert states[0].scores() == states[1].scores()
    finally:
        assert memory.close()


def test_action_history_accept_preserves_python_page_keys():
    memory, _ = fixture('partial', current='same')
    try:
        histories = []
        for segments in (reference(memory, 22, 10, 3), native(memory, 22, 10, 3)):
            state = ActionHistory(NS(action_scope='non-thinking', action_interval=1, prefill_attention=False),
                                  [10, 11, 12, 13, 14])
            state.begin('response', [])
            state.parser.prepare([20])
            state.accept(1, [0], segments[2], [[.25, .25, .25, .25, 0.]])
            histories.append(state.mass)
            assert all(type(key) is int for key in state.mass)
            json.dumps(state.mass)
        assert histories[0] == histories[1]
    finally:
        assert memory.close()


def test_option_is_default_off_and_profile_roundtrips_without_mutation():
    assert RuntimeOptions().native_action_metadata is False
    assert RuntimeOptions.from_vllm(config({})).native_action_metadata is False
    for enabled in (False, True):
        payload = {'kvmem': {'native_action_metadata': enabled}}
        assert RuntimeOptions.from_vllm(config(payload)).native_action_metadata is enabled
        assert payload == {'kvmem': {'native_action_metadata': enabled}}


@pytest.mark.parametrize('invalid', [0, 1, None, 'false'])
def test_option_rejects_non_boolean(invalid):
    with pytest.raises(ValueError, match='native_action_metadata must be boolean'):
        RuntimeOptions(native_action_metadata=invalid)
    with pytest.raises(ValueError, match='native_action_metadata must be boolean'):
        RuntimeOptions.from_vllm(config({'kvmem': {'native_action_metadata': invalid}}))
