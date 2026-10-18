"""M/P claims preserve only token-verified prefill evidence through model replay."""
from types import SimpleNamespace

import pytest

from vllm_kvmem.action_attention import ActionHistory
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool
from test_authority import setup, payload, drain
from test_context_continuation import Model
from test_core import Event
from test_prefill_attention import options
from test_prompt_checkpoints import ready


def captured_pm(tmp_path, *, terminal_end=12, native=True):
    authority, archive = setup(tmp_path)
    config = options(native_prefill_commit=native)
    config.memory = authority.memory.config
    state = authority.action_history = ActionHistory(config, [10, 11, 12, 13, 14])
    assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(authority)
    assert state.prefill.accept(8, 8, 8, [3, 7], [0, 4], [0, 4], 0,
        [[1., 0.], [.5, .5]])
    state.scores, state.rounds = {0: .25, 4: .75}, 1
    pool = PromptCheckpointPool(2, 8192, retain_prompt=True)
    prompt = pool.capture(('tenant', 'session'), 'a', authority, list(range(24)), Model())
    suffix = terminal_end-8
    assert authority.append(*payload(8, suffix), history_begin=8, local_rope_begin=8)
    drain(authority)
    pages = list(range(0, terminal_end, 4))
    assert state.prefill.accept(terminal_end, suffix, suffix, [suffix-1], pages, pages, 8,
        [[1./len(pages)]*len(pages)])
    state.scores, state.rounds = {0: .1, 8: .9}, 2
    terminal = pool.capture(prompt.namespace, 'a', authority, list(range(24)), Model(),
        replace=True, terminal=True)
    pool.retire(terminal, Event(True))
    ready(pool, terminal)
    return pool, prompt, terminal, authority, archive


def finish_import(pool, checkpoint, authority, *, retain=False):
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True), retain=retain) is authority


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('edited', [False, True])
def test_real_pm_claim_preserves_m_prefill_only_for_entire_unchanged_prefix(tmp_path, edited, native):
    pool, prompt, terminal, authority, archive = captured_pm(tmp_path, native=native)
    original_p, original_m = prompt.action_history, terminal.action_history
    tokens = list(range(24))
    if edited:
        tokens[9] = 999
    assert pool.claim(prompt.namespace, 'next', tokens, position=8) is prompt
    plan = pool.prepare_claim_window(prompt)
    state = authority.action_history
    expected = original_p if edited else original_m
    assert authority.history_end == 8
    assert state.prefill.snapshot() == expected[2]
    assert state.scores == expected[0] and state.rounds == expected[1]
    assert state.snapshot(8) == original_p
    assert state.snapshot(12) == expected
    assert len(state.snapshot()) == 3  # The durable action/prefill tuple did not change.
    assert state.prefill.end == (8 if edited else 12)
    finish_import(pool, prompt, authority)
    del plan
    assert pool.payload_bytes == 0 and authority.close() and archive.close()


def test_preserved_m_prefill_watermark_deduplicates_replayed_source_rows(tmp_path):
    pool, prompt, terminal, authority, archive = captured_pm(tmp_path)
    expected = terminal.action_history[2]
    assert pool.claim(prompt.namespace, 'next', list(range(24)), position=8) is prompt
    plan = pool.prepare_claim_window(prompt)
    state = authority.action_history
    finish_import(pool, prompt, authority)
    del plan
    assert state.prefill.rows(8, 4, 8) == []
    assert not state.prefill.accept(12, 4, 4, [3], [0, 4, 8], [0, 4, 8], 8,
        [[.2, .3, .5]])
    assert state.prefill.snapshot() == expected
    # Replay extends the physical authority to M but never adds feedback twice.
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    assert authority.append(*payload(12, 4), history_begin=12, local_rope_begin=12)
    drain(authority)
    rows = state.prefill.rows(12, 4, 12)
    assert rows == [3]
    assert state.prefill.accept(16, 4, 4, rows, [0, 4, 8, 12], [0, 4, 8, 12], 12,
        [[.25, .25, .25, .25]])
    assert state.prefill.end == 16
    assert state.prefill.samples == expected[2]+1 and state.prefill.chunks == expected[3]+1
    assert authority.close() and archive.close()


def test_intermediate_p_capture_does_not_publish_preserved_future_m_feedback(tmp_path):
    pool, prompt, terminal, authority, archive = captured_pm(tmp_path, terminal_end=16)
    original_p, original_m = prompt.action_history, terminal.action_history
    assert pool.claim(prompt.namespace, 'next', list(range(24)), position=8) is prompt
    plan = pool.prepare_claim_window(prompt)
    finish_import(pool, prompt, authority, retain=True)
    del plan
    assert authority.action_history.prefill.snapshot() == original_m[2]
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    intermediate = pool.capture(prompt.namespace, 'next', authority, list(range(24)), Model(), replace=True)
    assert intermediate.position == 12
    assert intermediate.action_history == original_p
    assert intermediate.action_history[2][1] == 8
    pool.retire(intermediate, Event(True))
    ready(pool, intermediate)
    edited = list(range(24))
    edited[13] = 999
    assert pool.claim(intermediate.namespace, 'edited', edited, position=12) is intermediate
    plan = pool.prepare_claim_window(intermediate)
    assert authority.action_history.snapshot() == original_p
    assert authority.action_history.prefill.end == 8
    finish_import(pool, intermediate, authority)
    del plan
    assert pool.payload_bytes == 0 and authority.close() and archive.close()


@pytest.mark.parametrize('validated_end', [None, 7, 11, True, 12.0])
def test_future_prefill_without_sufficient_explicit_proof_fails_before_mutation(validated_end):
    state = ActionHistory(options(), [10, 11, 12, 13, 14])
    state.scores, state.rounds = {0: .75}, 3
    assert state.prefill.accept(8, 8, 8, [7], [0], [0], 0, [[1.]])
    before = state.snapshot()
    stores = []
    authority = SimpleNamespace(history_end=8, records={0: SimpleNamespace(history_begin=0)},
        memory=SimpleNamespace(set_action_scores=stores.append))
    future = ({0: .25}, 99, ({0: (.1, .2)}, 12, 4, 2))
    with pytest.raises(ValueError, match='validated'):
        state.restore(future, authority, preserve_future=True, validated_end=validated_end)
    assert state.snapshot() == before and stores == []


def test_output_only_legacy_preserve_future_api_does_not_require_prefill_proof():
    state = ActionHistory(options(prefill_attention=False), [10, 11, 12, 13, 14])
    authority = SimpleNamespace(records={0: SimpleNamespace(history_begin=0)},
        memory=SimpleNamespace(set_action_scores=lambda _: None))
    state.restore(({0: .5, 8: .5}, 2), authority, preserve_future=True)
    assert state.snapshot() == ({0: .5, 8: .5}, 2)
