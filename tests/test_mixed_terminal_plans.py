"""KVMEM_MIXED_TERMINAL_PLANS: a mixed batch carrying a prompt's final chunk
keeps the fence-free candidate path; only that owner is planned (eagerly, from
CPU coordinates, behind its own fence) before the forward."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.action_attention import mixed_terminal_plans
from test_mixed_decode_candidates import Resolved, fixture


def terminal_fixture(*, prompt=1064, replay=None):
    action, batch, prefill, armed = fixture()
    action.prefill_plans = (batch, {})                  # the final chunk has no pipelined plan
    action.mixed_terminal = True
    action.runner._kvmem_window_adapter.offsets['p'] = 800
    action.runner.req_states = NS(req_id_to_index={'p': 0, 'd': 1}, prompt_len=NS(np=[prompt, 50]))
    planned, segments = [], []
    state = NS(prefill_replay=replay)
    authority = object()
    def plan_owner(request_id, prefilling, tokens, global_positions, local_positions, count):
        planned.append((request_id, prefilling, tokens, list(global_positions), list(local_positions), count))
        return dict(authority=authority, state=state, history_begin=int(global_positions[0]),
                    local_begin=int(local_positions[0]), rows=[31, 63], sources=())
    action._plan_owner = plan_owner
    def window_segments(owner, history_begin, local_begin, count):
        segments.append((owner, history_begin, local_begin, count))
        return [0, 192], [192, 8], [0, 992]
    action._window_segments = window_segments
    return action, batch, armed, planned, segments, authority


def test_final_chunk_gets_its_eager_plan_beside_decode_candidates():
    action, batch, armed, planned, segments, authority = terminal_fixture()
    assert action._prepare_mixed_candidates(batch)
    plans, = armed
    assert list(plans) == ['p', 'd']                     # batch order, as the eager planner armed
    terminal = plans['p']
    assert planned == [('p', True, (), list(range(1000, 1064)), list(range(200, 264)), 64)]
    assert segments == [(authority, 1000, 200, 64)]
    assert (terminal['index'], terminal['begin'], terminal['count'], terminal['terminal_exact']) == (0, 0, 64, True)
    assert (terminal['starts'], terminal['counts'], terminal['history']) == ([0, 192], [192, 8], [0, 992])
    assert '_layout' in plans['d'] and plans['d']['rows'] == [0, 1, 2, 3, 4]
    assert action.native_candidates == (batch, plans)
    assert action.native.counts == {'mixed_terminal_plans': 1}


@pytest.mark.parametrize('change,reason', [
    ('flag_off', 'prefill_unplanned'),
    ('padding_rows', 'terminal_shape'),      # scheduled rows beyond the prompt
    ('nonterminal', 'terminal_shape'),       # an ordinary chunk without a pipelined plan
    ('replay', 'terminal_replay'),
    ('logical', 'terminal_logical'),
    ('no_state', 'terminal_prompt'),
    ('decode_frame', 'decode_frame'),        # other owners are checked first
])
def test_anything_else_keeps_the_eager_planner(change, reason):
    action, batch, armed, planned, segments, _ = terminal_fixture(
        prompt={'padding_rows': 1063, 'nonterminal': 2000}.get(change, 1064),
        replay=(1032, 1064) if change == 'replay' else None)
    if change == 'flag_off': action.mixed_terminal = False
    elif change == 'logical': action.runner._kvmem_logical_prefill_groups = {'p': object()}
    elif change == 'no_state': action.runner.req_states.req_id_to_index = {'d': 1}
    elif change == 'decode_frame': action.native_cache = {}
    assert not action._prepare_mixed_candidates(batch)
    assert not armed and action.native_candidates is None
    assert action.native.counts == {'mixed_skip:' + reason: 1}
    if change != 'replay':
        assert not planned                   # no fence is taken unless the batch qualifies so far


def test_capture_reuses_the_armed_terminal_plan():
    action, batch, armed, planned, segments, _ = terminal_fixture()
    assert action._prepare_mixed_candidates(batch)
    terminal = armed[0]['p']
    action.metadata.host = NS(tolist=lambda: (list(range(69)), list(range(1000, 1064)) + list(range(100, 105)),
                                              list(range(200, 264)) + list(range(64, 69))))
    action.metadata_early_batches = 0
    del planned[:]
    resolved = []
    def resolve(plans):
        resolved.append(plans)
        raise Resolved
    action.native.resolve = resolve
    action._plan_owner = lambda request_id, prefilling, tokens, g, l, count: (
        planned.append(request_id) or dict(rows=[], sources=(), history_begin=100, local_begin=64))
    with pytest.raises(Resolved):
        action._capture([], batch)
    plans, = resolved
    assert planned == ['d'] and plans['p'] is terminal


def test_captured_coordinates_must_match_the_terminal_plan():
    action, *_ = terminal_fixture()
    plan = dict(count=4, history_begin=1000, local_begin=200)
    action._validate_prefill_plan(plan, [1000, 1001, 1002, 1003], [200, 201, 202, 203], 4)
    with pytest.raises(RuntimeError, match='coordinates differ'):
        action._validate_prefill_plan(plan, [1000, 1001, 1002, 1003], [201, 202, 203, 204], 4)


def test_flag_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_MIXED_TERMINAL_PLANS', raising=False)
    assert not mixed_terminal_plans()
    monkeypatch.setenv('KVMEM_MIXED_TERMINAL_PLANS', '1')
    assert mixed_terminal_plans()
    monkeypatch.setenv('KVMEM_MIXED_TERMINAL_PLANS', 'on')
    with pytest.raises(ValueError, match='0 or 1'):
        mixed_terminal_plans()


@pytest.mark.parametrize('drift', [None, 'rows', 'starts', 'local_begin'])
def test_check_mode_compares_the_armed_plan_with_the_eager_plan(drift):
    action, batch, armed, planned, segments, authority = terminal_fixture()
    assert action._prepare_mixed_candidates(batch)
    terminal = armed[0]['p']
    if drift == 'rows': terminal['rows'] = [63]
    elif drift == 'starts': terminal['starts'] = [0, 160]
    elif drift == 'local_begin': terminal['local_begin'] = 201
    g, l = list(range(1000, 1064)), list(range(200, 264))
    if drift is None:
        action._check_terminal_plan(terminal, 'p', [], g, l, 64)
        assert action.terminal_plan_checks == 1 and action.native.counts['mixed_terminal_checked'] == 1
    else:
        with pytest.raises(RuntimeError, match='differs from the eager plan: ' + drift):
            action._check_terminal_plan(terminal, 'p', [], g, l, 64)


@pytest.mark.parametrize('binding', ['session', 'missing', 'short'])
def test_prefill_fold_reads_the_session_prompt_tokens(monkeypatch, binding):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    action, batch, armed, planned, segments, _ = terminal_fixture()
    action.runner._kvmem_layerwise_pressure = NS(rolling_decode=True)
    tokens = {'session': tuple(range(5000, 6064)), 'short': tuple(range(10))}.get(binding)
    action.runner._kvmem_sessions = NS(bindings={} if tokens is None else {'p': NS(tokens=tokens)})
    if binding == 'session':
        assert action._prepare_mixed_candidates(batch)
        assert planned[0][2] == list(range(6000, 6064))
        assert armed[0]['p']['_terminal_tokens'] == list(range(6000, 6064))
    else:
        assert not action._prepare_mixed_candidates(batch)
        assert action.native.counts == {'mixed_skip:terminal_fold': 1} and not planned


def test_capture_rejects_terminal_tokens_that_differ_from_the_captured_input(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    action, batch, armed, planned, segments, _ = terminal_fixture()
    action.runner._kvmem_layerwise_pressure = NS(rolling_decode=True)
    action.runner._kvmem_sessions = NS(bindings={'p': NS(tokens=tuple(range(5000, 6064)))})
    assert action._prepare_mixed_candidates(batch)
    action.metadata.host = NS(tolist=lambda: (list(range(69)), list(range(1000, 1064)) + list(range(100, 105)),
                                              list(range(200, 264)) + list(range(64, 69))))
    action.metadata_early_batches = 0
    with pytest.raises(RuntimeError, match='tokens differ'):
        action._capture([], batch)
