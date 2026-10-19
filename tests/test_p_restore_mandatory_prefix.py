"""Historical mandatory coverage, not an edited future suffix, gates P reuse."""
import random

import pytest

from vllm_kvmem.prompt_checkpoints import token_bytes
from vllm_kvmem.request_plan import RequestPlan, request_plan
from test_session_control import planned_idle_session, request


@pytest.mark.parametrize('explicit', [False, True])
def test_mandatory_extension_after_P_keeps_current_query_and_historical_pins(explicit):
    previous = ((0, 4, 'root'), (7, 11, 'query'))
    plan = RequestPlan((9, 14), block_tokens=4, gen_budget=16,
        mandatory=((0, 4, 'root'), (7, 15, 'query')), explicit_mandatory=explicit)
    resolved, changed = plan.continuation(18, 4, 8, 8, 32, (9, 10), previous)
    assert not changed
    assert resolved.query == plan.query and resolved.replay_begin == 8
    assert resolved.mandatory == plan.mandatory


def test_historical_requirements_match_independent_per_token_coverage_oracle():
    rng = random.Random(180288)
    for _ in range(300):
        history = rng.randrange(17)
        def spans():
            result = []
            for _ in range(rng.randrange(6)):
                begin = rng.randrange(20)
                result.append((begin, rng.randrange(begin+1, 22), rng.choice(['root','query','file'])))
            return tuple(result)
        previous, current = spans(), spans()
        def coverage(values):
            return {(position, kind) for begin,end,kind in values
                    for position in range(begin, min(end,history))}
        expected = bool(coverage(current)-coverage(previous))
        plan = RequestPlan(None, block_tokens=4, gen_budget=16,
            mandatory=current, explicit_mandatory=True)
        _, changed = plan.continuation(24, 4, history, 4, 32, None, previous)
        assert changed == expected, (history, previous, current)


def branch_fixture():
    control, events, _, entry = planned_idle_session()
    entry.prompt_history = entry.prompt_window = 12
    entry.prompt_tokens = token_bytes(list(range(25)), 12)
    # The M prefix differs, while the complete saved P prefix is byte-identical.
    req = request('edited-query', tokens=list(range(14))+[999]*8)
    req.num_prompt_tokens = len(req.prompt_token_ids)
    req.max_tokens = 8
    req.kv_transfer_params = {'kvmem_request_plan': {
        'kvmem_query_span': [13, 19],
        'kvmem_mandatory_spans': [
            {'begin':0,'end':4,'kind':'root'},
            {'begin':8,'end':21,'kind':'query'}]}}
    return control, events, entry, req


def test_changed_query_crossing_P_reuses_verified_prefix_without_eviction():
    control, events, entry, req = branch_fixture()
    expected_mandatory = request_plan(req, 4, 32).mandatory
    assert control.lookup(req, 0) == 12
    assert not control.take_commands().evictions
    assert events == [('prepare', ('edited-query', 12, 12))]
    plan = request_plan(req, 4, 32)
    assert plan.query == (13, 19) and plan.replay_begin == 12
    assert plan.mandatory == expected_mandatory
    assert entry.tokens == entry.prompt_tokens
    # Cancellation keeps a retryable P lease; it must not fabricate allocation.
    control.finished(req)
    assert entry.phase == 'idle' and entry.history == 12
    assert events[-1] == ('abort', ('edited-query',))


@pytest.mark.parametrize('change', ['edit_before_P', 'new_historical_pin'])
def test_real_prefix_edit_and_new_historical_requirement_still_evict(change):
    control, events, entry, req = branch_fixture()
    if change == 'edit_before_P':
        req.prompt_token_ids[3] = 777
    else:
        req.kv_transfer_params['kvmem_request_plan']['kvmem_mandatory_spans'].append(
            {'begin':5,'end':6,'kind':'file'})
    assert control.lookup(req, 0) is None
    assert entry.phase == 'evicting' and not events
    assert len(control.take_commands().evictions) == 1
