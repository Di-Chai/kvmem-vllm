import asyncio
import copy
import json

import httpx
import pytest

from scripts import qualify_thinking_gc_preflight as gate


def profile_with_watermarks(low, high):
    return dict(served_model_name='model', max_model_len=10000000,
        kvmem=dict(repository_blocks=8192, block_tokens=32,
            historical_gc=dict(mode='delete', policy='redundancy', ready_low=low, ready_high=high,
                               policy_params=dict(thinking_only=True))))


def test_profile_sized_gate_crosses_256k_history_pressure_again_after_initial_retirement():
    profile = profile_with_watermarks(512, 1024)
    plan = gate.preflight_plan(profile)
    assert plan['target_tokens'] == 247808 > 65536
    assert plan['feedback_tokens'] == [9216, 9216, 32]
    # Simulate a full first retirement down to the configured target, then
    # both tool observations. The third turn must exceed the trigger again.
    retained = (8192-1024)*32
    trigger = (8192-512)*32
    retained += sum(plan['feedback_tokens'][:2])
    assert retained > trigger + 1024
    assert plan['logical_tokens_upper_bound'] < profile['max_model_len']
    legacy = gate.preflight_plan(profile_with_watermarks(7680, 7808))
    assert legacy['target_tokens'] == 18432
    assert legacy['feedback_tokens'] == [3072, 3072, 32]


@pytest.mark.parametrize('target', [18432, 245760, 246783, 264193, 10000000, True])
def test_preflight_rejects_non_pressuring_or_unbounded_initial_length(target):
    with pytest.raises(ValueError, match='length must cross history pressure'):
        gate.preflight_plan(profile_with_watermarks(512, 1024), target)


@pytest.mark.parametrize('target', [246784, 247808, 264192])
def test_explicit_length_is_allowed_within_profile_sized_bounds(target):
    assert gate.preflight_plan(profile_with_watermarks(512, 1024), target)['target_tokens'] == target


def test_preflight_rejects_logical_limit_that_fits_initial_prompt_but_not_all_four_turns():
    profile = profile_with_watermarks(512, 1024)
    profile['max_model_len'] = 262144
    with pytest.raises(ValueError, match='four-turn preflight'):
        gate.preflight_plan(profile)


@pytest.mark.parametrize('low,high', [(0,1024), (1024,512), (512,512), (512,8193), (True,1024)])
def test_preflight_requires_valid_capacity_watermarks(low, high):
    with pytest.raises(ValueError, match='capacity and GC watermarks'):
        gate.preflight_plan(profile_with_watermarks(low, high))


def empty():
    return dict(progress=dict(thinking_capture=dict(accepted_rows=0),redundancy_q_capture=dict(overflow_packets=0)),
        archive=dict(cpu_authority_bytes=0, cpu_index_bytes=0, cpu_index_owners=0, archive_used_slots=0,
            active_requests=0, retiring_requests=0, session_retiring_requests=0,
            historical_gc=dict(active_scopes=0, policy_failures=0)),
        scheduler_resources=dict(requests=0, owned_blocks=0, refresh_pending=0,
            connector_pending=0, reservations=0, session_owners=[]))


def active():
    value=empty()
    value['progress']['thinking_capture']=dict(enabled=True,layout_supported=True,accepted_rows=18000)
    value['archive']['historical_gc'].update(thinking_classified=560,thinking_pure=550,
        redundancy_retired=4,lru_fallback_retired=0,active_scopes=1)
    value['archive']['cpu_authority_bytes']=1024
    value['scheduler_resources']['session_owners']=[dict(namespace=['fixture','fixture'])]
    return value


@pytest.mark.parametrize('damage',[None,'no_classification','all_lru','policy_failure','old_failure','no_capture','q_overflow'])
def test_gate_requires_real_classification_and_redundancy_not_only_enabled_settings(damage):
    before, observed=empty(),active()
    if damage=='no_classification':observed['archive']['historical_gc']['thinking_pure']=0
    if damage=='all_lru':observed['archive']['historical_gc'].update(redundancy_retired=0,lru_fallback_retired=4)
    if damage=='policy_failure':observed['archive']['historical_gc']['policy_failures']=1
    if damage=='old_failure':before['archive']['historical_gc']['policy_failures']=1
    if damage=='no_capture':observed['progress']={}
    if damage=='q_overflow':observed['progress']['redundancy_q_capture']['overflow_packets']=1
    assert gate.verify(before,[observed])['passed']==(damage is None)


def test_release_proof_requires_all_capacity_fields_and_no_owned_state():
    assert gate.clean(empty())
    assert not gate.clean(active())
    missing=empty();del missing['archive']['cpu_index_bytes']
    assert not gate.clean(missing)



def trajectory_evidence():
    turns, records = [], []
    evidence = {'async_restore.jsonl': [], 'scheduler_refresh.jsonl': []}
    history = 18434
    for i in range(4):
        for part in ('thinking', 'eos'):
            after = active()
            after['archive']['historical_gc'].update(redundancy_retired=4+(4 if i >= 2 else 0),
                active_features=dict(question_ready=1, question_epoch=1, invalid_questions=0))
            rid = f'request-{i}-{part}'
            if records:
                evidence['async_restore.jsonl'].append(dict(request_id=rid+'-0-native',
                    history_tokens=records[-1]['terminal']['history_end'], cancelled=False))
            records.append(dict(turn=i+1, part=part, request_id=rid, after=after,
                terminal=dict(checkpoint_kind='M', history_end=history)))
            history += 64
        turns.append(dict(plan=dict(kvmem_query_id='task', kvmem_query_span=[3,20], kvmem_query_mode='agent'),
            exact_output_prefix_reused=i > 0, after=after))
        history += 3072
    # Initial long prefill and subsequent G-pressure refreshes are necessary
    # even with an unchanged task Query. Neither replays an old M prefix.
    evidence['scheduler_refresh.jsonl'] = [
        dict(request_id=records[0]['request_id']+'-0-native', history_begin=12224, replay_tokens=32),
        dict(request_id=records[4]['request_id']+'-0-native', history_begin=25000, replay_tokens=32)]
    return turns, records, [r['after'] for r in records], evidence


@pytest.mark.parametrize('damage', [None, 'ready', 'invalid', 'epoch', 'source', 'canonical_fallback',
    'missing_restore', 'rewind', 'wrong_checkpoint', 'cancelled', 'no_new_pressure', 'query_replay'])
def test_multiturn_gate_catches_continuation_failures_that_single_turn_misses(damage):
    turns, records, observations, evidence = trajectory_evidence()
    if damage == 'ready':
        records[2]['after']['archive']['historical_gc']['active_features']['question_ready'] = 0
    if damage == 'invalid':
        records[2]['after']['archive']['historical_gc']['active_features']['invalid_questions'] = 1
    if damage == 'epoch':
        records[2]['after']['archive']['historical_gc']['active_features']['question_epoch'] = 2
    if damage == 'source': turns[1]['plan']['kvmem_query_id'] = 'new-feedback'
    if damage == 'canonical_fallback': turns[1]['exact_output_prefix_reused'] = False
    if damage == 'missing_restore': evidence['async_restore.jsonl'].pop()
    if damage == 'rewind': evidence['async_restore.jsonl'][2]['history_tokens'] -= 32
    if damage == 'wrong_checkpoint': records[2]['terminal']['checkpoint_kind'] = 'P'
    if damage == 'cancelled': evidence['async_restore.jsonl'][2]['cancelled'] = True
    if damage == 'no_new_pressure': turns[2]['after']['archive']['historical_gc']['redundancy_retired'] = 4
    if damage == 'query_replay':
        evidence['scheduler_refresh.jsonl'].append(dict(request_id=records[2]['request_id']+'-0-native',
            history_begin=32, replay_tokens=17))
    assert gate.verify_multiturn(turns, records, observations, evidence)['passed'] == (damage is None)


@pytest.mark.parametrize('watermarks,expected_feedback', [((7680,7808),3072), ((512,1024),9216)])
def test_preflight_commits_only_real_accepted_ids_and_restores_before_renewed_pressure(tmp_path, monkeypatch, watermarks, expected_feedback):
    state, calls, commits, evidence = empty(), [], [], {'async_restore.jsonl': [], 'scheduler_refresh.jsonl': []}
    feedbacks = []
    history = 18434
    class Conversation:
        initial = dict(tokens=[1,2,3], plan=dict(kvmem_query_span=[0,1]))
        end = 999
        def __init__(self, *args): self.turn = 0
        def prepare(self):
            self.turn += 1
            return dict(tokens=[1,2,3], plan=dict(kvmem_query_span=[0,1], kvmem_query_id='source', kvmem_query_mode='agent'),
                        wire_proof={}, exact_output_prefix_reused=self.turn > 1)
        def feedback(self, target_tokens): feedbacks.append(target_tokens)
        def commit(self, ids): commits.append(ids); return {'content': []}
    monkeypatch.setattr(gate, 'snapshot', lambda path: copy.deepcopy(state))
    monkeypatch.setattr(gate, 'Conversation', Conversation)
    monkeypatch.setattr(gate, 'capture_evidence', lambda *args: evidence)
    async def upstream(request):
        nonlocal history
        body = json.loads(request.content); calls.append(body)
        if body.get('kv_transfer_params', {}).get('kvmem_release_session'):
            counters = state['archive']['historical_gc'].copy()
            state.clear(); state.update(empty()); state['archive']['historical_gc'].update(counters, active_scopes=0)
            return httpx.Response(200, json=dict(id='release'))
        turn = (len(calls)+1)//2
        part = 'thinking' if len(calls)%2 else 'eos'
        rid = f'request-{turn}-{part}'
        if len(calls) > 1:
            evidence['async_restore.jsonl'].append(dict(request_id=rid+'-0-native',
                history_tokens=history, cancelled=False))
        history += body['max_tokens']
        state.clear(); state.update(active())
        state['archive']['historical_gc'].update(redundancy_retired=4+(4 if turn >= 3 else 0),
            active_features=dict(question_ready=1, question_epoch=1, invalid_questions=0))
        state['archive']['completed'] = [dict(request_id=rid+'-0-native', checkpoint_kind='M', history_end=history)]
        return httpx.Response(200, json=dict(id=rid, choices=[dict(token_ids=body['allowed_token_ids']*body['max_tokens'])]))
    client = httpx.AsyncClient
    monkeypatch.setattr(gate.httpx, 'AsyncClient', lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(upstream)))
    profile = profile_with_watermarks(*watermarks)
    asyncio.run(gate.exercise('http://model', profile, tmp_path/'protocol', tmp_path/'capture', tmp_path/'result'))
    assert len(calls) == 9 and calls[0]['prompt'] == [1,2,3]
    assert len({c['session_id'] for c in calls}) == 1
    assert commits == [[830]*63+[999]]*4
    assert calls[1]['prompt'] == calls[0]['prompt']+[830]*63
    assert feedbacks == [expected_feedback, expected_feedback, 32]
    assert json.loads((tmp_path/'result/preflight-plan.json').read_text()) == gate.preflight_plan(profile)
    result = json.loads((tmp_path/'result/result.json').read_text())
    assert result['passed'] and result['clean_release'] and result['native_delta']['redundancy_retired'] == 8
    assert result['multi_turn']['checks']['m_restores_completed']
    assert (tmp_path/'result/single-turn-result.json').exists()


def test_native_capture_filters_only_exact_preflight_request_ids(tmp_path):
    path = tmp_path/'async_restore.jsonl'
    path.write_text(json.dumps(dict(request_id='old-0'))+'\n')
    offsets = gate.capture_offsets(tmp_path)
    with path.open('a') as stream:
        for rid in ('foreign-0', 'owned-0-native'):
            stream.write(json.dumps(dict(request_id=rid, history_tokens=12))+'\n')
        stream.write('{"request_id":')
    result = gate.capture_evidence(tmp_path, offsets, [dict(request_id='owned', turn=2, part='thinking')])
    assert result['async_restore.jsonl'] == [dict(request_id='owned-0-native', history_tokens=12, turn=2, part='thinking')]
