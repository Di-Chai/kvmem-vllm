"""CPU-only acceptance gates for the small real-model alignment client."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('context_alignment_qualification',
        SCRIPTS/'qualify_context_alignment.py')
    qual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qual)
finally:
    sys.path.pop(0)


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def profile(cap=1):
    return dict(port=12345, served_model_name='alignment', execution_mode='compiled_graph',
        draft_method='mtp', draft_tokens=4, verify_session_restore=True, max_num_seqs=2,
        max_model_len=32768, query_replay_budget=4096, kvmem=dict(select_budget=4096,
            gen_budget=4096, block_tokens=32, sink_blocks=41, session_capacity=4,
            session_checkpoint_policy='pm', session_capture_terminal=True, request_planning=True,
            archive_backend='cpu', action_attention=True, prefill_attention=True, query_history=True,
            action_scope='non-thinking', action_pressure=False, guided_query_tokens=cap))


def events(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def evidence(tmp_path, cap=1):
    rows, refresh, actions, prefill, states, targets, histories = [], [], [], [], [], [], []
    for agent in range(2):
        prompt, previous = list(range(100)), None
        qspan = [64,70]
        for turn in range(3):
            if turn:
                prompt = previous['input_ids']+previous['output_ids']+[20]*(2048 if turn == 1 else 10)
            response = f'user-{agent}-turn-{turn}'
            native = response+'-0-native'
            output = list(range(1000,1129))
            end = len(prompt)+len(output)-1
            row = dict(agent=agent, turn=turn, response_id=response, input_ids=prompt,
                output_ids=output, query_span=None if turn == 1 else qspan,
                terminal=dict(authority_namespace=f'authority-{agent}', checkpoint_kind='M',
                    checkpoint_tokens=end, history_end=end,
                    terminal_prefix=dict(history_end=end, output_tokens=len(output))))
            action = dict(request_id=native, prefill_end=64 if turn == 0 else 2277,
                prefill_samples=2 if turn == 0 else 5, prefill_chunks=1 if turn == 0 else 3)
            if turn != 1:
                refresh.append(dict(request_id=native, selection='query', query_span=qspan,
                    selected_tokens=32, query_history_enabled=True, query_history_pages=1,
                    prefill_attention_samples=action['prefill_samples'], prefill_attention_chunks=action['prefill_chunks'],
                    guided_published_tokens=0 if cap == 1 else 3, guided_consumed_tokens=1 if cap == 1 else 4,
                    guided_captured_tokens=1 if cap == 1 else 3, guided_natural_termination=cap != 1,
                    guided_fallback='unterminated-output' if cap == 1 else None,
                    query_score_tokens=6 if cap == 1 else 3))
            else:
                prefill.append(dict(request_id=native, end=2277, samples=5, chunks=3))
            if turn:
                restore_end = previous['terminal']['checkpoint_tokens'] if turn == 1 else 64
                states.append(dict(request_id=native, history_tokens=restore_end,
                    checkpoint_kind='M' if turn == 1 else 'P', recurrent_byte_equal=True, pressure_restore=False))
                targets.append(dict(request_id=native, history_tokens=restore_end, byte_equal=True,
                    working_attention_layers=17, mtp_attention_layers=1))
            histories.append(dict(request_id=native, round=turn+1, observed=turn != 1,
                score_sha256=f'{agent}-'+('initial' if turn < 2 else 'new')))
            rows.append(row)
            actions.append(action)
            previous = row
    for name, data in [('bounded_refresh',refresh), ('action_attention',actions),
            ('prefill_attention',prefill), ('verified_state_restores',states),
            ('verified_restores',targets), ('query_history',histories)]:
        events(tmp_path/(name+'.jsonl'), data)
    return rows


@pytest.mark.parametrize('cap', [1,128])
def test_fixed_model_and_two_session_parameters_allow_both_guided_modes(cap):
    qual.validate_profile(profile(cap))
    collect = profile(cap)
    collect['kvmem']['graph_mode'] = 'collect'
    qual.validate_profile(collect)


@pytest.mark.parametrize('key,value', [('gen_budget',8192), ('block_tokens',128),
    ('query_history',False), ('prefill_attention',False), ('session_capacity',2)])
def test_alignment_probe_rejects_missing_or_changed_controls(key, value):
    current = profile()
    current['kvmem'][key] = value
    with pytest.raises(ValueError):
        qual.validate_profile(current)


def test_fixture_and_closed_loop_prompts_keep_exact_history_and_complete_message_spans():
    fixture = qual.build_fixture(Tokenizer())
    assert fixture == qual.build_fixture(Tokenizer())
    qual.validate_fixture(fixture)
    agent = fixture['agents'][0]
    generated = [900001,900002,900003]
    prompt, spans = qual.append_turn(agent, agent['initial_tokens'], generated, agent['message_spans'], 1)
    prefix = agent['initial_tokens']+generated
    assert prompt[:len(prefix)] == prefix and len(prompt)-len(prefix) == 2048
    assert spans[-2]['role'] == 'tool' and spans[-1]['role'] == 'user'
    assert all(a['end'] <= b['begin'] for a,b in zip(spans,spans[1:]))
    forced, _ = qual.append_turn(agent, prompt, generated, spans, 2)
    assert forced[:len(prompt)+len(generated)] == prompt+generated
    fixture['agents'][0]['tool_append_tokens'][0] += 1
    with pytest.raises(ValueError, match='Frozen context fixture changed'):
        qual.validate_fixture(fixture)


def test_stable_body_has_no_query_and_forced_turn_uses_explicit_old_query():
    from vllm_kvmem.request_plan import FIELDS, RequestPlan
    body = qual.request_body(profile(), 'owner', list(range(8192)), turn=1, query=[1024,1088])
    assert 'kvmem_query_span' not in body
    assert body['kvmem_reselect'] == 'off' and body['kvmem_prefill_window'] == 'keep_selected'
    assert not body['ignore_eos'] and body['max_tokens'] == 257 and body['min_tokens'] == 129
    plan_fields = {k:v for k,v in body.items() if k in FIELDS}
    plan = RequestPlan.parse(plan_fields,8192,32,4096)
    assert plan.query is None and plan.mandatory == ((1024,1088,'query'),)
    assert plan.prefill_checkpoint(8192) is None
    del plan_fields['kvmem_mandatory_spans']
    assert RequestPlan.parse(plan_fields,8192,32,4096).prefill_checkpoint(8192) is not None
    force = qual.request_body(profile(), 'owner', [1,2], turn=2, query=[0,1])
    assert force['kvmem_query_span'] == [0,1] and force['kvmem_reselect'] == 'force'
    release = qual.request_body(profile(), 'owner', [0], release=True)
    assert release['kv_transfer_params'] == {'kvmem_release_session':True}


@pytest.mark.parametrize('cap', [1,128])
def test_complete_alignment_evidence_includes_mp_prefill_guided_and_owner_oracles(tmp_path, cap):
    result = qual.verify_evidence(tmp_path, evidence(tmp_path, cap), profile(cap))
    assert result['complete'] and result['isolated_authorities'] == 2
    assert result['guided_published_queries'] == (0 if cap == 1 else 4)


def test_large_cap_without_a_naturally_published_query_is_incomplete(tmp_path):
    result = qual.verify_evidence(tmp_path, evidence(tmp_path, 1), profile(128))
    assert not result['complete'] and result['guided_published_queries'] == 0


def test_initial_pressure_can_precede_one_explicit_query_without_duplicate_selection(tmp_path):
    rows = evidence(tmp_path)
    path = tmp_path/'bounded_refresh.jsonl'
    refresh = qual.records(path)
    events(path, [dict(request_id=rows[0]['response_id']+'-0-native', selection='pressure'),
        dict(request_id=rows[3]['response_id']+'-0-native', selection='pressure')]+refresh)
    assert qual.verify_evidence(tmp_path,rows,profile())['complete']


@pytest.mark.parametrize('second_turn', [False,True])
def test_stable_turn_pressure_or_duplicate_query_is_still_rejected(tmp_path, second_turn):
    rows = evidence(tmp_path)
    path = tmp_path/'bounded_refresh.jsonl'
    refresh = qual.records(path)
    extra = dict(request_id=rows[1]['response_id']+'-0-native',selection='pressure') if second_turn else refresh[0]
    events(path, refresh+[extra])
    with pytest.raises(ValueError, match='Stable tool|one bounded selection'):
        qual.verify_evidence(tmp_path,rows,profile())


@pytest.mark.parametrize('filename,index,changes', [
    ('action_attention',2,dict(prefill_samples=0,prefill_end=0)),
    ('action_attention',2,dict(prefill_samples=6)),
    ('verified_state_restores',1,dict(checkpoint_kind='M')),
    ('verified_restores',1,dict(mtp_attention_layers=0)),
    ('query_history',1,dict(score_sha256='unexpected-aging')),
    ('bounded_refresh',1,dict(prefill_attention_samples=2)),
    ('bounded_refresh',1,dict(guided_published_tokens=1,guided_natural_termination=False)),
])
def test_bad_feedback_restore_or_guided_evidence_never_passes(tmp_path, filename, index, changes):
    rows = evidence(tmp_path)
    path = tmp_path/(filename+'.jsonl')
    data = qual.records(path)
    data[index].update(changes)
    events(path, data)
    with pytest.raises(ValueError):
        qual.verify_evidence(tmp_path, rows, profile())


def test_foreign_authority_or_changed_own_prefix_is_rejected(tmp_path):
    rows = evidence(tmp_path)
    rows[-1]['terminal']['authority_namespace'] = 'authority-0'
    with pytest.raises(ValueError, match='authorities crossed'):
        qual.verify_evidence(tmp_path, rows, profile())
    rows = evidence(tmp_path)
    rows[1]['input_ids'] = rows[1]['input_ids'].copy()
    rows[1]['input_ids'][0] = 999
    with pytest.raises(ValueError, match='public token prefix'):
        qual.verify_evidence(tmp_path, rows, profile())


def test_invalid_force_requires_frontend_400_without_ever_accepting_success(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            body = json.loads(request.data)
            assert len(body['prompt']) == 7168 and body['max_tokens'] == 4096
            assert body['kvmem_query_span'] == [4096,7168] and body['kvmem_reselect'] == 'force'
            raise urllib.error.HTTPError(request.full_url,400,'bad request',{},
                io.BytesIO(b'Complete Query, output reserve and fixed sink/recent bands exceed active capacity '
                    b'(selected budget 992, required 1312)'))
    monkeypatch.setattr(qual.urllib.request, 'build_opener', lambda *a:Opener())
    assert qual.invalid_force(profile(), 'bad', list(range(8000)))['status'] == 400


def test_failed_evidence_still_releases_every_owned_session(monkeypatch, tmp_path):
    cfg, fixture = tmp_path/'profile.json', tmp_path/'fixture.json'
    qual.save(cfg, profile())
    qual.save(fixture, qual.build_fixture(Tokenizer()))
    args = SimpleNamespace(config=cfg, fixture=fixture, output=tmp_path/'result', capture=tmp_path, arm='test')
    released = []
    counters = {}
    def stream(profile, session, tokens, **kwargs):
        if kwargs.get('release'):
            released.append(session)
        counters[session] = counters.get(session,0)+1
        return dict(response_id=session+str(counters[session]), input_ids=list(tokens), output_ids=[1000]*129,
            text='', query_span=kwargs.get('query'), begin=1., end=2.)
    monkeypatch.setattr(qual, 'stream_request', stream)
    monkeypatch.setattr(qual, 'invalid_force', lambda *a:dict(status=400))
    monkeypatch.setattr(qual, 'wait_terminal_publication', lambda capture, rows:
        dict(completed=[dict(request_id=rows[0]['response_id']+'-native')]))
    monkeypatch.setattr(qual, 'verify_evidence', lambda *a:qual.require(False,'missing real oracle'))
    monkeypatch.setattr(qual, 'wait_released', lambda *a:dict(archive=dict(cpu_authority_bytes=0)))
    with pytest.raises(ValueError, match='missing real oracle'):
        qual.run(args)
    result = json.loads((args.output/'results.json').read_text())
    assert len(set(released)) == 3 and result['status'] == 'failed' and result['all_owned_sessions_released']
