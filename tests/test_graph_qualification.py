"""Do not allow missing/changed evidence to pass the real-model graph client."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('block_graph_qualification',
    Path(__file__).parents[1] / 'scripts/qualify_block_graph.py')
qual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qual)


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def profile(arm='multi'):
    return dict(port=18000, served_model_name='test', tokenizer='/unused', max_num_seqs=2,
        max_model_len=262144, draft_method='mtp', draft_tokens=4,
        execution_mode='compiled_graph', verify_session_restore=True,
        kvmem=dict(select_budget=2048, gen_budget=4096, block_tokens=32,
            session_capacity=2, session_checkpoint_policy='pm', session_capture_terminal=True,
            request_planning=True, archive_backend='cpu', repository_gc=False,
            query_history=False, action_pressure=False, guided_query_tokens=0,
            graph_mode=arm if arm in ('collect', 'locality', 'attention', 'multi') else 'off',
            action_attention=True, prefill_attention=True, action_scope='non-thinking',
            action_fraction=0.5 if arm == 'legacy' else 0.0))


@pytest.mark.parametrize('arm', qual.ARMS)
def test_all_six_arms_hold_generation_and_context_contract_fixed(arm):
    current = profile(arm)
    before = deepcopy(current)
    qual.validate_profile(current, arm, 2, 257, 129)
    assert current == before


@pytest.mark.parametrize('key,value', [('query_history', True), ('action_pressure', True),
    ('guided_query_tokens', 32), ('action_fraction', 0.5), ('repository_gc', True),
    ('session_capacity', 1), ('graph_mode', 'off'), ('block_tokens', 128),
    ('action_scope', 'tool-call'), ('archive_backend', 'nvme')])
def test_changed_or_missing_controls_cannot_be_called_a_graph_comparison(key, value):
    current = profile()
    current['kvmem'][key] = value
    with pytest.raises(ValueError):
        qual.validate_profile(current, 'multi', 2, 257, 129)


def test_fixture_is_identical_between_runs_and_immutable_hash_catches_edits():
    one = qual.build_fixture(Tokenizer(), prompt_tokens=12288, concurrency=2)
    two = qual.build_fixture(Tokenizer(), prompt_tokens=12288, concurrency=2)
    assert one == two
    qual.validate_fixture(one, profile(), 2, 257)
    assert one['agents'][0]['first_input_sha256'] != one['agents'][1]['first_input_sha256']
    one['agents'][0]['initial_tokens'][0] += 1
    with pytest.raises(ValueError, match='Fixture hash'):
        qual.validate_fixture(one, profile(), 2, 257)


def test_continuation_uses_exact_output_ids_and_edit_preserves_only_verified_prefix():
    fixture = qual.build_fixture(Tokenizer(), prompt_tokens=12288, concurrency=2)
    agent = fixture['agents'][0]
    accepted = [900001, 900002, 900003]
    prompt, query, spans, before = qual.next_turn(agent, agent['initial_tokens'], accepted)
    assert prompt[:len(agent['initial_tokens'])+3] == agent['initial_tokens']+accepted
    assert prompt[query[0]:query[1]] == agent['second_question']
    edited, edit_query, edit_spans = qual.edited_turn(agent, before, spans)
    assert prompt[:query[0]] == edited[:edit_query[0]]
    assert edited[edit_query[0]:edit_query[1]] == agent['edited_question']
    assert spans[:-1] == edit_spans[:-1]
    assert 900004 not in edited  # A later second-turn output is never inherited.
    assert spans[-2]['role'] == 'tool' and spans[-1]['role'] == 'user'
    for sequence in (spans, edit_spans):
        assert all(a['end'] <= b['begin'] for a, b in zip(sequence, sequence[1:]))


def write_events(path, values):
    path.write_text(''.join(json.dumps(row)+'\n' for row in values))


def test_native_lifecycle_gate_requires_each_query_and_both_restore_byte_oracles(tmp_path):
    rows = [dict(response_id='first', agent=0, turn=0, query_span=[100, 120]),
            dict(response_id='second', agent=0, turn=1, query_span=[140, 160])]
    refreshes = [dict(request_id=r['response_id']+'-0-abcdef', selection='query',
        query_span=r['query_span'], selected_tokens=2048) for r in rows]
    write_events(tmp_path / 'bounded_refresh.jsonl', refreshes)
    target = dict(request_id='second-0-abcdef', byte_equal=True,
        working_attention_layers=17, mtp_attention_layers=1)
    state = dict(request_id='second-0-abcdef', recurrent_byte_equal=True)
    write_events(tmp_path / 'verified_restores.jsonl', [target])
    write_events(tmp_path / 'verified_state_restores.jsonl', [state])
    assert len(qual.event_evidence(tmp_path, rows, profile())) == 2
    write_events(tmp_path / 'verified_restores.jsonl', [{**target, 'mtp_attention_layers': 0}])
    with pytest.raises(ValueError, match='MTP'):
        qual.event_evidence(tmp_path, rows, profile())
    write_events(tmp_path / 'verified_restores.jsonl', [target])
    write_events(tmp_path / 'bounded_refresh.jsonl', refreshes+refreshes[:1])
    with pytest.raises(ValueError, match='exactly once'):
        qual.event_evidence(tmp_path, rows, profile())


def test_step_summary_does_not_call_mixed_prefill_or_speculative_counts_pure_decode(tmp_path):
    write_events(tmp_path / 'model_step_latency.jsonl', [
        dict(started_at=10, prefill_tokens=100, decode_requests=0, model_ms=20),
        dict(started_at=11, prefill_tokens=300, decode_requests=2, model_ms=60),
        dict(started_at=12, prefill_tokens=0, decode_requests=2, model_ms=15),
        dict(started_at=20, prefill_tokens=500, decode_requests=0, model_ms=20)])
    result = qual.summarize_steps(tmp_path, 9, 15)
    assert result['pure_prefill_model_tokens_per_second'] == 5000
    assert result['mixed_steps'] == result['pure_decode_steps'] == 1
    assert result['pure_decode_batch_histogram'] == {'2': 1}
    assert 'decode_tokens_per_second' not in result


class Response:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, *args):
        return False


def test_stream_counts_only_accepted_returned_tokens_and_keeps_real_finish_reason(monkeypatch):
    data = [dict(id='response', choices=[dict(index=0, token_ids=[9, 10], text='ORCHID-4817')]),
            dict(id='response', choices=[dict(index=0, token_ids=[11], text='', finish_reason='stop')],
                usage=dict(completion_tokens=3))]
    chunks = [b'data: '+json.dumps(row).encode()+b'\n' for row in data]+[b'data: [DONE]\n']
    bodies = []
    class Opener:
        def open(self, request, timeout):
            bodies.append(json.loads(request.data))
            return Response(chunks)
    monkeypatch.setattr(qual.urllib.request, 'build_opener', lambda *args: Opener())
    row = qual.stream_request(profile(), 'owned', [1, 2], 5, 0, query=[0, 1])
    assert row['output_ids'] == [9, 10, 11] and row['finish_reason'] == 'stop'
    assert bodies[0]['ignore_eos'] is False and bodies[0]['kvmem_reselect'] == 'force'
    assert row['generation_period_tokens_per_second'] > 0
    data[-1]['usage']['completion_tokens'] = 4
    chunks[:] = [b'data: '+json.dumps(r).encode()+b'\n' for r in data]+[b'data: [DONE]\n']
    with pytest.raises(ValueError, match='accounting mismatch'):
        qual.stream_request(profile(), 'owned', [1, 2], 5, 0, query=[0, 1])


def test_live_jsonl_reader_ignores_only_incomplete_final_record(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('{"revision": 1}\n{"revi')
    assert qual.records(path) == [{'revision': 1}]
    path.write_text('{"revision": 1}\ninvalid\n')
    with pytest.raises(json.JSONDecodeError):
        qual.records(path)


def graph_evidence():
    rows, queries, actions, completions = [], [], [], []
    for turn, samples in enumerate((8, 9, 7)):
        row = dict(agent=0, turn=turn, response_id=f'turn-{turn}',
            input_tokens=12000+turn*300, output_ids=[1]*129, expected_output_samples=samples)
        rows.append(row)
        native = row['response_id']+'-0-native'
        end = row['input_tokens']+128
        graph = dict(mode='multi', accepted_through=row['input_tokens']-32,
            selected_tokens=4096, added_blocks=2 if turn == 1 else 0)
        queries.append(dict(agent=0, turn=turn, graph=[graph]))
        actions.append(dict(request_id=native, samples=samples, block_graph=dict(
            accepted_through=end, completed_through=end//32*32, total_bytes=1000,
            global_bytes=2000, output_samples=8+(samples if turn else 0),
            prefill_samples=200, observed_sources=50)))
        completions.append(dict(request_id=native, rejected_tokens=3,
            terminal_prefix=dict(history_end=end)))
    return rows, queries, actions, completions


def test_real_graph_checks_prove_accepted_samples_and_exclude_edited_future_branch():
    rows, queries, actions, completed = graph_evidence()
    result = qual.assert_real_graph_checks(rows, queries, actions, completed, profile())
    assert result['complete'] and result['selection_changed_observed'] and result['rejected_tokens'] == 9
    # Retaining the discarded second response would otherwise be an attractive
    # false positive: final graph is nonempty and its token boundary is correct.
    actions[-1]['block_graph']['output_samples'] += rows[1]['expected_output_samples']
    with pytest.raises(ValueError, match='edited-branch'):
        qual.assert_real_graph_checks(rows, queries, actions, completed, profile())


def test_unobserved_speculative_rejection_or_graph_ranking_change_is_not_a_pass():
    rows, queries, actions, completed = graph_evidence()
    for proof in queries:
        proof['graph'][0]['added_blocks'] = 0
    result = qual.assert_real_graph_checks(rows, queries, actions, completed, profile())
    assert not result['complete'] and not result['selection_changed_observed']
    for event in completed:
        event['rejected_tokens'] = 0
    result = qual.assert_real_graph_checks(rows, queries, actions, completed, profile())
    assert not result['complete'] and not result['mtp_rejections_observed']


@pytest.mark.parametrize('mutation,match', [
    (lambda a: a[-1].update(samples=99), 'parser oracle'),
    (lambda a: a[-1]['block_graph'].update(accepted_through=13000), 'watermark'),
    (lambda a: a[-1]['block_graph'].update(prefill_samples=0), 'prefill and output'),
    (lambda a: a[-1]['block_graph'].update(total_bytes=2**40), 'capacity'),
])
def test_missing_or_invalid_real_graph_evidence_fails(mutation, match):
    rows, queries, actions, completed = graph_evidence()
    mutation(actions)
    with pytest.raises(ValueError, match=match):
        qual.assert_real_graph_checks(rows, queries, actions, completed, profile())


def test_collect_baseline_comparison_requires_identical_model_input_and_selected_ranges(tmp_path):
    directories = [tmp_path/'baseline', tmp_path/'collect']
    for arm, directory in zip(('baseline', 'collect'), directories):
        directory.mkdir()
        qual.save(directory/'profile.json', profile(arm))
        qual.save(directory/'results.json', dict(arm=arm, fixture_sha256='fixed-input', evidence=[
            dict(agent=0, turn=0, query=dict(selected_history_ranges=[[0, 32], [512, 32]])),
            # Different continuations are intentionally excluded from this assertion.
            dict(agent=0, turn=1, query=dict(selected_history_ranges=[[0, 32], [1024 if arm == 'baseline' else 2048, 32]]))]))
    assert qual.compare_collect_baseline(*directories)['passed']
    changed = json.loads((directories[1]/'results.json').read_text())
    changed['evidence'][0]['query']['selected_history_ranges'][1][0] = 768
    qual.save(directories[1]/'results.json', changed)
    with pytest.raises(ValueError, match='Collect changed'):
        qual.compare_collect_baseline(*directories)


def test_client_waits_for_fenced_terminal_publication_after_http_completion(tmp_path, monkeypatch):
    rows = [dict(response_id='first'), dict(response_id='second')]
    archive = dict(completed=[dict(request_id='first-0-native')])
    qual.save(tmp_path/'archive.json', archive)
    waited = []
    def retire(seconds):
        waited.append(seconds)
        archive['completed'].append(dict(request_id='second-0-native'))
        qual.save(tmp_path/'archive.json', archive)
    monkeypatch.setattr(qual.time, 'sleep', retire)
    result = qual.wait_terminal_publication(tmp_path, rows, timeout=1)
    assert waited == [0.1] and len(result['completed']) == 2
