"""Keep the output triage strict and complete even when its control diverges."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_session_swap_outputs as diagnose
from test_qualify_session_swaps import worker_report
from test_stress_request_swaps import snapshot


def write_source(path):
    workload = dict(kind='session_workload', prompt_ids=[3, 4, 8, 9], output_tokens=2, seed=123)
    reference = dict(kind='completion', base_id='source-reference-cold', packet=dict(
        id='cmpl-source-reference-cold', choices=[dict(token_ids=[1, 2], finish_reason='length')],
        usage=dict(completion_tokens=2)))
    path.write_text('\n'.join(json.dumps(row) for row in (workload, reference)) + '\n')
    return workload, reference


def test_replay_preserves_exact_prompt_sampling_and_aligned_query_plan(tmp_path):
    source = tmp_path / 'events.jsonl'
    workload, reference = write_source(source)
    assert diagnose.source_workload(source) == (workload, reference)
    body = diagnose.request_body(workload, 'same-model', [8, 9], 2)
    assert body['prompt'] is workload['prompt_ids']
    assert body['seed'] == 123 and body['max_tokens'] == 2 and body['temperature'] == 0
    assert body['kvmem_query_span'] == [2, 4]
    assert body['kvmem_mandatory_spans'] == [dict(begin=2, end=4, kind='query')]
    with pytest.raises(ValueError, match='tokenizer'):
        diagnose.request_body(workload, 'different-tokenizer', [8, 10], 2)


@pytest.mark.parametrize('mutate', [
    lambda row: row['packet'].update(id='cmpl-unrelated'),
    lambda row: row['packet']['choices'][0].update(finish_reason='stop'),
    lambda row: row['packet']['choices'][0].update(token_ids=[1]),
    lambda row: row['packet']['usage'].update(completion_tokens=1),
])
def test_incomplete_or_wrong_identity_source_is_rejected(tmp_path, mutate):
    path = tmp_path / 'events.jsonl'
    workload, reference = write_source(path)
    mutate(reference)
    path.write_text('\n'.join(json.dumps(row) for row in (workload, reference)) + '\n')
    with pytest.raises(ValueError, match='exact, complete'):
        diagnose.source_workload(path)


def test_first_difference_includes_truncated_or_extended_sequences():
    assert diagnose.token_comparison([1, 2], [1])['first_difference'] == 1
    assert diagnose.token_comparison([1], [1, 2])['first_difference'] == 1
    assert diagnose.token_comparison([1, 2], [1, 3])['first_difference'] == 1
    assert diagnose.token_comparison([1, 2], [1, 2])['first_difference'] is None
    comparisons = diagnose.compare_outputs([1, 2], [[1, 2], [1, 9]], [[1, 2]] * 4)
    assert not diagnose.all_exact(comparisons)
    assert comparisons['concurrent_against_isolated'][1][0]['first_difference'] == 1


@pytest.mark.parametrize('http_failure', [False, True])
def test_all_concurrent_peers_finish_and_every_namespace_releases_on_difference_or_error(tmp_path, monkeypatch, http_failure):
    source = tmp_path / 'source.jsonl'
    write_source(source)
    probe = tmp_path / 'probe'; probe.mkdir()
    report = worker_report()
    (probe / 'archive_resources.json').write_text(json.dumps(report))
    (probe / 'archive.json').write_text(json.dumps(dict(memory_config=dict(block_tokens=2))))
    state = snapshot()
    state['request_swaps'].update(dict.fromkeys(
        ['capture_submissions', 'restore_submissions', 'pauses', 'restores', 'drops'], 0))
    state['request_swaps']['recent_transitions'] = []
    monkeypatch.setattr(diagnose, 'load_snapshot', lambda path: state)
    calls, released, completed = [], [], []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'tokens': [8, 9]}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return Response()
    monkeypatch.setattr(diagnose.httpx, 'AsyncClient', Client)
    async def completion(client, url, body, base, evidence):
        if body.get('kv_transfer_params'):
            released.append(body['session_id']); return [1]
        calls.append(base)
        return [1, 9] if base.endswith('-isolated-1') else [1, 2]
    async def stream(client, url, body, result, emit):
        await asyncio.sleep(.001)
        calls.append(result.base_id)
        result.token_ids = [1, 2]; result.completed = True; result.finish_reason = 'length'
        completed.append(result.base_id)
        if http_failure and result.base_id.endswith('-concurrent-0'):
            raise ValueError('injected HTTP failure')
    async def settled(*args, **kwargs):
        return state, {'completed': [dict(request_id='cmpl-' + base + '-0',
            checkpoint_kind='M', checkpoint_tokens=5) for base in calls]}
    async def drained(*args, **kwargs): return state
    async def worker(*args, **kwargs): return report
    monkeypatch.setattr(diagnose, 'completion', completion)
    monkeypatch.setattr(diagnose, 'stream_request', stream)
    monkeypatch.setattr(diagnose, 'settled', settled)
    monkeypatch.setattr(diagnose, 'wait_drained', drained)
    monkeypatch.setattr(diagnose, 'wait_worker_drained', worker)
    args = type('Args', (), dict(probe=probe, url='http://unused', model='same', timeout=2,
                                 source_events=source))()
    evidence = diagnose.Evidence(tmp_path / 'result')
    try: result = asyncio.run(diagnose.run(args, evidence))
    finally: evidence.file.close()
    assert len(calls) == 6 and len(completed) == 4
    assert len(released) == len(set(released)) == 6
    if http_failure:
        assert result is None
        saved = json.loads((evidence.path / 'result.json').read_text())
        assert saved['status'] == 'failed' and len(saved['released_namespaces']) == 6
        assert saved['no_swaps_observed']
        assert not saved['comparisons']['isolated_repeat']['exact']
    else:
        assert result['status'] == 'failed'
        assert result['no_swaps_observed'] and not result['comparisons']['isolated_repeat']['exact']
        assert len(result['terminals']) == 6
