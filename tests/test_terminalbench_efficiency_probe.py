import asyncio
import json

import httpx
import pytest

from scripts import terminalbench_efficiency_probe as probe


def fixture(tmp_path):
    path = tmp_path/'fixture.json'
    probe.save(path, dict(scope='test', owners=[dict(task='task', prefix=[1, 2], query=[0, 1],
        reasoning_tokens=[3, 4], initial_tail=[6, 7, 8], generation_begin=7,
        additions=[dict(head=[7], tail=[6, 7, 8], generation_begin=3,
            spans=[dict(begin=0, end=1, role='user'), dict(begin=1, end=2, role='assistant'),
                   dict(begin=2, end=3, role='user')])],
        initial_spans=[dict(begin=0, end=2, role='user'), dict(begin=2, end=6, role='assistant'),
                       dict(begin=6, end=7, role='user')])], turns=2, initial_tokens=8,
        history_step=4, output_tokens=3, output_token=5, eos_token=6,
        expected_final_history=20))
    return path


def test_probe_uses_real_eos_and_exact_prior_acceptance_for_every_continuation(tmp_path, monkeypatch):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if body.get('kv_transfer_params'):
            return httpx.Response(200, json={'released': True})
        return httpx.Response(200, json=dict(id=f'cmpl-{len(calls)}',
            choices=[dict(token_ids=body['allowed_token_ids']*body['max_tokens'])]))

    original = httpx.AsyncClient
    monkeypatch.setattr(probe.httpx, 'AsyncClient',
                        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    output = tmp_path/'run'
    asyncio.run(probe.exercise('http://model', 'model', fixture(tmp_path), output))
    summary = probe.read(output/'summary.json')
    assert summary['passed'] and summary['accepted_output_tokens'] == 8
    assert summary['completed_requests'] == 4
    assert [b['max_tokens'] for b in calls[:4]] == [3, 1, 3, 1]
    assert calls[1]['prompt'] == calls[0]['prompt']+[5]*3
    assert calls[2]['prompt'] == calls[1]['prompt']+[6, 7, 6, 7, 8]
    assert calls[3]['prompt'] == calls[2]['prompt']+[5]*3
    assert {tuple(b['kvmem_query_span']) for b in calls[:4]} == {(0, 1)}
    assert len({b['kvmem_query_id'] for b in calls[:4]}) == 1
    assert calls[1]['kvmem_message_spans'] == calls[0]['kvmem_message_spans']
    assert calls[2]['kvmem_message_spans'][3] == dict(begin=7, end=12, role='assistant')
    assert calls[3]['kvmem_message_spans'] == calls[2]['kvmem_message_spans']
    assert calls[1]['ignore_eos'] is False
    assert calls[-1]['kv_transfer_params']['kvmem_release_session']
    # Use the production checkpoint validator against every actual continuation.
    # The last generated ID is not forward-consumed at a length-capped stop.
    from vllm_kvmem.block_graph import _merge_spans
    saved, known, checkpoint = (), (), 0
    for body in calls[:4]:
        saved, known = _merge_spans(saved, known, body['kvmem_message_spans'], checkpoint)
        checkpoint = len(body['prompt']) + body['max_tokens'] - 1
    with pytest.raises(ValueError, match='Conflicting message boundaries'):
        _merge_spans(((2, 8, 'assistant'),), (True,), ((2, 11, 'assistant'),), 10)


def test_probe_rejects_changed_output_and_still_releases_its_session(tmp_path, monkeypatch):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if body.get('kv_transfer_params'):
            return httpx.Response(200, json={'released': True})
        return httpx.Response(200, json=dict(id='cmpl-bad', choices=[dict(token_ids=[99])]))

    original = httpx.AsyncClient
    monkeypatch.setattr(probe.httpx, 'AsyncClient',
                        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    output = tmp_path/'run'
    with pytest.raises(ValueError, match='fixed-output'):
        asyncio.run(probe.exercise('http://model', 'model', fixture(tmp_path), output))
    assert not probe.read(output/'summary.json')['passed']
    assert calls[-1]['kv_transfer_params']['kvmem_release_session']


def test_transport_disconnect_records_request_identity_and_never_retries(tmp_path, monkeypatch):
    calls, client_options = [], []

    def handle(request):
        assert request.headers['connection'] == 'close'
        body = json.loads(request.content)
        calls.append(body)
        if body.get('kv_transfer_params'):
            return httpx.Response(200, json={'released': True})
        try:
            raise ConnectionResetError(104, 'Connection reset by peer')
        except ConnectionResetError as cause:
            raise httpx.ReadError('socket reset', request=request) from cause

    original = httpx.AsyncClient
    def client(**kwargs):
        client_options.append(kwargs)
        return original(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(probe.httpx, 'AsyncClient', client)
    output = tmp_path/'run'
    with pytest.raises(httpx.ReadError):
        asyncio.run(probe.exercise('http://model', 'model', fixture(tmp_path), output))
    assert len(calls) == 2  # One generation attempt, then explicit release only.
    assert client_options[0]['limits'].max_keepalive_connections == 0
    failure = probe.read(output/'error-task-1-thinking.json')
    assert failure['task'] == 'task' and failure['turn'] == 1 and failure['part'] == 'thinking'
    assert not failure['cancelled'] and not failure['retried']
    assert [r['type'] for r in failure['exception_chain']] == ['ReadError', 'ConnectionResetError']
    assert 'ConnectionResetError' in failure['traceback']
    assert not probe.read(output/'summary.json')['passed']
