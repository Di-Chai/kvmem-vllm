import asyncio
import json

import pytest
httpx = pytest.importorskip('httpx')

from vllm_kvmem.elastic_stream import generate


def event(choices, usage=None):
    return 'data: ' + json.dumps({'id': 'test-response', 'created': 1,
        'choices': choices, 'usage': usage}) + '\n\n'


def test_restore_credit_released_once_at_first_accepted_token_and_usage_verified():
    async def scenario():
        calls = []
        text = event([{'index': 0, 'token_ids': [], 'text': ''}])
        text += event([{'index': 0, 'token_ids': [7, 8], 'text': 'result', 'finish_reason': 'length'}])
        text += event([], {'prompt_tokens': 2, 'completion_tokens': 2})
        text += 'data: [DONE]\n\n'
        async with httpx.AsyncClient(base_url='http://native', transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=text))) as client:
            result = await generate(client, {'model': 'native', 'prompt': [1, 2], 'max_tokens': 2}, lambda: calls.append('first'))
        assert calls == ['first']
        assert result['choices'][0]['token_ids'] == [7, 8]
        assert result['usage']['completion_tokens'] == 2
        assert result['metrics']['measurement_source'] == 'gateway_native_SSE_first_accepted_token'
    asyncio.run(scenario())


@pytest.mark.parametrize('tail', ['', 'data: [DONE]\n\n'])
def test_unverified_stream_never_becomes_accepted_history(tail):
    async def scenario():
        text = event([{'index': 0, 'token_ids': [7], 'text': 'result', 'finish_reason': 'stop'}]) + tail
        async with httpx.AsyncClient(base_url='http://native', transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=text))) as client:
            with pytest.raises(ValueError):
                await generate(client, {'model': 'native', 'prompt': [1], 'max_tokens': 2}, lambda: None)
    asyncio.run(scenario())


def test_long_prompt_id_echo_fits_bounded_stream_envelope():
    async def scenario():
        prompt = [12345] * 200000
        initial = {'id': 'test-response', 'created': 1, 'choices': [{'index': 0,
            'prompt_token_ids': prompt, 'token_ids': [7], 'text': 'ok', 'finish_reason': 'length'}]}
        text = 'data: ' + json.dumps(initial) + '\n\n'
        assert len(text) > 1024**2
        text += event([], {'prompt_tokens': len(prompt), 'completion_tokens': 1}) + 'data: [DONE]\n\n'
        async with httpx.AsyncClient(base_url='http://native', transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=text))) as client:
            result = await generate(client, {'model': 'native', 'prompt': prompt, 'max_tokens': 1}, lambda: None)
        assert result['choices'][0]['token_ids'] == [7]
    asyncio.run(scenario())


def test_unrelated_oversized_event_is_still_rejected():
    async def scenario():
        text = 'data: ' + json.dumps({'unexpected': 'x' * (1024**2 + 1024)}) + '\n\n'
        async with httpx.AsyncClient(base_url='http://native', transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=text))) as client:
            with pytest.raises(ValueError, match='bounded parser capacity'):
                await generate(client, {'model': 'native', 'prompt': [1], 'max_tokens': 1}, lambda: None)
    asyncio.run(scenario())
