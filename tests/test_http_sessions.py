import asyncio

import pytest

from vllm_kvmem.http_sessions import validate_session_payload, validate_prompt_session


def test_native_session_precedence_and_namespace_limits():
    validate_session_payload({'session_id': 'body'}, 'h'*1025)
    validate_session_payload({'vllm_xargs': {'session_id': 'x'*1025}}, 'header')
    validate_session_payload({'session_id': 's'*1024, 'cache_salt': 't'*1024}, None)
    validate_session_payload({'cache_salt': 't'*1025}, None)  # No session lookup.
    for payload, header in [({'session_id': 'x'*1025}, None),
                            ({}, 'x'*1025),
                            ({'vllm_xargs': {'session_id': 'x'*1025}}, None),
                            ({'session_id': 's', 'cache_salt': 'x'*1025}, None)]:
        with pytest.raises(ValueError, match='1024'):
            validate_session_payload(payload, header)


def test_native_http_middleware_rejects_before_dispatch_and_preserves_valid_streams():
    httpx = pytest.importorskip('httpx')
    pytest.importorskip('fastapi')
    from fastapi import FastAPI
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    received = []

    async def endpoint(request):
        received.append(await request.body())
        async def stream():
            yield b'data: first\n\n'
            await asyncio.sleep(0)
            yield b'data: [DONE]\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream')

    app = FastAPI(routes=[Route('/v1/chat/completions', endpoint, methods=['POST'])])
    app.middleware('http')(validate_prompt_session)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            raw = b'{ "session_id": "valid", "stream": true, "messages": [] }'
            valid, invalid = await asyncio.gather(
                client.post('/v1/chat/completions', content=raw),
                client.post('/v1/chat/completions', json={'session_id': 'x'*1025}))
            assert invalid.status_code == 400
            assert invalid.json()['error']['code'] == 400
            assert valid.status_code == 200
            assert valid.content == b'data: first\n\ndata: [DONE]\n\n'
            assert received == [raw]
            # A subsequent request still reaches the app unchanged; malformed
            # JSON belongs to the native endpoint rather than this guard.
            malformed = b'{not json'
            result = await client.post('/v1/chat/completions', content=malformed)
            assert result.status_code == 200 and received[-1] == malformed
            result = await client.post('/v1/chat/completions', content=raw,
                                       headers={'X-Session-ID': 'ignored'*200})
            assert result.status_code == 200 and received[-1] == raw

    asyncio.run(run())
