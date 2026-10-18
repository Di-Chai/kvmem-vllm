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


@pytest.mark.parametrize('change', [
    {'session_id':None}, {'prompt':[1,2]}, {'prompt':'text'}, {'prompt':[True]},
    {'max_tokens':2}, {'max_tokens':True},
    {'kv_transfer_params':{'kvmem_release_session':'true'}},
    {'kv_transfer_params':{'kvmem_release_session':True,'kvmem_request_plan':{'kvmem_query_span':[0,1]}}},
    {'kvmem_query_span':[0,1]},
])
def test_release_rejects_ambiguous_or_noncleanup_requests(change):
    payload = dict(session_id='s', prompt=[1], max_tokens=1,
        kv_transfer_params={'kvmem_release_session':True})
    payload.update(change)
    with pytest.raises(ValueError):
        validate_session_payload(payload, None)


def test_release_flag_survives_http_planning_and_header_namespace(monkeypatch):
    import json
    import httpx
    from fastapi import FastAPI
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(
        block_tokens=32, gen_budget=32768, select_budget=65536, rolling_active_window=True)))
    async def endpoint(request):
        return JSONResponse(await request.json())
    app = FastAPI(routes=[Route('/v1/completions',endpoint,methods=['POST'])])
    app.middleware('http')(validate_prompt_session)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            payload = dict(prompt=[1],max_tokens=1,kvmem_reselect='off',
                kv_transfer_params={'kvmem_release_session':True})
            response = await client.post('/v1/completions',json=payload,headers={'X-Session-ID':'s'})
            assert response.status_code == 200
            params = response.json()['kv_transfer_params']
            assert params['kvmem_release_session'] is True
            assert params['kvmem_request_plan'] == {'kvmem_reselect':'off'}
            response = await client.post('/v1/completions',json=payload)
            assert response.status_code == 400
    asyncio.run(run())


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


def test_query_metadata_body_replay_reaches_native_endpoint(monkeypatch):
    import json
    import httpx
    from fastapi import FastAPI
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'block_tokens':128,'gen_budget':4096,'select_budget':8192}))
    async def endpoint(request):
        return JSONResponse(await request.json())
    app=FastAPI(routes=[Route('/v1/completions',endpoint,methods=['POST'])])
    app.middleware('http')(validate_prompt_session)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.post('/v1/completions',json={'prompt':list(range(4096)),
                'max_tokens':None,'kvmem_query_span':[1001,3003],'kvmem_reselect':'force'})
            assert response.status_code==200
            body=response.json()
            assert body['max_tokens']==16 and 'kvmem_query_span' not in body
            assert body['kv_transfer_params']['kvmem_request_plan']['kvmem_query_span']==[1001,3003]
    asyncio.run(run())


def test_rolling_rejects_unplanned_entries_before_shared_engine_and_keeps_token_streams(monkeypatch):
    import json
    import httpx
    from fastapi import FastAPI
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(
        block_tokens=32, gen_budget=32768, select_budget=65536, rolling_active_window=True)))
    received = []
    async def endpoint(request):
        payload = await request.json()
        received.append(payload)
        return JSONResponse(payload)
    app = FastAPI(routes=[Route(path, endpoint, methods=['POST']) for path in
        ['/v1/completions', '/v1/chat/completions', '/v1/responses']])
    app.middleware('http')(validate_prompt_session)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            for path, payload in [('/v1/completions', {'prompt':'unplanned text'}),
                                  ('/v1/chat/completions', {'messages':[]}),
                                  ('/v1/responses', {'input':'unplanned text'})]:
                response = await client.post(path, json=payload)
                assert response.status_code == 400 and 'token-ID' in response.json()['error']['message']
            assert not received
            response = await client.post('/v1/completions', json=dict(
                prompt=[1, 2, 3], session_id='valid', max_tokens=32768))
            assert response.status_code == 200 and len(received) == 1
            assert received[0]['max_tokens'] == 32768
            assert received[0]['kv_transfer_params']['kvmem_request_plan'] == {}
    asyncio.run(run())


def test_rolling_cold_headroom_uses_actual_input_and_preserves_output_limit(monkeypatch):
    import json
    from vllm_kvmem.http_sessions import normalize_query_plan
    from vllm_kvmem.request_plan import RequestCapacityError
    cfg = dict(block_tokens=16, gen_budget=32, select_budget=32)
    payload = dict(prompt=[1]*40, max_tokens=32, kvmem_prefill_window='keep_selected')
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(cfg))
    with pytest.raises(RequestCapacityError,
                       match='Complete Query and output reserve exceed active capacity'):
        normalize_query_plan(dict(payload))
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(cfg, rolling_active_window=True)))
    assert normalize_query_plan(payload) and payload['max_tokens'] == 32
