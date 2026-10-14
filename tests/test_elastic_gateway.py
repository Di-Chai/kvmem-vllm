import asyncio
import json

import pytest
httpx = pytest.importorskip('httpx')
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from vllm_kvmem.elastic_gateway import create_app


def configuration(tmp_path):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'a', 'api_key': 'key-a'}, {'id': 'b', 'api_key': 'key-b'}]}))
    geometry = tmp_path / 'geometry.json'
    geometry.write_text(json.dumps({'groups': [{'block_tokens': 10, 'page_bytes': 10,
        'recurrent': False, 'speculative_blocks': 0}], 'capacity_bytes': 1000, 'cpu_offload_bytes': 2000}))
    return {'geometry_file': str(geometry), 'active_bytes': 100, 'token_history_bytes': 1000,
        'max_sessions': 4, 'session_ttl_seconds': 10, 'max_inflight': 2, 'max_queue': 2,
        'users_file': str(users), 'backend_url': 'http://native', 'turn_timeout_seconds': 10,
        'max_body_bytes': 1000, 'max_output_tokens': 10, 'queue_timeout_seconds': 5,
        'max_model_len': 100, 'served_model_name': 'native'}


def test_gateway_owns_namespace_raw_history_and_releases_active_budget(tmp_path):
    observed = []

    def upstream(request):
        data = json.loads(request.content)
        observed.append(data)
        return httpx.Response(200, json={'choices': [{'token_ids': [8], 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 1, 'prompt_tokens': len(data['prompt'])}})

    app = create_app(configuration(tmp_path))
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url='http://native')
        path = '/v1/kvmem/sessions/same/turn'
        assert client.post(path, json={'append_token_ids': [1]}).status_code == 401
        a, b = {'Authorization': 'Bearer key-a'}, {'Authorization': 'Bearer key-b'}
        r = client.post(path, headers=a, json={'append_token_ids': [1], 'max_tokens': 2, 'waiting_for_tool': True})
        assert r.status_code == 200 and r.json()['version'] == 1
        assert client.post(path, headers=a, json={'append_token_ids': [2], 'max_tokens': 2}).status_code == 409
        assert client.post(path, headers=a, json={'version': 1, 'append_token_ids': [2], 'max_tokens': 2}).status_code == 200
        assert client.post(path, headers=b, json={'append_token_ids': [3], 'max_tokens': 2}).status_code == 200
        assert [x['prompt'] for x in observed] == [[1], [1, 8, 2], [3]]
        assert observed[0]['cache_salt'] != observed[2]['cache_salt']
        state = client.get('/v1/kvmem/status', headers=b).json()
        assert state['active_reserved_bytes'] == state['active_requests'] == state['queued_requests'] == 0
        assert state['sessions'] == [{'session_id': 'same', 'version': 1, 'history_tokens': 2,
            'running': False, 'waiting_for_tool': False}]


def test_backend_failure_and_invalid_usage_preserve_history_and_credits(tmp_path):
    replies = [httpx.Response(503, json={}), httpx.Response(200,
        json={'choices': [{'token_ids': [1, 2]}], 'usage': {'completion_tokens': 1, 'prompt_tokens': 1}})]
    app = create_app(configuration(tmp_path))
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: replies.pop(0)), base_url='http://native')
        headers = {'Authorization': 'Bearer key-a'}
        body = {'append_token_ids': [1], 'max_tokens': 2}
        assert client.post('/v1/kvmem/sessions/s/turn', headers=headers, json=body).status_code == 502
        assert client.post('/v1/kvmem/sessions/s/turn', headers=headers, json=body).status_code == 400
        state = client.get('/v1/kvmem/status', headers=headers).json()
        assert state['sessions'] == [] and state['active_reserved_bytes'] == 0
        assert state['counters']['aborted'] == 2


def test_meank_gateway_passes_complete_history_and_stable_tenant_session_to_engine(tmp_path):
    cfg = configuration(tmp_path)
    cfg.update(context_mode='meank', active_context_tokens=4, gen_budget_tokens=4)
    observed = []
    def upstream(request):
        data = json.loads(request.content); observed.append(data)
        return httpx.Response(200, json={'choices': [{'token_ids': [8], 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 1, 'prompt_tokens': len(data['prompt'])}})
    app = create_app(cfg)
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url='http://native')
        path = '/v1/kvmem/sessions/same/turn'
        headers = {'Authorization': 'Bearer key-a'}
        response = client.post(path, headers=headers, json={'append_token_ids': list(range(16)), 'max_tokens': 2})
        assert response.status_code == 200
        assert response.json()['input_context']['mode'] == 'meank_historical_kv'
        response = client.post(path, headers=headers, json={'version': 1, 'append_token_ids': [20], 'max_tokens': 2})
        assert response.status_code == 200
        assert observed[1]['prompt'] == list(range(16)) + [8, 20]
        assert observed[0]['session_id'] == observed[1]['session_id'] == 'same'
        assert observed[0]['cache_salt'] == observed[1]['cache_salt']
        response = client.post(path, headers=headers, json={'version': 2, 'append_token_ids': [21],
            'active_ranges': [[0, 2]], 'max_tokens': 2})
        assert response.status_code == 400 and len(observed) == 2


def test_explicit_query_is_forwarded_and_invalid_force_preserves_history(tmp_path):
    cfg=configuration(tmp_path)
    cfg.update(context_mode='meank',request_planning=True,active_context_tokens=128,
        gen_budget_tokens=512,max_model_len=1024,active_bytes=1000,token_history_bytes=8000,max_body_bytes=10000)
    seen=[]
    def upstream(request):
        payload=json.loads(request.content);seen.append(payload)
        return httpx.Response(200,json={'choices':[{'token_ids':[8],'finish_reason':'stop'}],
            'usage':{'completion_tokens':1,'prompt_tokens':len(payload['prompt'])}})
    with TestClient(create_app(cfg)) as client:
        client.app.state.client=httpx.AsyncClient(transport=httpx.MockTransport(upstream),base_url='http://native')
        path='/v1/kvmem/sessions/query/turn';headers={'Authorization':'Bearer key-a'}
        result=client.post(path,headers=headers,json={'append_token_ids':list(range(300)),
            'max_tokens':2,'kvmem_query_span':[133,263],'kvmem_reselect':'force'})
        assert result.status_code==200
        assert seen[0]['kv_transfer_params']['kvmem_request_plan']=={
            'kvmem_query_span':[133,263],'kvmem_reselect':'force'}
        result=client.post(path,headers=headers,json={'version':1,'append_token_ids':[7],
            'max_tokens':2,'kvmem_reselect':'force'})
        assert result.status_code==400 and len(seen)==1
        state=client.get('/v1/kvmem/status',headers=headers).json()
        assert state['sessions'][0]['version']==1 and state['sessions'][0]['history_tokens']==301
        assert state['active_reserved_bytes']==0


class _Request:
    """Drive real ASGI receive/disconnect events without a network server."""
    def __init__(self, app, path, *, user='a', body=None, method='POST'):
        self.incoming, self.messages = asyncio.Queue(), []
        self.incoming.put_nowait({'type': 'http.request',
            'body': json.dumps(body or {}).encode(), 'more_body': False})
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
            'method': method, 'scheme': 'http', 'path': path, 'query_string': b'',
            'headers': [(b'authorization', f'Bearer key-{user}'.encode()),
                        (b'content-type', b'application/json')]}
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.send))

    async def send(self, message):
        self.messages.append(message)

    async def finish(self):
        await asyncio.wait_for(self.task, 2)
        return self

    @property
    def status(self):
        return next(m['status'] for m in self.messages if m['type'] == 'http.response.start')

    @property
    def data(self):
        return json.loads(b''.join(m.get('body', b'') for m in self.messages))


async def _status(app, user='a'):
    return (await _Request(app, '/v1/kvmem/status', user=user, method='GET').finish()).data


def _sse(ids, *, finished=False, prompt_tokens=1):
    data = {'id': 'test-stream', 'created': 1, 'choices': [{'index': 0,
        'text': 'x' if ids else '', 'token_ids': ids,
        'finish_reason': 'length' if finished else None}]}
    if finished:
        data['usage'] = {'completion_tokens': len(ids), 'prompt_tokens': prompt_tokens}
    return f'data: {json.dumps(data)}\n\n'.encode()


def test_progressing_stream_obeys_absolute_deadline_and_drains_before_releasing(tmp_path):
    cfg = configuration(tmp_path)
    cfg.update(turn_timeout_seconds=.15, max_restore_inflight=1)

    async def run():
        closing, allow_close = asyncio.Event(), asyncio.Event()
        produced = []
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(10):
                    await asyncio.sleep(.03)
                    produced.append(8)
                    yield _sse([8])
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                closing.set()
                await allow_close.wait()

        def upstream(request):
            payload = json.loads(request.content)
            if payload['prompt'] == [1]:
                return httpx.Response(200, content=_sse([8], finished=True) + b'data: [DONE]\n\n')
            return httpx.Response(200, stream=Stream())

        app = create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url='http://native') as native:
            app.state.client = native
            path = '/v1/kvmem/sessions/s/turn'
            first = await _Request(app, path, body={'append_token_ids': [1], 'max_tokens': 2}).finish()
            assert first.status == 200
            turn = _Request(app, path, body={'version': 1, 'append_token_ids': [2], 'max_tokens': 10})
            try:
                await asyncio.wait_for(closing.wait(), 2)
                assert 1 <= len(produced) < 10  # Tokens continue arriving, but wall time still expires.
                state = await _status(app)
                assert state['active_requests'] == 1 and state['active_reserved_bytes'] > 0
                assert state['restore_reserved_bytes'] == 0  # First output released restore credit.
                assert state['sessions'][0]['version'] == 1 and state['sessions'][0]['running']
            finally:
                allow_close.set()
            await turn.finish()
            assert turn.status == 504
            state = await _status(app)
            assert state['active_requests'] == state['active_reserved_bytes'] == 0
            assert state['sessions'] == [{'session_id': 's', 'version': 1, 'history_tokens': 2,
                'running': False, 'waiting_for_tool': False}]
    asyncio.run(run())


def test_disconnected_queued_turn_releases_history_and_queue_before_admission(tmp_path):
    cfg = configuration(tmp_path)
    cfg.update(max_inflight=1, token_history_bytes=24)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def upstream(request):
            payload = json.loads(request.content)
            seen.append(payload['prompt'])
            if payload['prompt'] == [1]:
                entered.set()
                await release.wait()
            return httpx.Response(200, json={'choices': [{'token_ids': [8]}],
                'usage': {'completion_tokens': 1, 'prompt_tokens': len(payload['prompt'])}})

        app = create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url='http://native') as native:
            app.state.client = native
            first = _Request(app, '/v1/kvmem/sessions/first/turn', body={'append_token_ids': [1], 'max_tokens': 2})
            await asyncio.wait_for(entered.wait(), 2)
            queued = _Request(app, '/v1/kvmem/sessions/queued/turn', user='b',
                              body={'append_token_ids': [2], 'max_tokens': 2})
            async def wait_queued():
                while (await _status(app))['queued_requests'] != 1:
                    await asyncio.sleep(0)
            try:
                await asyncio.wait_for(wait_queued(), 2)
                queued.incoming.put_nowait({'type': 'http.disconnect'})
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(queued.task), 1)
                state = await _status(app, 'b')
                assert state['sessions'] == [] and state['queued_requests'] == 0
                assert state['active_requests'] == 1 and seen == [[1]]
                # The released 12-byte history reservation can be reused while
                # the first native call still occupies the sole active slot.
                retry = _Request(app, '/v1/kvmem/sessions/queued/turn', user='b',
                                 body={'append_token_ids': [3], 'max_tokens': 2})
                await asyncio.wait_for(wait_queued(), 2)
            finally:
                release.set()
            assert (await first.finish()).status == 200
            assert (await retry.finish()).status == 200
            assert seen == [[1], [3]]
            state = await _status(app)
            assert state['active_requests'] == state['queued_requests'] == 0
    asyncio.run(run())


def test_disconnect_in_restore_queue_releases_its_active_lease_without_starting_native(tmp_path):
    cfg = configuration(tmp_path)
    cfg.update(max_restore_inflight=1)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await release.wait()
                yield _sse([8], finished=True, prompt_tokens=3)
                yield b'data: [DONE]\n\n'

        def upstream(request):
            prompt = json.loads(request.content)['prompt']
            if len(prompt) == 1:
                return httpx.Response(200, content=_sse([8], finished=True) + b'data: [DONE]\n\n')
            seen.append(prompt)
            return httpx.Response(200, stream=Stream())

        app = create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), base_url='http://native') as native:
            app.state.client = native
            path = '/v1/kvmem/sessions/s/turn'
            for user in ('a', 'b'):
                assert (await _Request(app, path, user=user,
                    body={'append_token_ids': [1], 'max_tokens': 2}).finish()).status == 200
            first = _Request(app, path, body={'version': 1, 'append_token_ids': [2], 'max_tokens': 2})
            await asyncio.wait_for(entered.wait(), 2)
            queued = _Request(app, path, user='b', body={'version': 1, 'append_token_ids': [3], 'max_tokens': 2})
            async def wait_active_pair():
                while (await _status(app))['active_requests'] != 2:
                    await asyncio.sleep(0)
            try:
                await asyncio.wait_for(wait_active_pair(), 2)
                queued.incoming.put_nowait({'type': 'http.disconnect'})
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(queued.task), 1)
                state = await _status(app, 'b')
                assert state['active_requests'] == state['restoring_requests'] == 1
                assert state['sessions'] == [{'session_id': 's', 'version': 1, 'history_tokens': 2,
                    'running': False, 'waiting_for_tool': False}]
                assert seen == [[1, 8, 2]]
                retry = _Request(app, path, user='b', body={'version': 1, 'append_token_ids': [4], 'max_tokens': 2})
                await asyncio.wait_for(wait_active_pair(), 2)
            finally:
                release.set()
            assert (await first.finish()).status == (await retry.finish()).status == 200
            assert seen == [[1, 8, 2], [1, 8, 4]]
            state = await _status(app)
            assert state['active_requests'] == state['restoring_requests'] == 0
            assert state['active_reserved_bytes'] == state['restore_reserved_bytes'] == 0
    asyncio.run(run())
