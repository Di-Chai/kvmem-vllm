import asyncio
import json

import pytest

from vllm_kvmem.multi_user import MultiUserMiddleware, validate_config


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'users.json'
    path.write_text(json.dumps({'users': [
        {'id': 'alice', 'api_key': 'alice-secret'},
        {'id': 'bob', 'api_key': 'bob-secret'},
        {'id': 'carol', 'api_key': 'carol-secret'},
    ]}))
    return {'users_file': str(path), 'max_inflight': 2, 'max_inflight_per_user': 1,
            'max_queue': 4, 'queue_timeout_seconds': 2, 'body_timeout_seconds': 30,
            'max_body_bytes': 4096}


class Request:
    """A transport that exposes body chunks, disconnects and streaming timing."""
    def __init__(self, app, payload=None, *, user='alice', path='/v1/chat/completions',
                 method='POST', headers=(), raw=None, chunks=None):
        self.incoming = asyncio.Queue()
        self.messages = []
        self.changed = asyncio.Event()
        auth = [] if user is None else [(b'authorization', f'Bearer {user}-secret'.encode())]
        self.scope = {'type': 'http', 'method': method, 'path': path,
                      'headers': auth + list(headers), 'state': {}, 'http_version': '1.1'}
        data = json.dumps(payload or {}).encode() if raw is None else raw
        if chunks is None:
            chunks = [data]
        for index, chunk in enumerate(chunks):
            self.incoming.put_nowait({'type': 'http.request', 'body': chunk,
                                     'more_body': index + 1 < len(chunks)})
        self.task = asyncio.create_task(app(self.scope, self.receive, self.send))

    async def receive(self):
        return await self.incoming.get()

    async def send(self, message):
        self.messages.append(message)
        self.changed.set()

    async def finish(self):
        await asyncio.wait_for(self.task, 2)
        return self

    async def disconnect(self):
        self.incoming.put_nowait({'type': 'http.disconnect'})
        await self.finish()

    @property
    def status(self):
        return next((m['status'] for m in self.messages if m['type'] == 'http.response.start'), None)

    @property
    def data(self):
        return b''.join(m.get('body', b'') for m in self.messages if m['type'] == 'http.response.body')


async def until(condition):
    async def check():
        while not condition():
            await asyncio.sleep(0)
    await asyncio.wait_for(check(), 2)


async def ok(scope, receive, send):
    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    await send({'type': 'http.response.body', 'body': b'ok'})


def test_browser_preflight_uses_native_cors_and_generation_still_requires_key(config):
    from starlette.middleware.cors import CORSMiddleware
    app = MultiUserMiddleware(CORSMiddleware(ok,
        allow_origins=['https://client.example'], allow_methods=['POST'],
        allow_headers=['Authorization', 'Content-Type']), config)

    async def run():
        preflight = await Request(app, method='OPTIONS', user=None, headers=[
            (b'origin', b'https://client.example'),
            (b'access-control-request-method', b'POST'),
            (b'access-control-request-headers', b'authorization,content-type')]).finish()
        assert preflight.status == 200
        response_headers = dict(preflight.messages[0]['headers'])
        assert response_headers[b'access-control-allow-origin'] == b'https://client.example'
        assert (await Request(app, user=None).finish()).status == 401
        assert (await Request(app).finish()).status == 200
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_config_validation_and_secret_free_errors(config, monkeypatch):
    canonical = validate_config(config)
    assert canonical == config
    assert 'alice-secret' not in repr(canonical)
    monkeypatch.setenv('KVMEM_MULTI_USER_CONFIG', json.dumps(config))
    assert MultiUserMiddleware(ok).config == canonical
    for change in ({'max_inflight': True}, {'max_inflight': 0},
                   {'max_inflight_per_user': 3}, {'max_queue': -1},
                   {'queue_timeout_seconds': float('nan')},
                   {'queue_timeout_seconds': float('inf')},
                   {'body_timeout_seconds': 0}, {'body_timeout_seconds': 10 ** 1000},
                   {'max_body_bytes': 0}, {'unknown': 'alice-secret'}):
        with pytest.raises(ValueError) as failure:
            validate_config({**config, **change})
        assert 'alice-secret' not in str(failure.value)
    for bad in ('null', '{', '{"users_file":"a","users_file":"b"}', None):
        with pytest.raises(ValueError):
            validate_config(bad)


@pytest.mark.parametrize('records', [
    [], [{'id': 'a', 'api_key': 'same'}, {'id': 'b', 'api_key': 'same'}],
    [{'id': 'a', 'api_key': 'one'}, {'id': 'a', 'api_key': 'two'}],
    [{'id': '', 'api_key': 'secret'}], [{'id': 'a', 'api_key': 'has space'}],
    [{'id': 'a', 'api_key': 'secret', 'unexpected': True}],
])
def test_invalid_users_fail_closed(config, records):
    from pathlib import Path
    Path(config['users_file']).write_text(json.dumps({'users': records}))
    with pytest.raises(ValueError):
        MultiUserMiddleware(ok, config)


def test_authentication_and_endpoint_allowlist(config):
    async def run():
        app = MultiUserMiddleware(ok, config)
        cases = [('/health', 'GET', None, 200),
                 ('/v1/models', 'GET', None, 401),
                 ('/v1/models', 'GET', 'alice', 200),
                 ('/v1/chat/completions', 'POST', 'invalid', 401),
                 ('/v1/chat/completions', 'POST', None, 401),
                 ('/v1/responses', 'POST', 'alice', 404),
                 ('/reset_prefix_cache', 'POST', 'alice', 404),
                 ('/load_lora_adapter', 'POST', 'alice', 404),
                 ('/metrics', 'GET', 'alice', 404),
                 ('/v1/chat/completions', 'GET', 'alice', 404)]
        for path, method, user, status in cases:
            response = await Request(app, path=path, method=method, user=user).finish()
            assert response.status == status
        response = await Request(app, headers=[(b'authorization', b'Bearer bob-secret')]).finish()
        assert response.status == 401
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_server_owns_tenant_namespace_and_engine_identity(config):
    seen = []
    async def endpoint(scope, receive, send):
        message = await receive()
        seen.append((scope, json.loads(message['body'])))
        assert dict(scope['headers'])[b'content-length'] == str(len(message['body'])).encode()
        await ok(scope, receive, send)

    async def run():
        app = MultiUserMiddleware(endpoint, config)
        payload = {'session_id': 'shared-session', 'cache_salt': 'spoofed',
                   'request_id': 'colliding-id', 'priority': -10,
                   'kv_transfer_params': {'remote_engine_id': 'spoof'},
                   'vllm_xargs': {'session_id': 'ignored', 'cache_salt': 'spoofed',
                                  'Cache-Salt': 'spoofed', 'request_id': 'colliding-id',
                                  'priority': -100, 'kv_transfer_params': 'spoof',
                                  'legitimate_extension': 1}}
        headers = [(b'X-Session-ID', b'ignored'), (b'X-Cache-Salt', b'spoofed'),
                   (b'cache-salt', b'spoofed'), (b'x-request-id', b'colliding-id'),
                   (b'x-vllm-priority', b'-100'), (b'content-length', b'1000')]
        for user in ('alice', 'bob', 'alice'):
            result = await Request(app, payload, user=user, headers=headers).finish()
            assert result.status == 200
        salts = [p['cache_salt'] for _, p in seen]
        assert salts[0] == salts[2] and salts[0] != salts[1]
        assert all(len(salt) == 64 and salt != 'spoofed' for salt in salts)
        assert len({p['request_id'] for _, p in seen}) == 3
        for scope, body in seen:
            assert body['session_id'] == 'shared-session'
            assert body['priority'] == 0
            assert 'kv_transfer_params' not in body
            assert body['vllm_xargs'] == {'legitimate_extension': 1}
            assert all(name.lower() not in {b'x-session-id', b'x-cache-salt', b'cache-salt',
                                           b'x-request-id', b'x-vllm-priority'}
                       for name, _ in scope['headers'])
        assert [scope['state']['kvmem_user_id'] for scope, _ in seen] == ['alice', 'bob', 'alice']
        assert 'kvmem_user_id' not in result.scope['state']
    asyncio.run(run())


def test_bad_bodies_sessions_and_sequence_amplification_rejected(config):
    seen = []
    async def endpoint(scope, receive, send):
        seen.append(json.loads((await receive())['body']))
        await ok(scope, receive, send)

    async def run():
        app = MultiUserMiddleware(endpoint, config)
        for raw in (b'{', b'[]', b'null', b'\xff', b'{"n":1,"n":2}',
                    b'{"x":NaN}', b'{"temperature":1e309}', b'{"temperature":-1e309}'):
            assert (await Request(app, raw=raw).finish()).status == 400
        for payload in ({'session_id': 's' * 1025}, {'session_id': 42},
                        {'session_id': 'line\nbreak'}, {'vllm_xargs': {'session_id': []}},
                        {'vllm_xargs': []}, {'n': 4}, {'best_of': 3}, {'n': True},
                        {'prompt': ['first', 'second']}, {'prompt': [[1], [2]]}):
            assert (await Request(app, payload).finish()).status == 400
        assert (await Request(app, headers=[(b'x-session-id', b's' * 1025)]).finish()).status == 400
        assert (await Request(app, raw=b'x' * 4097).finish()).status == 413
        assert (await Request(app, chunks=[b'x' * 2048, b'x' * 2049]).finish()).status == 413
        assert (await Request(app, headers=[(b'content-length', b'4097')]).finish()).status == 413
        assert (await Request(app, headers=[(b'content-length', b'-1')]).finish()).status == 400
        assert (await Request(app, headers=[(b'content-length', b'9' * 5000)]).finish()).status == 400
        assert not seen
        for payload, headers, session in [({'vllm_xargs': {'session_id': 'extras'}}, [], 'extras'),
                                          ({'prompt': [1, 2, 3]}, [(b'x-session-id', b'header')], 'header')]:
            assert (await Request(app, payload, headers=headers).finish()).status == 200
            assert seen[-1]['session_id'] == session
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_streaming_quota_fairness_queue_bound_and_timeout(config):
    async def run():
        started, gates = [], {}
        async def endpoint(scope, receive, send):
            name = json.loads((await receive())['body'])['prompt']
            started.append(name)
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'first', 'more_body': True})
            await gates[name].wait()
            await send({'type': 'http.response.body', 'body': b'last', 'more_body': False})
        app = MultiUserMiddleware(endpoint, {**config, 'max_queue': 3})
        def request(name, user):
            gates[name] = asyncio.Event()
            return Request(app, {'prompt': name}, user=user)
        alice = request('alice-live', 'alice')
        await until(lambda: alice.data == b'first')
        a2, a3 = request('alice-wait-1', 'alice'), request('alice-wait-2', 'alice')
        await until(lambda: app._admission.queued == 2)
        bob = request('bob-live', 'bob')
        await until(lambda: bob.data == b'first')
        carol = request('carol-wait', 'carol')
        await until(lambda: app._admission.queued == 3)
        overflow = await request('overflow', 'bob').finish()
        assert overflow.status == 429
        assert dict(overflow.messages[0]['headers'])[b'retry-after'] == b'1'
        assert started == ['alice-live', 'bob-live']
        assert app._admission.total == 2
        gates['bob-live'].set()
        await bob.finish()
        await until(lambda: carol.data == b'first')
        # Alice's blocked FIFO cannot strand an available slot needed by Carol.
        assert started == ['alice-live', 'bob-live', 'carol-wait']
        gates['alice-live'].set()
        await alice.finish()
        await until(lambda: a2.data == b'first')
        assert not a3.messages  # A response header did not release Alice's lease.
        gates['alice-wait-1'].set()
        await a2.finish()
        await until(lambda: a3.data == b'first')
        gates['alice-wait-2'].set()
        gates['carol-wait'].set()
        await asyncio.gather(a3.finish(), carol.finish())
        assert app._admission.total == app._admission.queued == 0

        app = MultiUserMiddleware(endpoint, {**config, 'queue_timeout_seconds': 0.02})
        live = request('timeout-holder', 'alice')
        await until(lambda: live.data == b'first')
        expired = await request('expires', 'alice').finish()
        assert expired.status == 429 and b'timed out' in expired.data
        assert app._admission.queued == 0 and app._admission.total == 1
        gates['timeout-holder'].set()
        await live.finish()
    asyncio.run(run())


def test_disconnects_and_caller_cancellation_release_after_cleanup(config):
    async def run():
        started, cleanup_started, cleaned = [], asyncio.Event(), []
        finish_cleanup = asyncio.Event()
        async def endpoint(scope, receive, send):
            name = json.loads((await receive())['body'])['prompt']
            started.append(name)
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'first', 'more_body': True})
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await finish_cleanup.wait()
                cleaned.append(name)
        app = MultiUserMiddleware(endpoint, config)
        first = Request(app, {'prompt': 'first'})
        await until(lambda: first.data == b'first')
        waiting = Request(app, {'prompt': 'waiting'})
        await until(lambda: app._admission.queued == 1)
        await waiting.disconnect()
        assert app._admission.queued == 0 and started == ['first']
        successor = Request(app, {'prompt': 'successor'})
        await until(lambda: app._admission.queued == 1)
        first.incoming.put_nowait({'type': 'http.disconnect'})
        await asyncio.wait_for(cleanup_started.wait(), 2)
        assert app._admission.total == 1 and started == ['first']
        finish_cleanup.set()
        await first.finish()
        await until(lambda: successor.data == b'first')
        assert cleaned == ['first']
        successor.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await successor.task
        assert cleaned == ['first', 'successor']
        assert app._admission.total == app._admission.queued == 0
        assert app._admission.waiting == {} and not app._admission.turns
    asyncio.run(run())


def test_queued_cancellation_and_downstream_errors_do_not_leak_slots(config):
    async def run():
        gate = asyncio.Event()
        async def endpoint(scope, receive, send):
            payload = json.loads((await receive())['body'])
            if payload.get('prompt') == 'raise':
                raise RuntimeError('downstream failed')
            await gate.wait()
            await ok(scope, receive, send)
        app = MultiUserMiddleware(endpoint, config)
        live = Request(app)
        await until(lambda: app._admission.total == 1)
        waiting = Request(app)
        await until(lambda: app._admission.queued == 1)
        waiting.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting.task
        assert app._admission.queued == 0
        gate.set()
        await live.finish()
        failure = Request(app, {'prompt': 'raise'})
        with pytest.raises(RuntimeError, match='downstream failed'):
            await failure.finish()
        assert app._admission.total == 0
        assert (await Request(app).finish()).status == 200
    asyncio.run(run())


def test_incomplete_body_deadline_and_disconnect_do_not_strand_slots(config):
    async def run():
        dispatched = []
        async def endpoint(scope, receive, send):
            dispatched.append(scope)
            await ok(scope, receive, send)
        app = MultiUserMiddleware(endpoint, {**config, 'body_timeout_seconds': 0.02})
        incomplete = Request(app, chunks=[])
        incomplete.incoming.put_nowait({'type': 'http.request', 'body': b'{', 'more_body': True})
        await incomplete.finish()
        assert incomplete.status == 408
        assert not dispatched and app._admission.total == 0

        incomplete = Request(app, chunks=[])
        await until(lambda: app._admission.total == 1)
        await incomplete.disconnect()
        assert not incomplete.messages and app._admission.total == 0
        valid = await Request(app).finish()
        assert valid.status == 200 and len(dispatched) == 1
    asyncio.run(run())


def test_repeated_cancellation_does_not_interrupt_engine_cleanup(config):
    async def run():
        running, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def endpoint(scope, receive, send):
            running.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()
        app = MultiUserMiddleware(endpoint, config)
        request = Request(app)
        await running.wait()
        request.task.cancel()
        await cleaning.wait()
        request.task.cancel()
        await asyncio.sleep(0)
        assert app._admission.total == 1
        assert not request.task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await request.task
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_many_users_complete_under_global_and_individual_limits(config):
    async def run():
        active, peak = {}, 0
        async def endpoint(scope, receive, send):
            nonlocal peak
            user = scope['state']['kvmem_user_id']
            active[user] = active.get(user, 0) + 1
            peak = max(peak, sum(active.values()))
            assert sum(active.values()) <= 2 and active[user] <= 1
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            try:
                for _ in range(3):
                    await asyncio.sleep(0)
                    await send({'type': 'http.response.body', 'body': b'token', 'more_body': True})
                await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            finally:
                active[user] -= 1
        app = MultiUserMiddleware(endpoint, {**config, 'max_queue': 100})
        requests = [Request(app, user=user) for _ in range(20) for user in ('alice', 'bob', 'carol')]
        await asyncio.gather(*(request.finish() for request in requests))
        assert peak == 2 and all(request.status == 200 for request in requests)
        assert all(request.data == b'token' * 3 for request in requests)
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


@pytest.mark.parametrize('source', ['body', 'header', 'extras'])
def test_session_restores_reject_explicit_and_implicit_prompt_logprobs(config, source):
    async def run():
        dispatched = []
        async def endpoint(scope, receive, send):
            dispatched.append(json.loads((await receive())['body']))
            await ok(scope, receive, send)
        app = MultiUserMiddleware(endpoint, config)
        session = {'session_id': 'saved'} if source == 'body' else (
            {'vllm_xargs': {'session_id': 'saved'}} if source == 'extras' else {})
        headers = [(b'x-session-id', b'saved')] if source == 'header' else []
        unsupported = [('/v1/chat/completions', {'prompt_logprobs': 0}),
                       ('/v1/completions', {'prompt_logprobs': 1}),
                       ('/v1/chat/completions', {'echo': True}),
                       ('/v1/chat/completions', {'echo': True, 'top_logprobs': 2}),
                       ('/v1/completions', {'echo': True, 'logprobs': 0}),
                       ('/v1/completions', {'echo': True, 'logprobs': 2})]
        for path, options in unsupported:
            response = await Request(app, {**session, **options}, path=path, headers=headers).finish()
            assert response.status == 400 and b'Prompt logprobs' in response.data
        assert not dispatched
        # Decode logprobs and completion text echo do not need old prompt scores.
        supported = [('/v1/chat/completions', {'logprobs': True, 'top_logprobs': 2}),
                     ('/v1/completions', {'logprobs': 2}),
                     ('/v1/completions', {'echo': True}),
                     ('/v1/chat/completions', {'echo': True, 'top_logprobs': None})]
        for path, options in supported:
            assert (await Request(app, {**session, **options}, path=path, headers=headers).finish()).status == 200
        # Without a checkpoint session, native inference has the whole prompt.
        for path, options in unsupported:
            assert (await Request(app, options, path=path).finish()).status == 200
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_only_text_inputs_and_single_sequence_sampling_reach_engine(config):
    async def run():
        dispatched = []
        async def endpoint(scope, receive, send):
            dispatched.append(json.loads((await receive())['body']))
            await ok(scope, receive, send)
        app = MultiUserMiddleware(endpoint, config)
        bad_inputs = [{'prompt_embeds': 'encoded-embedding'},
                      {'prompt_embeds': ['first-embedding', 'second-embedding']},
                      {'use_beam_search': True}, {'use_beam_search': 'true'},
                      {'use_beam_search': 1}]
        for part in ({'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image'}},
                     {'type': 'input_audio', 'input_audio': {'data': 'encoded', 'format': 'wav'}},
                     {'type': 'video_url', 'video_url': {'url': 'https://example.invalid/video'}},
                     {'type': 'image_embeds', 'image_embeds': 'encoded'},
                     {'type': 'unknown', 'data': 'unsupported'}):
            bad_inputs.append({'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe this'}, part]}]})
        bad_inputs.append({'messages': [{'role': 'assistant', 'content': None,
                                         'audio': {'id': 'audio-id'}}]})
        for payload in bad_inputs:
            response = await Request(app, payload).finish()
            assert response.status == 400
        assert not dispatched
        supported = [
            {'messages': [{'role': 'user', 'content': 'Plain text'}]},
            {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Text part'}]}]},
            {'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'tool-1', 'type': 'function', 'function': {'name': 'tool', 'arguments': '{}'}}]}]},
            {'prompt': [1, 2, 3], 'use_beam_search': False},
        ]
        for payload in supported:
            assert (await Request(app, payload).finish()).status == 200
        assert len(dispatched) == len(supported)
        assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())
