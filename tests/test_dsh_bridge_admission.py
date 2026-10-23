"""CPU-only bridge admission checks using an in-memory upstream transport."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

import scripts.dsh_bridge as bridge


class ByteTokenizer:
    backend_tokenizer = SimpleNamespace(
        decoder=SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}'))

    def get_added_vocab(self):
        return {}

    def convert_ids_to_tokens(self, token):
        return chr(token)


class Planner:
    def __init__(self, protocol, tokenizer, budget):
        self.protocol = protocol

    def prepare(self, payload, state):
        return ([1, 2], {}, dict(tools=[]), [], None, False, {})

    def response(self, raw, tools, model, usage, finish):
        return dict(model=model, content=[dict(type='text', text=raw)],
                    stop_reason='end_turn', usage=dict(
                        input_tokens=usage['prompt_tokens'],
                        output_tokens=usage['completion_tokens']))

    def commit(self, state, prepared, payload, response, output):
        state.turns += 1


class Upstream:
    def __init__(self):
        self.calls = []
        self.payloads = []
        self.starts = asyncio.Queue()
        self.gates = {}
        self.active = 0
        self.maximum = 0
        self.closed = []

    async def handle(self, request):
        data = json.loads(request.content)
        self.payloads.append(data)
        owner = data['session_id']
        segment = len(data['prompt']) - 2
        key = (owner, segment)
        gate = self.gates[key] = asyncio.Event()
        self.calls.append(key)
        await self.starts.put(key)
        source = self

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                source.active += 1
                source.maximum = max(source.maximum, source.active)
                await gate.wait()
                value = dict(id=owner, choices=[dict(
                    index=0, text=chr(97+segment), token_ids=[97+segment],
                    finish_reason='length' if segment == 0 else 'stop')])
                usage = dict(prompt_tokens=len(data['prompt']), completion_tokens=1,
                             total_tokens=len(data['prompt'])+1)
                for item in [value, dict(choices=[], usage=usage)]:
                    yield ('data: '+json.dumps(item)+'\n\n').encode()
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                source.closed.append(key)
                source.active -= 1

        return httpx.Response(200, stream=Stream())


def setup(monkeypatch, tmp_path, upstream, max_inflight=1, disable_thinking=False):
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle))
    monkeypatch.setattr(bridge, 'Planner', Planner)
    monkeypatch.setattr(bridge, 'Protocol', lambda _: None)
    monkeypatch.setattr(bridge.httpx, 'AsyncClient', lambda **_: client)
    args = SimpleNamespace(protocol=None, gen_budget=1, max_output_segments=2,
                           output=tmp_path, backend='http://test', model='local',
                           max_model_len=4096, max_sessions=8, disable_thinking=disable_thinking,
                           tokenizer_protocol=disable_thinking, tokenizer='unused')
    if max_inflight is not None:
        args.max_inflight = max_inflight
    app = bridge.create_app(args, ByteTokenizer())
    endpoint = next(r.endpoint for r in app.routes if r.path == '/v1/chat/completions')

    async def call(frame, *, stream=False):
        # Same DSH session, distinct system frame: exactly the title/agent case.
        body = json.dumps(dict(model='local', prompt_cache_key='same-session', stream=stream,
                               max_tokens=2, messages=[
                                   dict(role='system', content=frame),
                                   dict(role='user', content='task')])).encode()

        async def receive():
            return dict(type='http.request', body=body, more_body=False)

        return await endpoint(Request(dict(type='http', method='POST',
                              path='/v1/chat/completions', headers=[]), receive))

    return call, client


async def let_waiter_run():
    # Event-loop yields, not elapsed-time assumptions about upstream processing.
    for _ in range(4):
        await asyncio.sleep(0)


def test_c1_serializes_distinct_owners_across_all_reply_segments(monkeypatch, tmp_path):
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream)
        first = asyncio.create_task(call('agent'))
        a0 = await upstream.starts.get()
        second = asyncio.create_task(call('title'))
        await let_waiter_run()
        assert upstream.calls == [a0]
        upstream.gates[a0].set()
        a1 = await upstream.starts.get()
        assert a1 == (a0[0], 1)
        await let_waiter_run()
        assert upstream.calls == [a0, a1]  # No interleaving at segment boundary.
        upstream.gates[a1].set()
        assert (await first)['choices'][0]['message']['content'] == 'ab'
        b0 = await upstream.starts.get()
        assert b0[0] != a0[0] and b0[1] == 0
        upstream.gates[b0].set()
        b1 = await upstream.starts.get()
        upstream.gates[b1].set()
        await second
        assert upstream.maximum == 1 and upstream.active == 0
        assert upstream.closed == [a0, a1, b0, b1]
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


@pytest.mark.parametrize('cancel_waiting', [False, True])
def test_cancellation_releases_admission_after_upstream_cleanup(monkeypatch, tmp_path, cancel_waiting):
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream)
        first = asyncio.create_task(call('agent'))
        a0 = await upstream.starts.get()
        waiting = asyncio.create_task(call('title'))
        await let_waiter_run()
        if cancel_waiting:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert upstream.calls == [a0]
            waiting = asyncio.create_task(call('another-owner'))
            await let_waiter_run()
            assert upstream.calls == [a0]
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        b0 = await upstream.starts.get()
        assert upstream.closed == [a0]  # Admission follows old stream close.
        assert b0[0] != a0[0]
        upstream.gates[b0].set()
        b1 = await upstream.starts.get()
        upstream.gates[b1].set()
        await waiting
        assert upstream.maximum == 1 and upstream.active == 0
        assert upstream.closed == [a0, b0, b1]
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


@pytest.mark.parametrize('limit', [None, 0, 2])
def test_unlimited_default_and_explicit_capacity_preserve_concurrent_owners(monkeypatch, tmp_path, limit):
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream, max_inflight=limit)
        first = asyncio.create_task(call('agent'))
        a0 = await upstream.starts.get()
        second = asyncio.create_task(call('title'))
        b0 = await upstream.starts.get()
        assert a0[0] != b0[0]
        assert upstream.maximum == 2
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        assert upstream.active == 0
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


@pytest.mark.parametrize('limit', [-1, 1.5, True, '1'])
def test_invalid_admission_limits_rejected_before_loading_model_protocol(limit):
    with pytest.raises(ValueError, match='max_inflight'):
        bridge.create_app(SimpleNamespace(max_inflight=limit), None)


def test_stream_disconnect_releases_global_permit_after_upstream_close(monkeypatch, tmp_path):
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream)
        response = await call('agent', stream=True)
        iterator = response.body_iterator
        assert 'assistant' in await anext(iterator)
        a0 = await upstream.starts.get()
        waiting = asyncio.create_task(call('title'))
        await let_waiter_run()
        assert upstream.calls == [a0]
        await iterator.aclose()
        b0 = await upstream.starts.get()
        assert upstream.closed == [a0]
        upstream.gates[b0].set()
        b1 = await upstream.starts.get()
        upstream.gates[b1].set()
        await waiting
        assert upstream.maximum == 1 and upstream.active == 0
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


def test_asgi_disconnect_cancel_scope_cleans_up_and_releases_c1(monkeypatch, tmp_path):
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream)
        response = await call('agent', stream=True)
        disconnected = asyncio.Event()
        first_sent = asyncio.Event()

        async def receive():
            await disconnected.wait()
            return dict(type='http.disconnect')

        async def send(message):
            if message['type'] == 'http.response.body':
                first_sent.set()

        task = asyncio.create_task(response(
            dict(type='http', asgi=dict(version='3.0', spec_version='2.3')),
            receive, send))
        await first_sent.wait()
        a0 = await upstream.starts.get()
        waiting = asyncio.create_task(call('title'))
        await let_waiter_run()
        assert upstream.calls == [a0]
        disconnected.set()
        await task
        b0 = await upstream.starts.get()
        assert upstream.closed == [a0]
        upstream.gates[b0].set()
        b1 = await upstream.starts.get()
        upstream.gates[b1].set()
        await waiting
        assert upstream.maximum == 1 and upstream.active == 0
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


def test_cancel_during_commit_keeps_owner_and_c1_until_recovery_marker(monkeypatch, tmp_path):
    async def check():
        from threading import Event
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream)
        entered = asyncio.Event()
        finish_commit = Event()
        loop = asyncio.get_running_loop()
        original = Planner.commit

        def blocked_commit(self, *args):
            loop.call_soon_threadsafe(entered.set)
            assert finish_commit.wait(3)
            return original(self, *args)

        monkeypatch.setattr(Planner, 'commit', blocked_commit)
        first = asyncio.create_task(call('agent'))
        a0 = await upstream.starts.get()
        upstream.gates[a0].set()
        a1 = await upstream.starts.get()
        upstream.gates[a1].set()
        await entered.wait()
        first.cancel()
        second = asyncio.create_task(call('agent'))
        await let_waiter_run()
        assert not first.done() and upstream.calls == [a0, a1]
        assert not list(tmp_path.glob('**/accepted.json'))
        finish_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        b0 = await upstream.starts.get()
        assert len(list(tmp_path.glob('**/accepted.json'))) == 1
        assert b0[0] == a0[0]
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert upstream.active == 0 and upstream.maximum == 1
        await client.aclose()

    asyncio.run(asyncio.wait_for(check(), 5))


@pytest.mark.parametrize('disable_thinking', [False, True])
def test_disabled_thinking_removes_budget_from_every_native_segment(
        monkeypatch, tmp_path, disable_thinking):
    import scripts.dsh_model_protocol as model_protocol
    seen_modes = []
    def make_protocol(*args, **kwargs):
        seen_modes.append(kwargs['enable_thinking'])
        return None
    monkeypatch.setattr(model_protocol, 'TokenizerProtocol', make_protocol)
    async def check():
        upstream = Upstream()
        call, client = setup(monkeypatch, tmp_path, upstream, disable_thinking=disable_thinking)
        reply = asyncio.create_task(call('agent'))
        first = await upstream.starts.get()
        upstream.gates[first].set()
        second = await upstream.starts.get()
        upstream.gates[second].set()
        await reply
        await client.aclose()
        assert len(upstream.payloads) == 2
        if disable_thinking:
            assert seen_modes == [False]
            assert all('thinking_token_budget' not in value for value in upstream.payloads)
        else:
            assert upstream.payloads[0]['thinking_token_budget'] == 4096
            assert 'thinking_token_budget' not in upstream.payloads[1]
    asyncio.run(check())


def test_disable_thinking_rejects_frozen_protocol_before_loading():
    args = SimpleNamespace(disable_thinking=True, tokenizer_protocol=False)
    with pytest.raises(ValueError, match='requires --tokenizer-protocol'):
        bridge.create_app(args, ByteTokenizer())
