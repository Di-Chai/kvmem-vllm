import io
import json
import threading
import time

import pytest

from vllm_kvmem.apps.webui.chat import ChatBridge, _TextParser, _query_begin


class Tokenizer:
    all_special_tokens = ['~']

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking=False):
        text = ''
        for message in messages:
            content = message['content']
            if message.get('reasoning_content'):
                content = '<think>' + message['reasoning_content'] + '</think>' + content
            text += message['role'] + ':' + content + '~'
        if add_generation_prompt:
            text += 'assistant:' + ('<think>' if enable_thinking else '')
        return self.encode(text) if tokenize else text


class Manager:
    def __init__(self):
        self.lock = threading.RLock()
        self.instance = 'engine-1'
        self.base_url = 'http://127.0.0.1:8000'
        self.backend_headers = {'Authorization': 'Bearer only-for-native'}
        self.active_profile = {'model': '/model', 'tokenizer': '/tokenizer', 'served_model_name': 'kvmem',
                               'max_model_len': 4096, 'kvmem': {'request_planning': True,
                               'block_tokens': 1, 'select_budget': 2048, 'gen_budget': 2048,
                               'sink_blocks': 1, 'recent_blocks': 1,
                               'rolling_active_window': True}}

    def status(self):
        return {'state': 'ready', 'instance': self.instance}


def event(choices=None, usage=None, **extras):
    return ('data: ' + json.dumps({'id': 'native-1', 'choices': choices or [],
                                 'usage': usage, **extras}) + '\n\n').encode()


def reply(body, *, text='answer~', ids=True, done=True, usage=True):
    choice = {'index': 0, 'text': text, 'finish_reason': 'stop',
              'prompt_token_ids': body['prompt']}
    if ids:
        choice['token_ids'] = Tokenizer().encode(text)
    data = event([choice])
    if usage:
        data += event(usage={'prompt_tokens': len(body['prompt']), 'completion_tokens': len(text)})
    if done:
        data += b'data: [DONE]\n\n'
    return data


class Response(io.BytesIO):
    status = 200


class Connection:
    def __init__(self, factory):
        self.factory = factory
        self.sock = self
        self.closed = False
        self.shutdown_event = threading.Event()

    def connect(self):
        pass

    def shutdown(self, _):
        self.shutdown_event.set()

    def close(self):
        self.closed = True

    def request(self, method, path, body, headers):
        self.body = json.loads(body)
        self.factory.requests.append((self.body, headers))

    def getresponse(self):
        if self.body.get('kv_transfer_params', {}).get('kvmem_release_session'):
            return Response(b'{"id":"release"}')
        return self.factory.respond(self.body, self)


class Factory:
    def __init__(self, responder=None):
        self.requests = []
        self.connections = []
        self.responder = responder

    def __call__(self, *args, **kwargs):
        conn = Connection(self)
        self.connections.append(conn)
        return conn

    def respond(self, body, connection):
        return self.responder(body, connection) if self.responder else Response(reply(body))


def make_bridge(responder=None):
    manager, factory = Manager(), Factory(responder)
    return ChatBridge(manager, tokenizer_loader=lambda _: Tokenizer(), connection_factory=factory), factory


def payload(messages=None, **extras):
    return {'client_id': 'browser-a', 'conversation_id': 'conversation-1', 'model': 'kvmem',
            'messages': messages or [{'role': 'user', 'content': 'hello'}], 'max_tokens': 128, **extras}


def consume(bridge, body=None, request='request-1'):
    data = b''.join(bridge.stream(body or payload(), request, bridge.manager.instance))
    return [json.loads(row[6:]) for row in data.split(b'\n\n') if row.startswith(b'data: {')]


def generations(factory):
    return [body for body, _ in factory.requests if body.get('stream')]


def test_stream_translates_content_and_verifies_actual_token_prefix():
    bridge, factory = make_bridge()
    first = consume(bridge)
    assert ''.join(choice['delta'].get('content', '') for row in first
                   for choice in row.get('choices', [])) == 'answer'
    assert first[-1]['usage']['completion_tokens'] == len('answer~')
    assert generations(factory)[0]['return_token_ids'] is True
    assert factory.requests[0][1]['Authorization'] == 'Bearer only-for-native'
    messages = payload()['messages'] + [{'role': 'assistant', 'content': 'answer'},
                                       {'role': 'user', 'content': 'next'}]
    consume(bridge, payload(messages), 'request-2')
    one, two = generations(factory)
    accepted = one['prompt'] + Tokenizer().encode('answer~')
    assert two['prompt'][:len(accepted)] == accepted
    assert one['session_id'] == two['session_id']
    assert bridge.record('request-2', client_id='browser-a')['history_mode'] == 'exact_prefix'
    query = two['kvmem_query_span']
    assert ''.join(map(chr, two['prompt'][query[0]:query[1]])) == 'next~assistant:'


def test_edited_or_retokenized_history_releases_and_explicitly_rebuilds():
    bridge, factory = make_bridge()
    consume(bridge)
    messages = [{'role': 'user', 'content': 'edited'}, {'role': 'assistant', 'content': 'answer'},
                {'role': 'user', 'content': 'next'}]
    events = consume(bridge, payload(messages), 'request-2')
    one, two = generations(factory)
    assert one['session_id'] != two['session_id']
    assert factory.requests[1][0]['kv_transfer_params']['kvmem_release_session']
    assert factory.requests[1][0]['session_id'] == one['session_id']
    assert any(row.get('kvmem', {}).get('history_mode') == 'canonical_rebuild' for row in events)


@pytest.mark.parametrize('failure', ['ids', 'done', 'usage'])
def test_missing_acceptance_evidence_does_not_commit_and_next_turn_rebuilds(failure):
    bridge, factory = make_bridge(lambda body, _: Response(reply(body, **{failure: False})))
    events = consume(bridge)
    assert 'accepted token IDs/usage' in events[-1]['error']['message']
    conversation = next(iter(bridge.conversations.values()))
    assert not conversation.tokens and conversation.dirty
    assert bridge.record('request-1', client_id='browser-a')['state'] == 'error'
    factory.responder = None
    events = consume(bridge, request='request-2')
    assert any(row.get('kvmem', {}).get('history_mode') == 'canonical_rebuild' for row in events)
    assert factory.requests[1][0]['kv_transfer_params']['kvmem_release_session']


def test_browser_identity_isolates_sessions_records_cancel_and_release():
    bridge, factory = make_bridge()
    consume(bridge)
    consume(bridge, payload(client_id='browser-b'), 'request-2')
    one, two = generations(factory)
    assert one['session_id'] != two['session_id']
    assert one['cache_salt'] != two['cache_salt']
    assert bridge.record('request-1', client_id='browser-b') is None
    assert bridge.cancel('request-1', client_id='browser-b') == {'cancelled': False}
    assert bridge.release('conversation-1', client_id='browser-c') == {'released': False}
    assert bridge.release('conversation-1', client_id='browser-a') == {'released': True}
    assert ('browser-b', 'conversation-1') in bridge.conversations


def test_engine_restart_and_tokenizer_identity_do_not_reuse_old_namespace():
    bridge, factory = make_bridge()
    consume(bridge)
    bridge.manager.instance = 'engine-2'
    consume(bridge, request='request-2')
    assert len(generations(factory)) == len(factory.requests) == 2
    assert generations(factory)[0]['session_id'] != generations(factory)[1]['session_id']
    bridge.manager.active_profile['tokenizer'] = '/other-tokenizer'
    consume(bridge, request='request-3')
    assert factory.requests[2][0]['kv_transfer_params']['kvmem_release_session']
    assert bridge.record('request-3', client_id='browser-a')['history_mode'] == 'canonical_rebuild'
    with pytest.raises(ValueError, match='changed'):
        bridge.stream(payload(), 'stale', 'engine-1')


def test_split_reasoning_and_special_markers_are_not_exposed_as_content():
    parser = _TextParser(True)
    rows = []
    for text in ['rea', 'son</thi', 'nk>ans', 'wer<|im_', 'end|>']:
        rows.extend(parser.feed(text))
    rows.extend(parser.feed('', final=True))
    assert ''.join(text for key, text in rows if key == 'reasoning_content') == 'reason'
    assert ''.join(text for key, text in rows if key == 'content') == 'answer'


def test_query_coordinates_survive_templates_stripping_earlier_reasoning():
    class StrippingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            messages = [{k: v for k, v in row.items() if k != 'reasoning_content'} for row in messages]
            return super().apply_chat_template(messages, **kwargs)
    tokenizer = StrippingTokenizer()
    messages = [{'role': 'user', 'content': 'old'},
                {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'hidden history'},
                {'role': 'user', 'content': 'new'}]
    template = {'enable_thinking': False}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
    ids = tokenizer.encode(text)
    begin = _query_begin(tokenizer, messages, template, text, ids)
    assert text[begin:] == 'new~assistant:'


def test_slow_tokenizer_query_fallback_includes_cross_boundary_bpe_token():
    class SlowTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            raise NotImplementedError
        def encode(self, text, **kwargs):
            return [hash(text[i:i+2]) % (2**32) for i in range(0, len(text), 2)]
    tokenizer = SlowTokenizer()
    messages = [{'role': 'user', 'content': 'hello'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.encode(text)
    begin = _query_begin(tokenizer, messages, {}, text, prompt)
    assert begin == len('user:') // 2


def test_cancel_interrupts_upstream_prefill_and_waiting_same_conversation():
    entered = threading.Event()
    def blocked(body, conn):
        entered.set()
        conn.shutdown_event.wait(5)
        raise OSError('connection shut down')
    bridge, factory = make_bridge(blocked)
    results = {}
    first = threading.Thread(target=lambda: results.update(first=consume(bridge)))
    first.start()
    assert entered.wait(2)
    second = threading.Thread(target=lambda: results.update(second=consume(bridge, request='request-2')))
    second.start()
    deadline = time.monotonic() + 2
    while 'request-2' not in bridge.active and time.monotonic() < deadline:
        time.sleep(.01)
    assert bridge.cancel('request-2', client_id='browser-b') == {'cancelled': False}
    assert bridge.cancel('request-2', client_id='browser-a') == {'cancelled': True}
    second.join(2)
    assert not second.is_alive()
    assert len(factory.requests) == 1
    assert bridge.cancel('request-1', client_id='browser-a') == {'cancelled': True}
    first.join(2)
    assert not first.is_alive()
    assert not bridge.active
    assert all(row[-1]['error']['message'] == 'Chat cancelled' for row in results.values())


def test_conversations_do_not_hold_a_shared_streaming_lock():
    entered, proceed = threading.Event(), threading.Event()
    def responder(body, conn):
        if not entered.is_set():
            entered.set()
            assert proceed.wait(3)
        return Response(reply(body))
    bridge, _ = make_bridge(responder)
    first = threading.Thread(target=lambda: consume(bridge))
    first.start()
    assert entered.wait(2)
    result = consume(bridge, payload(conversation_id='other'), 'request-2')
    assert result[-1]['usage']['completion_tokens'] > 0
    proceed.set()
    first.join(2)
    assert not first.is_alive()


@pytest.mark.parametrize('extras', [{'tools': []}, {'temperature': float('nan')},
                                    {'max_tokens': True}, {'top_k': 0},
                                    {'chat_template_kwargs': {'preserve_thinking': True}},
                                    {'messages': [{'role': 'user', 'content': [{'type': 'image_url'}]}]}])
def test_unsupported_request_fails_before_registering(extras):
    bridge, _ = make_bridge()
    with pytest.raises(ValueError):
        bridge.stream(payload(**extras), 'request-1', 'engine-1')
    assert not bridge.records and not bridge.active


def test_context_discloses_supported_capabilities_without_backend_secrets():
    bridge, _ = make_bridge()
    context = bridge.context()
    assert context['backend'] == 'vllm'
    assert context['effective']['enable_thinking'] is False
    assert context['capabilities']['budget'] is False
    assert context['capabilities']['tools'] is False
    assert 'only-for-native' not in json.dumps(context)


def test_metadata_is_bounded_and_contains_no_prompt_or_output_tokens(monkeypatch):
    import vllm_kvmem.apps.webui.chat as module
    monkeypatch.setattr(module, 'MAX_RECORDS', 3)
    bridge, _ = make_bridge()
    for index in range(6):
        consume(bridge, payload(conversation_id=f'conversation-{index}'), f'request-{index}')
    assert len(bridge.records) == 3
    for row in bridge.snapshot()['recent']:
        assert not {'prompt', 'messages', 'token_ids', 'content', 'reasoning_content'} & row.keys()


def test_nonplanned_profile_uses_native_session_release_without_query_fields():
    bridge, factory = make_bridge()
    bridge.manager.active_profile['kvmem']['request_planning'] = False
    bridge.manager.active_profile['kvmem']['rolling_active_window'] = False
    consume(bridge)
    bridge.release('conversation-1', client_id='browser-a')
    assert 'kvmem_query_span' not in factory.requests[0][0]
    assert 'kvmem_reselect' not in factory.requests[1][0]


def test_output_budget_and_history_capacity_errors_do_not_launch_generation(monkeypatch):
    import vllm_kvmem.apps.webui.chat as module
    bridge, factory = make_bridge()
    monkeypatch.setattr(module, 'MAX_HISTORY_TOKENS', 1)
    result = consume(bridge)
    assert 'token memory is full' in result[-1]['error']['message']
    assert not factory.requests


def test_observed_decode_rate_excludes_entire_initial_speculative_batch():
    class DelayedResponse(Response):
        def readline(self, limit=-1):
            line = super().readline(limit)
            if b'"text": "later"' in line:
                time.sleep(.02)
            return line
    def responder(body, _):
        data = event([{'index': 0, 'text': 'initial', 'token_ids': [1, 2, 3, 4]}])
        data += event([{'index': 0, 'text': 'later', 'token_ids': [5, 6], 'finish_reason': 'stop'}])
        data += event(usage={'prompt_tokens': len(body['prompt']), 'completion_tokens': 6})
        return DelayedResponse(data + b'data: [DONE]\n\n')
    bridge, _ = make_bridge(responder)
    timings = consume(bridge)[-1]['timings']
    assert timings['predicted_per_second'] == pytest.approx(2000 / timings['predicted_ms'])
    assert timings['measurement_source'] == 'browser_observed_accepted_token_SSE'
    assert bridge.record('request-1', client_id='browser-a')['phase'] == 'completed'
    bridge2, _ = make_bridge()
    assert 'predicted_per_second' not in consume(bridge2)[-1]['timings']


@pytest.mark.parametrize('state', ['stopped', 'failed', 'exited'])
def test_release_after_confirmed_engine_exit_clears_local_session_without_http(state):
    bridge, factory = make_bridge()
    consume(bridge)
    bridge.manager.status = lambda: {'state': state, 'instance': 'engine-1', 'pid': None}
    bridge.manager.active_profile = None
    assert bridge.release('conversation-1', client_id='browser-a') == {'released': True}
    assert not bridge.conversations
    assert len(factory.requests) == 1


@pytest.mark.parametrize('state,pid', [('loading', 1), ('failed', 1)])
def test_release_does_not_forget_a_potentially_live_nonready_engine(state, pid):
    bridge, factory = make_bridge()
    consume(bridge)
    bridge.manager.status = lambda: {'state': state, 'instance': 'engine-1', 'pid': pid}
    with pytest.raises(ValueError, match='No ready model'):
        bridge.release('conversation-1', client_id='browser-a')
    assert bridge.conversations
    assert len(factory.requests) == 1


def test_coordinated_session_capacity_rejects_new_conversation_before_native_wait():
    bridge, factory = make_bridge()
    bridge.manager.active_profile['kvmem'].update(cpu_history_coordination=True, session_capacity=1)
    consume(bridge)
    result = consume(bridge, payload(conversation_id='second'), 'request-2')
    assert 'retained session limit (1)' in result[-1]['error']['message']
    assert len(factory.requests) == 1
    assert len(bridge.conversations) == 1
    bridge.release('conversation-1', client_id='browser-a')
    result = consume(bridge, payload(conversation_id='second'), 'request-3')
    assert result[-1]['usage']['completion_tokens'] == 7


def test_query_replay_uses_native_pressure_headroom_not_only_configured_replay_limit():
    bridge, factory = make_bridge()
    bridge.manager.active_profile['kvmem'].update(block_tokens=32, gen_budget=96)
    bridge.manager.active_profile['query_replay_budget'] = 96
    result = consume(bridge, payload([{'role': 'user', 'content': 'x' * 76}], max_tokens=8))
    assert 'one block of pressure headroom' in result[-1]['error']['message']
    assert not factory.requests
    assert not bridge.conversations


def test_guided_query_reserve_matches_final_query_and_prefill_policies():
    bridge, factory = make_bridge()
    bridge.manager.active_profile['kvmem'].update(block_tokens=32, gen_budget=512,
                                                guided_query_tokens=64, guided_reselect='boundary')
    body = payload([{'role': 'user', 'content': 'x' * 160}], max_tokens=8)
    result = consume(bridge, body)
    assert 'one block of pressure headroom' in result[-1]['error']['message']
    assert not factory.requests
    bridge.manager.active_profile['kvmem']['guided_reselect'] = 'prefill'
    result = consume(bridge, body, 'request-2')
    assert result[-1]['usage']['completion_tokens'] == 7


@pytest.mark.parametrize('failure', ['context', 'replay', 'memory', 'output'])
def test_invalid_edited_turn_does_not_release_previous_usable_checkpoint(monkeypatch, failure):
    import vllm_kvmem.apps.webui.chat as module
    bridge, factory = make_bridge()
    consume(bridge)
    conversation = next(iter(bridge.conversations.values()))
    previous_session, previous_tokens = conversation.session, list(conversation.tokens)
    bad = payload([{'role': 'user', 'content': 'edited history'}])
    if failure == 'context':
        bridge.manager.active_profile['max_model_len'] = 32
    elif failure == 'replay':
        bridge.manager.active_profile['query_replay_budget'] = 4
    elif failure == 'memory':
        monkeypatch.setattr(module, 'MAX_HISTORY_TOKENS', 1)
    else:
        bridge.manager.active_profile['kvmem']['rolling_active_window'] = False
        bad['max_tokens'] = 2049
    result = consume(bridge, bad, 'bad-turn')
    assert 'error' in result[-1]
    assert len(factory.requests) == 1
    assert conversation.session == previous_session and list(conversation.tokens) == previous_tokens
    assert conversation.reserved == 0 and not conversation.dirty
