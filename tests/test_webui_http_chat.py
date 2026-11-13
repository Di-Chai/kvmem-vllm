"""Real console -> chat bridge -> native HTTP/SSE integration, CPU only.

For manual browser checks after building the UI assets:
  PYTHONPATH=src python tests/test_webui_http_chat.py --port 18401

This is deliberately a fake model, never a quality/performance qualification.
Enter [fixture:error] for a native SSE error or [fixture:slow] to test Stop.
"""
import argparse
from contextlib import contextmanager
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import select
import signal
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from uuid import uuid4

from vllm_kvmem.apps.webui.chat import ChatBridge
from vllm_kvmem.apps.webui.server import ConsoleServer


class FakeTokenizer:
    """Explicit character tokenizer; exercises transport without model packages."""
    all_special_tokens = ['~']

    def encode(self, text, **_):
        return [ord(char) for char in text]

    def __call__(self, text, **_):
        return {'input_ids': self.encode(text),
                'offset_mapping': [(index, index + 1) for index in range(len(text))]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=False, **_):
        text = ''
        for message in messages:
            content = message['content']
            if message.get('reasoning_content'):
                content = '<think>' + message['reasoning_content'] + '</think>' + content
            text += message['role'] + ':' + content + '~'
        if add_generation_prompt:
            text += 'assistant:' + ('<think>' if enable_thinking else '')
        return self.encode(text) if tokenize else text


class FakeNativeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.secret = uuid4().hex
        self.rows = []
        self.lock = threading.RLock()
        self.prefill_started = threading.Event()
        self.prefill_cancelled = threading.Event()
        self.closing = threading.Event()
        super().__init__(('127.0.0.1', 0), NativeHandler)


class NativeHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def response(self, status, data, content_type='application/json'):
        body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def authenticated(self):
        if self.headers.get('Authorization') != 'Bearer ' + self.server.secret:
            self.response(401, {'error': 'Fake native server requires its private test key'})
            return False
        return True

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path == '/health':
            self.response(200, {})
        elif self.path == '/v1/models':
            self.response(200, {'data': [{'id': 'fixture'}]})
        elif self.path == '/metrics':
            self.response(200, '# CPU HTTP fixture: no GPU measurements\n', 'text/plain')
        else:
            self.response(404, {'error': 'Unknown fixture route'})

    def do_POST(self):
        if not self.authenticated():
            return
        if self.path != '/v1/completions':
            self.response(404, {'error': 'This fixture accepts token-ID completions only'})
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        prompt = body.get('prompt')
        if not isinstance(prompt, list) or any(type(value) is not int for value in prompt):
            self.response(400, {'error': 'Token IDs required'})
            return
        with self.server.lock:
            self.server.rows.append(body)
            del self.server.rows[:-100]
        if body.get('kv_transfer_params', {}).get('kvmem_release_session'):
            self.response(200, {'id': 'release-' + uuid4().hex, 'choices': [], 'usage': {}})
            return
        text = ''.join(map(chr, prompt))
        user = text.rsplit('user:', 1)[-1].split('~assistant:', 1)[0]
        native_id = 'native-' + uuid4().hex
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            if '[fixture:slow]' in user:
                self.server.prefill_started.set()
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not self.server.closing.is_set():
                    ready, _, _ = select.select([self.connection], [], [], .05)
                    if ready and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                        self.server.prefill_cancelled.set()
                        return
                if self.server.closing.is_set():
                    return
            if '[fixture:error]' in user:
                self.event({'error': {'message': 'Deliberate native SSE failure from HTTP fixture'}})
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
                return
            answer = 'fixture reply: ' + user
            output = ('reasoning from CPU fixture</think>' if text.endswith('<think>') else '') + answer + '~'
            output = output[:body['max_tokens']]
            pieces = [output[index:index + 4] for index in range(0, len(output), 4)]
            for index, piece in enumerate(pieces):
                choice = {'index': 0, 'text': piece, 'token_ids': FakeTokenizer().encode(piece),
                          'finish_reason': 'stop' if index == len(pieces) - 1 else None}
                if index == 0:
                    choice['prompt_token_ids'] = prompt
                self.event({'id': native_id, 'choices': [choice], 'usage': None})
                time.sleep(.005)
            self.event({'id': native_id, 'choices': [], 'usage': {
                'prompt_tokens': len(prompt), 'completion_tokens': len(output),
                'total_tokens': len(prompt) + len(output)}})
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (OSError, ValueError):
            self.server.prefill_cancelled.set()

    def event(self, value):
        self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
        self.wfile.flush()


class FixtureManager:
    """Simulates process controls only; all chat traffic uses real local sockets."""
    def __init__(self, native):
        self.native = native
        self.lock = threading.RLock()
        self._state = 'ready'
        self._instance = uuid4().hex
        self._started_at = int(time.time() * 1000)
        self._profile = {'model': '/fixture/CPU-HTTP-test-no-model', 'tokenizer': '/fixture/character-tokenizer',
                         'hf_config_path': '/fixture/CPU-HTTP-test-no-model', 'model_format': 'safetensors',
                         'draft_method': 'mtp', 'draft_tokens': 1, 'execution_mode': 'eager',
                         'served_model_name': 'fixture', 'host': '127.0.0.1', 'port': native.server_port,
                         'max_model_len': 65536, 'max_num_seqs': 2, 'max_num_batched_tokens': 4096,
                         'gpu_memory_utilization': .55, 'kvmem': {'request_planning': True,
                         'block_tokens': 1, 'select_budget': 2048, 'gen_budget': 2048,
                         'sink_blocks': 1, 'recent_blocks': 1,
                         'rolling_active_window': True, 'session_capacity': 8,
                         'cpu_read_cache_bytes': 2147483648}}
        self._draft = copy.deepcopy(self._profile)

    @property
    def active_profile(self):
        with self.lock:
            return copy.deepcopy(self._profile) if self._state == 'ready' else None

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.native.server_port}' if self._state == 'ready' else None

    @property
    def backend_headers(self):
        return {'Authorization': 'Bearer ' + self.native.secret}

    def draft(self):
        return copy.deepcopy(self._draft)

    def status(self):
        with self.lock:
            return {'state': self._state, 'instance': self._instance, 'started_at_ms': self._started_at,
                    'active_profile': self.active_profile, 'base_url': self.base_url, 'url': self.base_url,
                    'pid': None, 'error': None, 'exit_code': None,
                    'active_config': {'parameters': {'Model': self._profile['model'], 'Backend': 'vllm'}}}

    def preview(self, profile):
        return {'profile': copy.deepcopy(profile), 'scope': 'CPU HTTP test fixture; no model is loaded',
                'vllm_argv': ['fixture-only'], 'environment': {}}

    def save(self, profile):
        self._draft = copy.deepcopy(profile)
        return self.draft()

    def start(self, profile):
        with self.lock:
            if self._state == 'ready':
                raise ValueError('Stop the test fixture before starting it again')
            self.save(profile)
            self._profile = {**copy.deepcopy(profile), 'port': self.native.server_port,
                             'served_model_name': 'fixture', 'tokenizer': '/fixture/character-tokenizer'}
            self._state, self._instance = 'ready', uuid4().hex
            self._started_at = int(time.time() * 1000)
            return self.status()

    def stop(self):
        with self.lock:
            self._state = 'stopped'
            return self.status()

    def close(self):
        self.stop()

    def logs(self, revision=None):
        return {'protocol': 1, 'text': 'CPU HTTP fixture. No vLLM/GPU model has been loaded.\n',
                'revision': self._instance, 'available': True, 'running': self._state == 'ready',
                'backend': 'vllm', 'session': self._instance}


@contextmanager
def running_fixture(port=0, *, sample_telemetry=False):
    native = FakeNativeServer()
    native_thread = threading.Thread(target=native.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    native_thread.start()
    manager = FixtureManager(native)
    chat = ChatBridge(manager, tokenizer_loader=lambda _: FakeTokenizer())
    console = ConsoleServer(('127.0.0.1', port), manager, chat=chat)
    console_thread = threading.Thread(target=console.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    console_thread.start()
    if sample_telemetry:
        console.telemetry.start()
    try:
        yield SimpleNamespace(console=console, manager=manager, chat=chat, native=native)
    finally:
        native.closing.set()
        console.chat.cancel_all()
        console.shutdown()
        console.close()
        native.shutdown()
        native.server_close()
        console_thread.join(timeout=2)
        native_thread.join(timeout=2)


def http_request(fixture, method, path, body=None, *, request_id='request-1', client='browser-a'):
    connection = http.client.HTTPConnection('127.0.0.1', fixture.console.server_port, timeout=5)
    headers = {'Content-Type': 'application/json', 'X-KVMem-Token': fixture.console.token,
               'X-KVMem-Client': client, 'X-KVMem-Request': request_id,
               'X-KVMem-Instance': fixture.manager.status()['instance']}
    connection.request(method, path, json.dumps(body) if body is not None else None, headers)
    response = connection.getresponse()
    status, data = response.status, response.read()
    connection.close()
    return status, data


def chat_payload(text='hello', messages=None):
    return {'model': 'fixture', 'conversation_id': 'http-conversation', 'max_tokens': 1024,
            'messages': messages or [{'role': 'user', 'content': text}]}


def events(data):
    return [json.loads(row[6:]) for row in data.split(b'\n\n') if row.startswith(b'data: {')]


def content(data):
    return ''.join(choice.get('delta', {}).get('content', '') for row in events(data)
                   for choice in row.get('choices', []))


class HTTPChatTests(unittest.TestCase):
    def setUp(self):
        self.context = running_fixture()
        self.fixture = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_two_real_http_turns_preserve_verified_tokens_and_release(self):
        fixture = self.fixture
        status, first = http_request(fixture, 'POST', '/api/chat/completions', chat_payload())
        self.assertEqual(status, 200)
        self.assertEqual(content(first), 'fixture reply: hello')
        self.assertTrue(first.endswith(b'data: [DONE]\n\n'))
        messages = chat_payload()['messages'] + [{'role': 'assistant', 'content': content(first)},
                                               {'role': 'user', 'content': 'next'}]
        status, second = http_request(fixture, 'POST', '/api/chat/completions',
                                     chat_payload(messages=messages), request_id='request-2')
        self.assertEqual(status, 200)
        self.assertEqual(content(second), 'fixture reply: next')
        self.assertTrue(any(row.get('kvmem', {}).get('history_mode') == 'exact_prefix' for row in events(second)))
        one, two = fixture.native.rows
        accepted = one['prompt'] + FakeTokenizer().encode(content(first) + '~')
        self.assertEqual(two['prompt'][:len(accepted)], accepted)
        self.assertEqual(one['session_id'], two['session_id'])
        status, record = http_request(fixture, 'GET', '/api/chat/requests/request-2')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(record)['state'], 'completed')
        self.assertEqual(http_request(fixture, 'GET', '/api/chat/requests/request-2', client='browser-b')[0], 404)
        status, release = http_request(fixture, 'POST', '/api/chat/release', {'conversation_id': 'http-conversation'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(release), {'released': True})
        self.assertTrue(fixture.native.rows[-1]['kv_transfer_params']['kvmem_release_session'])
        self.assertEqual(fixture.native.rows[-1]['session_id'], one['session_id'])

    def test_upstream_sse_error_is_returned_without_committing_history(self):
        fixture = self.fixture
        status, body = http_request(fixture, 'POST', '/api/chat/completions', chat_payload('[fixture:error]'))
        self.assertEqual(status, 200)
        self.assertIn('Deliberate native SSE failure', events(body)[-1]['error']['message'])
        self.assertTrue(body.endswith(b'data: [DONE]\n\n'))
        _, record = http_request(fixture, 'GET', '/api/chat/requests/request-1')
        self.assertEqual(json.loads(record)['state'], 'error')
        self.assertFalse(next(iter(fixture.chat.conversations.values())).tokens)

    def test_cancel_endpoint_interrupts_native_http_prefill(self):
        fixture = self.fixture
        results = []
        worker = threading.Thread(target=lambda: results.append(http_request(
            fixture, 'POST', '/api/chat/completions', chat_payload('[fixture:slow]'))))
        worker.start()
        self.assertTrue(fixture.native.prefill_started.wait(3))
        status, wrong = http_request(fixture, 'POST', '/api/chat/cancel', {'id': 'request-1'}, client='browser-b')
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(wrong)['cancelled'])
        status, body = http_request(fixture, 'POST', '/api/chat/cancel', {'id': 'request-1'})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)['cancelled'])
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(fixture.native.prefill_cancelled.wait(3))
        self.assertEqual(events(results[0][1])[-1]['error']['message'], 'Chat cancelled')
        self.assertFalse(fixture.chat.active)

    def test_browser_disconnect_interrupts_native_http_prefill(self):
        fixture = self.fixture
        connection = http.client.HTTPConnection('127.0.0.1', fixture.console.server_port, timeout=5)
        connection.request('POST', '/api/chat/completions', json.dumps(chat_payload('[fixture:slow]')), {
            'Content-Type': 'application/json', 'X-KVMem-Token': fixture.console.token,
            'X-KVMem-Client': 'browser-a', 'X-KVMem-Request': 'disconnect-1',
            'X-KVMem-Instance': fixture.manager.status()['instance']})
        response = connection.getresponse()
        self.assertTrue(response.readline().startswith(b'data: '))
        self.assertTrue(fixture.native.prefill_started.wait(3))
        response.close()
        connection.close()
        self.assertTrue(fixture.native.prefill_cancelled.wait(3))
        deadline = time.monotonic() + 3
        while fixture.chat.active and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(fixture.chat.active)
        self.assertEqual(fixture.chat.record('disconnect-1', client_id='browser-a')['state'], 'cancelled')

    def test_thinking_stream_and_simulated_stop_start(self):
        fixture = self.fixture
        payload = {**chat_payload(), 'chat_template_kwargs': {'enable_thinking': True}}
        _, body = http_request(fixture, 'POST', '/api/chat/completions', payload)
        reasoning = ''.join(choice.get('delta', {}).get('reasoning_content', '')
                            for row in events(body) for choice in row.get('choices', []))
        self.assertEqual(reasoning, 'reasoning from CPU fixture')
        self.assertEqual(content(body), 'fixture reply: hello')
        instance = fixture.manager.status()['instance']
        self.assertEqual(http_request(fixture, 'POST', '/api/launcher/stop', {})[0], 200)
        self.assertEqual(fixture.manager.status()['state'], 'stopped')
        status, released = http_request(fixture, 'POST', '/api/chat/release', {'conversation_id': 'http-conversation'})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(released)['released'])
        self.assertEqual(len(fixture.native.rows), 1)  # Confirmed stopped: no native release attempt.
        self.assertEqual(http_request(fixture, 'POST', '/api/launcher/start', {'profile': fixture.manager.draft()})[0], 200)
        self.assertEqual(fixture.manager.status()['state'], 'ready')
        self.assertNotEqual(instance, fixture.manager.status()['instance'])


def main():
    parser = argparse.ArgumentParser(description='CPU-only browser/HTTP fixture, not a real model')
    parser.add_argument('--port', type=int, default=18401)
    args = parser.parse_args()
    ended = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: ended.set())
    with running_fixture(args.port, sample_telemetry=True) as fixture:
        print(f'CPU HTTP fixture: http://127.0.0.1:{fixture.console.server_port}/', flush=True)
        print('Real ConsoleServer and ChatBridge; fake character tokenizer/native SSE; no model/GPU.', flush=True)
        print('Chat commands: [fixture:error] or [fixture:slow]. Ctrl-C stops both owned servers.', flush=True)
        ended.wait()


if __name__ == '__main__':
    main()
