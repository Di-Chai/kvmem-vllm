import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading

import pytest

spec = importlib.util.spec_from_file_location('agent_relay', Path(__file__).parents[1] / 'scripts/agent_api_relay.py')
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def test_metrics_use_first_to_last_generation_interval_and_exclude_first_token():
    body = json.dumps({'usage': {'completion_tokens': 101}, 'metrics': {
        'generation_time_ms': 1000, 'tokens_per_second': 3.3, 'time_to_first_token_ms': 29000}})
    assert relay.response_measurement(body)['native_decode_tokens_per_second'] == 100
    for tokens, ms in [(1, 0), (10, None), (2, float('nan')), (4, -1)]:
        body = json.dumps({'usage': {'completion_tokens': tokens}, 'metrics': {'generation_time_ms': ms}})
        assert relay.response_measurement(body)['native_decode_tokens_per_second'] is None


@pytest.mark.parametrize('status', [200, 400])
def test_relay_preserves_request_response_and_records_native_metrics(tmp_path, status):
    received = []
    response = b'{"id":"native-id","usage":{"completion_tokens":11},"metrics":{"generation_time_ms":200}}'
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers['Content-Length'])), self.headers.get('x-kvmem-task-id')))
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *args):
            pass
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    journal = tmp_path / 'requests.jsonl'
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), relay.make_handler('127.0.0.1', upstream.server_port, journal))
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
    for thread in threads:
        thread.start()
    client = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
    try:
        payload = b'{"model":"qualification", "messages":[{"role":"user","content":"execute tools"}]}'
        client.request('POST', '/v1/chat/completions', payload,
                       {'Content-Type': 'application/json', 'x-kvmem-task-id': 'task-a'})
        result = client.getresponse()
        assert result.status == status and result.read() == response
        assert received == [('/v1/chat/completions', payload, 'task-a')]
        client.request('POST', '/v1/chat/completions', b'{"stream":true}')
        result = client.getresponse()
        assert result.status == 400
        result.read()
        assert len(received) == 1
    finally:
        client.close()
        for server in (proxy, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()
    record = json.loads(journal.read_text())
    assert record['task_id'] == 'task-a' and record['status'] == status
    assert record['native_decode_tokens_per_second'] == 50
    assert record['response_id'] == 'native-id'
    assert record['response_delivered'] and not record['downstream_disconnected']


def test_downstream_disconnect_interrupts_silent_upstream_and_does_not_count_delivery(tmp_path):
    import socket
    import time
    started, aborted = threading.Event(), threading.Event()
    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            started.set()
            self.connection.settimeout(3)
            if self.connection.recv(1) == b'':
                aborted.set()
        def log_message(self, *args): pass
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    journal = tmp_path / 'requests.jsonl'
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), relay.make_handler('127.0.0.1', upstream.server_port, journal))
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, proxy)]
    for t in threads: t.start()
    client = socket.create_connection(('127.0.0.1', proxy.server_port), timeout=3)
    try:
        client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nContent-Length: 2\r\n\r\n{}')
        assert started.wait(3)
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        assert aborted.wait(3)  # Cancellation before any response headers, not after model completion.
        deadline = time.monotonic() + 3
        while not journal.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        record = json.loads(journal.read_text())
        assert record['downstream_disconnected'] and not record['response_delivered']
    finally:
        client.close()
        for s in (proxy, upstream): s.shutdown(); s.server_close()
        for t in threads: t.join()
