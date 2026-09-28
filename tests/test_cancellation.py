from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
import time

from vllm_kvmem.cancellation import cancel_during_refresh


def test_socket_shutdown_interrupts_blocked_stream_without_waiting_for_http_read_lock(tmp_path):
    release = Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert payload['request_id'] == 'test-cancel'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.flush()
            # This simulates only the native trace signal, not a GPU test.
            (tmp_path / 'active.json').write_text(json.dumps({'started_at': time.perf_counter(),
                'requests': [{'request_id': 'cmpl-test-cancel-0-worker', 'command_id': 4,
                              'history_begin': 132096, 'query_tokens': 1024}]}))
            release.wait(5)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = cancel_during_refresh(f'http://127.0.0.1:{server.server_port}/v1/completions',
            {'request_id': 'test-cancel', 'prompt': [1, 2], 'max_tokens': 128},
            tmp_path / 'active.json', tmp_path / 'stream.jsonl', timeout=5)
        assert result['reader_ended'] and result['request_id'] == 'cmpl-test-cancel'
        assert result['cancelled_at'] >= result['trigger']['refresh_started_at']
        assert not release.is_set()  # Reader unblocked before server computation ended.
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
