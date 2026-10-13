"""Real loopback sockets with synthetic SSE; these are not GPU benchmarks."""
from contextlib import contextmanager
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import threading
import time

import pytest


spec = importlib.util.spec_from_file_location(
    'multi_user_gate', Path(__file__).parents[1] / 'scripts/qualify_multi_user.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


KEYS = ('secret-tenant-one-key', 'secret-tenant-two-key')


@contextmanager
def streaming_fixture(keys=KEYS):
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

        def handle_error(self, request, client_address):
            pass  # Expected connection resets from deliberate cancellation.

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def reject(self, status):
            self.send_response(status)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            key = self.headers.get('Authorization', '').removeprefix('Bearer ')
            if key not in keys:
                return self.reject(401)
            mode = payload.get('fixture_mode', 'normal')
            if mode == 'reject':
                return self.reject(429)
            if mode == 'bad_type':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'{}')
                return
            session = payload.get('session_id', '')
            tenant_session = (key, session)
            with server.lock:
                if session.startswith('cancel-') and tenant_session in server.active_sessions:
                    return self.reject(409)
                server.active_sessions.add(tenant_session)
                server.active += 1
                server.peak_active = max(server.peak_active, server.active)
                server.calls.append((key, payload))
            prompt = payload.get('prompt') or payload['messages'][0]['content']
            marker = re.search(r'private retrieval marker is (KVMEM_[A-Za-z0-9_]+)', prompt)
            answer = marker.group(1) if marker else 'fixture output'

            def event(value):
                wire = value if isinstance(value, str) else json.dumps(value)
                self.wfile.write(('data: ' + wire + '\r\n\r\n').encode())
                self.wfile.flush()

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            try:
                event({'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]})
                time.sleep(.025)
                if mode == 'invalid_json':
                    event('not JSON ' + key)
                    return
                if mode == 'stream_error':
                    event({'error': {'message': key}})
                    return
                event({'choices': [{'delta': {'content': answer}, 'finish_reason': None}]})
                if payload.get('ignore_eos'):
                    for _ in range(100):
                        time.sleep(.01)
                        event({'choices': [{'delta': {'content': ' more'}, 'finish_reason': None}]})
                if mode != 'missing_finish':
                    event({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
                if mode != 'no_usage':
                    event({'choices': [], 'usage': {'prompt_tokens': 31, 'completion_tokens': 7}})
                if mode != 'missing_done':
                    event('[DONE]')
            except OSError:
                with server.lock:
                    server.disconnects += 1
            finally:
                with server.lock:
                    server.active_sessions.discard(tenant_session)
                    server.active -= 1

    server = Server(('127.0.0.1', 0), Handler)
    server.lock = threading.Lock()
    server.active = server.peak_active = server.disconnects = 0
    server.active_sessions, server.calls = set(), []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def users_file(tmp_path):
    path = tmp_path / 'users.json'
    path.write_text(json.dumps({'users': [
        {'id': 'private-one', 'api_key': KEYS[0]},
        {'id': 'private-two', 'api_key': KEYS[1]}]}))
    return path


def test_sse_parser_handles_multiline_comments_and_crlf():
    data = io.BytesIO(b': heartbeat\r\nid: event-1\r\ndata: {"n":\r\ndata: 3}\r\n\r\n'
                      b'data:[DONE]\n\n')
    events = list(gate.iter_sse(data.readline))
    assert json.loads(events[0]) == {'n': 3}
    assert events[1] == '[DONE]'


@pytest.mark.parametrize('wire,error', [
    (b'data: {}\n', 'sse_event_truncated'),
    (b'data: {}', 'sse_line_truncated'),
    (b'data: \xff\n\n', 'sse_invalid_utf8'),
    (b'data: ' + b'x' * 64 + b'\n\n', 'sse_event_too_large'),
])
def test_sse_parser_rejects_incomplete_or_unbounded_events(wire, error):
    with pytest.raises(gate.StreamError, match=error):
        list(gate.iter_sse(io.BytesIO(wire).readline, max_event_bytes=32))


def test_load_user_validation_never_echoes_bad_credentials(tmp_path, users_file):
    assert len(gate.load_users(users_file)) == 2
    for entries in ([], [{'id': 'one', 'api_key': 'secret\nvalue'}],
                    [{'id': 'one', 'api_key': 'key'}, {'id': 'two', 'api_key': 'key'}],
                    [{'id': 'one', 'api_key': 'one'}, {'id': 'one', 'api_key': 'two'}]):
        path = tmp_path / 'invalid.json'
        path.write_text(json.dumps({'users': entries}))
        with pytest.raises(ValueError) as error:
            gate.load_users(path)
        assert 'secret\nvalue' not in str(error.value)


def test_summary_distinguishes_zero_success_overload_and_quality():
    def record(outcome, start, **extra):
        return dict(outcome=outcome, started_seconds=start, elapsed_seconds=.2,
                    http_status=200 if outcome == 'success' else 429,
                    completion_tokens=7 if outcome == 'success' else None,
                    ttft_seconds=.1 if outcome == 'success' else None, **extra)
    rejected = record('rejected', 0)
    assert gate.summarize([rejected], 1, .01)['status'] == 'failed'
    good = record('success', .1)
    summary = gate.summarize([rejected, good], 2)
    assert summary['status'] == 'capacity_limited'
    assert summary['arrival_rate_over_start_span'] == 10
    assert summary['successful_requests_per_second_over_wall'] == .5
    assert summary['output_tokens_per_second_over_wall'] == 3.5
    assert summary['rejected'] == 1 and summary['failure'] == 0
    assert gate.summarize([rejected, good], 2, .5)['status'] == 'passed'
    assert gate.summarize([record('success', 0, marker_exact=False)], 1)['status'] == 'failed'
    assert gate.summarize([record('success', 0, foreign_marker_observed=True)], 1)['status'] == 'failed'
    good['completion_tokens'] = None
    assert gate.summarize([good], 1)['output_tokens_per_second_over_wall'] is None
    assert gate.summarize([], 1)['status'] == 'failed'


def test_percentile_interpolates_and_handles_empty_singleton():
    assert gate.distribution([]) == {'count': 0, 'p50': None, 'p95': None, 'max': None}
    assert gate.percentile([9], .95) == 9
    assert gate.percentile([1, 2, 3, 4], .5) == 2.5
    assert gate.percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)


@pytest.mark.parametrize('mode,error', [
    ('missing_done', 'missing_done'), ('missing_finish', 'missing_finish_reason'),
    ('invalid_json', 'invalid_stream_json'), ('stream_error', 'stream_error_event'),
    ('bad_type', 'unexpected_content_type'),
])
def test_real_http_protocol_failures_do_not_become_success(mode, error):
    with streaming_fixture() as (_, base):
        result = gate.stream_request(base, 'completions', KEYS[0],
                                     {'prompt': 'hello', 'fixture_mode': mode})
    assert result['outcome'] == 'failure'
    assert result['error'] == error
    assert not any(key in json.dumps(result) for key in KEYS)


def test_real_http_latency_uses_first_content_and_usage_not_chunks():
    with streaming_fixture() as (_, base):
        result = gate.stream_request(base, 'completions', KEYS[0], {'prompt': 'hello'})
        no_usage = gate.stream_request(base, 'completions', KEYS[0],
                                      {'prompt': 'hello', 'fixture_mode': 'no_usage'})
        rejected = gate.stream_request(base, 'completions', KEYS[0],
                                       {'prompt': 'hello', 'fixture_mode': 'reject'})
        unauthenticated = gate.stream_request(base, 'completions', 'unknown', {'prompt': 'hello'})
    assert result['outcome'] == no_usage['outcome'] == 'success'
    assert result['content_chunks'] == 1 and result['completion_tokens'] == 7
    assert .02 <= result['ttft_seconds'] <= result['elapsed_seconds']
    assert no_usage['completion_tokens'] is None
    assert rejected['outcome'] == 'rejected' and rejected['http_status'] == 429
    assert unauthenticated['outcome'] == 'failure' and unauthenticated['http_status'] == 401


def test_complete_cli_with_concurrent_sockets_collision_and_cancel_recovery(tmp_path, users_file, capsys):
    out = tmp_path / 'run'
    with streaming_fixture() as (server, base):
        code = gate.main(['--api-base', base, '--users-file', str(users_file), '--output', str(out),
                          '--concurrency', '1', '4', '--rounds', '2', '--prompt-words', '4',
                          '--output-tokens', '64', '--endpoint', 'chat/completions',
                          '--disable-thinking', '--check-markers', '--probes', '--timeout', '5'])
        assert server.peak_active >= 2
        assert server.disconnects >= 1
        assert any(call[1].get('chat_template_kwargs') == {'enable_thinking': False}
                   for call in server.calls)
    assert code == 0
    serialized = (out / 'results.json').read_text()
    report = json.loads(serialized)
    assert report['status'] == 'passed'
    assert [level['summary']['success'] for level in report['levels']] == [2, 8]
    assert report['levels'][1]['distinct_users'] == 2
    assert report['probes']['status'] == 'passed'
    assert report['probes']['cancel']['outcome'] == 'cancelled'
    assert report['probes']['cancel']['saw_done'] is False
    assert report['probes']['recovery_attempts'][-1]['marker_exact']
    output = capsys.readouterr().out
    for secret in (*KEYS, 'private-one', 'private-two'):
        assert secret not in serialized + output


def test_cli_rejects_ambiguous_load_quality_and_secret_urls(tmp_path, users_file):
    common = ['--users-file', str(users_file), '--output', str(tmp_path / 'run')]
    for additional in (['--ignore-eos', '--check-markers'], ['--ignore-eos', '--probes'],
                       ['--concurrency', '0'], ['--rounds', '0'], ['--timeout', 'nan'],
                       ['--api-base', 'http://secret@127.0.0.1/v1'],
                       ['--api-base', 'http://127.0.0.1/v1?api_key=secret']):
        with pytest.raises(SystemExit):
            gate.parse_args(common + additional)


def test_foreign_marker_detection_requires_complete_identifier_boundaries():
    # Regression: old PRIVATE_1 was incorrectly found in user 10's valid output.
    assert not gate.contains_marker('KVMEM_run_PRIVATE_10', 'KVMEM_run_PRIVATE_1')
    assert not gate.contains_marker('prefixKVMEM_run_PRIVATE_1', 'KVMEM_run_PRIVATE_1')
    assert not gate.contains_marker('KVMEM_run_PRIVATE_001_END_extra', 'KVMEM_run_PRIVATE_001_END')
    assert gate.contains_marker('"KVMEM_run_PRIVATE_001_END".', 'KVMEM_run_PRIVATE_001_END')
    assert gate.contains_marker('KVMEM_run_PRIVATE_010_END\nKVMEM_run_PRIVATE_001_END',
                                'KVMEM_run_PRIVATE_001_END')


def test_twelve_users_do_not_report_tenant_one_as_leaked_in_tenant_ten(tmp_path):
    users = [{'id': f'user-{index}', 'api_key': f'secret-tenant-{index}-key'} for index in range(12)]
    with streaming_fixture(tuple(user['api_key'] for user in users)) as (server, base):
        args = gate.parse_args(['--api-base', base, '--users-file', str(tmp_path / 'unused.json'),
                               '--output', str(tmp_path / 'unused-output'), '--prompt-words', '4',
                               '--output-tokens', '64', '--timeout', '5'])
        report = gate.run_probes(args, users, 'twelvetenants')
        requests = [call[1] for call in server.calls if call[1]['session_id'].startswith('adversarial-')]
    assert report['status'] == 'passed'
    assert len(requests) == 24
    assert any('PRIVATE_001_END' in request['prompt'] for request in requests)
    assert any('PRIVATE_010_END' in request['prompt'] for request in requests)
    for wave in report['tenant_collision_waves']:
        assert len(wave) == 12
        assert all(result['marker_exact'] and not result['foreign_marker_observed'] for result in wave)


def test_redaction_preserves_json_numbers_and_escapes():
    result = gate.redact_strings({'count': 123, 'value': 'model-1-with-"secret"'}, ['1', '"secret"'])
    assert json.loads(json.dumps(result)) == {'count': 123, 'value': 'model-[REDACTED]-with-[REDACTED]'}
