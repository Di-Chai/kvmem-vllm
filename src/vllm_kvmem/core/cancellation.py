"""Interrupt an owned HTTP stream while a native semantic refresh is active.

Used by qualification only. Socket shutdown interrupts a blocked read without
waiting for HTTPResponse's read lock. No retry or server/process cancellation is
performed; native vLLM must observe the real client disconnect and reclaim state.
"""
import http.client
import json
from pathlib import Path
import socket
from threading import Event, Thread
import time
from urllib.parse import urlsplit


def cancel_during_refresh(url, payload, active_path, trace_path, *, timeout=3600):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or not payload.get('request_id'):
        raise ValueError('Cancellation probe requires local HTTP and an explicit request_id')
    request_id = 'cmpl-' + payload['request_id']
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    ready, ended = Event(), Event()
    state = {}
    cancelled_at = None
    trigger = None

    def reader():
        try:
            body = json.dumps({**payload, 'stream': True, 'return_token_ids': True})
            connection.request('POST', parsed.path, body, {'Content-Type': 'application/json'})
            state['socket'] = connection.sock
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f'Cancellation request returned HTTP {response.status}')
            ready.set()
            with Path(trace_path).open('x') as output:
                for line in response:
                    text = line.decode('utf-8').strip()
                    if text:
                        output.write(json.dumps({'at': time.perf_counter(), 'line': text}) + '\n')
                        output.flush()
                    if text == 'data: [DONE]':
                        state['done'] = True
            response.close()
        except Exception as error:
            state['read_error'] = repr(error)
        finally:
            connection.close()
            ended.set()

    thread = Thread(target=reader, name='kvmem-cancelled-http', daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if ended.is_set():
                raise RuntimeError(f'Request ended before refresh cancellation: {state.get("read_error")}')
            if ready.is_set():
                try:
                    active = json.loads(Path(active_path).read_text())
                    candidates = [item for item in active['requests'] if
                        item['request_id'].startswith(request_id + '-0-') and item['history_begin'] > 131072]
                    if candidates:
                        trigger = {**candidates[0], 'refresh_started_at': active['started_at']}
                        cancelled_at = time.perf_counter()
                        state['socket'].shutdown(socket.SHUT_RDWR)
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            ended.wait(0.02)
        else:
            raise TimeoutError('No live compressed refresh reached the cancellation trigger')
        thread.join(10)
        if thread.is_alive() or state.get('done'):
            raise RuntimeError('Owned stream did not terminate as an interrupted request')
        return {'request_id': request_id, 'prompt_tokens': len(payload['prompt']),
                'cancelled_at': cancelled_at, 'trigger': trigger,
                'reader_ended': True, 'read_error': state.get('read_error')}
    finally:
        sock = state.get('socket')
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        thread.join(10)
