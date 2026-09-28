"""Standard-library non-streaming relay for official agent calls and metrics.

Bind only to the intended local Docker bridge. Response JSON bytes are passed
unchanged; native --enable-per-request-metrics supplies decode timing. This is
not a general HTTP proxy or SSE measurement tool.
"""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import select
import socket
from pathlib import Path
import threading
import time


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}
MAX_BODY = 32 * 1024 * 1024


def response_measurement(body):
    value = json.loads(body)
    usage = value.get('usage') or {}
    metrics = value.get('metrics') or {}
    tokens = usage.get('completion_tokens')
    decode_ms = metrics.get('generation_time_ms')
    rate = None
    if (type(tokens) is int and tokens > 1 and type(decode_ms) in (int, float)
            and math.isfinite(decode_ms) and decode_ms > 0):
        rate = (tokens - 1) * 1000 / decode_ms
    return {'response_id': value.get('id'), 'usage': usage, 'native_metrics': metrics,
            'native_decode_tokens_per_second': rate,
            'finish_reasons': [choice.get('finish_reason') for choice in value.get('choices', [])],
            'error': value.get('error')}


def make_handler(upstream_host, upstream_port, output):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def log_message(self, *args):
            pass

        def forward(self):
            allowed = {'/v1/chat/completions', '/v1/completions', '/v1/models', '/health'}
            if self.path not in allowed:
                self.send_error(404)
                return
            if self.headers.get('Transfer-Encoding'):
                self.send_error(400, 'Chunked request bodies are not supported by this agent relay')
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= MAX_BODY:
                self.send_error(413)
                return
            payload = self.rfile.read(length)
            if len(payload) != length:
                self.close_connection = True
                return
            if self.command == 'POST':
                try:
                    request = json.loads(payload)
                except ValueError:
                    self.send_error(400)
                    return
                if not isinstance(request, dict) or request.get('stream'):
                    self.send_error(400, 'This relay only qualifies non-streaming agent calls')
                    return
            record = {'submitted_monotonic': time.monotonic(), 'submitted_unix': time.time(),
                      'path': self.path, 'task_id': self.headers.get('x-kvmem-task-id'),
                      'request_bytes': length, 'request_sha256': hashlib.sha256(payload).hexdigest()}
            conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=21600)
            replied = False
            stopped, disconnected = threading.Event(), threading.Event()
            watcher = None
            record['response_delivered'] = False
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS | {'accept-encoding'}}
                headers['Accept-Encoding'] = 'identity'
                conn.request(self.command, self.path, body=payload, headers=headers)
                upstream_socket = conn.sock
                def watch_disconnect():
                    while not stopped.wait(.05):
                        try:
                            readable, _, _ = select.select([self.connection], [], [], 0)
                            if not readable or self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                                continue
                        except BlockingIOError:
                            continue
                        except OSError:
                            pass
                        disconnected.set()
                        # close() alone can leave getresponse()/read() blocked on
                        # their socket file wrapper. Shutdown interrupts both and
                        # lets native ASGI cancellation abort the model request.
                        try:
                            upstream_socket.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        break
                watcher = threading.Thread(target=watch_disconnect, daemon=True)
                watcher.start()
                response = conn.getresponse()
                body = response.read(MAX_BODY + 1)
                if len(body) > MAX_BODY:
                    raise ValueError('Upstream response exceeded bounded relay buffer')
                record.update(upstream_finished_monotonic=time.monotonic(), status=response.status,
                              response_bytes=len(body), response_sha256=hashlib.sha256(body).hexdigest())
                try:
                    record.update(response_measurement(body))
                except (ValueError, TypeError, AttributeError) as error:
                    record['metrics_parse_error'] = str(error)
                self.send_response(response.status)
                for name, value in response.getheaders():
                    if name.lower() not in HOP_HEADERS:
                        self.send_header(name, value)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                replied = True
                self.wfile.write(body)
                self.wfile.flush()
                record['response_delivered'] = True
            except (OSError, ValueError, http.client.HTTPException) as error:
                record['relay_error'] = repr(error)
                if not replied and not disconnected.is_set():
                    try:
                        self.send_error(502)
                    except OSError:
                        disconnected.set()
                self.close_connection = True
            finally:
                stopped.set()
                if watcher is not None:
                    watcher.join(timeout=1)
                record['downstream_disconnected'] = disconnected.is_set()
                conn.close()
                record['finished_monotonic'] = time.monotonic()
                with lock, output.open('a') as stream:
                    stream.write(json.dumps(record) + '\n')

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen-host', required=True)
    parser.add_argument('--listen-port', type=int, default=443)
    parser.add_argument('--upstream-host', default='127.0.0.1')
    parser.add_argument('--upstream-port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadingHTTPServer((args.listen_host, args.listen_port), make_handler(
            args.upstream_host, args.upstream_port, args.output)) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
