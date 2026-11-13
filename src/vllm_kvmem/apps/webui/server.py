"""Loopback-only console hosting UI assets and the owned vLLM service."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from .telemetry import Telemetry

MAX_BODY = 32 * 1024 * 1024


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, manager, *, chat=None, telemetry=None, static_dir=None):
        if address[0] not in ('127.0.0.1', 'localhost'):
            raise ValueError('The management console listens on 127.0.0.1 only')
        if chat is None:
            from .chat import ChatBridge
            chat = ChatBridge(manager)
        self.manager, self.chat = manager, chat
        self.telemetry = telemetry or Telemetry(manager, chat)
        self.token = secrets.token_urlsafe(32)
        self.static_dir = Path(static_dir or Path(__file__).parent / 'static').resolve()
        super().__init__(address, Handler)

    def close(self):
        self.chat.cancel_all()
        self.manager.close()
        self.telemetry.close()
        self.server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'KVMemConsole/1'

    def log_message(self, *_):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status, payload, content_type='application/json; charset=utf-8'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def trusted(self, mutation=False):
        expected_port = self.server.server_port
        host = self.headers.get('Host', '')
        try:
            parsed = urlsplit('http://' + host)
            valid = (parsed.hostname in ('127.0.0.1', 'localhost') and
                     (parsed.port or 80) == expected_port and not parsed.username and
                     not parsed.password and not parsed.path and not parsed.query and not parsed.fragment)
        except ValueError:
            valid = False
        origin = self.headers.get('Origin')
        if origin is not None and origin != 'http://' + host:
            valid = False
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            valid = False
        if not valid:
            self.reply(403, {'error': 'Open this console on its local address'})
            return False
        token = self.headers.get('X-KVMem-Token', '')
        if mutation and (not token.isascii() or not hmac.compare_digest(token, self.server.token)):
            self.reply(403, {'error': 'Invalid console token; reload the page'})
            return False
        return True

    def client_id(self, payload=None):
        header = self.headers.get('X-KVMem-Client')
        body = payload.get('client_id') if isinstance(payload, dict) else None
        if header and body and header != body:
            raise ValueError('Conflicting client identity')
        return header or body

    def do_GET(self):
        if not self.trusted():
            return
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        manager = self.server.manager
        try:
            if path == '/api/launcher/info':
                self.reply(200, {'token': self.server.token, 'backend': 'vllm',
                    'backends': ['vllm'], 'console_protocol': 1, 'platform': 'linux',
                    'profile': manager.draft(), 'draft': manager.draft(),
                    'status': manager.status(), 'gpus': self.server.telemetry.hardware().get('gpus', []),
                    'capabilities': {'model_formats': ['safetensors', 'gguf'],
                        'draft_methods': ['mtp', 'dflash'], 'text_only': True,
                        'multi_gpu': False, 'script_export': False}})
            elif path == '/api/launcher/status':
                self.reply(200, manager.status())
            elif path in ('/api/launcher/logs', '/api/launcher/log'):
                revision = parse_qs(parsed.query).get('revision', [''])[0]
                self.reply(200, manager.logs(revision))
            elif path == '/api/launcher/hardware':
                self.reply(200, self.server.telemetry.hardware())
            elif path == '/api/monitor/snapshot':
                self.reply(200, self.server.telemetry.snapshot())
            elif path == '/props':
                self.reply(200, self.server.telemetry.props())
            elif path == '/slots':
                self.reply(200, [])
            elif path == '/api/chat/context':
                self.reply(200, self.server.chat.context())
            elif path.startswith('/api/chat/requests/'):
                record = self.server.chat.record(path.rsplit('/', 1)[1], client_id=self.client_id())
                self.reply(200 if record else 404, record or {'error': 'Unknown request'})
            else:
                self.asset(path)
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except (OSError, RuntimeError) as exc:
            self.reply(503, {'error': str(exc)})

    def asset(self, path):
        if path in ('/', '/monitor', '/monitor/', '/chat', '/chat/'):
            relative = 'monitor/index.html'
        elif path.startswith('/chat/view/'):
            relative = 'chat/view/' + (path[len('/chat/view/'):] or 'index.html')
        elif path.startswith('/monitor/'):
            relative = path[1:]
        elif path == '/history-export.html':
            relative = 'monitor/history-export.html'
        else:
            self.reply(404, {'error': 'Not found'})
            return
        candidate = (self.server.static_dir / relative).resolve()
        if (not candidate.is_relative_to(self.server.static_dir) or not candidate.is_file()
                or any(part.startswith('.') for part in Path(relative).parts)):
            self.reply(404, {'error': 'Not found'})
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or 'application/octet-stream'
        self.reply(200, candidate.read_bytes(), content_type)

    def read_payload(self):
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Expected application/json')
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('Chunked request bodies are not supported')
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError as exc:
            raise ValueError('A Content-Length is required') from exc
        if not 0 < length <= MAX_BODY:
            raise ValueError('Request body must be between 1 byte and 32 MiB')
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError('Incomplete request body')
        def invalid_constant(value):
            raise ValueError('Invalid JSON number: ' + value)
        payload = json.loads(data, parse_constant=invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        return payload

    def do_POST(self):
        if not self.trusted(mutation=True):
            return
        path = urlsplit(self.path).path
        try:
            payload = self.read_payload()
            manager, chat = self.server.manager, self.server.chat
            if path == '/api/chat/completions':
                payload['client_id'] = self.client_id(payload)
                request_id = self.headers.get('X-KVMem-Request', '')
                instance = self.headers.get('X-KVMem-Instance', '')
                self.stream(payload, request_id, instance)
                return
            if path == '/api/chat/cancel':
                result = chat.cancel(payload.get('id', ''), client_id=self.client_id(payload))
            elif path == '/api/chat/release':
                result = chat.release(payload.get('conversation_id', ''), client_id=self.client_id(payload))
            elif path == '/api/launcher/stop':
                chat.cancel_all()
                result = manager.stop()
            elif path in ('/api/launcher/save', '/api/launcher/preview', '/api/launcher/start'):
                profile = payload.get('profile')
                if not isinstance(profile, dict):
                    raise ValueError('profile must contain a serving configuration')
                if path.endswith('/start') and profile.get('port', 8000) == self.server.server_port:
                    raise ValueError('The engine and console must use different ports')
                operation = {'/api/launcher/save': manager.save, '/api/launcher/preview': manager.preview,
                             '/api/launcher/start': manager.start}[path]
                result = operation(profile)
            else:
                self.reply(404, {'error': 'Not found'})
                return
            self.reply(200, result if result is not None else {'ok': True})
        except (ValueError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
            self.reply(400, {'error': str(exc)})
        except (OSError, RuntimeError) as exc:
            self.reply(503, {'error': str(exc)})

    def stream(self, payload, request_id, instance):
        chat = self.server.chat
        client_id = payload['client_id']
        iterator = iter(chat.stream(payload, request_id, instance))
        # Validate before committing HTTP 200. The bridge yields an initial role
        # event before waiting for model prefill, so this cannot stall for prefill.
        first = next(iterator)
        ended = threading.Event()

        def disconnect_watch():
            while not ended.wait(.2):
                try:
                    ready, _, _ = select.select([self.connection], [], [], .2)
                    if ready and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                        chat.cancel(request_id, client_id=client_id)
                        return
                except BlockingIOError:
                    continue
                except OSError:
                    chat.cancel(request_id, client_id=client_id)
                    return

        watcher = threading.Thread(target=disconnect_watch, daemon=True, name='webui-disconnect')
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            watcher.start()
            self.wfile.write(first)
            self.wfile.flush()
            for chunk in iterator:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ValueError, RuntimeError):
            chat.cancel(request_id, client_id=client_id)
        finally:
            ended.set()
            iterator.close()
            if watcher.is_alive():
                watcher.join(timeout=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open the local KVMem for vLLM console')
    parser.add_argument('--config', type=Path, help='Initial serving JSON profile')
    parser.add_argument('--port', type=int, default=18400, help='Loopback console port (default: 18400)')
    parser.add_argument('--state-dir', type=Path, default=Path(os.environ.get('XDG_STATE_HOME',
        str(Path.home() / '.local/state'))) / 'kvmem-vllm', help='Private profiles, runs and logs')
    parser.add_argument('--no-browser', action='store_true', help='Print the address without opening a browser')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    assets = Path(__file__).parent / 'static'
    if not (assets / 'monitor/index.html').is_file() or not (assets / 'chat/view/index.html').is_file():
        parser.error('UI assets are missing. Build them with python scripts/build_webui.py or reinstall a built package.')
    from .manager import EngineManager
    manager = EngineManager(args.state_dir, config_path=args.config)
    try:
        server = ConsoleServer(('127.0.0.1', args.port), manager)
    except BaseException:
        manager.close()
        raise
    url = f'http://127.0.0.1:{server.server_port}/'
    print('KVMem console: ' + url, flush=True)
    print('Choose a model and start the engine from the configuration page.', flush=True)
    server.telemetry.start()
    old_handlers = {}
    def stop(*_):
        # HTTPServer.shutdown must be called from outside serve_forever's thread.
        threading.Thread(target=server.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
