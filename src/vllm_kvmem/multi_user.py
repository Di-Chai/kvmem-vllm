"""Single-process tenant isolation and bounded admission for the public API.

This is pure ASGI middleware: an admission lease covers the complete downstream
application, including streamed response bodies and cancellation cleanup. One
reader owns the transport so queued and running disconnects cannot be lost to
competing ``receive`` calls. Deploy with one API server process.
"""
import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


_DEFAULTS = {
    'max_inflight': 4,
    'max_inflight_per_user': 2,
    'max_queue': 64,
    'queue_timeout_seconds': 60,
    'body_timeout_seconds': 30,
    'max_body_bytes': 8 * 1024 * 1024,
}
_GENERATE_PATHS = {'/v1/chat/completions', '/v1/completions'}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError('Non-finite JSON number')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite JSON number')
    return result


def _loads(value):
    return json.loads(value, object_pairs_hook=_object,
                      parse_constant=_invalid_constant, parse_float=_finite_float)


def _load_users(path):
    try:
        data = _loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ValueError('Cannot read a valid multi-user users file') from None
    if (not isinstance(data, dict) or set(data) != {'users'}
            or not isinstance(data['users'], list) or not data['users']):
        raise ValueError('Users file must contain a nonempty users array')
    users, identifiers = {}, set()
    for entry in data['users']:
        if not isinstance(entry, dict) or set(entry) != {'id', 'api_key'}:
            raise ValueError('Each user must contain only id and api_key')
        identifier, key = entry['id'], entry['api_key']
        if (not isinstance(identifier, str) or not identifier.strip()
                or len(identifier) > 128
                or any(ord(c) < 32 or ord(c) == 127
                       or 0xD800 <= ord(c) <= 0xDFFF for c in identifier)):
            raise ValueError('User IDs must be nonempty strings of at most 128 characters')
        if (not isinstance(key, str) or not key or len(key) > 4096
                or any(not 33 <= ord(c) <= 126 for c in key)):
            raise ValueError('API keys must be nonempty printable ASCII without whitespace')
        digest = hashlib.sha256(key.encode('ascii')).digest()
        if identifier in identifiers or digest in users:
            raise ValueError('User IDs and API keys must be unique')
        identifiers.add(identifier)
        users[digest] = identifier
    return users


def validate_config(settings, check_users=True):
    """Return canonical settings, never user records or API keys.

    ``settings`` is a dictionary (or its JSON representation). ``check_users``
    is useful to the launcher, which validates credentials before loading a GPU.
    Middleware construction always validates and loads the credential file.
    """
    if isinstance(settings, str):
        try:
            settings = _loads(settings)
        except (ValueError, RecursionError):
            raise ValueError('Multi-user configuration must be a JSON object') from None
    if (not isinstance(settings, dict)
            or set(settings) - (set(_DEFAULTS) | {'users_file'})):
        raise ValueError('Invalid or unknown multi-user configuration fields')
    path = settings.get('users_file')
    if not isinstance(path, str) or not path.strip() or '\x00' in path:
        raise ValueError('Multi-user configuration requires users_file')
    result = {**_DEFAULTS, **settings}
    for field in ('max_inflight', 'max_inflight_per_user', 'max_body_bytes'):
        if type(result[field]) is not int or result[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if type(result['max_queue']) is not int or result['max_queue'] < 0:
        raise ValueError('max_queue must be a nonnegative integer')
    for field in ('queue_timeout_seconds', 'body_timeout_seconds'):
        timeout = result[field]
        try:
            valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'{field} must be finite and positive')
    if result['max_inflight_per_user'] > result['max_inflight']:
        raise ValueError('max_inflight_per_user must not exceed max_inflight')
    result['users_file'] = str(Path(path).expanduser().resolve())
    if check_users:
        _load_users(result['users_file'])
    return result


@dataclass(eq=False)
class _Ticket:
    user: str
    ready: asyncio.Future
    started: float
    admitted: bool = False
    released: bool = False


class _Admission:
    """Event-loop-local round robin between eligible users, FIFO within a user."""

    def __init__(self, config):
        self.limit = config['max_inflight']
        self.per_user = config['max_inflight_per_user']
        self.queue_limit = config['max_queue']
        self.active = {}
        self.total = self.queued = 0
        self.waiting = {}
        self.turns = deque()

    def enter(self, user):
        loop = asyncio.get_running_loop()
        ticket = _Ticket(user, loop.create_future(), loop.time())
        if self.total < self.limit and self.active.get(user, 0) < self.per_user:
            self._grant(ticket)
        else:
            if self.queued >= self.queue_limit:
                return None
            if user not in self.waiting:
                self.waiting[user] = deque()
                self.turns.append(user)
            self.waiting[user].append(ticket)
            self.queued += 1
        return ticket

    def _grant(self, ticket):
        ticket.admitted = True
        self.total += 1
        self.active[ticket.user] = self.active.get(ticket.user, 0) + 1
        ticket.ready.set_result(None)

    def release(self, ticket):
        if ticket.released:
            return
        ticket.released = True
        if ticket.admitted:
            self.total -= 1
            self.active[ticket.user] -= 1
            if not self.active[ticket.user]:
                del self.active[ticket.user]
        else:
            queue = self.waiting[ticket.user]
            queue.remove(ticket)
            self.queued -= 1
            ticket.ready.cancel()
            if not queue:
                del self.waiting[ticket.user]
                self.turns.remove(ticket.user)
        self._drain()

    def _drain(self):
        while self.total < self.limit and self.turns:
            for _ in range(len(self.turns)):
                user = self.turns.popleft()
                if self.active.get(user, 0) >= self.per_user:
                    self.turns.append(user)
                    continue
                queue = self.waiting[user]
                ticket = queue.popleft()
                self.queued -= 1
                if queue:
                    self.turns.append(user)
                else:
                    del self.waiting[user]
                self._grant(ticket)
                break
            else:
                break  # Every queued user is already at its individual limit.


@dataclass
class _RequestError:
    status: int
    message: str


async def _error(send, status, message):
    body = json.dumps({'error': {'message': message, 'type': {
        401: 'AuthenticationError', 429: 'RateLimitError',
    }.get(status, 'BadRequestError'), 'code': status}}).encode('utf-8')
    headers = [(b'content-type', b'application/json'),
               (b'content-length', str(len(body)).encode('ascii'))]
    if status == 429:
        headers.append((b'retry-after', b'1'))
    if status == 401:
        headers.append((b'www-authenticate', b'Bearer'))
    await send({'type': 'http.response.start', 'status': status, 'headers': headers})
    await send({'type': 'http.response.body', 'body': body})


class MultiUserMiddleware:
    def __init__(self, app, config=None):
        self.app = app
        self.config = validate_config(
            os.environ.get('KVMEM_MULTI_USER_CONFIG') if config is None else config,
            check_users=False)
        self._users = _load_users(self.config['users_file'])
        self._admission = _Admission(self.config)

    def _authenticate(self, headers):
        values = [value for name, value in headers if name.lower() == b'authorization']
        if len(values) != 1:
            return None
        parts = values[0].split()
        if len(parts) != 2 or parts[0].lower() != b'bearer':
            return None
        return self._users.get(hashlib.sha256(parts[1]).digest())

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        path, method = scope['path'].rstrip('/'), scope['method']
        if path == '/health' and method == 'GET':
            await self.app(scope, receive, send)
            return
        if method == 'OPTIONS' and path in _GENERATE_PATHS | {'/v1/models'}:
            # Browser preflights carry no Bearer key. Native CORS policy handles
            # their headers; the subsequent generation still authenticates.
            await self.app(scope, receive, send)
            return
        headers = scope.get('headers', [])
        user = self._authenticate(headers)
        if user is None:
            await _error(send, 401, 'A valid Bearer API key is required')
            return
        if path == '/v1/models' and method == 'GET':
            await self.app(scope, receive, send)
            return
        if path not in _GENERATE_PATHS or method != 'POST':
            await _error(send, 404, 'Endpoint unavailable in multi-user mode')
            return
        lengths = [value for name, value in headers if name.lower() == b'content-length']
        if lengths:
            if len(lengths) != 1 or len(lengths[0]) > 20 or not lengths[0].isdigit():
                await _error(send, 400, 'Invalid Content-Length')
                return
            if int(lengths[0]) > self.config['max_body_bytes']:
                await _error(send, 413, 'Request body exceeds the configured limit')
                return
        ticket = self._admission.enter(user)
        if ticket is None:
            await _error(send, 429, 'Generation queue is full; retry later')
            return
        loop = asyncio.get_running_loop()
        body = loop.create_future()
        disconnected = asyncio.Event()
        reader = asyncio.create_task(self._read(receive, body, disconnected))
        handler = asyncio.create_task(self._dispatch(
            scope, send, ticket, body, disconnected))
        disconnect = asyncio.create_task(disconnected.wait())
        tasks = (handler, reader, disconnect)
        try:
            done, _ = await asyncio.wait((handler, disconnect),
                                         return_when=asyncio.FIRST_COMPLETED)
            if handler in done:
                await handler  # Preserve downstream exceptions for server logging.
        finally:
            # Cancellation must finish downstream generator/engine cleanup before
            # another request can consume this lease.
            for task in tasks:
                if not task.done():
                    task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # A second cancellation must not interrupt engine cleanup
                    # and release a slot while the first generation still runs.
                    cancelled = True
            self._admission.release(ticket)
            if cancelled:
                raise asyncio.CancelledError

    async def _read(self, receive, body, disconnected):
        chunks = bytearray()
        try:
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    disconnected.set()
                    return
                if message['type'] != 'http.request':
                    if not body.done():
                        body.set_result(_RequestError(400, 'Invalid request body'))
                    return
                if body.done():
                    continue
                chunk = message.get('body', b'')
                if len(chunks) + len(chunk) > self.config['max_body_bytes']:
                    body.set_result(_RequestError(413, 'Request body exceeds the configured limit'))
                    return
                chunks.extend(chunk)
                if not message.get('more_body', False):
                    body.set_result(bytes(chunks))
                    chunks.clear()
        except Exception:
            # A closed transport need not provide a final http.disconnect.
            disconnected.set()

    def _prepare(self, scope, raw, user):
        try:
            payload = _loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            return _RequestError(400, 'Request body must be valid JSON')
        if not isinstance(payload, dict):
            return _RequestError(400, 'Request body must be a JSON object')
        extras = payload.get('vllm_xargs')
        if extras is not None and not isinstance(extras, dict):
            return _RequestError(400, 'vllm_xargs must be an object')
        extras = dict(extras or {})
        sessions = [value.decode('latin-1') for name, value in scope.get('headers', [])
                    if name.lower() == b'x-session-id']
        candidates = [payload.get('session_id'), extras.get('session_id'), *sessions]
        if (len(sessions) > 1 or any(value is not None and (
                not isinstance(value, str) or len(value) > 1024
                or any(ord(c) < 32 or ord(c) == 127
                       or 0xD800 <= ord(c) <= 0xDFFF for c in value))
                for value in candidates)):
            return _RequestError(400, 'Session IDs must be strings of at most 1024 characters without controls')
        session = payload.get('session_id') or (sessions[0] if sessions else None) or extras.get('session_id')
        completion = scope['path'].rstrip('/') == '/v1/completions'
        echo_logprobs = (payload.get('logprobs') if completion
                         else payload.get('top_logprobs', 0))
        if session and (payload.get('prompt_logprobs') is not None
                        or (payload.get('echo') and echo_logprobs is not None)):
            # Restored checkpoints do not retain old prompt-token logprobs.
            # Native echo converts completion logprobs / chat top_logprobs to
            # prompt_logprobs, including chat's implicit top_logprobs=0 default.
            return _RequestError(400, 'Prompt logprobs are unavailable when session_id is set')
        if 'use_beam_search' in payload and payload['use_beam_search'] is not False:
            return _RequestError(400, 'Beam search is unavailable in multi-user mode')
        if payload.get('prompt_embeds') is not None:
            return _RequestError(400, 'Prompt embeddings are unavailable; only text inputs are supported')
        messages = payload.get('messages')
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue  # Native request validation handles malformed messages.
                content = message.get('content')
                if (message.get('audio') is not None
                        or (content is not None and not isinstance(content, (str, list)))
                        or (isinstance(content, list) and any(
                            not isinstance(part, dict) or part.get('type') != 'text'
                            for part in content))):
                    return _RequestError(400, 'Only text chat message content is supported')
        for name in ('n', 'best_of'):
            value = payload.get(name)
            if value is not None and (type(value) is not int or value != 1):
                return _RequestError(400, 'Multi-user requests support only n=1 and best_of=1')
        prompt = payload.get('prompt')
        if isinstance(prompt, list) and (not prompt or any(type(token) is not int for token in prompt)):
            return _RequestError(400, 'Batch prompts are unavailable in multi-user mode')
        # Normalize all namespace and engine identity inputs before forwarding.
        # Client IDs are not globally unique, even within one tenant.
        payload['cache_salt'] = hashlib.sha256(
            b'kvmem-tenant-v1\0' + user.encode('utf-8')).hexdigest()
        payload['request_id'] = uuid.uuid4().hex
        payload['priority'] = 0
        payload.pop('kv_transfer_params', None)
        payload.pop('session_id', None)
        if session:
            payload['session_id'] = session
        reserved = {'cache_salt', 'session_id', 'request_id', 'priority', 'kv_transfer_params'}
        for name in list(extras):
            if name.lower().replace('-', '_') in reserved:
                del extras[name]
        if 'vllm_xargs' in payload:
            payload['vllm_xargs'] = extras
        rewritten = json.dumps(payload, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
        hidden = {b'content-length', b'transfer-encoding', b'cache-salt', b'x-cache-salt',
                  b'cache_salt', b'x-cache_salt', b'x-session-id', b'x-request-id',
                  b'x-vllm-priority'}
        forwarded = dict(scope)
        forwarded['headers'] = [(name, value) for name, value in scope.get('headers', [])
                                if name.lower() not in hidden]
        forwarded['headers'].append((b'content-length', str(len(rewritten)).encode('ascii')))
        forwarded['state'] = {**scope.get('state', {}), 'kvmem_user_id': user}
        return forwarded, rewritten

    async def _dispatch(self, scope, send, ticket, body, disconnected):
        prepared = None
        deadline = ticket.started + self.config['queue_timeout_seconds']
        body_deadline = ticket.started + self.config['body_timeout_seconds']
        while True:
            if body.done() and prepared is None:
                value = body.result()
                prepared = value if isinstance(value, _RequestError) else self._prepare(scope, value, ticket.user)
                if isinstance(prepared, _RequestError):
                    await _error(send, prepared.status, prepared.message)
                    return
            if ticket.admitted and prepared is not None:
                break
            pending = [future for future in (ticket.ready, body) if not future.done()]
            deadlines = ([] if ticket.admitted else [deadline]) + ([] if body.done() else [body_deadline])
            remaining = max(0, min(deadlines) - asyncio.get_running_loop().time()) if deadlines else None
            done, _ = await asyncio.wait(pending, timeout=remaining,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                if not body.done() and asyncio.get_running_loop().time() >= body_deadline:
                    await _error(send, 408, 'Request body read timed out')
                else:
                    await _error(send, 429, 'Generation queue wait timed out; retry later')
                return
        forwarded, rewritten = prepared
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {'type': 'http.request', 'body': rewritten, 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}

        await self.app(forwarded, replay, send)
