"""Text chat adapter for an owned vLLM server, with verified token histories.

The browser owns durable messages. This bridge retains only bounded, in-memory
accepted token prefixes for the current engine. A canonical template render that
does not match that prefix starts a new engine session and reports the rebuild.
"""
from array import array
from contextlib import nullcontext
from dataclasses import dataclass, field
import copy
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from ..config import KVMemConfig


MAX_BODY = 16 * 1024 * 1024
MAX_RECORDS = 100
MAX_CONVERSATIONS = 32
MAX_HISTORY_TOKENS = 4 * 1024 * 1024
PARAMS = {'temperature', 'top_p', 'top_k', 'min_p', 'seed', 'max_tokens',
          'presence_penalty', 'frequency_penalty', 'repetition_penalty'}


def _identity(value, name):
    if (not isinstance(value, str) or not value or len(value) > 200
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f'Invalid {name}')
    return value


def _tokens(value):
    if not isinstance(value, list) or any(type(t) is not int or not 0 <= t < 2**32 for t in value):
        raise ValueError('Engine/tokenizer did not return valid token IDs')
    return value


def _sse(value):
    return ('data: ' + json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n\n').encode()


def _load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def _query_begin(tokenizer, messages, template, rendered, prompt):
    """Anchor the last user's content in the actual complete template encoding.

    Rendering a truncated conversation is insufficient: Qwen templates can strip
    earlier reasoning depending on the final user turn. The sentinel render only
    locates this turn's character boundary; all token coordinates refer to the
    original complete prompt, and a BPE boundary token stays inside the query.
    """
    marker = '__kvmem_query_' + uuid4().hex + '__'
    marked = [*messages[:-1], {**messages[-1], 'content': marker}]
    probe = tokenizer.apply_chat_template(marked, tokenize=False, add_generation_prompt=True, **template)
    if not isinstance(probe, str) or probe.count(marker) != 1:
        raise ValueError('This chat template cannot identify the final user query')
    prefix, suffix = probe.split(marker)
    if not rendered.startswith(prefix) or (suffix and not rendered.endswith(suffix)):
        raise ValueError('This chat template changes structure with user content')
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        if encoded['input_ids'] == prompt:
            offsets = encoded['offset_mapping']
            if len(offsets) == len(prompt):
                for index, (begin, end) in enumerate(offsets):
                    if end > len(prefix):
                        return index
    except (TypeError, ValueError, KeyError, NotImplementedError, AttributeError):
        pass
    # Slow tokenizers may lack offsets. The proven token-prefix intersection is
    # conservative and includes any BPE token that crosses the content boundary.
    prefix_ids = _tokens(tokenizer.encode(prefix, add_special_tokens=False))
    begin = 0
    for left, right in zip(prefix_ids, prompt):
        if left != right:
            break
        begin += 1
    if begin >= len(prompt):
        raise ValueError('The chat template did not expose a new user query')
    return begin


@dataclass
class _Conversation:
    instance: str
    tokenizer_key: tuple
    session: str = field(default_factory=lambda: 'webui-' + uuid4().hex)
    tokens: array = field(default_factory=lambda: array('I'))
    lock: threading.Lock = field(default_factory=threading.Lock)
    reserved: int = 0
    users: int = 0
    dirty: bool = False


class _TextParser:
    """Split Qwen reasoning and hide special markers across arbitrary SSE cuts."""
    def __init__(self, thinking, special_tokens=()):
        self.reasoning = thinking
        self.buffer = ''
        self.markers = set(special_tokens) | {'<think>', '</think>', '<|im_end|>', '<|endoftext|>'}
        self.markers.discard('')

    def feed(self, text, final=False):
        self.buffer += text
        result = []
        while self.buffer:
            found = [(self.buffer.find(mark), mark) for mark in self.markers if mark in self.buffer]
            if found:
                offset, mark = min(found, key=lambda item: item[0])
                if offset:
                    result.append(('reasoning_content' if self.reasoning else 'content', self.buffer[:offset]))
                self.buffer = self.buffer[offset + len(mark):]
                if mark == '<think>':
                    self.reasoning = True
                elif mark == '</think>':
                    self.reasoning = False
                continue
            keep = 0
            if not final:
                for mark in self.markers:
                    for size in range(1, min(len(mark), len(self.buffer) + 1)):
                        if self.buffer.endswith(mark[:size]):
                            keep = max(keep, size)
            ready = self.buffer[:-keep] if keep else self.buffer
            if ready:
                result.append(('reasoning_content' if self.reasoning else 'content', ready))
            self.buffer = self.buffer[-keep:] if keep else ''
            break
        return result


class ChatBridge:
    def __init__(self, manager, *, tokenizer_loader=None, connection_factory=None):
        self.manager = manager
        self.lock = threading.RLock()
        self.active = {}
        self.records = {}
        self.conversations = {}
        self.tokenizer_loader = tokenizer_loader or _load_tokenizer
        self.connection_factory = connection_factory or http.client.HTTPConnection
        self._tokenizer_key = None
        self._tokenizer = None

    def _target(self, expected=None):
        state = self.manager.status()
        if state.get('state') != 'ready':
            raise ValueError('No ready model; start a model first')
        if expected is not None and state.get('instance') != expected:
            raise ValueError('The running model changed; refresh chat settings')
        return state

    def context(self):
        with getattr(self.manager, 'lock', nullcontext()):
            state = self.manager.status()
            profile = copy.deepcopy(state.get('active_profile') or self.manager.active_profile)
        result = {'schema_version': 1, 'state': state.get('state'), 'instance': state.get('instance'),
                  'backend': 'vllm', 'model': None, 'effective': {}, 'sources': {},
                  'capabilities': {'thinking': True, 'budget': False, 'preserve': False,
                                   'efforts': [], 'vision': False, 'tools': False},
                  'sampling_by_mode': {}, 'sampling_by_effort': {}, 'budget_min': 0,
                  'history_export_url': None}
        if state.get('state') != 'ready':
            return result
        if not isinstance(profile, dict):
            raise ValueError('The running model changed; refresh chat settings')
        memory = profile.get('kvmem', {})
        limit = profile.get('max_model_len', 16384)
        output = limit if memory.get('rolling_active_window') else memory.get('gen_budget', KVMemConfig.gen_budget)
        result.update(model=profile.get('served_model_name', 'kvmem'), context=limit, limit=output,
                      display_model=Path(profile.get('model', '')).name,
                      effective={'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                                 'max_tokens': min(1024, output), 'enable_thinking': False})
        result['sources'] = {name: 'console' for name in result['effective']}
        return result

    def _get_tokenizer(self, profile, instance):
        path = str(Path(profile['tokenizer']).expanduser().resolve())
        stamps = []
        for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'tokenizer.model', 'special_tokens_map.json', 'added_tokens.json'):
            source = Path(path) / name
            try:
                stat = source.stat()
                stamps.append((name, stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                pass
        key = (instance, path, tuple(stamps))
        with self.lock:
            if key != self._tokenizer_key:
                self._tokenizer = self.tokenizer_loader(path)
                self._tokenizer_key = key
            return self._tokenizer, key

    def _validate(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Chat request must be an object')
        allowed = PARAMS | {'model', 'messages', 'stream', 'client_id', 'conversation_id',
                            'chat_template_kwargs'}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError('Unsupported chat fields: ' + ', '.join(sorted(unknown)))
        client = _identity(payload.get('client_id'), 'browser identity')
        conversation = _identity(payload.get('conversation_id'), 'conversation identity')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages or len(messages) > 10000:
            raise ValueError('messages must be a nonempty text conversation')
        for index, message in enumerate(messages):
            if (not isinstance(message, dict) or set(message) - {'role', 'content', 'reasoning_content'}
                    or message.get('role') not in ('system', 'user', 'assistant')
                    or not isinstance(message.get('content'), str)
                    or ('reasoning_content' in message and not isinstance(message['reasoning_content'], str))
                    or (message['role'] != 'assistant' and 'reasoning_content' in message)):
                raise ValueError('This console supports text system/user/assistant messages only')
            if message['role'] == 'system' and index != 0:
                raise ValueError('The system message must be first')
        if messages[-1]['role'] != 'user':
            raise ValueError('The last message must be a user message')
        template = payload.get('chat_template_kwargs', {})
        if (not isinstance(template, dict) or set(template) - {'enable_thinking'}
                or ('enable_thinking' in template and type(template['enable_thinking']) is not bool)):
            raise ValueError('Only the boolean enable_thinking template setting is supported')
        template = {'enable_thinking': False, **template}
        for name in PARAMS & payload.keys():
            value = payload[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'Invalid {name}')
            if name in ('max_tokens', 'top_k', 'seed') and type(value) is not int:
                raise ValueError(f'{name} must be an integer')
            bounds = {'temperature': (0, 2), 'top_p': (0, 1), 'min_p': (0, 1),
                      'presence_penalty': (-2, 2), 'frequency_penalty': (-2, 2),
                      'repetition_penalty': (0.001, 100), 'max_tokens': (1, 2**31 - 1),
                      'top_k': (-1, 2**31 - 1), 'seed': (0, 2**32 - 1)}
            if not bounds[name][0] <= value <= bounds[name][1] or (name == 'top_k' and value == 0):
                raise ValueError(f'{name} is out of range')
        if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_BODY:
            raise ValueError('Chat request exceeds 16 MiB')
        return client, conversation, messages, template

    def _connect(self, instance, active=None):
        with getattr(self.manager, 'lock', nullcontext()):
            self._target(instance)
            parsed = urlsplit(self.manager.base_url)
            if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
                raise ValueError('The console requires an owned loopback HTTP backend')
            conn = self.connection_factory(parsed.hostname, parsed.port, timeout=300)
            if active is not None:
                active['connection'] = conn
            try:
                conn.connect()
                if active is not None:
                    active['socket'] = conn.sock
                    if active['cancel'].is_set():
                        raise InterruptedError('Chat cancelled')
                self._target(instance)
                return conn
            except BaseException:
                conn.close()
                raise

    def _headers(self):
        return {'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                **getattr(self.manager, 'backend_headers', {})}

    def _release_native(self, conversation, client):
        with getattr(self.manager, 'lock', nullcontext()):
            state = self.manager.status()
            if state.get('instance') != conversation.instance:
                return
            if state.get('state') in ('stopped', 'failed', 'exited') and not state.get('pid'):
                # The owned worker is gone, so its in-memory KV session is gone.
                # Keep loading/live processes fenced; a transient lack of ready
                # status is not permission to forget a live retained session.
                return
            profile = copy.deepcopy(state.get('active_profile') or self.manager.active_profile)
            conn = self._connect(conversation.instance)
        try:
            body = {'model': profile.get('served_model_name', 'kvmem'), 'prompt': [1],
                    'session_id': conversation.session, 'cache_salt': self._salt(client),
                    'max_tokens': 1, 'temperature': 0,
                    'kv_transfer_params': {'kvmem_release_session': True}}
            if profile.get('kvmem', {}).get('request_planning'):
                body.update(kvmem_reselect='off', kvmem_mandatory_spans=[])
            conn.request('POST', '/v1/completions', json.dumps(body), self._headers())
            response = conn.getresponse()
            response.read(65537)
            if response.status != 200:
                raise ValueError('Engine session release failed; retry closing the conversation')
        finally:
            conn.close()

    @staticmethod
    def _salt(client):
        return 'webui-' + hashlib.sha256(client.encode()).hexdigest()

    def record(self, request_id, *, client_id=None):
        _identity(client_id, 'browser identity')
        with self.lock:
            row = self.records.get(request_id)
            return self._record_view(row) if row and row['client_id'] == client_id else None

    @staticmethod
    def _record_view(row):
        result = copy.deepcopy(row)
        result['elapsed_ms'] = max(0, result.get('finished_at_ms', int(time.time() * 1000)) - result['started_at_ms'])
        return result

    def snapshot(self):
        with self.lock:
            return {'active': [self._record_view(row) for key, row in self.records.items() if key in self.active],
                    'recent': [self._record_view(row) for key, row in reversed(self.records.items()) if key not in self.active][:MAX_RECORDS]}

    def cancel(self, request_id, *, client_id=None):
        _identity(client_id, 'browser identity')
        with self.lock:
            active = self.active.get(request_id)
            if not active or active['client_id'] != client_id:
                return {'cancelled': False}
            active['cancel'].set()
            sock = active.get('socket')
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if active.get('connection'):
                active['connection'].close()
            return {'cancelled': True}

    def cancel_all(self):
        with self.lock:
            pending = [(key, row['client_id']) for key, row in self.active.items()]
        for key, client in pending:
            self.cancel(key, client_id=client)

    def release(self, conversation_id, *, client_id=None):
        key = (_identity(client_id, 'browser identity'), _identity(conversation_id, 'conversation identity'))
        with self.lock:
            conversation = self.conversations.get(key)
            if conversation is None:
                return {'released': False}
            if conversation.users or not conversation.lock.acquire(blocking=False):
                raise ValueError('Cancel the active reply before closing this conversation')
        try:
            self._release_native(conversation, client_id)
            with self.lock:
                self.conversations.pop(key, None)
            return {'released': True}
        finally:
            conversation.lock.release()

    def stream(self, payload, request_id, instance):
        client, conversation_id, messages, template = self._validate(payload)
        _identity(request_id, 'request identity')
        _identity(instance, 'engine identity')
        with getattr(self.manager, 'lock', nullcontext()):
            state = self._target(instance)
            profile = copy.deepcopy(state.get('active_profile') or self.manager.active_profile)
        if not isinstance(profile, dict):
            raise ValueError('The running model changed; refresh chat settings')
        if payload.get('model', profile.get('served_model_name', 'kvmem')) != profile.get('served_model_name', 'kvmem'):
            raise ValueError('Requested model differs from the loaded model')
        active = {'client_id': client, 'cancel': threading.Event()}
        record = {'id': request_id, 'client_id': client, 'conversation_id': conversation_id,
                  'instance': instance, 'backend': 'vllm', 'state': 'waiting',
                  'phase': 'queued', 'started_at_ms': int(time.time() * 1000),
                  'finish_reason': None, 'timings': None}
        with self.lock:
            if request_id in self.records:
                raise ValueError('Duplicate request identity')
            if len(self.active) >= MAX_CONVERSATIONS:
                raise ValueError('Too many active chat requests')
            self.active[request_id] = active
            self.records[request_id] = record
            for key in list(self.records):
                if len(self.records) <= MAX_RECORDS:
                    break
                if key not in self.active:
                    self.records.pop(key)
        return self._stream(payload, request_id, instance, client, conversation_id, messages,
                            template, profile, active, record)

    def _stream(self, payload, request_id, instance, client, conversation_id, messages,
                template, profile, active, record):
        conn = None
        conversation = None
        acquired = False
        committed = False
        submitted = False
        started = time.monotonic()
        model = profile.get('served_model_name', 'kvmem')
        chunk = {'id': 'chatcmpl-' + request_id, 'object': 'chat.completion.chunk',
                 'created': int(time.time()), 'model': model}
        def delta(value):
            return _sse({**chunk, 'choices': [{'index': 0, 'delta': value, 'finish_reason': None}]})
        try:
            yield delta({'role': 'assistant'})
            tokenizer, tokenizer_key = self._get_tokenizer(profile, instance)
            key = (client, conversation_id)
            memory = profile.get('kvmem', {})
            session_limit = MAX_CONVERSATIONS
            if memory.get('cpu_history_coordination') and memory.get('session_capacity', 0):
                session_limit = min(session_limit, memory['session_capacity'])
            with self.lock:
                for old_key, old in list(self.conversations.items()):
                    if old.instance != instance and not old.users:
                        self.conversations.pop(old_key)
                conversation = self.conversations.get(key)
                if conversation is None:
                    retained = sum(c.instance == instance for c in self.conversations.values())
                    if len(self.conversations) >= MAX_CONVERSATIONS or retained >= session_limit:
                        raise ValueError(f'The retained session limit ({session_limit}) is full; '
                                         'close an unused conversation before opening another')
                    conversation = _Conversation(instance, tokenizer_key)
                    self.conversations[key] = conversation
                conversation.users += 1
            while not conversation.lock.acquire(timeout=.1):
                if active['cancel'].is_set():
                    raise InterruptedError('Chat cancelled')
            acquired = True
            if active['cancel'].is_set():
                raise InterruptedError('Chat cancelled')
            self._target(instance)
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
            prompt = _tokens(tokenizer.apply_chat_template(messages, tokenize=True,
                                                           add_generation_prompt=True, **template))
            if not prompt or not isinstance(rendered, str):
                raise ValueError('The tokenizer produced an empty/invalid chat prompt')
            query_begin = _query_begin(tokenizer, messages, template, rendered, prompt)
            same_identity = conversation.instance == instance and conversation.tokenizer_key == tokenizer_key
            reused = bool(same_identity and not conversation.dirty and conversation.tokens and len(prompt) >= len(conversation.tokens)
                          and all(a == b for a, b in zip(conversation.tokens, prompt)))
            history_mode = 'exact_prefix' if reused else ('canonical_rebuild' if conversation.tokens or conversation.dirty else 'new')
            reset = not same_identity or conversation.dirty or (conversation.tokens and not reused)
            maximum = profile.get('max_model_len', 16384)
            generation_budget = memory.get('gen_budget', KVMemConfig.gen_budget)
            max_tokens = payload.get('max_tokens', min(1024, generation_budget))
            if len(prompt) + max_tokens > maximum:
                raise ValueError('The complete conversation plus reply exceeds the model context limit')
            if not memory.get('rolling_active_window') and max_tokens > generation_budget:
                raise ValueError('The reply limit exceeds this profile’s generation budget')
            planning = bool(memory.get('request_planning'))
            if planning:
                from ..request_plan import RequestPlan
                config = KVMemConfig(**{name: memory[name] for name in KVMemConfig.__dataclass_fields__ if name in memory})
                block = config.block_tokens
                guided = memory.get('guided_query_tokens', 0)
                reserve = ((256 + guided + block - 1) // block) * block if guided else 0
                if memory.get('guided_reselect') == 'prefill':
                    reserve = 0  # Matches http_sessions: scratch precedes the final Query.
                plan = RequestPlan.parse({'kvmem_query_span': [query_begin, len(prompt)]},
                                         len(prompt), block, config.gen_budget, reserve)
                replay = plan.replay_end - plan.replay_begin
                if replay > (profile.get('query_replay_budget') or config.gen_budget):
                    raise ValueError('The new user query exceeds this profile’s atomic query replay budget')
                if reserve and plan.replay_end + reserve > maximum:
                    raise ValueError('Guided Query scratch exceeds the model position limit')
                if not reused:
                    plan.validate_capacity(len(prompt), config.select_budget, output_tokens=max_tokens,
                                           sink_blocks=config.sink_blocks, recent_blocks=config.recent_blocks,
                                           rolling=memory.get('rolling_active_window', False),
                                           pressure_policy=memory.get('pressure_policy', 'recency'))
            with self.lock:
                amount = len(prompt) + max_tokens
                used = sum(max(len(c.tokens), c.reserved) for c in self.conversations.values() if c is not conversation)
                if used + amount > MAX_HISTORY_TOKENS:
                    raise ValueError('Chat token memory is full; close unused conversations')
                conversation.reserved = amount
            if active['cancel'].is_set():
                raise InterruptedError('Chat cancelled')
            # Every local validation and memory reservation precedes releasing a
            # usable checkpoint. A bad edit must leave the prior turn resumable.
            if reset:
                self._release_native(conversation, client)
                conversation.session = 'webui-' + uuid4().hex
                conversation.tokens = array('I')
                conversation.instance = instance
                conversation.tokenizer_key = tokenizer_key
                conversation.dirty = False
            with self.lock:
                record.update(state='generating', phase='prefill', history_mode=history_mode, limit=max_tokens,
                              exact_prefix_tokens=len(conversation.tokens) if reused else 0,
                              prompt_tokens=len(prompt))
            yield _sse({**chunk, 'choices': [], 'kvmem': {'history_mode': history_mode}})
            body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
                    'temperature': .6, 'top_p': .95, 'top_k': 20, 'min_p': 0.0,
                    **{name: payload[name] for name in PARAMS if name in payload},
                    'session_id': conversation.session, 'cache_salt': self._salt(client),
                    'return_token_ids': True, 'skip_special_tokens': False, 'stream': True,
                    'stream_options': {'include_usage': True}}
            if planning:
                body.update(kvmem_query_span=[query_begin, len(prompt)], kvmem_reselect='auto')
            raw = json.dumps(body, ensure_ascii=False).encode()
            conn = self._connect(instance, active)
            submitted = True
            conn.request('POST', '/v1/completions', raw, self._headers())
            response = conn.getresponse()
            if response.status != 200:
                detail = response.read(65537).decode(errors='replace')[:2000]
                raise ValueError(f'Engine rejected the request ({response.status}): {detail}')
            parser = _TextParser(rendered.rfind('<think>') > rendered.rfind('</think>'),
                                 getattr(tokenizer, 'all_special_tokens', ()))
            output, usage, finish, native_id = [], None, None, None
            first_at = last_at = None
            first_count = 0
            done = False
            lines, event_size = [], 0
            max_event = 1024 * 1024 + 16 * len(prompt)
            while True:
                if active['cancel'].is_set():
                    raise InterruptedError('Chat cancelled')
                line = response.readline(max_event + 1)
                if not line:
                    break
                event_size += len(line)
                if event_size > max_event:
                    raise ValueError('Engine stream event is too large')
                if line.strip():
                    if line.startswith(b'data:'):
                        lines.append(line[5:].strip())
                    continue
                if not lines:
                    event_size = 0
                    continue
                data = b'\n'.join(lines)
                lines, event_size = [], 0
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise ValueError('Engine stream error: ' + str(event['error'])[:2000])
                if not event.get('id') or native_id not in (None, event['id']):
                    raise ValueError('Engine stream response identity changed')
                native_id = event['id']
                record['native_id'] = native_id
                if event.get('usage') is not None:
                    usage = event['usage']
                choices = event.get('choices', [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError('Engine returned unexpected choices')
                for choice in choices:
                    if choice.get('index') != 0:
                        raise ValueError('Engine returned an unexpected choice identity')
                    if choice.get('prompt_token_ids') is not None and choice['prompt_token_ids'] != prompt:
                        raise ValueError('Engine prompt token IDs differ from the submitted prompt')
                    ids = choice.get('token_ids')
                    if ids is not None:
                        ids = _tokens(ids)
                        if ids:
                            last_at = time.monotonic()
                            if first_at is None:
                                first_at, first_count = last_at, len(ids)
                            record.update(phase='decode', metrics={'generated_tokens': len(output) + len(ids),
                                          'ttft_ms': (first_at - started) * 1000})
                        output.extend(ids)
                        if len(output) > max_tokens:
                            raise ValueError('Engine output exceeds the requested token limit')
                    text = choice.get('text', '')
                    if not isinstance(text, str):
                        raise ValueError('Engine returned non-text output')
                    for name, piece in parser.feed(text):
                        yield delta({name: piece})
                    if choice.get('finish_reason') is not None:
                        finish = choice['finish_reason']
                record['generated_tokens'] = len(output)
            if active['cancel'].is_set():
                raise InterruptedError('Chat cancelled')
            if (not done or not isinstance(usage, dict) or usage.get('prompt_tokens') != len(prompt)
                    or usage.get('completion_tokens') != len(output) or not output
                    or finish not in ('stop', 'length')):
                raise ValueError('The engine did not return complete accepted token IDs/usage; '
                                 'exact continuation is unavailable and this reply was not committed')
            self._target(instance)
            for name, piece in parser.feed('', final=True):
                yield delta({name: piece})
            elapsed = time.monotonic() - started
            decode_ms = max(0, (last_at - first_at) * 1000)
            timings = {'predicted_n': len(output), 'measurement_source': 'browser_observed_accepted_token_SSE',
                       'time_to_first_token_ms': (first_at - started) * 1000,
                       'total_ms': elapsed * 1000}
            if decode_ms > 0:
                timings.update(predicted_ms=decode_ms,
                               predicted_per_second=max(0, len(output) - first_count) * 1000 / decode_ms)
            with self.lock:
                conversation.tokens = array('I', prompt + output)
                conversation.dirty = False
                committed = True
                record.update(state='completed', phase='completed', usage=usage, timings=timings, finish_reason=finish,
                              metrics={'generated_tokens': len(output), 'prompt_tokens': len(prompt),
                                       'ttft_ms': timings['time_to_first_token_ms'],
                                       'decode_tps_mean': timings.get('predicted_per_second')})
            yield _sse({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                        'usage': usage, 'timings': timings})
            yield b'data: [DONE]\n\n'
        except GeneratorExit:
            active['cancel'].set()
            record.update(state='cancelled', phase='cancelled')
            raise
        except Exception as error:
            cancelled = active['cancel'].is_set() or isinstance(error, InterruptedError)
            record.update(state='cancelled' if cancelled else 'error', phase='cancelled' if cancelled else 'error',
                          error='Chat cancelled' if cancelled else str(error)[:2000])
            yield _sse({'error': {'message': record['error']}})
            yield b'data: [DONE]\n\n'
        finally:
            if conn is not None:
                conn.close()
            if conversation is not None and acquired:
                if not committed:
                    # A partially streamed reply is not a verified exact prefix.
                    # Keep the namespace so the next turn can explicitly release it.
                    record['history_mode'] = 'uncommitted'
                    conversation.dirty = conversation.dirty or submitted
                conversation.reserved = 0
                conversation.lock.release()
            with self.lock:
                if conversation is not None:
                    conversation.users -= 1
                    if not conversation.users and not conversation.tokens and not conversation.dirty:
                        self.conversations.pop((client, conversation_id), None)
                record['finished_at_ms'] = int(time.time() * 1000)
                self.active.pop(request_id, None)
