"""Scheduler-owned exact-prefix session leases, carried by the KV connector.

An idle hit exists only after a worker retirement acknowledgment. All other
lookups either wait for that particular session or use ordinary native prefill.
Generation-qualified messages cannot release a newer incarnation of a session.
"""
from dataclasses import dataclass, field

from .prompt_checkpoints import token_bytes


def session_namespace(session, salt):
    if not session:
        return None
    salt = salt or ''
    if (not isinstance(session, str) or not isinstance(salt, str)
            or len(session) > 1024 or len(salt) > 1024):
        raise ValueError('Session ID and cache salt must be strings of at most 1024 characters')
    return salt, session


@dataclass(frozen=True)
class SessionBinding:
    namespace: tuple[str, str]
    generation: int
    request_id: str
    tokens: tuple[int, ...]
    history: int = 0
    window: int = 0


@dataclass(frozen=True)
class SessionCapture:
    request_id: str
    generation: int
    position: int


@dataclass(frozen=True)
class SessionEviction:
    namespace: tuple[str, str]
    generation: int


@dataclass(frozen=True)
class SessionNotice:
    kind: str  # retired, restored, evicted
    namespace: tuple[str, str]
    generation: int
    request_id: str
    history: int = 0
    window: int = 0
    ready_records: int = 0


@dataclass
class SessionCommands:
    bindings: list[SessionBinding] = field(default_factory=list)
    captures: list[SessionCapture] = field(default_factory=list)
    evictions: list[SessionEviction] = field(default_factory=list)


@dataclass
class _Entry:
    namespace: tuple[str, str]
    generation: int
    request_id: str | None
    phase: str = 'lookup'
    allocated: bool = False
    history: int = 0
    window: int = 0
    tokens: bytes = b''
    candidate_position: int = 0
    candidate_tokens: bytes = b''
    ready_records: int = 0


class SessionLedger:
    def __init__(self, capacity, block_tokens, prepare_import, abort_import, commit_import):
        if any(type(v) is not int or v <= 0 for v in (capacity, block_tokens)):
            raise ValueError('Session capacity and checkpoint block size must be positive')
        self.capacity, self.block_tokens = capacity, block_tokens
        self.prepare_import, self.abort_import, self.commit_import = prepare_import, abort_import, commit_import
        self.entries = {}  # Insertion order is the idle LRU order.
        self.requests = {}
        self.generation = 0
        self.commands = SessionCommands()
        self.hits = self.misses = self.waits = 0

    @staticmethod
    def namespace(request):
        session = getattr(request, 'session_id', None)
        if not session:
            return None
        if (not isinstance(session, str) or getattr(request, 'resumable', False) or
                getattr(request, 'prompt_embeds', None) is not None or
                getattr(request, 'mm_features', None) or getattr(request, 'lora_request', None)):
            raise ValueError('Prompt sessions require non-streaming, text-only native requests')
        return session_namespace(session, getattr(request, 'cache_salt', None))

    def lookup(self, request, local_tokens):
        namespace = self.namespace(request)
        if namespace is None:
            return 0
        if local_tokens:
            raise ValueError('Prompt session import requires native prefix caching disabled')
        entry = self.entries.get(namespace)
        if entry is not None and entry.request_id == request.request_id:
            if entry.phase != 'lookup':
                raise ValueError('Native request repeated session lookup after allocation')
            return entry.history
        if entry is not None and entry.phase != 'idle':
            self.waits += 1
            return None
        if entry is not None:
            prompt = request.prompt_token_ids
            if len(prompt) <= entry.history or token_bytes(prompt, entry.history) != entry.tokens:
                self._evict(entry)
                self.waits += 1
                return None
            entry.phase, entry.request_id = 'lookup', request.request_id
            self.requests[request.request_id] = entry
            try:
                self.prepare_import(request.request_id, entry.history, entry.window)
            except Exception:
                entry.phase, entry.request_id = 'idle', None
                del self.requests[request.request_id]
                raise
            return entry.history
        if len(self.entries) >= self.capacity:
            idle = next((e for e in self.entries.values() if e.phase == 'idle'), None)
            if idle is not None:
                self._evict(idle)
                self.waits += 1
                return None
            # All cache slots are in use. A new session may still run normally;
            # it cannot create a checkpoint or touch any existing namespace.
            return 0
        self.generation += 1
        entry = _Entry(namespace, self.generation, request.request_id)
        self.entries[namespace] = self.requests[request.request_id] = entry
        return 0

    def allocated(self, request, external_tokens):
        entry = self.requests.get(request.request_id)
        if entry is None:
            if external_tokens:
                raise ValueError('Unowned external prompt checkpoint')
            return
        if entry.phase != 'lookup' or entry.allocated or external_tokens != entry.history:
            raise ValueError('Session allocation differs from its exact prefix lease')
        entry.phase, entry.allocated = 'active', True
        self.commands.bindings.append(SessionBinding(entry.namespace, entry.generation,
            request.request_id, tuple(request.prompt_token_ids), entry.history, entry.window))
        self.hits += int(bool(external_tokens))
        self.misses += int(not external_tokens)

    def capture_before_final_prefill(self, request, query_tokens):
        entry = self.requests.get(request.request_id)
        position = request.num_computed_tokens
        if (entry is None or entry.phase != 'active' or entry.candidate_position or
                not 0 < position < request.num_prompt_tokens or position % self.block_tokens or
                position + query_tokens < request.num_prompt_tokens):
            return
        entry.candidate_position = position
        entry.candidate_tokens = token_bytes(request.prompt_token_ids, position)
        self.commands.captures.append(SessionCapture(request.request_id, entry.generation, position))

    def finished(self, request):
        entry = self.requests.get(request.request_id)
        if entry is None:
            return
        if not entry.allocated:
            del self.requests[request.request_id]
            if entry.history:
                self.abort_import(request.request_id)
                entry.phase, entry.request_id = 'idle', None
            else:
                del self.entries[entry.namespace]
            return
        if entry.phase != 'active':
            raise ValueError('Duplicate session terminal transition')
        entry.phase = 'retiring'

    def _evict(self, entry):
        if entry.phase != 'idle':
            raise ValueError('Cannot evict a leased or retiring session')
        entry.phase = 'evicting'
        self.commands.evictions.append(SessionEviction(entry.namespace, entry.generation))

    def consume(self, notice):
        entry = self.entries.get(notice.namespace)
        if entry is None or entry.generation != notice.generation:
            raise ValueError('Stale or foreign session completion')
        if notice.kind == 'evicted':
            if entry.phase != 'evicting' or notice.request_id or notice.history or notice.window:
                raise ValueError('Unexpected session eviction completion')
            del self.entries[entry.namespace]
            return
        if entry.request_id != notice.request_id:
            raise ValueError('Session completion belongs to another native request')
        if notice.kind == 'restored':
            if (entry.phase not in ('active', 'retiring') or not entry.history or
                    (notice.history, notice.window) != (entry.history, entry.window)):
                raise ValueError('Unexpected prompt import completion')
            self.commit_import(notice.request_id, notice.history)
            entry.history = entry.window = 0
            entry.tokens = b''
            return
        if notice.kind != 'retired' or entry.phase != 'retiring':
            raise ValueError('Unexpected session retirement completion')
        if entry.history:
            raise ValueError('Retirement arrived before prompt import restoration')
        if notice.history:
            if (notice.history != entry.candidate_position or not 0 < notice.window <= notice.history or
                    type(notice.ready_records) is not int or notice.ready_records < 0):
                raise ValueError('Worker retained a different prompt checkpoint')
            entry.history, entry.window = notice.history, notice.window
            entry.ready_records = notice.ready_records
            entry.tokens = entry.candidate_tokens
            entry.candidate_position, entry.candidate_tokens = 0, b''
            entry.request_id, entry.phase, entry.allocated = None, 'idle', False
            # Most recently retired session is the last idle eviction candidate.
            del self.entries[entry.namespace]
            self.entries[entry.namespace] = entry
        else:
            if notice.window:
                raise ValueError('Empty checkpoint has a nonempty target frame')
            del self.entries[entry.namespace]
        del self.requests[notice.request_id]

    def take_commands(self):
        result, self.commands = self.commands, SessionCommands()
        return result

    def has_pending_work(self):
        return any(e.phase in ('retiring', 'evicting') for e in self.entries.values())

    def import_scheduled(self, request_id):
        return any(b.request_id == request_id and b.history for b in self.commands.bindings)
