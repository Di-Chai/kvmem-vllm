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
    pressure_restore: bool = False


@dataclass(frozen=True)
class SessionCapture:
    request_id: str
    generation: int
    position: int
    tokens: tuple[int, ...] = ()


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
    terminals: list[SessionCapture] = field(default_factory=list)


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
    retained_position: int = 0
    retained_tokens: bytes = b''
    pressure_restore: bool = False


class SessionLedger:
    def __init__(self, capacity, block_tokens, prepare_import, abort_import, commit_import, *, terminal_capture=False,
                 select_budget=None, gen_budget=None):
        if any(type(v) is not int or v <= 0 for v in (capacity, block_tokens)):
            raise ValueError('Session capacity and checkpoint block size must be positive')
        self.capacity, self.block_tokens = capacity, block_tokens
        self.prepare_import, self.abort_import, self.commit_import = prepare_import, abort_import, commit_import
        self.entries = {}  # Insertion order is the idle LRU order.
        self.requests = {}
        # Checkpoint capacity does not limit independent cold requests. Keep a
        # lightweight lease for those overflow sessions nevertheless: otherwise
        # concurrent turns can bypass per-session serialization when slots fill.
        self.uncached_sessions = {}
        self.uncached_requests = {}
        self.generation = 0
        self.commands = SessionCommands()
        self.terminal_capture = terminal_capture
        self.select_budget, self.gen_budget = select_budget, gen_budget
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
        owner = self.uncached_sessions.get(namespace)
        if owner is not None:
            if owner == request.request_id:
                return 0
            self.waits += 1
            return None
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
            from .request_plan import request_plan
            plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
            if len(prompt) <= entry.history or token_bytes(prompt, entry.history) != entry.tokens:
                self._evict(entry)
                self.waits += 1
                return None
            if plan is not None and plan.query is not None and entry.history > plan.replay_begin:
                self._evict(entry)  # Never reuse state causally after the Query boundary.
                self.waits += 1
                return None
            if plan is not None and plan.prefill_window != 'keep_selected' and entry.window > self.select_budget:
                entry.window = self.select_budget
                entry.pressure_restore = True
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
            # it cannot create a checkpoint or touch any existing namespace,
            # but must exclude another turn until this request finishes.
            self.uncached_sessions[namespace] = request.request_id
            self.uncached_requests[request.request_id] = namespace
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
            request.request_id, tuple(request.prompt_token_ids), entry.history, entry.window,entry.pressure_restore))
        self.hits += int(bool(external_tokens))
        self.misses += int(not external_tokens)

    def capture_before_final_prefill(self, request, query_tokens):
        entry = self.requests.get(request.request_id)
        position = request.num_computed_tokens
        from .request_plan import request_plan
        plan = request_plan(request,self.block_tokens,self.gen_budget) if self.gen_budget else None
        at_query = plan is not None and plan.replay_begin == position
        if (entry is None or entry.phase != 'active' or entry.candidate_position or
                not 0 < position < request.num_prompt_tokens or position % self.block_tokens or
                (position + query_tokens < request.num_prompt_tokens and not at_query)):
            return
        entry.candidate_position = position
        entry.candidate_tokens = token_bytes(request.prompt_token_ids, position)
        self.commands.captures.append(SessionCapture(request.request_id, entry.generation, position))

    def capture_decode_boundary(self, request):
        """Advance the persistent accepted prefix, never a speculative suffix."""
        entry = self.requests.get(request.request_id)
        position = request.num_computed_tokens
        if (entry is None or entry.phase != 'active' or entry.candidate_position or
                position < request.num_prompt_tokens or position <= entry.retained_position or
                not position or position % self.block_tokens):
            return
        tokens = tuple(request.all_token_ids)
        entry.candidate_position = position
        entry.candidate_tokens = token_bytes(tokens, position)
        self.commands.captures.append(SessionCapture(request.request_id, entry.generation, position, tokens))

    def finished(self, request):
        namespace = self.uncached_requests.pop(request.request_id, None)
        if namespace is not None:
            del self.uncached_sessions[namespace]
            return
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
        if self.terminal_capture and request.status.name in ('FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED'):
            from .terminal import TerminalPrefix
            position = TerminalPrefix.from_request(request).history_end
            if position:
                entry.retained_position = entry.retained_position or entry.candidate_position
                entry.retained_tokens = entry.retained_tokens or entry.candidate_tokens
                entry.candidate_position = position
                tokens = tuple(request.all_token_ids)
                entry.candidate_tokens = token_bytes(tokens, position)
                self.commands.terminals.append(SessionCapture(request.request_id, entry.generation, position, tokens))

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
        if notice.kind in ('captured', 'capture_deferred'):
            if (entry.phase not in ('active', 'retiring') or
                    notice.history != entry.candidate_position or not notice.history):
                raise ValueError('Checkpoint acknowledgment differs from the requested accepted prefix')
            if notice.kind == 'captured':
                entry.retained_position = notice.history
                entry.retained_tokens = entry.candidate_tokens
            entry.candidate_position, entry.candidate_tokens = 0, b''
            return
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
            expected = entry.retained_position or entry.candidate_position
            identity = entry.retained_tokens or entry.candidate_tokens
            if (notice.history != expected or not 0 < notice.window <= notice.history or
                    type(notice.ready_records) is not int or notice.ready_records < 0):
                raise ValueError('Worker retained a different prompt checkpoint')
            entry.history, entry.window = notice.history, notice.window
            entry.ready_records = notice.ready_records
            entry.tokens = identity
            entry.candidate_position, entry.candidate_tokens = 0, b''
            entry.retained_position, entry.retained_tokens = 0, b''
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
