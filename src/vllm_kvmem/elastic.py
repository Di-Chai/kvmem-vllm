"""Bounded turn histories and deadline-aware admission, independent of CUDA.

The native offload connector owns KV/state bytes. This ledger owns accepted
token histories; it never treats a completed HTTP turn as proof of DMA completion.
"""
import asyncio
from array import array
from collections import deque
from dataclasses import dataclass, field
import hashlib
import time


@dataclass(frozen=True)
class CacheGeometry:
    groups: tuple[dict, ...]
    capacity_bytes: int

    def active_bytes(self, tokens):
        if type(tokens) is not int or tokens < 1:
            raise ValueError('Token extent must be positive')
        total = 0
        for group in self.groups:
            block = group['block_tokens']
            pages = (tokens + block - 1) // block
            if group['recurrent']:
                # Running state, boundary CoW and speculative state rows.
                pages = 2 + group['speculative_blocks']
            total += pages * group['page_bytes']
        return total


def select_context(tokens, ranges, limit):
    """Explicit turn-boundary selection; full CPU history is never modified.

    Selected spans are freshly interpreted as the model input. Prefix reuse is
    permitted only for exact selected token prefixes. This does not reuse old
    recurrent state across a changed context or claim full-context equivalence.
    """
    if not isinstance(ranges, list) or not ranges or type(limit) is not int or limit < 1:
        raise ValueError('Active context requires nonempty ordered token ranges and a limit')
    selected, previous = [], 0
    for span in ranges:
        if (not isinstance(span, list) or len(span) != 2
                or any(type(t) is not int for t in span)
                or not previous <= span[0] < span[1] <= len(tokens)):
            raise ValueError('Active ranges must be ordered, disjoint and within full history')
        if len(selected) + span[1] - span[0] > limit:
            raise BufferError('Selected context exceeds active token limit')
        selected.extend(tokens[span[0]:span[1]])
        previous = span[1]
    return selected


@dataclass
class Session:
    tokens: array = field(default_factory=lambda: array('I'))
    version: int = 0
    busy: bool = False
    waiting_for_tool: bool = False
    last_used: float = 0


class HistoryStore:
    def __init__(self, max_bytes, max_sessions, ttl_seconds, *, clock=time.monotonic,
            per_user_bytes=None, per_user_sessions=None):
        if min(max_bytes, max_sessions, ttl_seconds) <= 0:
            raise ValueError('History limits must be positive')
        self.max_bytes, self.max_sessions, self.ttl = max_bytes, max_sessions, ttl_seconds
        self.clock = clock
        self.entries, self.reservations = {}, {}
        self.pending_appends = {}
        self.user_bytes = {}
        self.per_user_bytes = per_user_bytes or max_bytes
        self.per_user_sessions = per_user_sessions or max_sessions
        self.bytes = 0

    def expire(self):
        for key, session in list(self.entries.items()):
            if not session.busy and self.clock() - session.last_used >= self.ttl:
                self.bytes -= len(session.tokens) * session.tokens.itemsize
                self.user_bytes[key[0]] -= len(session.tokens) * session.tokens.itemsize
                del self.entries[key]

    def begin(self, user, session_id, version, append, max_output):
        if (not isinstance(session_id, str) or not session_id or len(session_id) > 128
                or type(version) is not int or version < 0
                or type(max_output) is not int or max_output < 1
                or not isinstance(append, list) or not append
                or any(type(t) is not int or not 0 <= t < 2**32 for t in append)):
            raise ValueError('Invalid session version, append_token_ids or output limit')
        self.expire()
        key = (user, session_id)
        session = self.entries.get(key)
        if session is None:
            if version:
                raise LookupError('Session expired or does not exist; restart at version zero')
            if len(self.entries) >= self.max_sessions:
                raise BufferError('Historical session capacity exceeded')
            if sum(k[0] == user for k in self.entries) >= self.per_user_sessions:
                raise BufferError('User historical session capacity exceeded')
            session = Session(last_used=self.clock())
        if session.busy or session.version != version:
            raise RuntimeError('Session already running or version changed')
        reservation = 4 * (len(append) + max_output)
        if self.bytes + sum(self.reservations.values()) + reservation > self.max_bytes:
            raise BufferError('Accepted token history budget exceeded')
        user_pending = sum(v for k, v in self.reservations.items() if k[0] == user)
        if self.user_bytes.get(user, 0) + user_pending + reservation > self.per_user_bytes:
            raise BufferError('User accepted token history budget exceeded')
        self.entries[key] = session
        self.user_bytes.setdefault(user, 0)
        self.reservations[key] = reservation
        self.pending_appends[key] = tuple(append)
        session.busy = True
        return key, session, list(session.tokens) + append

    def finish(self, key, append, output, *, waiting_for_tool=False):
        session = self.entries[key]
        reservation = self.reservations[key]
        if (not session.busy or tuple(append) != self.pending_appends[key]
                or any(type(t) is not int or not 0 <= t < 2**32 for t in output)
                or 4 * (len(append) + len(output)) > reservation):
            raise ValueError('Invalid accepted output or historical reservation overrun')
        session.tokens.extend(append)
        session.tokens.extend(output)
        self.bytes += 4 * (len(append) + len(output))
        self.user_bytes[key[0]] += 4 * (len(append) + len(output))
        session.version += 1
        session.busy = False
        session.waiting_for_tool = bool(waiting_for_tool)
        session.last_used = self.clock()
        del self.reservations[key]
        del self.pending_appends[key]

    def abort(self, key):
        session = self.entries[key]
        session.busy = False
        session.last_used = self.clock()
        self.reservations.pop(key, None)
        self.pending_appends.pop(key, None)
        if not session.version:
            del self.entries[key]

    def delete(self, user, session_id):
        key = (user, session_id)
        session = self.entries.get(key)
        if session is None:
            return False
        if session.busy:
            raise RuntimeError('Cannot delete a running session')
        self.bytes -= 4 * len(session.tokens)
        self.user_bytes[user] -= 4 * len(session.tokens)
        del self.entries[key]
        return True

    @staticmethod
    def cache_salt(user, session_id):
        return hashlib.sha256((user + '\0' + session_id).encode()).hexdigest()


@dataclass(eq=False)
class Ticket:
    user: str
    cost: int
    deadline: float
    ready: asyncio.Future
    granted: bool = False


class ActiveBudget:
    """Whole-turn byte reservations; EDF across eligible users, one per user.

    A waiting ticket never owns GPU credit. A lease survives cancellation until
    the upstream HTTP request has closed. Token/state geometry includes output
    growth, so a turn cannot silently consume another session's reservation.
    """
    def __init__(self, max_bytes, max_requests, max_queue):
        if min(max_bytes, max_requests) <= 0 or max_queue < 0:
            raise ValueError('Invalid active budget')
        self.max_bytes, self.max_requests, self.max_queue = max_bytes, max_requests, max_queue
        self.bytes = 0
        self.active, self.waiting = set(), deque()
        self.peak_bytes = self.peak_requests = 0

    def enter(self, user, cost, timeout):
        if type(cost) is not int or not 0 < cost <= self.max_bytes or timeout <= 0:
            raise BufferError('Turn exceeds active byte budget or deadline')
        loop = asyncio.get_running_loop()
        ticket = Ticket(user, cost, loop.time() + timeout, loop.create_future())
        self._schedule()
        if len(self.waiting) >= self.max_queue and not self._eligible(ticket):
            raise BufferError('Active queue is full')
        self.waiting.append(ticket)
        self._schedule()
        return ticket

    def _eligible(self, ticket):
        return (len(self.active) < self.max_requests and self.bytes + ticket.cost <= self.max_bytes
                and all(t.user != ticket.user for t in self.active))

    def _schedule(self):
        now = asyncio.get_running_loop().time()
        for ticket in list(self.waiting):
            if ticket.deadline <= now or ticket.ready.cancelled():
                self.waiting.remove(ticket)
                if not ticket.ready.done():
                    ticket.ready.set_exception(TimeoutError('Active admission deadline exceeded'))
        while True:
            eligible = [t for t in self.waiting if self._eligible(t)]
            if not eligible:
                return
            ticket = min(eligible, key=lambda t: t.deadline)
            self.waiting.remove(ticket)
            self.active.add(ticket)
            self.bytes += ticket.cost
            ticket.granted = True
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.peak_requests = max(self.peak_requests, len(self.active))
            ticket.ready.set_result(None)

    def release(self, ticket):
        if ticket in self.active:
            self.active.remove(ticket)
            self.bytes -= ticket.cost
        elif ticket in self.waiting:
            self.waiting.remove(ticket)
            ticket.ready.cancel()
        self._schedule()
