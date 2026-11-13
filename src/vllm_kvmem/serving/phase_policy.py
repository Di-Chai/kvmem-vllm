"""Single-GPU prefill/decode phases over already eligible native requests.

This policy neither admits requests nor owns KV. The caller excludes native
dependencies, in-flight work and KVMem transactions, and only supplies waiting
requests with an available native slot and successful resource previews.
``record`` consumes actual work feedback, never the proposed token budget.
It is not a GPU completion fence; the caller must label its feedback boundary.
"""
from dataclasses import asdict, dataclass
import math
import time


@dataclass(frozen=True)
class PhaseDecision:
    phase: str
    reason: str
    changed: bool
    prefill_budget: int
    oldest_wait_ms: float
    wave_tokens: int
    wave_batches: int
    sequence: int
    eligible_prefill_count: int
    eligible_decode_count: int
    wave_id: int


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _request_ids(values, name):
    values = tuple(values)
    if any(not isinstance(rid, str) or not rid for rid in values):
        raise ValueError(name + ' must contain nonempty request IDs')
    if len(set(values)) != len(values):
        raise ValueError(name + ' contains duplicate request IDs')
    return values


class PhasePolicy:
    """Bound a prefill wave while giving pending input an aging escape.

    The threshold counts eligible running and waiting prefills together.
    Missing/ineligible requests leave the aging set; time behind a dependency
    is not starvation caused by this policy. A progressing request starts a new
    wait interval, so a long prompt does not immediately retrigger every wave.
    """

    def __init__(self, max_seqs, prefill_wave_tokens, max_wait_ms, clock=time.monotonic):
        self.max_seqs = _positive_integer(max_seqs, 'max_seqs')
        self.prefill_wave_tokens = _positive_integer(prefill_wave_tokens, 'prefill_wave_tokens')
        if (isinstance(max_wait_ms, bool) or not isinstance(max_wait_ms, (int, float)) or
                not math.isfinite(max_wait_ms) or max_wait_ms <= 0):
            raise ValueError('max_wait_ms must be finite and positive')
        if not callable(clock):
            raise ValueError('clock must be callable')
        self.max_wait_ms, self.clock = float(max_wait_ms), clock
        self.prefill_trigger = (self.max_seqs + 1) // 2
        self.phase = 'decode'
        self.sequence = self.transitions = self.waves = 0
        self.wave_tokens = self.wave_batches = 0
        self.total_prefill_tokens = self.total_atomic_query_tokens = 0
        self.total_decode_tokens = 0
        self.prefill_budget_overshoot_tokens = self.atomic_query_overshoot_tokens = 0
        self.ordinary_budget_overshoot_tokens = 0
        self.max_prefill_wait_ms = 0.
        self._waiting_since = {}
        self._running_remaining = {}
        self._last_time = None
        self._decision = None
        self._last_recorded = 0
        self._last_feedback = None
        self._last_transition = None
        self._ages = {}
        self._decode_ids = set()
        self._wave_started = None
        self._budget_blocked = False
        self._awaiting_decode_progress = False
        self._last_wave = None

    def _now(self):
        now = self.clock()
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('phase clock must be finite and monotonic')
        self._last_time = now
        return now

    def decide(self, decode_ids, running_prefill, waiting_prefill_ids):
        decode = _request_ids(decode_ids, 'decode_ids')
        running = dict(running_prefill)
        _request_ids(running, 'running_prefill')
        for remaining in running.values():
            _positive_integer(remaining, 'remaining prefill tokens')
        waiting = _request_ids(waiting_prefill_ids, 'waiting_prefill_ids')
        prefills = set(running) | set(waiting)
        if set(running) & set(waiting) or set(decode) & prefills:
            raise ValueError('Request phases must be disjoint')
        if len(decode) + len(prefills) > self.max_seqs:
            raise ValueError('Eligible requests exceed native max_seqs')
        now = self._now()
        since = {rid: self._waiting_since.get(rid, now) for rid in prefills}
        for rid, remaining in running.items():
            previous = self._running_remaining.get(rid)
            if previous is not None and remaining < previous:
                since[rid] = now
        self._waiting_since, self._running_remaining = since, running
        self._ages = {rid: (now - began) * 1000 for rid, began in since.items()}
        oldest = max(self._ages.values(), default=0.)
        self.max_prefill_wait_ms = max(self.max_prefill_wait_ms, oldest)
        previous_phase = self.phase
        reason = 'decode_ready' if decode else 'idle'
        renew_wave = False
        if self.phase == 'prefill':
            if decode and not prefills:
                # The previous proposal may have been stripped for async
                # restore. Never wait for a first batch that cannot run.
                self.phase, reason = 'decode', 'prefill_unavailable'
            elif decode and self.wave_tokens >= self.prefill_wave_tokens:
                self.phase, reason = 'decode', 'prefill_wave_complete'
            elif decode and self._budget_blocked:
                self.phase, reason = 'decode', 'prefill_alignment_tail'
            elif (decode and self.wave_batches and
                  (now-self._wave_started)*1000 >= self.max_wait_ms):
                self.phase, reason = 'decode', 'prefill_wave_time_limit'
            elif prefills and not decode and (self.wave_tokens >= self.prefill_wave_tokens or self._budget_blocked):
                renew_wave, reason = True, 'prefill_wave_renewed_no_decode'
            else:
                reason = 'prefill_wave' if prefills else 'idle'
        elif prefills:
            if not decode:
                self.phase, reason = 'prefill', 'no_decode_ready'
            elif self._awaiting_decode_progress:
                reason = 'awaiting_decode_progress'
            elif len(prefills) >= self.prefill_trigger:
                self.phase, reason = 'prefill', 'prefill_backlog'
            elif oldest >= self.max_wait_ms:
                self.phase, reason = 'prefill', 'prefill_aged'
        changed = self.phase != previous_phase
        if changed:
            self.transitions += 1
            self._last_transition = dict(at=now, previous=previous_phase, phase=self.phase,
                reason=reason, wave_tokens=self.wave_tokens, wave_batches=self.wave_batches,
                oldest_wait_ms=oldest, eligible_wait_ms=dict(self._ages))
            if self.phase == 'decode':
                self._awaiting_decode_progress = True
        if renew_wave or (changed and self.phase == 'prefill'):
            if self.waves:
                self._last_wave = dict(wave_id=self.waves, tokens=self.wave_tokens,
                    batches=self.wave_batches, budget_overshoot_tokens=max(0, self.wave_tokens-self.prefill_wave_tokens))
            self.waves += 1
            self.wave_tokens = self.wave_batches = 0
            self._wave_started = now
            self._budget_blocked = False
            self._awaiting_decode_progress = False
        # A no-decode wave renews in place; it still has one cumulative budget.
        budget = self.prefill_wave_tokens-self.wave_tokens if self.phase == 'prefill' and prefills else 0
        self._decode_ids = set(decode)
        self.sequence += 1
        self._decision = PhaseDecision(self.phase, reason, changed, budget, oldest,
            self.wave_tokens, self.wave_batches, self.sequence, len(prefills), len(decode), self.waves)
        return self._decision

    def record(self, prefill_tokens_by_request, atomic_query_tokens=0, *,
               decode_tokens_by_request=None, blocked_by_budget=False):
        """Count one final batch after transaction filtering and prompt clamping.

        Query tokens are included in the per-request values and reported again
        as a subset, not added twice. Atomic Query may exceed the ordinary
        budget. No phase change occurs until the next ``decide`` call.
        """
        if self._decision is None or self._last_recorded == self.sequence:
            raise RuntimeError('record requires a new phase decision')
        tokens = dict(prefill_tokens_by_request)
        _request_ids(tokens, 'prefill_tokens_by_request')
        if any(type(n) is not int or n < 0 for n in tokens.values()):
            raise ValueError('Recorded prefill tokens must be nonnegative integers')
        if any(rid not in self._waiting_since and n for rid, n in tokens.items()):
            raise ValueError('Recorded prefill was not eligible in this decision')
        total = sum(tokens.values())
        if type(atomic_query_tokens) is not int or not 0 <= atomic_query_tokens <= total:
            raise ValueError('Atomic Query tokens must be a subset of recorded prefill')
        decodes = dict(decode_tokens_by_request or {})
        _request_ids(decodes, 'decode_tokens_by_request')
        if any(type(n) is not int or n < 0 for n in decodes.values()):
            raise ValueError('Recorded decode tokens must be nonnegative integers')
        if any(rid not in self._decode_ids and n for rid, n in decodes.items()):
            raise ValueError('Recorded decode was not eligible in this decision')
        if type(blocked_by_budget) is not bool:
            raise ValueError('blocked_by_budget must be boolean')
        decode_tokens = sum(decodes.values())
        now = self._now()
        served_ages = {rid: (now-self._waiting_since[rid])*1000
                       for rid, n in tokens.items() if n}
        self.max_prefill_wait_ms = max(self.max_prefill_wait_ms, max(served_ages.values(), default=0.))
        for rid, n in tokens.items():
            if n:
                self._waiting_since[rid] = now
                self._ages[rid] = 0.
                # Suppress a second aging reset when the next real snapshot
                # catches up with the already-recorded progress.
                if rid in self._running_remaining:
                    self._running_remaining[rid] = max(0, self._running_remaining[rid] - n)
        if total:
            self.wave_tokens += total
            self.wave_batches += 1
        self.total_prefill_tokens += total
        self.total_atomic_query_tokens += atomic_query_tokens
        self.total_decode_tokens += decode_tokens
        if self.phase == 'decode' and decode_tokens:
            self._awaiting_decode_progress = False
        if self.phase == 'prefill' and blocked_by_budget and not total:
            self._budget_blocked = True
        over = max(0, total-self._decision.prefill_budget)
        atomic_over = min(over, atomic_query_tokens)
        ordinary_over = max(0, total-atomic_query_tokens-self._decision.prefill_budget)
        self.prefill_budget_overshoot_tokens += over
        self.atomic_query_overshoot_tokens += atomic_over
        self.ordinary_budget_overshoot_tokens += ordinary_over
        self._last_recorded = self.sequence
        self._last_feedback = dict(at=now, sequence=self.sequence, phase=self.phase,
            prefill_tokens=total, atomic_query_tokens=atomic_query_tokens,
            decode_tokens=decode_tokens, blocked_by_budget=blocked_by_budget,
            prefill_budget=self._decision.prefill_budget, budget_overshoot_tokens=over,
            atomic_query_overshoot_tokens=atomic_over, ordinary_budget_overshoot_tokens=ordinary_over,
            wave_tokens=self.wave_tokens, wave_batches=self.wave_batches,
            wave_id=self.waves,
            wave_budget_overshoot_tokens=max(0, self.wave_tokens-self.prefill_wave_tokens),
            served_wait_ms=served_ages)
        return dict(self._last_feedback)

    def snapshot(self):
        return dict(phase=self.phase, max_seqs=self.max_seqs, prefill_trigger=self.prefill_trigger,
            prefill_wave_tokens=self.prefill_wave_tokens, max_wait_ms=self.max_wait_ms,
            sequence=self.sequence, transitions=self.transitions, waves=self.waves,
            wave_tokens=self.wave_tokens, wave_batches=self.wave_batches,
            total_prefill_tokens=self.total_prefill_tokens,
            total_atomic_query_tokens=self.total_atomic_query_tokens,
            total_decode_tokens=self.total_decode_tokens,
            prefill_budget_overshoot_tokens=self.prefill_budget_overshoot_tokens,
            atomic_query_overshoot_tokens=self.atomic_query_overshoot_tokens,
            ordinary_budget_overshoot_tokens=self.ordinary_budget_overshoot_tokens,
            max_prefill_wait_ms=self.max_prefill_wait_ms, eligible_wait_ms=dict(self._ages),
            awaiting_decode_progress=self._awaiting_decode_progress,
            wave_started=self._wave_started, budget_blocked=self._budget_blocked,
            last_wave=self._last_wave,
            decision=asdict(self._decision) if self._decision is not None else None,
            last_feedback=self._last_feedback, last_transition=self._last_transition)
