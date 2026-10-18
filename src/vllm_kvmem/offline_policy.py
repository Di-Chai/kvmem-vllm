"""Offline ready-work ordering; native allocation and KVMem semantics stay authoritative.

No model forwards, prefix mutations, CUDA synchronization or sleeping occur here.
Cost hints only order candidates. Unknown costs retain FIFO; aging bounds bypass.
"""
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
import math
import os
import statistics
import time


def context_bucket(tokens):
    return 1 << max(10, max(1, int(tokens)).bit_length() - 1)


class ServiceCosts:
    def __init__(self):
        self.decode = defaultdict(lambda: deque(maxlen=32))
        self.prefill = defaultdict(lambda: deque(maxlen=32))
        self.restore = defaultdict(lambda: deque(maxlen=32))
        self.sequence = 0

    def observe_steps(self, rows):
        for row in rows:
            if row['sequence'] <= self.sequence:
                continue
            self.sequence = row['sequence']
            ms = row.get('model_ms')
            if ms is None or not math.isfinite(ms) or ms <= 0:
                continue
            d, p, n = row['decode_requests'], row.get('prefill_requests', 0), row['prefill_tokens']
            db, pb = row.get('decode_context_bucket'), row.get('prefill_context_bucket')
            if d and not n and db is not None:
                self.decode[d, db].append(ms)
            # A single prefill participant provides an attributable token rate.
            # Fixed mixed/model overhead remains conservatively in the rate;
            # host maintenance is excluded. This model never shrinks chunks.
            elif p == 1 and n and pb is not None:
                baseline = self.decode.get((d, db))
                if d and (not baseline or len(baseline) < 3):
                    continue
                extra = ms - (statistics.median(baseline) if d else 0.)
                if extra > 0:
                    self.prefill[pb].append(extra / n)

    def observe_restores(self, rows):
        for row in rows:
            ms, tokens = row['service_seconds'] * 1000, row['active_tokens']
            if not math.isfinite(ms) or ms <= 0 or tokens <= 0:
                raise ValueError('Invalid restore service feedback')
            mode = 'resident' if row['resident'] else 'cpu'
            # Recurrent-state restore has a fixed payload in the resident path.
            self.restore[mode, context_bucket(tokens)].append(ms if row['resident'] else ms / tokens)

    def estimate(self, remaining, context, restore_kind, restore_tokens):
        prefill = self.prefill.get(context_bucket(context))
        restore = self.restore.get((restore_kind, context_bucket(restore_tokens)))
        if remaining and (not prefill or len(prefill) < 3):
            return None
        if restore_tokens and (not restore or len(restore) < 3):
            return None
        return ((statistics.median(prefill) * remaining if remaining else 0.) +
                (statistics.median(restore) * (1 if restore_kind == 'resident' else restore_tokens)
                 if restore_tokens else 0.))


@dataclass
class Candidate:
    request_id: str
    phase: str
    remaining_tokens: int
    restore_tokens: int
    estimated_ms: float | None
    waiting_seconds: float
    reason: str | None = None


class OfflineAgentPolicy:
    def __init__(self, scheduler, options, output, *, clock=time.monotonic):
        self.scheduler, self.clock, self.output = scheduler, clock, output
        self.aging = options.maintenance_aging_ms / 1000
        self.chunk = int(os.getenv('KVMEM_PREFILL_CHUNK_TOKENS', '0')) or options.prefill_max_tokens
        self.costs = ServiceCosts()
        self.progress = {}
        self.hints = {}
        self.rows = {}
        self.ranks = {}
        self.steps = self.reorders = self.aged_choices = 0
        self.last_signature = None

    def _hint(self, request):
        """Read-only exact M hint; real lookup still validates and owns the import.

        P branches and new mandatory-content selections remain unknown here.
        Token identity is checked once per immutable request/checkpoint pair.
        """
        ledger = self.scheduler.connector.sessions
        entry = ledger.entries.get(ledger.namespace(request))
        if entry is None:
            return 0, 0, 'none', None
        if entry.request_id == request.request_id and entry.phase == 'lookup':
            history, window = entry.history, entry.window
        elif entry.phase != 'idle':
            return 0, 0, 'unknown', 'session_retirement_or_owner'
        else:
            key = (entry.generation, entry.history, entry.window, entry.tokens,
                   id((request.kv_transfer_params or {}).get('kvmem_request_plan')))
            cached = self.hints.get(request.request_id)
            if cached is None or cached[0] != key:
                from .prompt_checkpoints import token_bytes
                from .request_plan import request_plan
                prompt = request.prompt_token_ids
                matches = len(prompt) > entry.history and token_bytes(prompt, entry.history) == entry.tokens
                plan = request_plan(request, ledger.block_tokens, ledger.gen_budget)
                if matches and plan is not None:
                    effective, changed = plan.continuation(len(prompt), ledger.output_tokens(request),
                        entry.history, entry.window, ledger.select_budget + ledger.gen_budget,
                        entry.query, entry.mandatory)
                    # Conservative: a legal P branch or new pin needs the real
                    # lookup before a reliable remaining-work estimate exists.
                    matches = not changed and not (effective.query is not None and entry.history > effective.replay_begin)
                self.hints[request.request_id] = key, matches
            if not self.hints[request.request_id][1]:
                return 0, 0, 'unknown', None
            history, window = entry.history, entry.window
        residency = self.scheduler.connector.active_residency
        view = residency.entries.get(entry.namespace) if residency is not None else None
        resident = (view is not None and view.expires > self.clock() and
                    (view.generation, view.history, view.window) == (entry.generation, history, window))
        return history, window, 'resident' if resident else 'cpu', None

    def _candidate(self, request, *, running):
        s, rid, now = self.scheduler, request.request_id, self.clock()
        computed = request.num_computed_tokens
        previous, since = self.progress.get(rid, (computed, now))
        if computed != previous:
            since = now
        self.progress[rid] = computed, since
        remaining = max(0, request.num_prompt_tokens - computed)
        age = now - since
        if rid in s.transactions:
            return Candidate(rid, 'restore_or_reselect', remaining, 0, None, age, 'completion_fence')
        if request.is_finished():
            return Candidate(rid, 'retiring', 0, 0, None, age, 'history_publication')
        if request.num_in_flight_tokens or request.next_decode_eligible_step > s.current_step + 1:
            return Candidate(rid, 'in_flight', remaining, 0, None, age, 'native_step_fence')
        window, mode, reason = 0, 'none', None
        offset = next(iter(s.window_managers.values())).offsets.get(rid, 0)
        context = computed - offset
        if not running:
            if s._is_blocked_waiting_status(request.status):
                reason = 'native_waiting_dependency'
            elif not s.connector.history_admission.can_admit(request):
                reason = 'cpu_history_budget'
            else:
                reason = s.kv_cache_manager.admission_preview(request)
            if reason is None:
                history, window, mode, reason = self._hint(request)
                remaining = request.num_prompt_tokens - max(computed, history)
                context = window
        phase = 'decode' if running and not remaining else ('prefill' if running else ('cold' if mode == 'none' else mode) + '_admission')
        estimate = None if reason or mode == 'unknown' else self.costs.estimate(
            remaining, min(s.query_policy.capacity, context + min(remaining, self.chunk)),
            mode, window)
        return Candidate(rid, phase, remaining, window, estimate, age, reason)

    def _order(self, requests):
        ready = [r for r in requests if self.rows[r.request_id].reason is None]
        # Unknown costs retain FIFO across the entire competing set. Do not
        # silently treat missing samples as zero-time work.
        measured = bool(ready) and all(self.rows[r.request_id].estimated_ms is not None for r in ready)
        def key(request):
            row = self.rows[request.request_id]
            arrival = getattr(request, 'arrival_time', 0.)
            if row.reason:
                return 2, arrival
            if row.waiting_seconds >= self.aging:
                return 0, -row.waiting_seconds
            return 1, row.estimated_ms if measured else arrival
        return sorted(requests, key=key)

    def prepare(self):
        s = self.scheduler
        self.steps += 1
        waiting = [*s.waiting, *s.skipped_waiting]
        active_ids = {r.request_id for r in [*s.running, *waiting]}
        self.progress = {k:v for k,v in self.progress.items() if k in active_ids}
        self.hints = {k:v for k,v in self.hints.items() if k in active_ids}
        self.rows = {r.request_id:self._candidate(r, running=True) for r in s.running}
        self.rows.update((r.request_id,self._candidate(r, running=False)) for r in waiting)
        decode = [r for r in s.running if self.rows[r.request_id].phase == 'decode' and self.rows[r.request_id].reason is None]
        rest = [r for r in s.running if r not in decode]
        ordered = decode + self._order(rest)
        self.reorders += int([r.request_id for r in ordered] != [r.request_id for r in s.running])
        s.running[:] = ordered
        ranked = self._order(waiting)
        self.ranks = {r.request_id:i for i,r in enumerate(ranked)}
        for queue in (s.waiting, s.skipped_waiting):
            ordered = sorted(queue, key=lambda r:self.ranks[r.request_id])
            queue.clear()
            queue.extend(ordered)

    def waiting_queue(self):
        s = self.scheduler
        queues = [q for q in (s.waiting, s.skipped_waiting) if q]
        # Native admission may have consumed credits earlier in this same step.
        # Recheck before selecting a head, otherwise a now-too-large request
        # makes native allocation break ahead of a smaller runnable neighbour.
        pending = [r for q in queues for r in q]
        for request in pending:
            row = self.rows[request.request_id]
            if row.reason in (None, 'cpu_history_budget', 'gpu_lifetime_budget', 'retirement_or_active_limit'):
                row.reason = ('cpu_history_budget' if not s.connector.history_admission.can_admit(request)
                              else s.kv_cache_manager.admission_preview(request))
        ordered = self._order(pending)
        self.ranks = {r.request_id:i for i,r in enumerate(ordered)}
        for queue in queues:
            ordered = sorted(queue, key=lambda r:self.ranks[r.request_id])
            queue.clear();queue.extend(ordered)
        return min(queues, key=lambda q:self.ranks[q.peek_request().request_id]) if queues else None

    def maintenance_costs(self):
        """Use the same remaining-work estimates for the background executor."""
        from .connector import RestoreCommand
        result = {}
        for rid,(command,_) in self.scheduler.transactions.items():
            request = self.scheduler.requests[rid]
            estimate = None
            if isinstance(command, RestoreCommand):
                remaining = max(0, request.num_prompt_tokens - command.history_end)
                estimate = self.costs.estimate(remaining,
                    min(self.scheduler.query_policy.capacity, command.window_tokens + min(remaining,self.chunk)),
                    'resident' if command.resident else 'cpu', command.window_tokens)
            result[rid] = estimate
        return result

    def record(self, output):
        selected = set(output.num_scheduled_tokens)
        for rid in selected:
            row = self.rows.get(rid)
            if row is not None and row.waiting_seconds >= self.aging and row.phase != 'decode':
                self.aged_choices += 1
        # Emit state/order changes, not every speculative decode shape.
        signature = tuple((rid, row.phase, row.reason, self.ranks.get(rid), rid in selected,
                           rid in self.scheduler.transactions) for rid,row in self.rows.items())
        if signature == self.last_signature:
            return
        self.last_signature = signature
        if self.output is not None:
            row = dict(at=self.clock(), step=self.scheduler.current_step,
                candidates=[asdict(r) for r in self.rows.values()], waiting_order=list(self.ranks),
                scheduled=output.num_scheduled_tokens,
                maintenance_costs=self.maintenance_costs(),
                cpu=self.scheduler.connector.history_admission.snapshot(),
                active_residency=self.scheduler.connector.active_residency.snapshot(),
                free_gpu_blocks=self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
            with self.output.open('a') as stream:
                stream.write(json.dumps(row) + '\n')

    def snapshot(self):
        return dict(mode='offline_ready_work', steps=self.steps, running_reorders=self.reorders,
                    aged_choices=self.aged_choices, waiting_order=list(self.ranks),
                    states={rid:row.phase for rid,row in self.rows.items()},
                    waits={rid:row.reason for rid,row in self.rows.items() if row.reason})
