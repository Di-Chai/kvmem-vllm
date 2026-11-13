"""Scheduler control of fenced page swaps, retaining native request slots.

Only this explicit controller may reclaim a running request's GPU pages. Each
control command completes before another is sent, while unrelated public work
can continue. Call ``before_schedule`` before constructing any public batch;
``on_pressure`` inside allocation only records a hint and never mutates pages.
"""
from dataclasses import dataclass, field
from itertools import count
from collections import deque
from time import monotonic

from .request_swap import FencedPageSwap, SwapPhase
from .request_swap_worker import SwapCommand
from ..core.resource_ledger import ResourceLedger
from .session_swap import session_swap_candidate, validate_session_swap


class _Ready:
    @staticmethod
    def query():
        return True


@dataclass
class _SwapState:
    ticket: object
    grant: object
    checkpoint_bytes: int
    paused_step: int
    paused_at: float
    paused_progress: int = 0
    session_identity: object = None
    phase: str = 'capturing'
    command: object = None
    cancelled: bool = False
    delay_free_blocks: bool = False
    grace_used: bool = False
    grace_expired: bool = False
    last_grace_progress: int = -1
    grace_finishers: dict = field(default_factory=dict)


class RequestSwapController:
    def __init__(self, scheduler, *, checkpoint_bytes, page_bytes, finish_callback,
                 min_resident_steps=32, max_pause_steps=128, checkpoint_cpu_bytes=None):
        checkpoint_cpu_bytes = checkpoint_bytes if checkpoint_cpu_bytes is None else checkpoint_cpu_bytes
        if (any(type(value) is not int or value <= 0 for value in
                (checkpoint_bytes, checkpoint_cpu_bytes, page_bytes, min_resident_steps, max_pause_steps)) or
                max_pause_steps < min_resident_steps or not callable(finish_callback)):
            raise ValueError('Invalid request swap capacity, fairness or finish callback')
        self.scheduler = scheduler
        self.checkpoint_bytes, self.page_bytes = checkpoint_bytes, page_bytes
        self.checkpoint_cpu_bytes = checkpoint_cpu_bytes
        self.finish_callback = finish_callback
        self.min_resident_steps, self.max_pause_steps = min_resident_steps, max_pause_steps
        self.cache_manager = self.swaps = self.ledger = None
        self._states, self._pressure, self._resumed = {}, {}, {}
        self._restored_history = {}
        self._owner_progress = {}
        self._progress_round = self._idle_polls = self._forward_turns = self._forward_tokens = 0
        self._progress_pending = False
        self._trace = deque(maxlen=128)
        self._commands = count(1 << 60)
        self._inflight = None
        self._restore_target = None
        self.progress_reserve_bytes = self.worst_checkpoint_bytes = 0
        self.captures = self.restores = self.drops = self.capacity_deferrals = 0
        self.resume_forward_completions = self.resume_forward_tokens = 0
        self.max_pause_steps_observed = 0
        self.max_pause_seconds_observed = 0.
        self.max_pause_progress_observed = 0
        self.waiter_deferrals = self.cooldown_overrides = 0
        self.capture_transfer_bound_bytes = self.restore_transfer_bound_bytes = 0
        self.completion_grace_rounds = self.completion_grace_polls = self.completion_grace_expirations = 0
        self.completion_grace_retirement_rounds = self.completion_grace_retirement_polls = 0

    def bind(self, cache_manager):
        """Bind once after constructing the incremental allocator adapter."""
        if self.cache_manager is not None:
            raise ValueError('Request swap controller was already bound')
        worst_pages = cache_manager._demand(cache_manager.native.max_model_len)
        if cache_manager.block_pool.num_gpu_blocks - 1 < worst_pages:
            raise ValueError('Incremental GPU pool cannot fit one complete request operation window')
        worst = worst_pages * self.page_bytes + 4096
        if self.checkpoint_cpu_bytes < worst:
            raise ValueError(f'Request checkpoint CPU capacity cannot fit one complete checkpoint: need at least {worst}')
        slots = getattr(getattr(self.scheduler, 'scheduler_config', None), 'max_num_seqs',
                        getattr(cache_manager, 'max_authorities', None) or 1)
        # Several small victims can be needed to restore one larger owner.
        # Their page sum is less than target + last victim, bounded by 2*worst;
        # retain a per-slot counter/alignment allowance as well.
        reserve = 2 * worst + slots * 4096
        if self.checkpoint_bytes < reserve + worst:
            raise ValueError('Request checkpoint byte capacity cannot fit one retained checkpoint '
                             f'plus progress reserve: need at least {reserve + worst}')
        self.cache_manager = cache_manager
        self.swaps = FencedPageSwap(cache_manager)
        self.worst_checkpoint_bytes, self.progress_reserve_bytes = worst, reserve
        self.ledger = ResourceLedger({'checkpoint_bytes': self.checkpoint_bytes},
                                    emergency={'checkpoint_bytes': reserve})

    def is_suspended(self, request_id):
        return request_id in self._states

    def has_pending_work(self):
        return bool(self._states or self._pressure or self._inflight)

    def reserved_pages(self, request_id):
        rid = self._restore_target
        if rid is None or request_id == rid:
            return 0
        state = self._states.get(rid)
        if state is None and rid not in self._restored_history:
            return 0
        quota = self.cache_manager.reservations[rid][1]
        remaining = max(0, quota - self.cache_manager._owned(rid))
        return min(self.cache_manager.block_pool.get_num_free_blocks(), remaining)

    def on_pressure(self, request, required_pages, available_pages):
        if (any(type(value) is not int or value < 0 for value in (required_pages, available_pages)) or
                request.is_finished()):
            return
        rid = request.request_id
        previous = self._pressure.get(rid)
        first_step = previous[3] if previous is not None and previous[0] is request else self._step()
        first_progress = previous[4] if previous is not None and previous[0] is request else self._progress_round
        self._pressure[rid] = request, required_pages, request.num_computed_tokens, first_step, first_progress

    def on_allocation(self, request, history_end):
        """Register the pre-forward baseline; imported history is not progress."""
        row = self._owner_progress.get(request.request_id)
        if row is None or row[0] is not request:
            self._owner_progress[request.request_id] = request, history_end, 0

    def _record_progress(self, request):
        rid = request.request_id
        previous = self._owner_progress.get(rid)
        if previous is None or previous[0] is not request:
            self.on_allocation(request, request.num_computed_tokens)
            return False
        if request.num_in_flight_tokens or request.num_computed_tokens <= previous[1]:
            return False
        self._owner_progress[rid] = request, request.num_computed_tokens, previous[2] + 1
        self._forward_turns += 1
        self._forward_tokens += request.num_computed_tokens - previous[1]
        return True

    def _observe_progress(self):
        progressed = self._progress_pending
        self._progress_pending = False
        connector = self.scheduler.connector
        for rid, request in self.scheduler.requests.items():
            if rid not in self._states and rid not in connector.pending and rid not in self.scheduler.transactions:
                progressed = self._record_progress(request) or progressed
        if progressed:
            self._progress_round += 1
        else:
            self._idle_polls += 1
        self._owner_progress = {rid: row for rid, row in self._owner_progress.items()
                                if rid in self.scheduler.requests}

    def _turns(self, request_id):
        row = self._owner_progress.get(request_id)
        return row[2] if row is not None else 0

    def _step(self):
        return self.scheduler.current_step + 1

    def _active(self, request):
        return (self.scheduler.requests.get(request.request_id) is request and
                not request.is_finished())

    def _eligible(self, request, *, ignore_cooldown=False):
        rid = request.request_id
        scheduler, connector = self.scheduler, self.scheduler.connector
        if (not self._active(request) or rid in self._states or
                request.num_computed_tokens <= 0 or
                request.num_in_flight_tokens or request.next_decode_eligible_step > self._step() or
                rid in scheduler.transactions or rid in connector.pending or
                rid in getattr(connector, 'publication_blocked', ()) or
                rid in getattr(scheduler, 'prepared_refreshes', {}) or
                request.num_computed_tokens <= self._restored_history.get(rid, -1) or
                (not ignore_cooldown and rid in self._resumed and
                 self._turns(rid) - self._resumed[rid] < self.min_resident_steps)):
            return False
        if not session_swap_candidate(scheduler, request)[0]:
            return False
        for manager in self.swaps.groups:
            blocks = manager.req_to_blocks.get(rid, ())
            if (not blocks or any(b.is_null or b.ref_cnt != 1 for b in blocks) or
                    rid in getattr(manager, 'pending', {}) or rid in getattr(manager, 'imports', {})):
                return False
        return True

    def _size(self, request):
        return self.cache_manager._owned(request.request_id) * self.page_bytes + 4096

    def _capture(self, request, *, emergency=False, ignore_cooldown=False, reason='pressure'):
        if not self._eligible(request, ignore_cooldown=ignore_cooldown):
            return False
        size = self._size(request)
        available = self.ledger.snapshot()['resources']['checkpoint_bytes']['emergency_available']
        # Replenish the physical progress margin even when a previous emergency
        # grant is still retained. Borrowing status is not newly freed RAM.
        if not emergency and size + self.progress_reserve_bytes > available:
            self.capacity_deferrals += 1
            return False
        grant = self.ledger.prepare(request.request_id, 0, {'checkpoint_bytes': size}, emergency=emergency)
        if grant is None:
            self.capacity_deferrals += 1
            return False
        try:
            ticket = self.swaps.prepare(request)
        except BaseException:
            self.ledger.abort(grant)
            raise
        state = _SwapState(ticket, grant, size, self._step(), monotonic(),
                           self._progress_round, session_swap_candidate(self.scheduler, request)[1])
        self._states[request.request_id] = state
        self.ledger.dispatch(grant)
        try:
            self._queue(state, 'capture', ticket.original_blocks)
        except BaseException:
            # queue() is local, before native metadata is published. If it
            # partially changed pending, retain ownership and fail closed.
            if request.request_id not in self.scheduler.connector.pending:
                self.swaps.abort_capture(ticket, _Ready())
                self.ledger.settle(grant, _Ready(), retained={})
                del self._states[request.request_id]
                self._inflight = None
            raise
        self.captures += 1
        self.capture_transfer_bound_bytes += size
        self.cooldown_overrides += int(ignore_cooldown and request.request_id in self._resumed and
            self._turns(request.request_id) - self._resumed[request.request_id] < self.min_resident_steps)
        self._trace[-1]['reason'] = reason
        return True

    def _queue(self, state, action, blocks=()):
        if self._inflight is not None:
            raise RuntimeError('Only one request swap command may be in flight')
        ticket = state.ticket
        validate_session_swap(self.scheduler, ticket.request, state.session_identity)
        command = SwapCommand(ticket.request_id, next(self._commands), ticket.operation_id,
            action, ticket.history_end, blocks, state.checkpoint_bytes if action == 'capture' else 0,
            session_identity=state.session_identity)
        state.command = command
        self._inflight = ticket.request_id, command.command_id
        self.scheduler.connector.queue(command)
        state.phase = {'capture': 'capturing', 'restore': 'restoring', 'drop': 'dropping'}[action]
        self._trace.append(dict(step=self._step(), request_id=ticket.request_id,
            progress_round=self._progress_round, owner_forward_turns=self._turns(ticket.request_id),
            action=action, history_end=ticket.history_end, pages=ticket.pages))

    def _restore(self, state):
        # One restored owner must make a real forward before a second restore.
        # Reserve its complete existing operation ceiling until that progress,
        # since the next operation may be a private Query/refresh or MTP frame.
        if (self._restored_history or self.cache_manager.block_pool.get_num_free_blocks() <
                self.cache_manager.reservations[state.ticket.request_id][1]):
            return False
        self._restore_target = state.ticket.request_id
        blocks = self.swaps.prepare_restore(state.ticket)
        if blocks is None:
            return False
        try:
            self._queue(state, 'restore', blocks)
        except BaseException:
            if state.ticket.request_id not in self.scheduler.connector.pending:
                self.swaps.abort_restore(state.ticket)
                self._inflight = None
            raise
        self.swaps.restore_dispatched(state.ticket)
        self.restores += 1
        self.restore_transfer_bound_bytes += state.checkpoint_bytes
        return True

    def _all_residents_blocked(self):
        owners = [r for r in self.scheduler.running if self._active(r) and r.request_id not in self._states]
        return bool(owners) and all(r.request_id in self._pressure and
            self._pressure[r.request_id][2] == r.num_computed_tokens for r in owners)

    def _victims(self, required, *, exclude=None, ignore_cooldown=False):
        candidates = [r for r in self.scheduler.running if r is not exclude and
                      self._eligible(r, ignore_cooldown=ignore_cooldown)]
        def near_finish(request):
            return (request.num_computed_tokens >= request.num_prompt_tokens and
                    request.max_tokens - request.num_output_tokens <= self.min_resident_steps)
        fitting = [r for r in candidates if self.cache_manager._owned(r.request_id) >= required]
        if fitting:
            # Prefer a cheaper one-command reclaim, while allowing a near-finished
            # decode to release its own pages when another suitable owner exists.
            return sorted(fitting, key=lambda r: (near_finish(r), self.cache_manager._owned(r.request_id),
                self._turns(r.request_id), r.request_id))
        return sorted(candidates, key=lambda r: (-self.cache_manager._owned(r.request_id),
                       near_finish(r), self._turns(r.request_id), r.request_id))

    def _make_restore_room(self, target, *, urgent=False):
        if self._restored_history:
            return False
        required = (self.cache_manager.reservations[target.ticket.request_id][1] -
                    self.cache_manager.block_pool.get_num_free_blocks())
        if required <= 0:
            return self._restore(target)
        override = urgent or self._all_residents_blocked()
        candidates = self._victims(required)
        if sum(self.cache_manager._owned(r.request_id) for r in candidates) < required and override:
            candidates = self._victims(required, ignore_cooldown=True)
        chosen, pages, size = [], 0, 0
        for request in candidates:
            chosen.append(request)
            pages += self.cache_manager._owned(request.request_id)
            size += self._size(request)
            if pages >= required:
                break
        available = self.ledger.snapshot()['resources']['checkpoint_bytes']['emergency_available']
        if pages < required or size > available:
            self.capacity_deferrals += 1
            return False
        # Protect the future destination pages before capturing the first
        # victim. Otherwise other growth could consume every reclaimed page.
        self._restore_target = target.ticket.request_id
        return self._capture(chosen[0], emergency=True, ignore_cooldown=override,
            reason=('cancelled_session_restore' if target.cancelled else
                    'fair_restore_after_grace' if target.grace_expired else 'fair_restore')
                    if urgent else 'restore_capacity')

    def _completion_grace(self, target):
        """Briefly wait for bounded decode completion before a fairness swap.

        These potential frees never become allocation credit. The next native
        operation still checks physical pages, and the fixed progress deadline
        eventually selects an urgent victim even if completion did not occur.
        """
        age = self._progress_round - target.paused_progress
        if age >= self.max_pause_steps + self.min_resident_steps:
            if target.grace_used and not target.grace_expired:
                target.grace_expired = True
                self.completion_grace_expirations += 1
            return False
        if target.cancelled or self._all_residents_blocked():
            return False
        required = (self.cache_manager.reservations[target.ticket.request_id][1] -
                    self.cache_manager.block_pool.get_num_free_blocks())
        if required <= 0:
            target.grace_finishers.clear()
            return False
        # Only owners already qualified by a prior grace decision may bridge
        # the terminal ACK gap. Native retirement removes them from running
        # before the archive fence permits their actual page release. Keep
        # exact request identity, never a reusable request-id or quota forecast.
        target.grace_finishers = {rid: request for rid, request in target.grace_finishers.items()
            if request.request_id == rid and self.scheduler.requests.get(rid) is request and
            self.cache_manager.reservations.get(rid, (None,))[0] is request and
            self.cache_manager._owned(rid) > 0}
        retiring = {rid: request for rid, request in target.grace_finishers.items()
                    if request.is_finished() and not request.num_in_flight_tokens}
        finishers = dict(retiring)
        for request in self.scheduler.running:
            rid = request.request_id
            remaining = request.max_tokens - request.num_output_tokens
            pressure = self._pressure.get(rid)
            if (request.num_computed_tokens < request.num_prompt_tokens or
                    not 0 < remaining <= self.min_resident_steps or
                    (pressure is not None and pressure[2] == request.num_computed_tokens) or
                    not self._eligible(request, ignore_cooldown=True) or
                    self.cache_manager.reservations.get(rid, (None,))[0] is not request):
                continue
            finishers[rid] = request
        if sum(self.cache_manager._owned(rid) for rid in finishers) < required:
            return False
        target.grace_finishers = finishers
        target.grace_used = True
        self.completion_grace_polls += 1
        self.completion_grace_retirement_polls += bool(retiring)
        if target.last_grace_progress != self._progress_round:
            target.last_grace_progress = self._progress_round
            self.completion_grace_rounds += 1
            self.completion_grace_retirement_rounds += bool(retiring)
            self._trace.append(dict(step=self._step(), progress_round=self._progress_round,
                request_id=target.ticket.request_id, action='completion_grace',
                history_end=target.ticket.history_end, pages=target.ticket.pages,
                deadline_progress_round=target.paused_progress+self.max_pause_steps+self.min_resident_steps,
                finishers=list(finishers), retiring_finishers=list(retiring)))
        return True

    def before_schedule(self):
        """Advance maintenance before native scheduling chooses public rows."""
        if self.cache_manager is None:
            raise RuntimeError('Request swap controller has not been bound')
        connector = self.scheduler.connector
        self._observe_progress()
        for rid, state in list(self._states.items()):
            if state.phase != 'done' or rid in connector.pending:
                continue
            del self._states[rid]
            self._pressure.pop(rid, None)
            if self._restore_target == rid and state.cancelled:
                self._restore_target = None
            if state.cancelled:
                self.finish_callback(state.ticket.request, delay_free_blocks=state.delay_free_blocks)
            else:
                self._resumed[rid] = self._turns(rid)
                self._restored_history[rid] = state.ticket.history_end
                self.max_pause_steps_observed = max(self.max_pause_steps_observed,
                    self._step() - state.paused_step)
                self.max_pause_seconds_observed = max(self.max_pause_seconds_observed,
                    monotonic() - state.paused_at)
                self.max_pause_progress_observed = max(self.max_pause_progress_observed,
                    self._progress_round - state.paused_progress)
        for rid, (request, _, history, _, _) in list(self._pressure.items()):
            if not self._active(request) or request.num_computed_tokens != history:
                self._pressure.pop(rid, None)
        self._resumed = {rid: step for rid, step in self._resumed.items() if rid in self.scheduler.requests}
        for rid, history in list(self._restored_history.items()):
            request = self.scheduler.requests.get(rid)
            if request is None or request.is_finished():
                del self._restored_history[rid]
                if self._restore_target == rid:
                    self._restore_target = None
            elif request.num_computed_tokens > history:
                self.resume_forward_completions += 1
                self.resume_forward_tokens += request.num_computed_tokens - history
                del self._restored_history[rid]
                if self._restore_target == rid:
                    self._restore_target = None
        if self._inflight is not None or connector.pending or self.scheduler.transactions:
            return
        # After restore, keep scheduling blocked until the checkpoint owner is
        # dropped. A cancellation at any phase follows this same drop fence.
        for state in self._states.values():
            if state.phase == 'restored' or (state.phase == 'paused' and state.cancelled and
                                            state.session_identity is None):
                self._queue(state, 'drop')
                self.drops += 1
                return
        # Session terminal publication reads exact P/M state. A canceled session
        # must regain its pages before DROP and the original session finish path.
        for state in self._states.values():
            if state.phase == 'paused' and state.cancelled and state.session_identity is not None:
                self._make_restore_room(state, urgent=True)
                return
        paused = [state for state in self._states.values() if state.phase == 'paused']
        paused.sort(key=lambda state: (state.paused_progress, state.ticket.operation_id))
        target = self._states.get(self._restore_target)
        if target is not None and target.phase == 'paused' and not self._restored_history:
            self._make_restore_room(target)
            return
        overdue = paused and self._progress_round - paused[0].paused_progress >= self.max_pause_steps
        if overdue and not self._restored_history:
            if self._completion_grace(paused[0]):
                return
            self._make_restore_room(paused[0], urgent=True)
            return
        all_blocked = self._all_residents_blocked()
        for request, required, _, _, first_progress in sorted(self._pressure.values(),
                key=lambda item: (item[0].request_id not in self._restored_history,
                                  item[0].request_id not in self.cache_manager.reservations, item[4])):
            if request.request_id in self._states:
                continue
            if self.cache_manager.block_pool.get_num_free_blocks() >= required:
                return  # Let the waiting owner use the pages before restoring its victim.
            waiting = request.request_id not in self.cache_manager.reservations
            overdue_waiter = self._progress_round - first_progress >= self.max_pause_steps
            if waiting and not overdue_waiter and not all_blocked:
                # Existing operations can use the newly available concurrency.
                # A newcomer does not immediately evict a progressing owner.
                self.waiter_deferrals += 1
                continue
            required -= self.cache_manager.block_pool.get_num_free_blocks()
            override = all_blocked or overdue_waiter
            candidates = self._victims(required, exclude=request)
            if not candidates and override:
                candidates = self._victims(required, exclude=request, ignore_cooldown=True)
            for victim in candidates:
                if self._capture(victim, emergency=request.request_id in self._restored_history,
                                 ignore_cooldown=override,
                                 reason='waiter_fairness' if waiting else 'resident_growth'):
                    return
            # Too much RAM is held by paused checkpoints: restore one using
            # the protected progress margin before attempting further growth.
            if paused:
                self._make_restore_room(paused[0])
            return
        if (paused and not self._restored_history and
                self.cache_manager.block_pool.get_num_free_blocks() >=
                self.cache_manager.reservations[paused[0].ticket.request_id][1]):
            self._restore_target = paused[0].ticket.request_id
            self._restore(paused[0])

    def complete(self, request_id, command_id):
        """Consume the connector's verified worker ACK; never queue recursively."""
        state = self._states.get(request_id)
        if state is None:
            return False
        if (self._inflight != (request_id, command_id) or state.command is None or
                state.command.command_id != command_id):
            raise ValueError('Stale or foreign request swap completion')
        validate_session_swap(self.scheduler, state.ticket.request, state.session_identity)
        action = state.command.action
        if action == 'capture':
            self.swaps.checkpoint_complete(state.ticket, _Ready())
            self.ledger.allocated(state.grant, {'checkpoint_bytes': state.checkpoint_bytes})
            self.ledger.settle(state.grant, _Ready())
            state.phase = 'paused'
        elif action == 'restore':
            self.swaps.restore_complete(state.ticket, _Ready())
            state.phase = 'restored'
        else:
            if state.ticket.phase is SwapPhase.PAUSED:
                self.swaps.discard_paused(state.ticket)
            self.ledger.release(state.grant, _Ready())
            state.phase = 'done'
        self._inflight = None
        return True

    def defer_free(self, request, delay_free_blocks=False):
        if request.is_finished():
            self._progress_pending = self._record_progress(request) or self._progress_pending
        state = self._states.get(request.request_id)
        self._pressure.pop(request.request_id, None)
        if state is None:
            if request.is_finished():
                self._restored_history.pop(request.request_id, None)
                self._resumed.pop(request.request_id, None)
                if self._restore_target == request.request_id:
                    self._restore_target = None
            return False
        if state.ticket.request is not request or not request.is_finished():
            raise ValueError('Deferred swap cleanup requires the exact finished request')
        state.cancelled = True
        state.delay_free_blocks |= bool(delay_free_blocks)
        return True

    def snapshot(self):
        return dict(capture_submissions=self.captures, restore_submissions=self.restores, drops=self.drops,
            pauses=self.swaps.pauses if self.swaps is not None else 0,
            restores=self.swaps.restores if self.swaps is not None else 0,
            restore_deferrals=self.swaps.restore_deferrals if self.swaps is not None else 0,
            capacity_deferrals=self.capacity_deferrals,
            resume_forward_completions=self.resume_forward_completions,
            resume_forward_tokens=self.resume_forward_tokens,
            max_pause_steps_observed=self.max_pause_steps_observed,
            max_pause_progress_rounds_observed=self.max_pause_progress_observed,
            max_pause_seconds_observed=self.max_pause_seconds_observed,
            progress_rounds=self._progress_round, idle_poll_calls=self._idle_polls,
            observed_forward_turns=self._forward_turns, observed_forward_tokens=self._forward_tokens,
            cooldown_overrides=self.cooldown_overrides, waiter_deferrals=self.waiter_deferrals,
            completion_grace_rounds=self.completion_grace_rounds,
            completion_grace_polls=self.completion_grace_polls,
            completion_grace_expirations=self.completion_grace_expirations,
            completion_grace_retirement_rounds=self.completion_grace_retirement_rounds,
            completion_grace_retirement_polls=self.completion_grace_retirement_polls,
            capture_transfer_bound_bytes=self.capture_transfer_bound_bytes,
            restore_transfer_bound_bytes=self.restore_transfer_bound_bytes,
            progress_clock_scope='quiescent positive history advances; imported and restored history excluded',
            checkpoint_capacity_bytes=self.checkpoint_bytes,
            checkpoint_cpu_capacity_bytes=self.checkpoint_cpu_bytes,
            worst_checkpoint_bytes=self.worst_checkpoint_bytes,
            progress_reserve_bytes=self.progress_reserve_bytes,
            resources=self.ledger.snapshot() if self.ledger is not None else None,
            restore_target=self._restore_target, inflight=self._inflight,
            restore_progress_reserved_pages=self.reserved_pages('') if self.cache_manager is not None else 0,
            recent_transitions=list(self._trace),
            awaiting_forward_progress=list(self._restored_history),
            pressure={rid: dict(required_pages=value[1], history_end=value[2], first_step=value[3],
                               first_progress_round=value[4])
                      for rid, value in self._pressure.items()},
            requests={rid: dict(phase=state.phase, operation_id=state.ticket.operation_id,
                history_end=state.ticket.history_end, pages=state.ticket.pages,
                checkpoint_bytes=state.checkpoint_bytes, cancelled=state.cancelled,
                session_bound=state.session_identity is not None,
                paused_progress_rounds=self._progress_round-state.paused_progress)
                for rid, state in self._states.items()})
