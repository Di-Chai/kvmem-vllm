"""Bounded logical publications sharing one physical DMA lease.

The default-disabled physical coalescing path retains CaptureRuntime's owner
FIFO/consuming claim and validates the whole physical DMA and score ticket
before calling ``advance``. The returned receipt replaces one original
logical metadata row, not one completed physical owner. Only ``owner_done``
permits marking the owner published and releasing its physical download.

Archive append owns MeanK finalization, sealing and (for native history) its
``gc_begin_step`` call. This cursor never duplicates that call. The runtime must
apply each receipt's accepted/rejected/discarded counts, logical last_steps and
capacity snapshot before it permits another policy operation. Exceptions after
append are fail-stop: neither GraphUpdate.abort nor this cursor rolls back KV.
"""
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace

import torch

from .authority import append_authority_batch
from .logical_action import LogicalActionSlice
from .logical_prefill import LogicalPrefillGroup
from .terminal import TerminalPrefix


@dataclass(frozen=True)
class LogicalPublicationReceipt:
    logical_index: int
    history_begin: int
    local_rope_begin: int
    scheduled: int
    accepted: int
    rejected: int
    terminal_discarded: int
    published_history_end: int
    ready_records: int
    owner_done: bool
    logical_commits: int = 1
    # These are logical calls, even though there is just one physical packet.
    append_calls: int = 1
    append_requests: int = 1


class LogicalPublicationCursor:
    """One owner, bounded spans, at most one reserved GraphUpdate, no new buffers.

    ``packet`` is the existing CapturedPacket, not a copied download. The caller
    must set packet.confirmed through the scheduler's normal begin_step before
    confirm(). Explicit receipts are monotonic; reading a mutable terminal map
    during a native append is deliberately unsupported. Concurrent calls return
    BlockingIOError so the engine can defer the receipt without waiting for CPU
    copy. The caller retains the physical owner until this cursor is complete.
    """

    def __init__(self, group, packet, authority, action_runtime, slices, *,
                 owner_epoch, source_generation):
        if not isinstance(group, LogicalPrefillGroup):
            raise ValueError('Publication requires an existing logical group')
        if (type(slices) is not tuple or len(slices) != len(group.spans) or
                any(not isinstance(part, LogicalActionSlice) or part.group is not group or
                    part.plan.span != span for part, span in zip(slices, group.spans))):
            raise ValueError('Publication requires every validated logical score slice')
        if (tuple(packet.request_ids) != (group.request_id,) or packet.tokens != group.count or
                getattr(packet, 'request_tokens', ()) not in ((), (group.count,)) or
                getattr(packet, 'query_inputs', None) or getattr(packet, 'source_released', False)):
            raise ValueError('Logical publication requires one intact ordinary physical packet')
        if type(source_generation) is not int or source_generation < 0:
            raise ValueError('Physical source generation must be a nonnegative integer')
        state = getattr(authority, 'action_history', None)
        graph = getattr(authority, 'block_graph', None)
        if (state is None or getattr(state, 'prefill', None) is None or graph is None or
                not callable(getattr(action_runtime, 'prepare_accept', None))):
            raise ValueError('Logical publication requires active prefill attention and graph transactions')
        self.group, self.packet, self.authority = group, packet, authority
        self.action, self.slices = action_runtime, slices
        self.state, self.graph = state, graph
        self.source_generation = source_generation
        self.graph_binding = self.graph.binding
        self.next_index = 0
        self.phase = 'unconfirmed'
        self.confirmation_epoch = None
        self._confirmation = None
        self.retained = None
        self.error = self.failure_stage = None
        self.archive_done = self.semantic_done = False
        self.published_history_end = group.history_begin
        self.prepared = None
        self._child = None
        self._payload = None
        self._pending_receipt = self._last_receipt = None
        self._lock = Lock()
        self._validate_binding(owner_epoch)
        if authority.history_end != group.history_begin:
            raise ValueError('Logical group does not begin at the published authority prefix')

    @property
    def done(self):
        return self.phase in ('done', 'released')

    def _validate_binding(self, owner_epoch):
        if (owner_epoch != self.group.owner_epoch or
                self.packet.source_generation != self.source_generation or
                self.authority.memory.request_id != self.group.request_id or
                self.authority.action_history is not self.state or
                self.authority.block_graph is not self.graph or
                self.packet.source_released):
            raise RuntimeError('Logical publication lost its owner or physical source generation')
        self.graph.validate_binding(self.graph_binding)

    def _check_open(self):
        if self.error is not None:
            raise self.error
        if self.phase == 'released':
            raise RuntimeError('Logical publication lease has been released')

    def confirm(self, scheduler_step, *, owner_epoch, terminal_prefix=None, native_accepted=None):
        """Install one scheduler receipt before any logical append.

        Later epochs may trim only the still-unpublished suffix. A terminal
        cannot resurrect discarded rows or retract an already committed span.
        Repeating the same epoch is permitted only with identical contents.
        """
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError('Defer scheduler confirmation until this owner yields')
        try:
            self._check_open()
            self._validate_binding(owner_epoch)
            if not self.packet.confirmed:
                raise RuntimeError('Scheduler must confirm the physical packet first')
            if type(scheduler_step) is not int or scheduler_step <= self.group.scheduler_step:
                raise ValueError('Confirmation must follow the physical scheduler step')
            if native_accepted is None:
                native_accepted = self.group.count
            if type(native_accepted) is not int or native_accepted != self.group.count:
                raise ValueError('Logical ordinary prefill requires no native rejected rows')
            if terminal_prefix is not None and not isinstance(terminal_prefix, TerminalPrefix):
                raise ValueError('Terminal confirmation requires a scheduler TerminalPrefix')
            receipt = (native_accepted, terminal_prefix)
            if self.confirmation_epoch is not None:
                if scheduler_step < self.confirmation_epoch:
                    raise ValueError('Scheduler confirmation epoch moved backwards')
                if scheduler_step == self.confirmation_epoch:
                    if receipt != self._confirmation:
                        raise ValueError('Conflicting scheduler confirmation receipt')
                    return
            retained = native_accepted if terminal_prefix is None else max(0, min(
                native_accepted, terminal_prefix.history_end-self.group.history_begin))
            if (terminal_prefix is not None and terminal_prefix.history_end < self.published_history_end or
                    self.retained is not None and retained > self.retained):
                raise ValueError('Terminal confirmation would retract or resurrect published history')
            self.confirmation_epoch, self._confirmation = scheduler_step, receipt
            self.retained = retained
            if self.phase == 'unconfirmed':
                self.phase = 'ready'
        finally:
            self._lock.release()

    def _bind_payload(self, raw, values):
        expected = (self.authority.shape[0], self.group.count,
                    self.authority.shape[2], self.authority.shape[3])
        if (tuple(raw.shape) != expected or tuple(values.shape) != expected or
                raw.device.type != 'cpu' or values.device.type != 'cpu' or
                raw.dtype != self.authority.raw_dtype or values.dtype != torch.uint8):
            raise ValueError('Physical payload differs from the original authority codec')
        if self._payload is None:
            self._payload = raw, values
        elif raw is not self._payload[0] or values is not self._payload[1]:
            raise ValueError('A logical cursor cannot switch its physical payload lease')

    def advance(self, raw, values, *, owner_epoch):
        """Commit at most one original logical span; None means capacity retry.

        Caller bookkeeping is mandatory: apply the returned receipt exactly
        once before its next advance. If that bookkeeping fails, call fail()
        and quarantine the physical packet through the runtime's normal error
        path. Never use the number of logical commits as completed-owner count.
        """
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError('Logical owner already has a publication consumer')
        try:
            self._check_open()
            if self.phase == 'unconfirmed':
                raise RuntimeError('Logical publication is not scheduler-confirmed')
            if self.done:
                raise RuntimeError('Physical owner has already completed publication')
            if self._pending_receipt is not None:
                raise RuntimeError('Acknowledge the preceding logical runtime counters first')
            try:
                self._validate_binding(owner_epoch)
                self._bind_payload(raw, values)
                span = self.group.spans[self.next_index]
                if self.authority.history_end != self.published_history_end:
                    raise RuntimeError('Authority advanced outside its logical publication cursor')
                accepted = self.group.accepted_counts(self.retained)[self.next_index]
                self.archive_done = self.semantic_done = False
                self._child = SimpleNamespace(action_inputs={self.group.request_id:
                    self.slices[self.next_index].action_packet(self.state)}, query_inputs={})
                self.phase = 'preparing'
                self.prepared = self.action.prepare_accept(self._child, self.group.request_id, accepted,
                    graph=self.graph, history_begin=span.history_begin)
                self.prepared.graph_update.detach()
                self.phase = 'appending'
                begin, end = span.physical_offset, span.physical_offset+span.count
                consumed = append_authority_batch([(self.authority, raw[:, begin:end], values[:, begin:end],
                    dict(history_begin=span.history_begin, local_rope_begin=span.local_begin, accepted=accepted))])
                if not consumed:
                    self.prepared.abort()
                    self.prepared = self._child = None
                    self.phase = 'ready'
                    return None
                if consumed != 1:
                    raise RuntimeError('Logical append consumed an invalid owner count')
                self.archive_done = True
                self.phase = 'archive_done'
                self.phase = 'committing'
                self.prepared.commit()
                self.semantic_done = True
                self.published_history_end = self.authority.history_end
                self.graph_binding = self.graph.binding
                receipt = LogicalPublicationReceipt(self.next_index, span.history_begin, span.local_begin,
                    span.count, accepted, 0, span.count-accepted, self.published_history_end,
                    self.authority.memory.repository.ready_slots, self.next_index+1 == len(self.group.spans))
                self.next_index += 1
                self.prepared = self._child = None
                self._pending_receipt = receipt
                self.phase = 'awaiting_ack'
                return receipt
            except BaseException as error:
                self._fail(error)
                raise
        finally:
            self._lock.release()

    def ack(self, receipt):
        """After original counters/last_steps/capacity are updated exactly once.

        Runtime must make its update and this acknowledgement under its owner
        publication lock; if the update raises, fail() instead of retrying it.
        """
        with self._lock:
            self._check_open()
            if receipt is self._last_receipt and self._pending_receipt is None:
                return
            if receipt is not self._pending_receipt:
                raise ValueError('Acknowledgement does not own the pending logical receipt')
            self._last_receipt, self._pending_receipt = receipt, None
            self.phase = 'done' if receipt.owner_done else 'ready'

    def _fail(self, error):
        if self.error is not None:
            return
        self.error, self.failure_stage = error, self.phase
        self.phase = 'failed'
        if self.prepared is not None:
            try:
                self.prepared.abort()
            except BaseException as cleanup_error:
                error.add_note('Logical reservation cleanup also failed: '+repr(cleanup_error))
        # Retain prepared/child/payload/score/packet owners after every failure.

    def fail(self, error):
        """Quarantine a runtime bookkeeping failure after a returned receipt."""
        if not isinstance(error, BaseException):
            raise TypeError('Publication failure must retain its original exception')
        with self._lock:
            self._fail(error)

    def release(self):
        """Drop cursor-owned references only; caller still owns pool.release()."""
        with self._lock:
            self._check_open()
            if not self.done:
                raise RuntimeError('The physical packet still has unpublished logical spans')
            self.packet = self.authority = self.action = self.state = self.graph = None
            self.slices = self._payload = None
            self.phase = 'released'
