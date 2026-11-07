"""Fenced native-page swaps while retaining the request's runner slot.

This is an explicit control primitive, not ordinary vLLM preemption. Callers
must stop public scheduling, drain publication/private work, checkpoint all
native cache groups, and retain the runner/sampler/authority state. No K/G,
window offsets, selection state or refresh boundary is changed here.
"""
from dataclasses import dataclass
from enum import Enum
from itertools import count


class SwapPhase(str, Enum):
    CAPTURING = 'capturing'
    PAUSED = 'paused'
    RESTORING = 'restoring'
    COMPLETE = 'complete'
    DISCARDED = 'discarded'


@dataclass(eq=False)
class PageSwap:
    request: object
    operation_id: int
    history_end: int
    previous_eligible_step: int
    original_blocks: tuple[tuple[int, ...], ...]
    offsets: tuple[int | None, ...]
    phase: SwapPhase = SwapPhase.CAPTURING
    destination_blocks: tuple[tuple[int, ...], ...] | None = None
    restore_submitted: bool = False

    @property
    def request_id(self):
        return self.request.request_id

    @property
    def pages(self):
        return sum(map(len, self.original_blocks))


class FencedPageSwap:
    """Detach and rebind exclusive pages, preserving all native logical state.

    Supported managers are the existing bounded full-attention wrapper and
    non-prefix-cached ``mamba_cache_mode=none`` state manager. Constructing this
    object does not turn on overcommit or relax a lifetime capacity guarantee.
    The owner still controls admission and must reserve checkpoint bytes before
    ``prepare``. Completion objects must fence *all* relevant worker streams.
    """

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.pool = cache_manager.block_pool
        self.groups = tuple(cache_manager.coordinator.single_type_managers)
        if not self.groups or any(m.block_pool is not self.pool for m in self.groups):
            raise ValueError('Page swap requires one shared native page pool')
        for manager in self.groups:
            if (getattr(manager, 'enable_caching', False) or
                    getattr(manager, 'mamba_cache_mode', 'none') != 'none'):
                raise ValueError('Page swap requires APC off and non-aligned state')
        self.entries = {}
        self._operations = count(1)
        self.pauses = self.restores = self.restore_deferrals = 0

    def _rows(self, request_id):
        return tuple(tuple(m.req_to_blocks.get(request_id, ())) for m in self.groups)

    @staticmethod
    def _ids(rows):
        return tuple(tuple(b.block_id for b in row) for row in rows)

    def _offsets(self, request_id):
        return tuple(getattr(m, 'offsets', {}).get(request_id) for m in self.groups)

    def _check_request(self, ticket):
        if self.entries.get(ticket.request_id) is not ticket:
            raise ValueError('Stale or foreign page swap')
        request = ticket.request
        if (request.num_computed_tokens != ticket.history_end or
                request.num_in_flight_tokens or
                request.next_decode_eligible_step != 2**62 or
                self._offsets(ticket.request_id) != ticket.offsets):
            raise RuntimeError('Suspended request advanced or changed its window')

    def _check_rows(self, ticket, expected):
        self._check_request(ticket)
        rows = self._rows(ticket.request_id)
        if self._ids(rows) != expected:
            raise RuntimeError('Page swap allocation changed before completion')
        self._exclusive(rows)
        return rows

    @staticmethod
    def _exclusive(rows):
        blocks = [block for row in rows for block in row]
        if (not rows or any(not row for row in rows) or
                any(b.is_null or b.ref_cnt != 1 for b in blocks) or
                len({b.block_id for b in blocks}) != len(blocks)):
            raise ValueError('Page swap requires complete exclusive non-null groups')

    def prepare(self, request):
        """Freeze a quiescent request after checkpoint capacity was admitted."""
        rid = request.request_id
        if (rid in self.entries or request.num_in_flight_tokens or
                request.num_computed_tokens <= 0 or
                request.next_decode_eligible_step >= 2**62 or request.is_finished()):
            raise ValueError('Page swap requires a live quiescent request')
        for manager in self.groups:
            if (rid in getattr(manager, 'pending', {}) or rid in getattr(manager, 'imports', {}) or
                    rid in getattr(manager, '_partial_hit_reqs', {})):
                raise ValueError('Page swap cannot overlap refresh or checkpoint import')
        rows = self._rows(rid)
        self._exclusive(rows)
        ticket = PageSwap(request, next(self._operations), request.num_computed_tokens,
                          request.next_decode_eligible_step, self._ids(rows), self._offsets(rid))
        self.entries[rid] = ticket
        request.next_decode_eligible_step = 2**62
        return ticket

    def checkpoint_complete(self, ticket, completion):
        """Return pages only after the independent checkpoint and read fences."""
        if ticket.phase is not SwapPhase.CAPTURING:
            raise ValueError('Page swap checkpoint was already completed')
        rows = self._check_rows(ticket, ticket.original_blocks)
        if not completion.query():
            return False
        # Keep num_cached_block, native state bookkeeping and window offsets.
        # Ordinary free/pop would destroy state needed for this same request.
        for manager in self.groups:
            del manager.req_to_blocks[ticket.request_id]
        self.pool.free_blocks(reversed([b for row in rows for b in row]))
        ticket.phase = SwapPhase.PAUSED
        self.pauses += 1
        return True

    def abort_capture(self, ticket, completion):
        """Keep original pages after a failed/unpublished capture has drained."""
        if ticket.phase is not SwapPhase.CAPTURING:
            raise ValueError('Only an unfinished capture can be aborted')
        self._check_rows(ticket, ticket.original_blocks)
        if not completion.query():
            return False
        ticket.request.next_decode_eligible_step = ticket.previous_eligible_step
        ticket.phase = SwapPhase.COMPLETE
        del self.entries[ticket.request_id]
        return True

    def prepare_restore(self, ticket):
        """Atomically obtain all replacement groups, or allocate nothing."""
        if ticket.phase is not SwapPhase.PAUSED:
            raise ValueError('Page swap restore requires a paused request')
        self._check_request(ticket)
        if any(self._rows(ticket.request_id)):
            raise RuntimeError('Paused request unexpectedly owns native pages')
        if self.pool.get_num_free_blocks() < ticket.pages:
            self.restore_deferrals += 1
            return None
        blocks = self.pool.get_new_blocks(ticket.pages)
        rows, start = [], 0
        for original in ticket.original_blocks:
            end = start + len(original)
            rows.append(blocks[start:end])
            start = end
        self._exclusive(rows)
        for manager, row in zip(self.groups, rows):
            manager.req_to_blocks[ticket.request_id] = list(row)
        ticket.destination_blocks = self._ids(rows)
        ticket.phase = SwapPhase.RESTORING
        return ticket.destination_blocks

    def restore_dispatched(self, ticket):
        if ticket.phase is not SwapPhase.RESTORING or ticket.restore_submitted:
            raise ValueError('Duplicate or premature page swap restore dispatch')
        self._check_rows(ticket, ticket.destination_blocks)
        ticket.restore_submitted = True

    def abort_restore(self, ticket):
        """Undo an allocation only when no restore work was published."""
        if ticket.phase is not SwapPhase.RESTORING or ticket.restore_submitted:
            raise ValueError('Published restore must drain before releasing pages')
        rows = self._check_rows(ticket, ticket.destination_blocks)
        for manager in self.groups:
            del manager.req_to_blocks[ticket.request_id]
        self.pool.free_blocks(reversed([b for row in rows for b in row]))
        ticket.destination_blocks = None
        ticket.phase = SwapPhase.PAUSED

    def restore_complete(self, ticket, completion):
        if ticket.phase is not SwapPhase.RESTORING or not ticket.restore_submitted:
            raise ValueError('Page swap restore has no dispatched completion')
        self._check_rows(ticket, ticket.destination_blocks)
        if not completion.query():
            return False
        ticket.request.next_decode_eligible_step = ticket.previous_eligible_step
        ticket.phase = SwapPhase.COMPLETE
        del self.entries[ticket.request_id]
        self.restores += 1
        return True

    def discard_paused(self, ticket):
        """Forget a cancelled paused request after its worker checkpoint drops.

        No physical pages or active DMA may exist. The caller then performs
        ordinary request cleanup to remove retained native logical bookkeeping.
        """
        if ticket.phase is not SwapPhase.PAUSED:
            raise ValueError('Only fully paused swaps can be discarded')
        self._check_request(ticket)
        if not ticket.request.is_finished() or any(self._rows(ticket.request_id)):
            raise ValueError('Discard requires a cancelled request without pages')
        ticket.phase = SwapPhase.DISCARDED
        del self.entries[ticket.request_id]

    def snapshot(self):
        return dict(pauses=self.pauses, restores=self.restores,
                    restore_deferrals=self.restore_deferrals,
                    requests={rid: dict(phase=t.phase.value, operation_id=t.operation_id,
                                        history_end=t.history_end, pages=t.pages)
                              for rid, t in self.entries.items()})
