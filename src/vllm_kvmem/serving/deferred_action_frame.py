"""One whole-frame validation barrier before any accepted-history mutation.

The publication queue owns the physical KV prefix. FrozenScorePool owns copies
of overwriteable Q, page-table and scale inputs until this barrier completes.
No worker touches the live NativeAttentionStats bank or its global counters.
"""
from threading import Condition


class DeferredActionResult(dict):
    def __init__(self, packets, validation):
        super().__init__(packets)
        self.validation = validation


class DeferredActionFrame:
    def __init__(self, pool):
        self.pool = pool
        self.slot = None
        self.ticket = None
        self.outcome = None
        self.condition = Condition()
        self.state = 'private'
        self.error = None
        self.applied = False

    def seal(self, slot, ticket):
        if self.state != 'private' or self.slot is not slot:
            raise RuntimeError('Deferred score frame lost its private snapshot')
        self.ticket = ticket
        self.pool.seal(slot)
        self.state = 'pending'

    @property
    def ready(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Deferred score frame is quarantined') from self.error
            return self.state == 'ready'

    def quarantine(self, error):
        # Keep the slot, Q snapshots and physical-prefix lease. Even a CUDA
        # event failure must never turn into a slot reuse or release ACK.
        with self.condition:
            if self.slot is not None and hasattr(self.pool, 'quarantine'):
                self.pool.quarantine(self.slot, error)
            self.error = error
            self.state = 'failed'
            self.condition.notify_all()

    def resolve(self, *, wait):
        with self.condition:
            while self.state == 'resolving':
                if not wait:
                    return False
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError('Deferred score frame is quarantined') from self.error
            if self.state == 'ready':
                return True
            if self.state != 'pending':
                raise RuntimeError('Deferred score frame was not sealed')
            if not wait:
                # Maintenance is only a poll. In particular a ready flags DMA
                # with an invalid score cannot launch fallback on this path.
                return False
            self.state = 'resolving'
        try:
            outcome = self.ticket.resolve()
            if not outcome.valid:
                self.pool.fallback(self.slot)
            self.pool.release(self.slot)
        except BaseException as error:
            self.quarantine(error)
            raise
        with self.condition:
            self.slot = None
            self.outcome = outcome
            self.state = 'ready'
            self.condition.notify_all()
        return True

    def apply(self, native):
        # Engine thread only; completed outcomes are immutable and applying an
        # old failure cannot change plans/flags belonging to the new forward.
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Deferred score frame is quarantined') from self.error
            if self.state != 'ready':
                return False
            if self.applied:
                raise RuntimeError('Deferred score outcome already applied')
            native.apply_detached_validation(self.ticket, self.outcome)
            self.applied = True
            return True
