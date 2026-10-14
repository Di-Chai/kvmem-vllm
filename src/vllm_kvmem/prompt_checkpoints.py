"""Bounded worker ownership for exact, sealed prompt checkpoints.

The serving coordinator must supply an authenticated session namespace, serialize
turns within it, and detach the authority before retire(). This pool owns no GPU
pages. Claims transfer an idle authority only after native reconstruction has
completed; the scheduler must not advertise a hit before idle() reports readiness.
"""
from array import array
from dataclasses import dataclass
from enum import Enum, auto


class CheckpointPhase(Enum):
    LIVE = auto()       # Capture runtime still owns the executing authority.
    RETIRING = auto()   # Detached; rewind/publication/readers must drain.
    IDLE = auto()       # Exact saved authority prefix is available to import.
    CLAIMED = auto()    # New native allocation/restore owns the checkpoint.
    CLOSING = auto()    # Evicted; no new lookup may see it, readers still drain.


def token_bytes(tokens, count):
    if count <= 0 or len(tokens) <= count:
        raise ValueError('Prompt checkpoint must leave at least one token for native prefill')
    prefix = tokens[:count]
    if any(type(token) is not int or not 0 <= token < 2**32 for token in prefix):
        raise ValueError('Prompt identity requires exact unsigned token IDs')
    result = array('I', prefix)
    if result.itemsize != 4:
        raise RuntimeError('Unqualified token identity storage width')
    return result.tobytes()


@dataclass(eq=False)
class PromptCheckpoint:
    namespace: object
    request_id: str
    authority: object
    window: object
    model: object
    tokens: bytes
    payload_bytes: int
    phase: CheckpointPhase = CheckpointPhase.LIVE
    fence: object = None
    rewind_started: bool = False
    claimant: str | None = None
    restore_started: bool = False
    restore_records: tuple | None = None

    @property
    def position(self):
        return self.window.position


class PromptCheckpointPool:
    def __init__(self, capacity, max_payload_bytes):
        if any(type(v) is not int or v <= 0 for v in (capacity, max_payload_bytes)):
            raise ValueError('Checkpoint pool needs positive entry and payload limits')
        self.capacity = capacity
        self.max_payload_bytes = max_payload_bytes
        self.entries = {}
        self.payload_bytes = 0

    def _owned(self, checkpoint):
        if self.entries.get(checkpoint.namespace) is not checkpoint:
            raise ValueError('Stale or foreign prompt checkpoint lease')

    def capture(self, namespace, request_id, authority, tokens, model_adapter, *, replace=False, terminal=False):
        previous = self.entries.get(namespace)
        if previous is not None and not replace:
            raise ValueError('Session already owns a checkpoint; serialize its turns')
        if previous is not None and (previous.phase is not CheckpointPhase.LIVE or
                previous.request_id != request_id or previous.authority is not authority):
            raise ValueError('Replacement requires the same live session authority')
        if previous is None and len(self.entries) >= self.capacity:
            raise BufferError('Prompt checkpoint entry capacity is exhausted')
        position = authority.history_end
        # Validate token IDs before acquiring native prefix leases or GPU copies.
        if (type(position) is not int or position <= 0 or
                (position % authority.memory.config.block_tokens and not terminal)):
            raise ValueError('Prompt checkpoint requires a sealed authority boundary')
        token_size = position * 4
        retained_bytes = self.payload_bytes - (previous.payload_bytes if previous else 0)
        if token_size >= self.max_payload_bytes - retained_bytes:
            raise BufferError('Prompt checkpoint payload capacity is exhausted')
        identity = token_bytes(tokens, position)
        window = authority.memory.checkpoint_window(allow_partial=terminal)
        # Account for the serialized frame geometry as well as token/model data.
        # Python object overhead and authority/archive storage are separate quotas.
        fixed = len(identity) + len(window.records) * 5 * 8
        available = self.max_payload_bytes - retained_bytes - fixed
        if available <= 0:
            raise BufferError('Prompt checkpoint payload capacity is exhausted')
        model = model_adapter.capture(request_id, position, max_bytes=available,
                                      **({'terminal': True} if terminal else {}))
        checkpoint = PromptCheckpoint(namespace, request_id, authority, window,
                                      model, identity, fixed + model.bytes)
        # Keep the previous snapshot available if any validation/copy/admission
        # above fails. Only a complete CPU-owned replacement releases its leases.
        if previous is not None:
            self._remove(previous)
        self.entries[namespace] = checkpoint
        self.payload_bytes += checkpoint.payload_bytes
        return checkpoint

    def drop_live(self, checkpoint):
        """Discard a snapshot while capture runtime keeps its live authority."""
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.LIVE:
            raise ValueError('Only a live borrowed snapshot may be dropped')
        self._remove(checkpoint)

    def update_live_window(self, checkpoint):
        """Keep the pre-Query model state with the committed replay KV frame.

        A short turn may end before the next sealed decode checkpoint. Its
        replay must use the selected frame used by the original durable Query,
        without rerunning selection from the older pressure frame.
        """
        self._owned(checkpoint)
        if (checkpoint.phase is not CheckpointPhase.LIVE or
                checkpoint.authority.history_end != checkpoint.position):
            raise ValueError('Window update requires the exact live checkpoint boundary')
        window = checkpoint.authority.memory.checkpoint_window()
        delta = (len(window.records) - len(checkpoint.window.records)) * 5 * 8
        if self.payload_bytes + delta > self.max_payload_bytes:
            raise BufferError('Updated checkpoint window exceeds its payload budget')
        checkpoint.window = window
        checkpoint.payload_bytes += delta
        self.payload_bytes += delta

    def retire(self, checkpoint, completion):
        """Caller has detached capture and all further suffix execution."""
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.LIVE:
            raise ValueError('Prompt authority already detached or claimed')
        checkpoint.fence = completion
        checkpoint.phase = CheckpointPhase.RETIRING

    def idle(self, checkpoint):
        self._owned(checkpoint)
        if checkpoint.phase is CheckpointPhase.IDLE:
            return True
        if checkpoint.phase is not CheckpointPhase.RETIRING:
            return False
        if checkpoint.fence is not None and not checkpoint.fence.query():
            return False
        authority = checkpoint.authority
        if authority.history_end == checkpoint.position and not checkpoint.rewind_started:
            # At a terminal checkpoint every accepted raw K/V row is retained,
            # including the mutable partial tail. No branching or rewrite of an
            # immutable archive generation is necessary before transfer.
            checkpoint.phase = CheckpointPhase.IDLE
            checkpoint.fence = None
            return True
        if not checkpoint.rewind_started:
            spans = authority.begin_rewind(checkpoint.window.prefix)
            if spans is None:
                return False
            authority.attach_rewind_completion(checkpoint.fence)
            checkpoint.rewind_started = True
        if not authority.poll_rewind():
            return False
        checkpoint.phase = CheckpointPhase.IDLE
        checkpoint.fence = None
        return True

    def claim(self, namespace, request_id, tokens):
        checkpoint = self.entries.get(namespace)
        if checkpoint is None or checkpoint.phase is not CheckpointPhase.IDLE:
            return None
        # No hash-only equality and no reuse of reserialized generated suffixes.
        if len(tokens) <= checkpoint.position or token_bytes(tokens, checkpoint.position) != checkpoint.tokens:
            return None
        if not request_id or request_id == checkpoint.request_id:
            raise ValueError('Checkpoint import requires a fresh native request identity')
        checkpoint.phase = CheckpointPhase.CLAIMED
        checkpoint.claimant = request_id
        return checkpoint

    def finish_claim(self, checkpoint, completion):
        """After exact target/GDN/draft restoration, return authority ownership.

        The caller owns fresh native pages through this fence and must already
        have committed RequestMemory.restore_window. A failed import instead calls
        evict with its cleanup fence and must never resume native execution.
        """
        from .memory import Phase
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLAIMED:
            raise ValueError('Checkpoint is not claimed by a native import')
        if checkpoint.fence is not None and checkpoint.fence is not completion:
            raise ValueError('Checkpoint import completion fence changed')
        checkpoint.fence = completion
        if not completion.query():
            return None
        authority = checkpoint.authority
        if (not checkpoint.restore_started or authority.memory.phase is not Phase.READY or
                authority.history_end != checkpoint.position or
                tuple(authority.memory._store.working_records()) != checkpoint.restore_records):
            raise ValueError('Checkpoint import did not commit its exact target prefix')
        self._remove(checkpoint)
        return authority

    def prepare_claim_window(self, checkpoint, *, pressure=False):
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLAIMED or checkpoint.restore_started:
            raise ValueError('No unsubmitted checkpoint import')
        memory = checkpoint.authority.memory
        if pressure:
            memory._validate_window_checkpoint(checkpoint.window)
            if memory.history_end != checkpoint.position:
                raise ValueError('Pressure restore requires the exact saved history prefix')
            # Native allocation is fresh: neither same-position K nor resident V
            # from the retired request is still resident in these physical pages.
            for slot, generation, *_ in memory._store.working_records():
                memory._store.working_resident(memory._store.handle(slot), False)
            plan = memory.select({r.handle.slot: float(r.history_begin)
                for r in checkpoint.authority.records.values()})
        else:
            plan = memory.restore_window(checkpoint.window)
        checkpoint.restore_records = tuple(memory._store.working_records())
        checkpoint.restore_started = True
        return plan

    def evict(self, checkpoint, completion=None):
        self._owned(checkpoint)
        if checkpoint.phase is CheckpointPhase.LIVE:
            raise ValueError('Live capture still owns authority; use drop_live')
        if checkpoint.phase is CheckpointPhase.CLAIMED:
            if completion is None:
                raise ValueError('Allocated import eviction requires its cleanup fence')
            checkpoint.fence = completion
        elif completion is not None:
            raise ValueError('Unexpected replacement of authority cleanup fence')
        checkpoint.phase = CheckpointPhase.CLOSING
        return self.poll_eviction(checkpoint)

    def poll_eviction(self, checkpoint):
        self._owned(checkpoint)
        if checkpoint.phase is not CheckpointPhase.CLOSING:
            raise ValueError('Checkpoint has not been evicted')
        if checkpoint.fence is not None and not checkpoint.fence.query():
            return False
        # Release our prefix pins before closing. External readers still keep
        # the entry and its credits unavailable until authority.close succeeds.
        checkpoint.window = checkpoint.model = None
        checkpoint.tokens = b''
        if not checkpoint.authority.close():
            return False
        self._remove(checkpoint)
        return True

    def _remove(self, checkpoint):
        del self.entries[checkpoint.namespace]
        self.payload_bytes -= checkpoint.payload_bytes
        # Do not let an obsolete Python lease retain native prefix/storage pins.
        checkpoint.window = checkpoint.model = checkpoint.authority = None
        checkpoint.tokens = b''
