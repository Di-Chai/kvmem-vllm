"""Explicit semantic evidence for the native statistical history policy.

Call this controller on an owning request thread at step/refresh boundaries.
It neither examines token text nor runs a model. Observations must describe an
actual measurement; absence from the current GPU window is not an observation.
"""
from dataclasses import dataclass, field
import math


def _key(handle):
    return handle.slot, handle.generation


def _score(value, name):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return float(value)


@dataclass
class _Span:
    handles: dict
    kind: str
    complete: bool = False
    observations: int = 0
    cold_observations: int = 0
    importance: float = 1.0
    observation_id: int = -1
    model_id: int = -1
    model_score: float | None = None
    revision: int = 0
    pending: set = field(default_factory=set)
    update_hold: object = None
    result_hold: object = None


class HistoricalEvidenceLedger:
    """Bounded, generation-qualified evidence and retained result witnesses.

    A completion event permits consideration, not deletion by itself. Native
    thresholds, mandatory/P/M guards and in-flight selection guards still apply.
    Result witnesses remain protected until this ledger is reset or closed.
    Each ledger is bound to one branch epoch; late signals cannot cross reset.
    """

    KINDS = frozenset(('thinking', 'completed_step', 'tool_detail', 'superseded'))

    def __init__(self, scope, repository, records, *, block_tokens, epoch,
                 max_spans=64, max_blocks=4096, next_revision=None):
        if type(max_spans) is not int or max_spans <= 0 or type(max_blocks) is not int or max_blocks <= 0:
            raise ValueError('Evidence ledger limits must be positive integers')
        self.scope, self.repository, self.records = scope, repository, records
        self.block_tokens, self.epoch = block_tokens, epoch
        self.max_spans, self.max_blocks = max_spans, max_blocks
        self.spans, self.owners = {}, {}
        self.witness_keys = set()
        self._revision = 0
        self._next_revision = next_revision
        self.closed = self.resetting = False

    def _span(self, span_id, epoch):
        if self.closed or self.resetting or epoch != self.epoch:
            return None
        return self.spans.get(span_id)

    def register_span(self, span_id, begin, end, *, kind, epoch):
        """Register only complete archived blocks wholly inside [begin, end).

        Mixed boundary blocks and unpublished tails are intentionally retained.
        Registration is explicit and can scan the archive; never call per token.
        """
        if self.closed or self.resetting or epoch != self.epoch:
            return False
        if kind not in self.KINDS:
            raise ValueError('Unknown semantic span kind')
        if type(begin) is not int or type(end) is not int or not 0 <= begin < end:
            raise ValueError('Span must have nonnegative ordered token positions')
        if not isinstance(span_id, str) or not span_id or span_id in self.spans:
            raise ValueError('Span id must be a new nonempty string')
        if len(self.spans) >= self.max_spans:
            return False
        handles = {_key(r.handle): r.handle for r in self.records.values()
                   if r.archived and r.count == self.block_tokens and
                   begin <= r.history_begin and r.history_begin+r.count <= end}
        if not handles or len(self.owners)+len(handles) > self.max_blocks:
            return False
        if any(key in self.owners or key in self.witness_keys for key in handles):
            raise ValueError('A block cannot belong to overlapping semantic spans')
        # Reject stale metadata, without treating a read lease as GC protection.
        hold = self.scope.protect_selection(list(handles.values()))
        if hold is None:
            return False
        span = _Span(handles, kind)
        self._revision = self._revision+1 if self._next_revision is None else self._next_revision()
        span.revision = self._revision
        self.spans[span_id] = span
        self.owners.update((key, span_id) for key in handles)
        del hold
        return True

    def _begin_change(self, span):
        if span.update_hold is not None:
            return True
        if not span.handles:
            return False
        span.update_hold = self.scope.protect_selection(list(span.handles.values()))
        return span.update_hold is not None

    def _changed(self, span):
        self._revision = self._revision+1 if self._next_revision is None else self._next_revision()
        span.revision = self._revision
        span.pending = set(span.handles)

    def complete_span(self, span_id, result_records, *, epoch):
        """Retain explicit result KV before publishing completion eligibility."""
        span = self._span(span_id, epoch)
        if span is None or span.complete:
            return False
        results = list(result_records)
        if not results or any(not r.archived or r.count != self.block_tokens for r in results):
            raise ValueError('Completion requires sealed, archived result witnesses')
        handles = {_key(r.handle): r.handle for r in results}
        if any(self.records.get(_key(r.handle)) is not r for r in results):
            raise ValueError('Result witnesses must belong to this authority')
        if any(key in self.owners for key in handles):
            raise ValueError('Result witnesses cannot be reclaimed with their source')
        if len(self.witness_keys | handles.keys()) > self.max_blocks:
            return False
        witness = self.scope.protect_selection(list(handles.values()))
        if witness is None:
            return False
        if not self._begin_change(span):
            return False
        span.result_hold, span.complete = witness, True
        span.model_score = None
        self.witness_keys.update(handles)
        self._changed(span)
        return True

    def observe(self, span_id, *, epoch, observation_id, importance, cold):
        """Record one valid observation, using monotone ids to reject duplicates.

        The caller defines/calibrates the importance estimator and cold test.
        An unobserved block must never be supplied as a zero-importance sample.
        """
        importance = _score(importance, 'importance')
        if type(observation_id) is not int or observation_id < 0 or type(cold) is not bool:
            raise ValueError('Observation id must be nonnegative and cold must be boolean')
        span = self._span(span_id, epoch)
        if span is None or observation_id <= span.observation_id or not self._begin_change(span):
            return False
        span.observation_id = observation_id
        span.observations += 1
        span.cold_observations = span.cold_observations+1 if cold else 0
        span.importance = importance
        span.model_score = None
        self._changed(span)
        return True

    def model_signal(self, span_id, *, epoch, based_on_revision, decision_id, reclaim_score):
        """Accept an optional advisory model score; it cannot bypass other gates."""
        reclaim_score = _score(reclaim_score, 'reclaim_score')
        if type(decision_id) is not int or decision_id < 0:
            raise ValueError('Model decision id must be nonnegative')
        span = self._span(span_id, epoch)
        if (span is None or type(based_on_revision) is not int or
                based_on_revision != span.revision or decision_id <= span.model_id or
                not self._begin_change(span)):
            return False
        span.model_id, span.model_score = decision_id, reclaim_score
        self._changed(span)
        return True

    def revision(self, span_id, *, epoch):
        """Capture the evidence revision for an asynchronous model signal."""
        span = self._span(span_id, epoch)
        return None if span is None else span.revision

    def flush(self, max_blocks=32):
        """Publish bounded metadata without waiting for native policy contention.

        A partially published update keeps a hard guard until every block sees
        the revision. Retrying at a later refresh boundary is safe and bounded.
        """
        if type(max_blocks) is not int or max_blocks <= 0:
            raise ValueError('Publication budget must be a positive integer')
        attempted = 0
        for span in self.spans.values():
            for key in tuple(span.pending):
                if attempted == max_blocks:
                    return False
                attempted += 1
                if self.scope.publish_evidence(span.handles[key], span.revision,
                        span.complete and not self.resetting, span.observations,
                        span.cold_observations, span.importance, span.model_score):
                    span.pending.remove(key)
            if not span.pending:
                span.update_hold = None
        return not any(span.pending for span in self.spans.values())

    def retired(self, record):
        """Forget retired source handles; keep their result witnesses alive."""
        key = _key(record.handle)
        owner = self.owners.pop(key, None)
        if owner is None:
            return
        span = self.spans[owner]
        span.handles.pop(key, None)
        span.pending.discard(key)
        if not span.pending:
            span.update_hold = None

    def reset(self):
        """Revoke live evidence before branch rewind, yielding on contention."""
        if self.closed:
            return True
        self.resetting = True
        prepared = True
        for span in self.spans.values():
            if not span.complete and not span.pending:
                continue
            # Explicit rewind/retirement may already have removed generations.
            for key, handle in tuple(span.handles.items()):
                if not self.repository.pin(handle):
                    span.handles.pop(key)
                    span.pending.discard(key)
                    self.owners.pop(key, None)
            if not span.handles:
                span.pending.clear()
                span.update_hold = None
                span.complete = False
            elif span.complete:
                if self._begin_change(span):
                    span.complete = False
                    self._changed(span)
                else:
                    prepared = False
        if not self.flush() or not prepared:
            return False
        self._release()
        return True

    def _release(self):
        """Authority close only; live callers must use nonblocking reset()."""
        self.closed = True
        self.spans.clear()
        self.owners.clear()
        self.witness_keys.clear()
