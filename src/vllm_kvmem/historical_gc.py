"""Authority integration for the native archive-wide history worker.

The engine consumes bounded completion descriptors. Policy, irreversible
retirement and large CPU payload destruction remain on native workers.
"""
import time
import math
from itertools import count
from threading import Event
from dataclasses import dataclass, field


# Public schemas mirror the native factory. No Python policy callbacks execute
# on the worker; adding a policy requires registering its native constructor.
_POLICY_DEFAULTS = {
    'lru': {'keep_mandatory': True},
    'statistical': {'min_observations': 3, 'min_cold_observations': 3,
                    'max_importance': 0.1, 'proactive': True,
                    'min_model_reclaim_score': -1.0},
}


def historical_record_charge(record_bytes, block_tokens, raw_element_bytes=1):
    """KV plus worst-case FP32 sum and FP16 MeanK for every target/draft layer."""
    elements, remainder = divmod(record_bytes, block_tokens*(raw_element_bytes+1))
    if remainder or not elements:
        raise ValueError('Historical record geometry cannot hold a complete index')
    return record_bytes + elements*6


@dataclass(frozen=True)
class HistoricalGcSettings:
    mode: str = 'off'
    poll_ms: int = 10
    scan_blocks: int = 256
    delete_blocks: int = 64
    ready_low: int = 0
    ready_high: int = 0
    policy: str = 'lru'
    policy_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ('off', 'shadow', 'delete'):
            raise ValueError('historical_gc.mode must be off, shadow or delete')
        if not isinstance(self.policy, str) or self.policy not in _POLICY_DEFAULTS:
            raise ValueError(f'Unknown historical_gc.policy: {self.policy!r} (supported: lru, statistical)')
        if not isinstance(self.policy_params, dict):
            raise ValueError('historical_gc.policy_params must be an object')
        defaults = _POLICY_DEFAULTS[self.policy]
        for key, value in self.policy_params.items():
            if key not in defaults:
                raise ValueError(f'Unknown historical_gc.policy_params.{self.policy} parameter: {key!r}')
            default = defaults[key]
            if type(default) is bool:
                valid = type(value) is bool
            elif type(default) is int:
                valid = type(value) is int and 1 <= value <= 2**31-1
            else:
                valid = (type(value) in (int, float) and
                         (0 <= value <= 1 or key == 'min_model_reclaim_score' and value == -1) and
                         math.isfinite(value))
            if not valid:
                raise ValueError(f'Invalid historical_gc.policy_params.{key}: {value!r}')
        # Own the configuration and expand defaults without changing the input.
        object.__setattr__(self, 'policy_params', {**defaults, **self.policy_params})
        for name in ('poll_ms', 'scan_blocks', 'delete_blocks'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65536:
                raise ValueError(f'historical_gc.{name} must be a positive bounded integer')
        if (type(self.ready_low) is not int or type(self.ready_high) is not int or
                (self.ready_low, self.ready_high) != (0, 0) and not 0 < self.ready_low < self.ready_high):
            raise ValueError('historical_gc ready watermarks must be automatic (0,0) or 0 < low < high')

    @classmethod
    def parse(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError('historical_gc must be an object or null')
        try:
            return cls(**value)
        except TypeError as error:
            raise ValueError(f'Unknown historical_gc setting: {error}') from error

    def native(self, memory, max_scopes):
        from . import _native
        config = _native.HistoricalGcConfig()
        config.mode = getattr(_native.HistoricalGcMode, self.mode.upper())
        config.policy, config.policy_params = self.policy, self.policy_params
        config.max_scopes = max_scopes
        config.scope_capacity = memory.repository_blocks
        config.poll_ms, config.scan_blocks, config.delete_blocks = self.poll_ms, self.scan_blocks, self.delete_blocks
        config.ready_low = self.ready_low or max(1, memory.repository_blocks//20)
        config.ready_high = self.ready_high or max(config.ready_low+1, memory.repository_blocks//10)
        if config.ready_high > memory.repository_blocks:
            raise ValueError('historical_gc watermarks exceed repository capacity')
        config.background = True
        return config


class HistoricalAuthorityGc:
    native_history = True

    def __init__(self, memory, archive):
        self.memory, self.archive = memory, archive
        self.scope = archive.register_history_scope(memory.request_id, memory.repository)
        memory._store.attach_history(self.scope)
        memory._history_scope = self.scope
        self.reclaimed_records = 0
        self.retired_index = {}
        self.pending_index = {}
        self.stopped = False
        self.on_retire = None
        self.evidence = None
        self._evidence_versions = count(1)

    def open_evidence(self, records, *, epoch, max_spans=64):
        """Create an explicit semantic ledger for the current accepted branch."""
        if self.archive._history_policy != 'statistical':
            raise ValueError('Semantic evidence requires historical_gc.policy=statistical')
        if self.stopped:
            raise RuntimeError('Historical GC is stopped')
        if self.evidence is not None and not self.evidence.closed:
            raise RuntimeError('An evidence ledger already owns this branch')
        from .historical_evidence import HistoricalEvidenceLedger
        self.evidence = HistoricalEvidenceLedger(self.scope, self.memory.repository,
            records, block_tokens=self.memory.config.block_tokens, epoch=epoch,
            max_spans=max_spans, max_blocks=self.memory.config.repository_blocks,
            next_revision=lambda: next(self._evidence_versions))
        return self.evidence

    def prepare_rewind(self):
        if self.evidence is not None:
            if not self.evidence.reset():
                return False
            self.evidence = None
        return True

    def _record_retired(self, record):
        if self.evidence is not None:
            self.evidence.retired(record)
        if self.on_retire is not None:
            self.on_retire(record)

    def defer_index(self, key, record):
        if record.index_owner is not None:
            self.pending_index[key] = record.index_owner.completion

    def index_released(self):
        for key, fence in tuple(self.pending_index.items()):
            if fence.query():
                del self.pending_index[key]
        return not self.pending_index

    def drain(self, records, discarded=None):
        discarded = {} if discarded is None else discarded
        if self.stopped:
            # Finalization freezes the logical live view independently of CPU
            # readers still draining. Keep their small cleanup owners until
            # the actual release event; verification must not reopen them.
            for key, record in tuple(records.items()):
                if record.archived and not self.memory.repository.pin(record.handle):
                    self.retired_index[key] = records.pop(key)
                    self._record_retired(record)
                    self.reclaimed_records += 1
        for event in self.scope.completions(64):
            key = event.handle.slot, event.handle.generation
            record = records.get(key)
            removed = discarded.get(key)
            # I/O can publish native backing before its Future is observed.
            # Publication bookkeeping and staging must complete before its
            # descriptor/index entry is consumed or its slot is reused.
            if event.cpu:
                if record is not None and not record.archived:
                    continue
                if not self.archive.retire_cpu_record(self.memory.request_id, event.handle):
                    continue
                if removed is not None:
                    self.defer_index(key, removed)
                    del discarded[key]
                elif record is not None:
                    self.defer_index(key, record)
                    del records[key]
                    self._record_retired(record)
                    self.reclaimed_records += 1
                retired = self.retired_index.pop(key, None)
                if retired is not None:
                    self.defer_index(key, retired)
                # Completion is reliable until ACK. Python snapshots can still
                # own the Record; no reusable metadata credit precedes actual
                # index storage release by the native worker.
                fence = self.pending_index.get(key)
                if fence is not None:
                    if not fence.query():
                        continue
                    del self.pending_index[key]
            if not self.scope.acknowledge(event):
                raise RuntimeError('Historical completion acknowledgement was rejected')
        # Native policy failures pause new decisions and remain visible in
        # scope stats. They do not abort decode or stop already committed work.

    def stop(self):
        """Finalization barrier, never called to supply decode capacity."""
        if self.stopped:
            return
        self.scope.pause_policy(True)
        deadline, retry = time.monotonic() + 30, Event()
        while not self.scope.policy_paused:
            if time.monotonic() >= deadline:
                raise TimeoutError('Historical policy did not acknowledge finalization pause')
            if not self.archive._history_background:
                self.archive._history_worker.advance()
            else:
                retry.wait(.001)
        self.stopped = True

    def close(self):
        if self.evidence is not None:
            self.evidence._release()
            self.evidence = None
        if not self.archive._history_background:
            self.archive._history_worker.advance()
        if not self.scope.close():
            return False
        for key, record in tuple(self.retired_index.items()):
            if not self.archive.retire_cpu_record(self.memory.request_id, record.handle):
                return False
            self.defer_index(key, record)
            del self.retired_index[key]
        return self.index_released()
