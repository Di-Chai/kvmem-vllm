"""Worker controls for exact page swaps with a retained native request slot.

The initial path synchronizes CUDA before capture for correctness. It never
re-runs selection, changes a working window, removes the request, or resamples
tokens. The scheduler must keep paused requests out of public batches and hold
source/destination pages until this module's completion is acknowledged.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

import torch

from .request_checkpoint import NativeRequestCheckpointAdapter
from .resource_ledger import ResourceLedger
from .session_swap import SessionSwapIdentity
from . import settings as _settings


@dataclass(frozen=True)
class SwapCommand:
    request_id: str
    command_id: int
    operation_id: int
    action: str
    history_end: int
    block_ids: tuple[tuple[int, ...], ...] = ()
    checkpoint_bytes: int = 0
    session_identity: SessionSwapIdentity | None = None

    def __post_init__(self):
        if self.session_identity is not None:
            if not isinstance(self.session_identity, SessionSwapIdentity):
                raise ValueError('Swap session identity must be immutable and generation qualified')
        if (not isinstance(self.request_id, str) or not self.request_id or
                any(type(value) is not int or value <= 0 for value in
                    (self.command_id, self.operation_id, self.history_end)) or
                self.action not in ('capture', 'restore', 'drop') or
                type(self.checkpoint_bytes) is not int or self.checkpoint_bytes < 0):
            raise ValueError('Invalid native request swap command')
        if self.action == 'capture' and not self.checkpoint_bytes:
            raise ValueError('Swap capture requires admitted checkpoint bytes')
        if self.action != 'drop' and (not self.block_ids or any(
                not row or any(type(page) is not int or page <= 0 for page in row)
                or len(set(row)) != len(row) for row in self.block_ids)):
            raise ValueError('Swap command requires complete non-null native pages')


class _Ready:
    @staticmethod
    def query():
        return True


@dataclass
class _HeldCheckpoint:
    command: SwapCommand
    index: int
    generation: int
    grant: object
    snapshot: object = None
    phase: str = 'capturing'
    restore: object = None
    command_ids: set = field(default_factory=set)
    restored: bool = False
    stored: object = None
    capture_started: bool = False
    session_started: bool = False
    layerwise: object = None
    layerwise_capture_seconds: float = 0.0
    layerwise_check_seconds: float = 0.0
    layerwise_deep_check_seconds: float = 0.0
    layerwise_deep_checks: int = 0


class _SwapCompletion:
    def __init__(self, worker, command, held):
        self.worker, self.command, self.held = worker, command, held
        self.done = False

    def query(self):
        if self.done:
            return True
        self.done = self.worker._poll(self.command, self.held)
        return self.done


class RequestSwapWorker:
    def __init__(self, runner, max_checkpoint_bytes, *, adapter=None,
                 checkpoint_ssd_bytes=0, checkpoint_directory=None):
        if type(max_checkpoint_bytes) is not int or max_checkpoint_bytes <= 0:
            raise ValueError('Exact swaps require a positive shared checkpoint byte cap')
        self.runner = runner
        self.verify_request_swap = _settings.get('KVMEM_VERIFY_REQUEST_SWAP') == '1'
        self.verification_successes = self.verification_failures = 0
        self.verification_path = None
        if self.verify_request_swap:
            output = getattr(runner, '_kvmem_probe_output', None)
            if output is None:
                raise ValueError('Request swap byte verification requires a diagnostic output directory')
            self.verification_path = Path(output)/'request_swap_verified.jsonl'
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.adapter = adapter if adapter is not None else NativeRequestCheckpointAdapter(runner)
        if type(checkpoint_ssd_bytes) is not int or checkpoint_ssd_bytes < 0:
            raise ValueError('Checkpoint SSD quota must be a nonnegative integer')
        self.store = None
        if checkpoint_ssd_bytes:
            if checkpoint_directory is None:
                raise ValueError('SSD checkpoints require a process-local directory')
            from .checkpoint_storage import CheckpointStore
            self.store = CheckpointStore(checkpoint_directory, max_checkpoint_bytes, checkpoint_ssd_bytes)
        self.ledger = ResourceLedger({'checkpoint_bytes': checkpoint_ssd_bytes or max_checkpoint_bytes})
        self.checkpoints = {}
        self.capture_count = self.restore_count = self.drop_count = 0
        self.capture_waits = 0
        self.layerwise_capture_seconds = self.layerwise_check_seconds = 0.0
        self.layerwise_deep_check_seconds = 0.0
        self.layerwise_deep_checks = 0

    def _session(self, method, command):
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if sessions is None:
            if command.session_identity is not None:
                raise ValueError('Session swap has no retained session worker')
            return True
        return getattr(sessions, method)(command.request_id, command.session_identity)

    def validate_public_step(self, output):
        """Fail before native cleanup or forward can touch a suspended slot."""
        control = getattr(output, 'kv_connector_metadata', None)
        held = set(self.checkpoints)
        held.update(command.request_id for command in getattr(control, 'commands', ())
                    if isinstance(command, SwapCommand))
        changed = set(output.num_scheduled_tokens) | set(output.finished_req_ids)
        changed.update(getattr(output, 'preempted_req_ids', ()) or ())
        changed.update(getattr(getattr(output, 'scheduled_cached_reqs', None), 'req_ids', ()))
        changed.update(getattr(control, 'block_table_updates', {}) or {})
        changed.update(request.req_id for request in getattr(output, 'scheduled_new_reqs', ()))
        if held & changed:
            raise RuntimeError('Public update or native retirement precedes request swap drop acknowledgement')

    def _identity(self, command, held=None, *, check_history=True):
        runner = self.runner
        index = runner.req_states.req_id_to_index.get(command.request_id)
        generation = runner._kvmem_request_generations.get(command.request_id)
        if (index is None or generation is None or
                (check_history and int(runner.req_states.num_computed_tokens_np[index]) != command.history_end) or
                (held is not None and (held.index, held.generation) != (index, generation))):
            raise ValueError('Swap request slot, generation or accepted history changed')
        return index, generation

    def execute(self, command):
        if not isinstance(command, SwapCommand):
            raise TypeError('Unexpected native swap command')
        rid = command.request_id
        held = self.checkpoints.get(rid)
        if command.action == 'capture':
            if held is not None:
                raise ValueError('Request already owns a swap checkpoint')
            if command.checkpoint_bytes > self.max_checkpoint_bytes:
                raise BufferError('One complete swap checkpoint exceeds the CPU staging cap')
            index, generation = self._identity(command, check_history=False)
            grant = self.ledger.prepare(rid, generation, {'checkpoint_bytes': command.checkpoint_bytes})
            if grant is None:
                raise BufferError('Shared swap checkpoint byte capacity exhausted')
            held = _HeldCheckpoint(command, index, generation, grant)
            held.command_ids.add(command.command_id)
            self.checkpoints[rid] = held
        else:
            if (held is None or held.command.operation_id != command.operation_id or
                    held.command.history_end != command.history_end or
                    held.command.session_identity != command.session_identity):
                raise ValueError('Stale or foreign native swap checkpoint operation')
            if held.phase != 'paused':
                raise ValueError('Swap restore/drop requires a completed checkpoint')
            if command.command_id in held.command_ids:
                raise ValueError('Swap transport command ID was reused')
            if command.action == 'restore':
                if held.restored:
                    raise ValueError('Swap checkpoint was already restored')
                self._identity(command, held)
                self._session('validate_swap', command)
                self._validate_layerwise(held, deep=True)
                held.command_ids.add(command.command_id)
                held.phase = 'restoring'
                # Complete row installation is independent of public forward.
                # The native journal observes this exact overwrite and commit.
                if self.store is None:
                    self._dispatch_restore(command, held, held.snapshot, layerwise_validated=True)
            else:
                held.command_ids.add(command.command_id)
                held.phase = 'dropping'
        return _SwapCompletion(self, command, held)

    def _dispatch_restore(self, command, held, snapshot, *, layerwise_validated=False):
        self._identity(command, held)
        self._session('validate_swap', command)
        if not layerwise_validated:
            self._validate_layerwise(held, deep=True)
        self.runner.block_tables.append_block_ids(held.index, command.block_ids, overwrite=True)
        self.runner.block_tables.apply_staged_writes()
        held.restore = self.adapter.restore(snapshot, command.request_id,
                                            command.operation_id, command.block_ids)

    def _validate_layerwise(self, held, *, deep=False):
        if held.layerwise is None:
            return
        start = time.perf_counter()
        try:
            held.layerwise.validate(self.runner, deep=deep)
        finally:
            elapsed = time.perf_counter() - start
            held.layerwise_check_seconds += elapsed
            self.layerwise_check_seconds += elapsed
            if deep:
                held.layerwise_deep_check_seconds += elapsed
                held.layerwise_deep_checks += 1
                self.layerwise_deep_check_seconds += elapsed
                self.layerwise_deep_checks += 1

    def _verify_restore(self, command, held):
        snapshot = held.stored.snapshot if self.store is not None else held.snapshot
        try:
            report = self.adapter.verify_restore(snapshot, command.request_id,
                                                  command.operation_id, command.block_ids)
            report = dict(report, at=time.perf_counter(), request_id=command.request_id,
                generation=held.generation, operation_id=command.operation_id,
                command_id=command.command_id, history_end=command.history_end,
                source_tier=('ssd' if held.stored is not None and held.stored.disk is not None else 'cpu'))
            if command.session_identity is not None:
                report['session_identity'] = dict(namespace=command.session_identity.namespace,
                                                generation=command.session_identity.generation)
            self.verification_path.parent.mkdir(parents=True, exist_ok=True)
            with self.verification_path.open('a') as output:
                output.write(json.dumps(report, separators=(',', ':'))+'\n')
            self.verification_successes += 1
        except BaseException:
            self.verification_failures += 1
            raise
        finally:
            snapshot = None

    def _poll(self, command, held):
        rid = command.request_id
        if self.checkpoints.get(rid) is not held:
            raise ValueError('Swap completion belongs to a retired checkpoint')
        if command.action == 'capture':
            if held.phase != 'capturing':
                raise ValueError('Unexpected capture completion phase')
            self._identity(command, held, check_history=held.capture_started)
            if not held.capture_started:
                from .layerwise_swap import ready as layerwise_ready
                # First exclude private writers. Publication/APPENDING may
                # need the existing drain below to make progress.
                if not layerwise_ready(self.runner, rid, accepted=False):
                    self.capture_waits += 1
                    return False
                runtime = self.runner._kvmem_capture_runtime
                if not runtime.drain_requests([rid], wait=True):
                    # Scheduler confirmation is never replaced by a stream
                    # synchronization or a guessed accepted token count.
                    self.capture_waits += 1
                    return False
                if not layerwise_ready(self.runner, rid):
                    self.capture_waits += 1
                    return False
                if not self._session('swap_ready', command):
                    self.capture_waits += 1
                    return False
                if self.store is not None and held.stored is None:
                    held.stored = self.store.reserve_capture(
                        (rid, held.generation, command.operation_id), command.checkpoint_bytes)
                    if held.stored is None:
                        self.capture_waits += 1
                        return False
                if torch.device(self.runner.device).type == 'cuda':
                    torch.cuda.synchronize(self.runner.device)
                authority = runtime.authorities.get(rid)
                computed = self.runner.req_states.num_computed_tokens.gpu[held.index]
                if (authority is None or authority.history_end != command.history_end or
                        int(computed.item()) != command.history_end):
                    raise ValueError('Swap capture differs from confirmed authority or native GPU history')
                # Native postprocess advances the GPU counter; its independent
                # NumPy hint changes only when a cached request is scheduled.
                # Pausing skips that scheduled update, so reconcile it using
                # the two exact accepted-boundary witnesses above.
                self.runner.req_states.num_computed_tokens_np[held.index] = command.history_end
                self.ledger.dispatch(held.grant)
                snapshot = None
                try:
                    self._session('begin_swap', command)
                    held.session_started = True
                    from .layerwise_swap import capture as capture_layerwise
                    started = time.perf_counter()
                    held.layerwise = capture_layerwise(self.runner, rid)
                    held.layerwise_capture_seconds = time.perf_counter() - started
                    self.layerwise_capture_seconds += held.layerwise_capture_seconds
                    snapshot = self.adapter.capture(rid, command.operation_id,
                        command.block_ids, max_bytes=command.checkpoint_bytes)
                    if self.store is not None:
                        self.store.install(held.stored, snapshot)
                    else:
                        held.snapshot = snapshot
                    self.ledger.allocated(held.grant, {'checkpoint_bytes': snapshot.bytes})
                    held.capture_started = True
                except BaseException:
                    # The adapter drains partially submitted copies before
                    # raising. No checkpoint or destination survives failure.
                    if snapshot is not None:
                        completion = getattr(snapshot, 'completion', None)
                        if completion is not None:
                            completion.synchronize()
                        elif not snapshot.query():
                            raise RuntimeError('Failed checkpoint publication still owns unfinished copies')
                        if held.stored is not None and held.stored.state != 'reserved':
                            if not self.store.drop(held.stored):
                                raise RuntimeError('Failed checkpoint publication retained I/O')
                            held.stored = None
                        self.adapter.release(rid)
                    if held.session_started:
                        self._session('end_swap', command)
                        held.session_started = False
                    if held.stored is not None and held.stored.state == 'reserved':
                        self.store.abort_capture(held.stored)
                    self.ledger.settle(held.grant, _Ready(), retained={})
                    del self.checkpoints[rid]
                    raise
                finally:
                    snapshot = None
            if self.store is not None and not self.store.capture_ready(held.stored):
                return False
            self._session('validate_swap', command)
            self._validate_layerwise(held)
            if not self.ledger.settle(held.grant, _Ready() if self.store is not None else held.snapshot):
                return False
            self._validate_layerwise(held, deep=True)
            held.phase = 'paused'
            self.capture_count += 1
            return True
        if command.action == 'restore':
            if held.phase != 'restoring':
                raise ValueError('Unexpected restore completion phase')
            self._identity(command, held)
            self._session('validate_swap', command)
            self._validate_layerwise(held)
            if held.restore is None and self.store is not None:
                snapshot = self.store.acquire_restore(held.stored)
                if snapshot is None:
                    return False
                self._dispatch_restore(command, held, snapshot)
                snapshot = None
            if not held.restore.query():
                return False
            self._validate_layerwise(held, deep=True)
            if self.verify_request_swap:
                # Keep both the source checkpoint and restore/page ownership
                # on mismatch or diagnostic write failure; never publish ACK.
                self._verify_restore(command, held)
            held.restore = None
            if self.store is not None:
                self.store.restore_complete(held.stored)
            held.phase = 'paused'
            held.restored = True
            self.restore_count += 1
            # Keep the source snapshot until an explicit drop, including on
            # cancellation racing the restore completion notification.
            return True
        if held.phase != 'dropping':
            return False
        self._session('validate_swap', command)
        if self.store is not None:
            if not self.store.drop(held.stored):
                return False
        elif not held.snapshot.query():
            return False
        # Drop Python/native owners before returning their live byte credit.
        self.adapter.release(rid)
        self._session('end_swap', command)
        held.snapshot = None
        # Cancellation may retire logical state after the source-copy fence;
        # dropping an exact checkpoint never restores or reselects that state.
        held.layerwise = None
        if not self.ledger.release(held.grant, _Ready()):
            raise RuntimeError('A drained checkpoint failed to release its byte credit')
        del self.checkpoints[rid]
        self.drop_count += 1
        return True

    def snapshot(self):
        return dict(captures=self.capture_count, restores=self.restore_count, drops=self.drop_count,
                    capture_waits=self.capture_waits, resources=self.ledger.snapshot(),
                    layerwise_capture_seconds=self.layerwise_capture_seconds,
                    layerwise_check_seconds=self.layerwise_check_seconds,
                    layerwise_deep_check_seconds=self.layerwise_deep_check_seconds,
                    layerwise_deep_checks=self.layerwise_deep_checks,
                    verification_enabled=self.verify_request_swap,
                    verification_successes=self.verification_successes,
                    verification_failures=self.verification_failures,
                    storage=self.store.snapshot() if self.store is not None else None,
                    requests={rid: dict(operation_id=held.command.operation_id,
                                        phase=held.phase, history_end=held.command.history_end,
                                        layerwise_capture_seconds=held.layerwise_capture_seconds,
                                        layerwise_check_seconds=held.layerwise_check_seconds,
                                        layerwise_deep_check_seconds=held.layerwise_deep_check_seconds,
                                        layerwise_deep_checks=held.layerwise_deep_checks,
                                        bytes=(held.stored.size if held.stored is not None else
                                               held.snapshot.bytes if held.snapshot is not None else 0))
                              for rid, held in self.checkpoints.items()})

    def close(self):
        if self.checkpoints:
            return False
        return self.store.close() if self.store is not None else True
