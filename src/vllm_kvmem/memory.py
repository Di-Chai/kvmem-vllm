"""Per-request host transactions. No model execution or GPU allocation here.

The engine must retain a RequestMemory until close() succeeds. Completion fences
must include rejected-page cleanup as well as DMA/model execution. This interface
is deliberately nonblocking: an unready request yields to the native scheduler.
"""
from enum import Enum, auto
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import os
import time
from typing import Protocol

from . import _native
from .config import KVMemConfig



def _native_publish_selection():
    """KVMEM_NATIVE_PUBLISH_SELECTION=1: RequestMemory._publish_selection's mapping
    loop runs as one native call (same store calls and order)."""
    value = os.getenv('KVMEM_NATIVE_PUBLISH_SELECTION', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_PUBLISH_SELECTION must be 0 or 1')
    if value == '0':
        return None
    function = getattr(_native, 'publish_selection', None)
    if function is None:
        raise RuntimeError('KVMEM_NATIVE_PUBLISH_SELECTION requires a rebuilt native extension')
    return function

class Fence(Protocol):
    def query(self) -> bool: ...


def _native_lookups():
    return os.getenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '0') == '1'


class Phase(Enum):
    READY = auto()
    APPENDING = auto()
    ASSEMBLING = auto()
    REWINDING = auto()
    FAILED = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class WindowCheckpoint:
    """Original dense frame and pinned authority; model state is separate."""
    request_id: str
    prefix: object
    # slot, generation, logical begin, token count, original dense position
    records: tuple[tuple[int, int, int, int, int], ...]
    # A rebased partial checkpoint may describe a different frame from the
    # current working set. Retain its exact generations explicitly.
    leases: tuple = field(default=(), repr=False, compare=False)
    policy_hold: object = field(default=None, repr=False, compare=False)

    @property
    def position(self):
        return self.prefix.position

    @property
    def window_tokens(self):
        return sum(record[3] for record in self.records)


class RequestMemory:
    # Strategy layer: None runs the native store picker unchanged. A picker from
    # vllm_kvmem.strategy replaces only the scores-to-blocks step below.
    picker = None
    # Optional slot -> (history_begin, count), for pickers that use positions.
    block_info = None

    def __init__(self, request_id: str, config: KVMemConfig):
        if not request_id:
            raise ValueError("request_id must be nonempty")
        self.request_id = request_id
        self.config = config
        self.repository = _native.Repository(config.repository_blocks, config.block_tokens)
        self._store = _native.Store(config.native(), self.repository)
        self.phase = Phase.READY
        self._append = None
        self._plan = None
        self._rewind = None
        self._fence = None
        self._submitted = False
        self._action_scores = {}
        self._layerwise_materialization = None

    def _require(self, phase: Phase) -> None:
        if self.phase is not phase:
            raise RuntimeError(f"{self.request_id}: expected {phase.name}, got {self.phase.name}")

    @property
    def history_end(self) -> int:
        return self._store.history_end

    def begin_append(self, count: int, local_rope_begin: int):
        self._require(Phase.READY)
        if count <= 0 or local_rope_begin < 0:
            raise ValueError("Invalid append extent")
        if local_rope_begin + count > self.config.select_budget + self.config.gen_budget:
            raise ValueError("Append exceeds K + G; refresh or yield before execution")
        append = self._store.prepare_append(count, local_rope_begin)
        if append is None:
            return None  # Backpressure, never trigger foreground GC.
        self._append = append
        self.phase = Phase.APPENDING
        self._submitted = False
        return append.spans

    def discarded_pages(self, accepted: int, page_tokens: int):
        self._require(Phase.APPENDING)
        return self._append.discarded_page_ranges(accepted, page_tokens)

    def attach_completion(self, fence: Fence) -> None:
        if self.phase not in (Phase.APPENDING, Phase.ASSEMBLING, Phase.REWINDING) or self._submitted:
            raise RuntimeError("No unsubmitted transaction")
        self._fence = fence
        self._submitted = True

    def fail_transaction(self, fence: Fence) -> None:
        """Retain partially submitted work through its cleanup fence; never commit."""
        if self.phase not in (Phase.APPENDING, Phase.ASSEMBLING, Phase.REWINDING) or self._submitted:
            raise RuntimeError("No unsubmitted transaction to fail")
        self._fence = fence
        self._submitted = True
        self.phase = Phase.FAILED

    def _completed(self) -> bool:
        if not self._submitted:
            raise RuntimeError("No completion fence; refusing to publish/recycle unwritten data")
        return self._fence.query()

    def commit_append(self, accepted: int) -> bool:
        self._require(Phase.APPENDING)
        if not 0 <= accepted <= self._append.end - self._append.begin:
            raise ValueError("Accepted prefix outside prepared extent")
        if not self._completed():
            return False
        self._append.commit(accepted)
        self._append = None
        self._reset()
        return True

    def set_action_scores(self, scores):
        self._require(Phase.READY)
        self._store.set_action_scores(scores)
        self._action_scores = dict(scores)

    def select(self, scores: dict[int, float], mandatory_slots: tuple[int, ...] = (), *, action=True,
               prefill_scores=None, pressure_scores=None, max_tokens=None):
        """Scores are for this request's current snapshot; call only when READY.

        Async scorers must retain a storage read and use generation-qualified
        native handles before entering this synchronous convenience method.
        """
        self.first_pick = None
        for _ in range(self.config.repository_blocks + 1):
            selected = self.preview_selection(scores, mandatory_slots, action=action,
                prefill_scores=prefill_scores, pressure_scores=pressure_scores, max_tokens=max_tokens)
            if getattr(self, 'tp_records', None) is not None:
                from .tensor_parallel import selected_slots
                selected = selected_slots(self.tp_records, selected)
            if self.first_pick is None:
                # The pure (const) native pick for these inputs; a report may reuse it.
                self.first_pick = selected
            try:
                plan = self._admit_slots(selected, use_native_lookup=_native_lookups())
                break
            except _native.HistoryAdmissionRetry:
                # No window mutation or GPU work preceded this failure. Each
                # failed admission lost at least one candidate to retirement;
                # pick again from the remaining live generations, without
                # waiting for GC or granting old reads execution permission.
                continue
        else:
            raise RuntimeError('Historical execution admission did not converge')
        self._plan = plan  # Retains previous-window leases until assembly completes.
        self.phase = Phase.ASSEMBLING
        self._submitted = False
        return plan

    @staticmethod
    def _plan_read(plan, handle):
        """plan.admitted_read, except that under opt-in KVMEM_LAYERWISE_HISTORICAL_GC
        a record outside the native (layer-0) admitted window reads by a plain pin.

        Layerwise selection holds those records from retirement (paused
        selection, then a current-window hold over every layer), and prepared
        readers take their own GC read rights (tier.borrow_records).
        """
        try:
            return plan.admitted_read(handle)
        except ValueError:
            from .pressure_reader import layerwise_gc
            if not layerwise_gc():
                raise
            return None

    def admitted_read(self, handle):
        """Clone CPU/execution rights of this already admitted assembly only."""
        if self.phase is Phase.ASSEMBLING and self._plan is not None:
            result = self._plan_read(self._plan, handle)
            if result is not None:
                return result
        if self._layerwise_materialization is not None:
            result = self._plan_read(self._layerwise_materialization.plan, handle)
            if result is not None:
                return result
        return self.repository.pin(handle), None

    def admitted_pin_set(self, handles):
        """admitted_read_batch for the plain-pin case as one native LeaseSet, or
        None when a live plan carries a read window (callers then use the
        original per-handle path) or the native set is unavailable."""
        pin_set = getattr(self.repository, 'pin_set', None)
        if pin_set is None:
            return None
        plans = []
        if self.phase is Phase.ASSEMBLING and self._plan is not None:
            plans.append(self._plan)
        if self._layerwise_materialization is not None:
            plans.append(self._layerwise_materialization.plan)
        for plan in plans:
            probe = self._plan_read(plan, handles[0]) if handles else None
            if probe is not None:
                for owner in probe:
                    if owner is not None and hasattr(owner, 'reset'):
                        owner.reset()
                return None
        return pin_set(list(handles))

    def admitted_read_batch(self, handles):
        """[admitted_read(h) for h in handles] with one call for the common case.

        When no live plan carries a repository read window, every handle takes
        the plain repository pin, exactly as admitted_read would.
        """
        plans = []
        if self.phase is Phase.ASSEMBLING and self._plan is not None:
            plans.append(self._plan)
        if self._layerwise_materialization is not None:
            plans.append(self._layerwise_materialization.plan)
        for plan in plans:
            probe = self._plan_read(plan, handles[0]) if handles else None
            if probe is not None:
                for owner in probe:
                    if owner is not None and hasattr(owner, 'reset'):
                        owner.reset()
                return [self.admitted_read(handle) for handle in handles]
        many = getattr(self.repository, 'pin_many', None)
        if many is not None:
            return [(lease, None) for lease in many(list(handles))]
        pin = self.repository.pin
        return [(pin(handle), None) for handle in handles]

    def select_groups(self, groups, group_scores, mandatory_slots=(), *, max_tokens=None,
                      expected_slots=None):
        """Select complete logical groups through ordinary assembly admission.

        The caller must retain its generation-qualified history snapshot while
        choosing groups. Retirement may make a whole group unavailable between
        preview and admission; retry the native picker, never admit its surviving
        members individually or reinterpret a recycled slot as the old history.
        If evidence was exported from an earlier preview, expected_slots binds
        admission to that selection; a changed retry fails before any assembly.
        """
        # Materialize once so a retry sees the same candidates and priorities.
        groups, group_scores, mandatory_slots = tuple(groups), tuple(group_scores), tuple(mandatory_slots)
        expected = None if expected_slots is None else frozenset(expected_slots)
        for _ in range(self.config.repository_blocks + 1):
            selected = self.preview_group_selection(groups, group_scores, mandatory_slots,
                max_tokens=max_tokens)
            if getattr(self, 'tp_records', None) is not None:
                from .tensor_parallel import selected_slots
                selected = selected_slots(self.tp_records, selected)
            try:
                plan = self._admit_slots(selected, expected=expected)
                break
            except _native.HistoryAdmissionRetry:
                # The native failure precedes mutation/assembly, just as in
                # select(). Rebuild whole groups from currently live history.
                continue
        else:
            raise RuntimeError('Historical group execution admission did not converge')
        self._plan = plan
        self.phase = Phase.ASSEMBLING
        self._submitted = False
        return plan

    def _admit_slots(self, selected, *, use_native_lookup=False, expected=None):
        def admit():
            if expected is not None and frozenset(selected) != expected:
                raise RuntimeError('Historical group selection changed after preview; refusing stale selection evidence')
            by_slots = getattr(self._store, 'select_slots', None) if use_native_lookup else None
            if by_slots is not None:
                return by_slots(list(selected))
            return self._store.select([self._store.handle(slot) for slot in selected])
        if getattr(self, 'tp_records', None) is None:
            return admit()
        from .tensor_parallel import admit_window
        return admit_window(admit)

    def preview_group_selection(self, groups, group_scores, mandatory_slots=(), *, max_tokens=None):
        """Native atomic groups within the existing block budget and keep bands.

        Spans are half-open original logical token positions, not physical slot
        ranges. Shared members cost one physical block. This does not change
        stored per-block scores, action scores, or the working-frame transaction.
        All-fit history follows the original native all-fit behavior.
        """
        self._require(Phase.READY)
        if self.picker is not None:
            raise ValueError('Group selection implements the native picker only')
        groups, group_scores = tuple(groups), tuple(group_scores)
        if len(groups) != len(group_scores):
            raise ValueError('Group spans and scores must have equal lengths')
        for span in groups:
            if (not isinstance(span, (tuple, list)) or len(span) != 2
                    or any(type(position) is not int for position in span)
                    or not 0 <= span[0] < span[1] <= self.history_end):
                raise ValueError('Group span is outside accepted logical history')
        if any(not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) for score in group_scores):
            raise ValueError('Group scores must be finite numbers')
        previous_budget = self._store.runtime_select_budget
        if max_tokens is not None:
            self._store.set_runtime_select_budget(max_tokens)
        try:
            selected = self._store.pick_groups(groups, group_scores, list(mandatory_slots))
        finally:
            if max_tokens is not None:
                self._store.set_runtime_select_budget(previous_budget)
        return tuple(selected)

    def preview_selection(self, scores, mandatory_slots=(), *, action=True,
                          prefill_scores=None, pressure_scores=None, max_tokens=None):
        """Use the original native picker without changing the working frame.

        Query graph diagnostics compare two score vectors with the same picker,
        keep bands and runtime budget. Retrieval metadata is overwritten by the
        following real selection; no KV leases or assembly transaction is begun.
        """
        self._require(Phase.READY)
        if any(not math.isfinite(score) for score in scores.values()):
            raise ValueError("Retrieval scores must be finite")
        if self.picker is not None:
            return self._strategy_pick(scores, mandatory_slots, prefill_scores, pressure_scores, max_tokens)
        handles = [self._store.handle(slot) for slot in scores]
        self._store.scores(handles, list(scores.values()))
        previous_budget = self._store.runtime_select_budget
        scoped_query_budget = max_tokens is not None and pressure_scores is None
        if scoped_query_budget:
            self._store.set_runtime_select_budget(max_tokens)
        try:
            if pressure_scores is not None:
                selected = self._store.pick_pressure(list(mandatory_slots), list(pressure_scores), max_tokens or 0)
            elif prefill_scores:
                selected = self._store.pick_with_prefill(list(mandatory_slots), prefill_scores)
            elif not action and self._action_scores:
                self._store.set_action_scores({})
                try:
                    selected = self._store.pick(list(mandatory_slots))
                finally:
                    self._store.set_action_scores(self._action_scores)
            else:
                selected = self._store.pick(list(mandatory_slots))
        finally:
            if scoped_query_budget:
                self._store.set_runtime_select_budget(previous_budget)
        return tuple(selected)

    def _strategy_pick(self, scores, mandatory_slots, prefill_scores, pressure_scores, max_tokens):
        """preview_selection through a strategy picker, checked against the contract."""
        from .strategy import PickView, check_pick
        if prefill_scores:
            raise ValueError('Prefill-fused picks implement the native picker only')
        view = PickView('pressure' if pressure_scores is not None else 'query', scores,
                        tuple(pressure_scores) if pressure_scores is not None else (),
                        tuple(mandatory_slots),
                        max_tokens if max_tokens is not None else self._store.runtime_select_budget,
                        self.block_info, self.config.block_tokens)
        return check_pick(view, self.picker.pick(view))

    def commit_selection(self) -> bool:
        self._require(Phase.ASSEMBLING)
        if not self._completed():
            return False
        self._publish_selection()
        return True

    def commit_layerwise_selection(self, materialization):
        """Publish host mapping for explicitly guarded, partially ready KV.

        This is deliberately separate from commit_selection: an early first
        layer event is not a completion fence for the whole window.
        """
        from .pressure_pipeline import LayerMaterialization
        self._require(Phase.ASSEMBLING)
        if (not isinstance(materialization, LayerMaterialization) or self._submitted or
                self._layerwise_materialization is not None):
            raise RuntimeError('Invalid or overlapping layer materialization')
        materialization.validate_dispatch(self)
        self._layerwise_materialization = materialization
        # The immutable reader must see its retained plan even if the engine
        # yields the GIL during native host-map publication.
        self._publish_selection()

    def finish_layerwise_selection(self, materialization):
        if self._layerwise_materialization is not materialization or not materialization.query():
            raise RuntimeError('Layer materialization cannot release before its final fence')
        self._layerwise_materialization = None

    def _require_materialized(self):
        if self._layerwise_materialization is not None and not self._layerwise_materialization.query():
            raise RuntimeError('Whole-window access preceded layer materialization completion')

    def _publish_selection(self):
        from . import publication_trace
        traced = publication_trace.enabled()
        if traced:
            t0 = time.perf_counter(), time.thread_time()
        validator = getattr(self, '_selection_validator', None)
        if validator is not None:
            validator()
        if traced:
            t1 = time.perf_counter(), time.thread_time()
            stage_out, remaps = len(self._plan.stage_out), len(self._plan.remaps)
        native = None if traced else _native_publish_selection()
        if native is not None and isinstance(self._store, _native.Store) and isinstance(self._plan, _native.Plan):
            native(self._store, self._plan)   # opt-in KVMEM_NATIVE_PUBLISH_SELECTION: same calls, same order
        else:
            for slot in self._plan.stage_out:
                self._store.working_resident(self._store.handle(slot), False)
            for remap in self._plan.remaps:
                handle = self._store.handle(remap.block_id)
                self._store.working_resident(handle, True)
                self._store.set_baked(handle.slot, remap.to_base, handle.generation)
        if traced:
            t2 = time.perf_counter(), time.thread_time()
        self._plan = None
        self._selection_validator = None
        self._reset()
        if traced:
            t3 = time.perf_counter(), time.thread_time()
            publication_trace.emit('publish_selection_sections', stage_out=stage_out, remaps=remaps,
                validator=[t1[0]-t0[0], t1[1]-t0[1]], mapping=[t2[0]-t1[0], t2[1]-t1[1]],
                reset=[t3[0]-t2[0], t3[1]-t2[1]])

    def read(self, begin: int, count: int):
        self._require(Phase.READY)
        return self._store.read(begin, count)

    def checkpoint(self):
        """Pin the current working prefix; model/draft tensors are separate.

        This host handle is not an executable session checkpoint. Serving must
        also snapshot exact model state, tokens, position maps and mutable tails.
        """
        self._require(Phase.READY)
        self._require_materialized()
        return self._store.checkpoint(self.history_end)

    def checkpoint_valid(self, checkpoint):
        return self._store is not None and self._store.checkpoint_valid(checkpoint)

    def checkpoint_window(self, *, allow_partial=False, native_validation=False):
        """Capture a sealed prompt boundary including its exact selected frame."""
        self._require(Phase.READY)
        self._require_materialized()
        if self.history_end == 0 or (self.history_end % self.config.block_tokens and not allow_partial):
            raise ValueError('Window checkpoint requires a nonempty sealed prefix')
        records = tuple(self._store.working_records())
        hold = self._protect_checkpoint(records)
        prefix = self.checkpoint()
        checkpoint = WindowCheckpoint(self.request_id, prefix,
                                      records, policy_hold=hold)
        self._validate_window_checkpoint(checkpoint, native=native_validation)
        return checkpoint

    def _protect_checkpoint(self, records):
        scope = getattr(self, '_history_scope', None)
        if scope is None or not records:
            return None
        hold = scope.protect([_native.Handle(slot, generation) for slot, generation, *_ in records])
        if hold is None:
            raise BufferError('Historical checkpoint admission must retry before model capture')
        return hold

    def _validate_window_checkpoint(self, checkpoint, *, timing=None, native=False):
        with timing.stage('window_validate') if timing is not None else nullcontext():
            if (not isinstance(checkpoint, WindowCheckpoint) or checkpoint.request_id != self.request_id
                    or not self.checkpoint_valid(checkpoint.prefix)):
                raise ValueError('Stale or foreign window checkpoint')
            if type(native) is not bool:
                raise ValueError('native_validation must be boolean')
            if native:
                self._store.validate_frame(checkpoint.prefix, checkpoint.records)
                return
            dense, logical_end, seen = 0, 0, set()
            for slot, generation, begin, count, position in checkpoint.records:
                if (position != dense or begin < logical_end or begin % self.config.block_tokens
                        or not 0 < count <= self.config.block_tokens
                        or (count != self.config.block_tokens and begin + count != checkpoint.position)
                        or begin + count > checkpoint.position or (slot, generation) in seen):
                    raise ValueError('Checkpoint window is not a complete ordered dense frame')
                handle = self._store.handle(slot)
                if (handle.slot, handle.generation) != (slot, generation):
                    raise ValueError('Checkpoint window contains a recycled generation')
                read = self._store.read(begin, count)
                spans = read.spans
                if len(spans) != 1 or spans[0].handle != handle or spans[0].token_count != count:
                    raise ValueError('Checkpoint window no longer matches immutable history')
                dense += count
                logical_end = begin + count
                seen.add((slot, generation))
            if not 0 < dense <= min(checkpoint.position, self.config.select_budget + self.config.gen_budget):
                raise ValueError('Checkpoint window exceeds its original bounded extent')

    def checkpoint_frame(self, records):
        """Pin an exact CPU frame after replacing a cancelled partial tail."""
        self._require(Phase.READY)
        self._require_materialized()
        hold = self._protect_checkpoint(records)
        leases = tuple(self.repository.pin(_native.Handle(slot, generation))
                       for slot, generation, *_ in records)
        if not all(leases):
            raise ValueError('Checkpoint frame contains a retired generation')
        result = WindowCheckpoint(self.request_id,
            self._store.checkpoint(self.history_end, False), tuple(records), leases, hold)
        self._validate_window_checkpoint(result)
        return result

    def restore_window(self, checkpoint, *, timing=None, native_restore_frame=False):
        """Prepare complete reassembly into fresh native pages after a rewind.

        Both K and V must reload: a matching dense position does not mean that
        newly allocated pages contain the checkpoint's bytes. The caller holds
        GPU pages and fences assembly before commit_selection and native resume.
        The opt-in native adapter changes only host frame preparation. Request
        identity, transaction state and completion publication remain here.
        """
        self._require(Phase.READY)
        if type(native_restore_frame) is not bool:
            raise ValueError('native_restore_frame must be boolean')
        if native_restore_frame:
            if not isinstance(checkpoint, WindowCheckpoint) or checkpoint.request_id != self.request_id:
                raise ValueError('Stale or foreign window checkpoint')
            restore = getattr(self._store, 'restore_frame', None)
            if restore is None:
                raise RuntimeError('native_restore_frame requires a rebuilt native adapter; refusing silent fallback')
            with timing.stage('native_restore_frame') if timing is not None else nullcontext():
                plan = restore(checkpoint.prefix, checkpoint.records)
        else:
            if timing is None:
                self._validate_window_checkpoint(checkpoint)
            else:
                self._validate_window_checkpoint(checkpoint, timing=timing)
            if self.history_end != checkpoint.position:
                raise ValueError('Rewind authority before restoring its checkpoint window')
            with timing.stage('saved_handles') if timing is not None else nullcontext():
                saved = [_native.Handle(slot, generation) for slot, generation, *_ in checkpoint.records]
            with timing.stage('native_select') if timing is not None else nullcontext():
                # Native selection owns the residency reset: first admit the
                # saved frame, then hand existing (possibly retired) GPU leases
                # to the plan until the assembly completion fence has passed.
                plan = self._store.restore_selection(saved)
        self._plan = plan
        self.phase = Phase.ASSEMBLING
        self._submitted = False
        return self._plan

    def begin_rewind(self, checkpoint):
        """Prepare a sealed-record boundary; caller quiesces all suffix users.

        Partial-tail branching needs a new immutable archive generation and is
        deliberately refused by this first host adapter. The completion fence
        must include physical suffix cleanup and GDN/draft state restoration.
        """
        self._require(Phase.READY)
        if not self.checkpoint_valid(checkpoint):
            raise ValueError('Stale or foreign repository checkpoint')
        if checkpoint.position % self.config.block_tokens:
            raise ValueError('Rewind requires a sealed-record boundary')
        rewind = self._store.prepare_rewind(checkpoint.position)
        self._rewind = rewind
        self.phase = Phase.REWINDING
        self._submitted = False
        # Copy value fields: exposing pybind views would let callers keep the
        # transaction's suffix leases alive after commit without realizing it.
        return tuple((span.history_begin, span.storage_begin, span.token_count,
                      _native.Handle(span.handle.slot, span.handle.generation))
                     for span in rewind.discarded_spans)

    def commit_rewind(self):
        self._require(Phase.REWINDING)
        if not self._completed():
            return False
        self._rewind.commit()
        self._rewind = None
        self._reset()
        return True

    def _reset(self):
        self._fence = None
        self._submitted = False
        self.phase = Phase.READY

    def close(self, *, background_retire=False) -> bool:
        if self.phase is Phase.CLOSED:
            return True
        if self._layerwise_materialization is not None and not self._layerwise_materialization.query():
            return False
        if self._submitted and not self._fence.query():
            return False
        # Before submission, cancelling reservations is safe. After submission,
        # callers' completion fence includes GPU cleanup for all discarded pages.
        if self._append is not None:
            self._append.commit(0)
            self._append = None
        self._plan = None
        self._rewind = None
        self._store = None
        # Retirement hides cancelled request data immediately. Existing read/DMA
        # leases still prevent the resource bundle from being reclaimed.
        for slot in range(0 if background_retire else self.config.repository_blocks):
            lease = self.repository.pin_slot(slot)
            if lease:
                self.repository.retire(lease.handle)
        self._fence = None
        self.phase = Phase.CLOSED
        return True
