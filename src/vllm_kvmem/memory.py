"""Per-request host transactions. No model execution or GPU allocation here.

The engine must retain a RequestMemory until close() succeeds. Completion fences
must include rejected-page cleanup as well as DMA/model execution. This interface
is deliberately nonblocking: an unready request yields to the native scheduler.
"""
from enum import Enum, auto
from dataclasses import dataclass
import math
from typing import Protocol

from . import _native
from .config import KVMemConfig


class Fence(Protocol):
    def query(self) -> bool: ...


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

    @property
    def position(self):
        return self.prefix.position

    @property
    def window_tokens(self):
        return sum(record[3] for record in self.records)


class RequestMemory:
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

    def select(self, scores: dict[int, float], mandatory_slots: tuple[int, ...] = ()):
        """Scores are for this request's current snapshot; call only when READY.

        Async scorers must retain a storage read and use generation-qualified
        native handles before entering this synchronous convenience method.
        """
        self._require(Phase.READY)
        if any(not math.isfinite(score) for score in scores.values()):
            raise ValueError("Retrieval scores must be finite")
        handles = [self._store.handle(slot) for slot in scores]
        self._store.scores(handles, list(scores.values()))
        selected = self._store.pick(list(mandatory_slots))
        plan = self._store.select([self._store.handle(slot) for slot in selected])
        self._plan = plan  # Retains previous-window leases until assembly completes.
        self.phase = Phase.ASSEMBLING
        self._submitted = False
        return plan

    def commit_selection(self) -> bool:
        self._require(Phase.ASSEMBLING)
        if not self._completed():
            return False
        for slot in self._plan.stage_out:
            self._store.working_resident(self._store.handle(slot), False)
        for remap in self._plan.remaps:
            handle = self._store.handle(remap.block_id)
            self._store.working_resident(handle, True)
            self._store.set_baked(handle.slot, remap.to_base, handle.generation)
        self._plan = None
        self._reset()
        return True

    def read(self, begin: int, count: int):
        self._require(Phase.READY)
        return self._store.read(begin, count)

    def checkpoint(self):
        """Pin the current working prefix; model/draft tensors are separate.

        This host handle is not an executable session checkpoint. Serving must
        also snapshot exact model state, tokens, position maps and mutable tails.
        """
        self._require(Phase.READY)
        return self._store.checkpoint(self.history_end)

    def checkpoint_valid(self, checkpoint):
        return self._store is not None and self._store.checkpoint_valid(checkpoint)

    def checkpoint_window(self, *, allow_partial=False):
        """Capture a sealed prompt boundary including its exact selected frame."""
        self._require(Phase.READY)
        if self.history_end == 0 or (self.history_end % self.config.block_tokens and not allow_partial):
            raise ValueError('Window checkpoint requires a nonempty sealed prefix')
        prefix = self.checkpoint()
        checkpoint = WindowCheckpoint(self.request_id, prefix,
                                      tuple(self._store.working_records()))
        self._validate_window_checkpoint(checkpoint)
        return checkpoint

    def _validate_window_checkpoint(self, checkpoint):
        if (not isinstance(checkpoint, WindowCheckpoint) or checkpoint.request_id != self.request_id
                or not self.checkpoint_valid(checkpoint.prefix)):
            raise ValueError('Stale or foreign window checkpoint')
        dense, logical_end, seen = 0, 0, set()
        for slot, generation, begin, count, position in checkpoint.records:
            if (position != dense or begin < logical_end or not 0 < count <= self.config.block_tokens
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

    def restore_window(self, checkpoint):
        """Prepare complete reassembly into fresh native pages after a rewind.

        Both K and V must reload: a matching dense position does not mean that
        newly allocated pages contain the checkpoint's bytes. The caller holds
        GPU pages and fences assembly before commit_selection and native resume.
        """
        self._require(Phase.READY)
        self._validate_window_checkpoint(checkpoint)
        if self.history_end != checkpoint.position:
            raise ValueError('Rewind authority before restoring its checkpoint window')
        saved = [_native.Handle(slot, generation) for slot, generation, *_ in checkpoint.records]
        current = [_native.Handle(slot, generation) for slot, generation, *_ in self._store.working_records()]
        for handle in current + saved:
            self._store.working_resident(handle, False)
        self._plan = self._store.select(saved, True)
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

    def close(self) -> bool:
        if self.phase is Phase.CLOSED:
            return True
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
        for slot in range(self.config.repository_blocks):
            lease = self.repository.pin_slot(slot)
            if lease:
                self.repository.retire(lease.handle)
        self._fence = None
        self.phase = Phase.CLOSED
        return True
