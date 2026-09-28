"""Per-request host transactions. No model execution or GPU allocation here.

The engine must retain a RequestMemory until close() succeeds. Completion fences
must include rejected-page cleanup as well as DMA/model execution. This interface
is deliberately nonblocking: an unready request yields to the native scheduler.
"""
from enum import Enum, auto
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
    FAILED = auto()
    CLOSED = auto()


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
        if self.phase not in (Phase.APPENDING, Phase.ASSEMBLING) or self._submitted:
            raise RuntimeError("No unsubmitted transaction")
        self._fence = fence
        self._submitted = True

    def fail_transaction(self, fence: Fence) -> None:
        """Retain partially submitted work through its cleanup fence; never commit."""
        if self.phase not in (Phase.APPENDING, Phase.ASSEMBLING) or self._submitted:
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
        for remap in self._plan.remaps:
            handle = self._store.handle(remap.block_id)
            self._store.set_baked(handle.slot, remap.to_base, handle.generation)
        self._plan = None
        self._reset()
        return True

    def read(self, begin: int, count: int):
        self._require(Phase.READY)
        return self._store.read(begin, count)

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
