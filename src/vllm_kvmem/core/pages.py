"""Scheduler-side leases over vLLM's allocator, never a second GPU allocator.

Only the scheduler owner thread calls these methods. Worker completion messages
must carry the exact request/lease identity. Cancellation marks a lease retiring;
it does not return pages until the worker confirms all GPU/transfer use finished.
"""
from dataclasses import dataclass
from threading import get_ident


@dataclass(frozen=True)
class PageLease:
    request_id: str
    lease_id: int
    block_ids: tuple[int, ...]
    engine_block_tokens: int
    kernel_page_tokens: int

    @property
    def kernel_pages(self):
        ratio = self.engine_block_tokens // self.kernel_page_tokens
        return tuple(block * ratio + offset for block in self.block_ids for offset in range(ratio))


class EnginePageLeases:
    def __init__(self, block_pool, engine_block_tokens: int, kernel_page_tokens: int):
        if min(engine_block_tokens, kernel_page_tokens) <= 0 or engine_block_tokens % kernel_page_tokens:
            raise ValueError("Kernel pages must divide engine allocation blocks")
        self.pool = block_pool
        self.engine_block_tokens = engine_block_tokens
        self.kernel_page_tokens = kernel_page_tokens
        self.owner_thread = get_ident()
        self._next_id = 1
        self._live = {}
        self._retiring = set()

    def _owner(self):
        if get_ident() != self.owner_thread:
            raise RuntimeError("GPU allocator may only be used by its scheduler owner")

    def reserve(self, request_id: str, tokens: int) -> PageLease | None:
        self._owner()
        if not request_id or tokens <= 0:
            raise ValueError("Invalid request reservation")
        count = (tokens + self.engine_block_tokens - 1) // self.engine_block_tokens
        if count > self.pool.get_num_free_blocks():
            return None
        blocks = self.pool.get_new_blocks(count)
        lease = PageLease(request_id, self._next_id, tuple(b.block_id for b in blocks),
                          self.engine_block_tokens, self.kernel_page_tokens)
        self._next_id += 1
        self._live[lease.lease_id] = (lease, blocks)
        return lease

    def retire(self, lease: PageLease):
        self._owner()
        entry = self._live.get(lease.lease_id)
        if entry is None or entry[0] is not lease:
            raise ValueError("Unknown or foreign page lease")
        self._retiring.add(lease.lease_id)

    def acknowledge_release(self, request_id: str, lease_id: int) -> bool:
        self._owner()
        entry = self._live.get(lease_id)
        if entry is None:
            return False  # Duplicate/stale completion is harmless.
        if entry[0].request_id != request_id:
            raise ValueError("Cross-request worker completion")
        if lease_id not in self._retiring:
            raise ValueError("Completion cannot release an active working set")
        self.pool.free_blocks(entry[1])
        del self._live[lease_id]
        self._retiring.remove(lease_id)
        return True

    @property
    def live_blocks(self):
        return sum(len(blocks) for _, blocks in self._live.values())
