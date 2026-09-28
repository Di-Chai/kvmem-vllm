"""Bounded full-attention allocation adapter for the native vLLM coordinator.

Composition preserves the native manager/pool and all its allocation bookkeeping.
Only full-attention token extents are translated. Recurrent managers and Request
logical counters are untouched. Scheduler wiring must suspend a pending request
and send complete replacement block-table rows after commit.
"""
from dataclasses import dataclass
from itertools import count


class RefreshPending(RuntimeError):
    pass


class RefreshRequired(RuntimeError):
    pass


_ticket_ids = count(1)


@dataclass(frozen=True)
class RefreshTicket:
    request_id: str
    ticket_id: int
    history_end: int
    window_tokens: int
    original_block_ids: tuple[int, ...]
    retained_block_ids: tuple[int, ...]


class WorkingSetManager:
    def __init__(self, native, select_budget: int, gen_budget: int):
        from .integration import check_engine_version
        check_engine_version()
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
        if type(native) is not FullAttentionManager:
            raise ValueError("Only native full-attention groups may use this adapter")
        if native.enable_caching or native.dcp_world_size != 1 or native.pcp_world_size != 1:
            raise ValueError("First integration requires prefix caching off and single context-parallel rank")
        if min(select_budget, gen_budget) <= 0:
            raise ValueError("K/G must be positive")
        self.native = native
        self.select_budget = select_budget
        self.capacity = select_budget + gen_budget
        self.offsets: dict[str, int] = {}
        self.pending: dict[str, RefreshTicket] = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _local(self, request_id: str, tokens: int, *, admission=False):
        if request_id in self.pending:
            raise RefreshPending(request_id)
        value = tokens - self.offsets.get(request_id, 0)
        if value < 0:
            raise ValueError("Logical position predates committed working set; restore a checkpoint first")
        if admission:
            return min(value, self.capacity)
        if value > self.capacity:
            raise RefreshRequired(request_id)
        return value

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                                   total_computed_tokens, num_local_computed_tokens,
                                   num_tokens_main_model, apply_admission_cap=False):
        if new_computed_blocks:
            raise ValueError("Working-set manager cannot consume ordinary prefix-cache hits")
        return self.native.get_num_blocks_to_allocate(
            request_id, self._local(request_id, num_tokens, admission=apply_admission_cap),
            new_computed_blocks,
            self._local(request_id, total_computed_tokens, admission=apply_admission_cap),
            self._local(request_id, num_local_computed_tokens, admission=apply_admission_cap),
            self._local(request_id, num_tokens_main_model, admission=apply_admission_cap),
            apply_admission_cap=apply_admission_cap,
        )

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        return self.native.allocate_new_blocks(request_id, self._local(request_id, num_tokens),
                                               self._local(request_id, num_tokens_main_model))

    def prepare_refresh(self, request_id: str, history_end: int, window_tokens: int):
        if request_id in self.pending:
            raise RefreshPending(request_id)
        old_window = self._local(request_id, history_end)
        if not 0 < window_tokens <= min(old_window, self.select_budget):
            raise ValueError("Refresh must select a nonempty subset within K")
        blocks = self.native.req_to_blocks.get(request_id)
        if not blocks or old_window > len(blocks) * self.native.block_size:
            raise ValueError("Refresh references unallocated history")
        count_kept = (window_tokens + self.native.block_size - 1) // self.native.block_size
        ticket = RefreshTicket(request_id, next(_ticket_ids), history_end, window_tokens,
                               tuple(b.block_id for b in blocks),
                               tuple(b.block_id for b in blocks[:count_kept]))
        self.pending[request_id] = ticket
        return ticket

    def commit_refresh(self, ticket: RefreshTicket, completion) -> bool:
        if self.pending.get(ticket.request_id) is not ticket:
            raise ValueError("Stale or foreign refresh completion")
        if not completion.query():
            return False
        blocks = self.native.req_to_blocks[ticket.request_id]
        if tuple(b.block_id for b in blocks) != ticket.original_block_ids:
            raise RuntimeError("Native allocation changed while refresh was pending")
        keep = len(ticket.retained_block_ids)
        removed = blocks[keep:]
        del blocks[keep:]
        self.native.block_pool.free_blocks(reversed(removed))
        self.offsets[ticket.request_id] = ticket.history_end - ticket.window_tokens
        del self.pending[ticket.request_id]
        return True

    def pop_blocks_for_free(self, request_id: str):
        if request_id in self.pending:
            raise RefreshPending("Drain refresh before cancelling/freeing request pages")
        self.offsets.pop(request_id, None)
        return self.native.pop_blocks_for_free(request_id)

    def free(self, request_id: str):
        blocks = self.pop_blocks_for_free(request_id)
        self.native.block_pool.free_blocks(reversed(blocks))
