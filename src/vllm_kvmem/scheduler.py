"""Native scheduler subclass for transaction suspension and atomic commit.

Initial live qualification uses an explicitly named identity-refresh probe.
Automatic compressed refresh and admission clipping are not enabled yet.
"""
import os
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from .connector import KVMemConnector, RefreshCommand
from .config import KVMemConfig
from .window_manager import WorkingSetManager


class _Complete:
    def query(self):
        return True


class KVMemScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.connector, KVMemConnector):
            raise ValueError("KVMemScheduler requires KVMemConnector")
        if self.scheduler_config.async_scheduling or self.vllm_config.speculative_config is not None:
            raise ValueError("First scheduler transaction qualification requires synchronous target execution")
        if os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") != "1":
            raise ValueError("Compressed runtime not yet enabled; scheduler currently exposes the explicit identity probe")
        config = KVMemConfig()
        coordinator = self.kv_cache_manager.coordinator
        managers = list(coordinator.single_type_managers)
        self.window_managers = {}
        for group_id, manager in enumerate(managers):
            if type(manager) is FullAttentionManager:
                managers[group_id] = WorkingSetManager(manager, config.select_budget, config.gen_budget)
                self.window_managers[group_id] = managers[group_id]
        if not self.window_managers:
            raise ValueError("No full-attention cache group found")
        coordinator.single_type_managers = tuple(managers)
        self.connector.scheduler_completion = self._complete_refresh
        self.transactions = {}
        self.table_updates = {}
        self._probe_submitted = False

    def begin_refresh(self, request, window_tokens):
        if getattr(request, "num_in_flight_tokens", 0):
            raise RuntimeError("Wait for all target/speculation work before refreshing")
        if request.request_id in self.transactions or request.request_id in self.connector.pending:
            raise RuntimeError("Request already has a refresh transaction")
        prepared = []
        try:
            for group_id, manager in self.window_managers.items():
                prepared.append((group_id, manager.prepare_refresh(
                    request.request_id, request.num_computed_tokens, window_tokens)))
        except Exception:
            for group_id, ticket in prepared:
                self.window_managers[group_id].abort_preparation(ticket)
            raise
        groups = tuple(prepared)
        command = RefreshCommand(request.request_id, groups[0][1].ticket_id, groups)
        self.transactions[request.request_id] = (command, request.next_decode_eligible_step)
        request.next_decode_eligible_step = 2**62
        self.connector.queue(command)

    def _request_blocks_can_be_freed(self, request):
        if request.request_id in self.transactions:
            return False
        return super()._request_blocks_can_be_freed(request)

    def _complete_refresh(self, request_id, command_id):
        command, previous_step = self.transactions[request_id]
        if command.command_id != command_id:
            raise ValueError("Refresh command identity mismatch")
        # Validate every group before freeing any pages; no partial commit on a
        # stale ticket or unexpected native allocation change.
        for group_id, ticket in command.groups:
            self.window_managers[group_id].validate_refresh(ticket)
        for group_id, ticket in command.groups:
            if not self.window_managers[group_id].commit_refresh(ticket, _Complete()):
                raise RuntimeError("Acknowledged refresh could not commit")
        request = self.requests[request_id]
        request.next_decode_eligible_step = previous_step
        if not request.is_finished():
            self.table_updates[request_id] = tuple(
                [b.block_id for b in manager.req_to_blocks.get(request_id, [])]
                for manager in self.kv_cache_manager.coordinator.single_type_managers)
        del self.transactions[request_id]

    def schedule(self, throttle_prefills=False):
        armed = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "armed"
        if not self._probe_submitted and armed.is_file():
            for request in self.running:
                if request.num_computed_tokens == request.num_prompt_tokens and request.num_output_tokens:
                    self.begin_refresh(request, request.num_computed_tokens)
                    self._probe_submitted = True
                    break
        output = super().schedule(throttle_prefills)
        if self.table_updates:
            # Scheduling can allocate another block after the refresh commit.
            # Publish the complete current row, including those new allocations.
            updates = {
                key: tuple([b.block_id for b in manager.req_to_blocks.get(key, [])]
                           for manager in self.kv_cache_manager.coordinator.single_type_managers)
                for key in self.table_updates
                if key in self.requests and not self.requests[key].is_finished()}
            output.block_table_updates = {**(output.block_table_updates or {}), **updates}
            self.table_updates = {}
        return output
