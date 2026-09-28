"""Native scheduler subclass for transaction suspension and atomic commit.

Experimental bounded target mode clips admission and schedules semantic refresh.
Identity-refresh qualification remains separately available.
"""
import os
import hashlib
import inspect
import json
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from .connector import KVMemConnector, RefreshCommand
from .options import RuntimeOptions
from .window_manager import WorkingSetManager
from .admission import QueryPolicy

SCHEDULER_SHA256 = "c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884"


class _Complete:
    def query(self):
        return True


class KVMemScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.connector, KVMemConnector):
            raise ValueError("KVMemScheduler requires KVMemConnector")
        if self.scheduler_config.async_scheduling:
            raise ValueError("Scheduler refresh transactions require synchronous execution")
        self.draft_contract = None
        if self.vllm_config.speculative_config is not None:
            if os.getenv("KVMEM_BOUNDED_DFLASH") != "1" or os.getenv("KVMEM_BOUNDED_TARGET") != "1":
                raise ValueError("Bounded speculation requires its explicit experimental opt-in")
            from .draft_contract import DraftContract
            self.draft_contract = DraftContract.from_vllm(
                self.vllm_config, RuntimeOptions.from_vllm(self.vllm_config).memory)
        self._bounded_target = os.getenv("KVMEM_BOUNDED_TARGET") == "1"
        if not self._bounded_target and os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") != "1":
            raise ValueError("Compressed runtime not yet enabled; scheduler currently exposes the explicit identity probe")
        if self._bounded_target and hashlib.sha256(Path(inspect.getfile(Scheduler)).read_bytes()).hexdigest() != SCHEDULER_SHA256:
            raise ValueError("Unqualified scheduler token-admission/connector-metadata hooks")
        config = RuntimeOptions.from_vllm(self.vllm_config).memory
        self.query_policy = QueryPolicy(config.block_tokens, config.select_budget, config.gen_budget)
        if self.draft_contract is not None:
            from .speculative_admission import SpeculativeQueryPolicy
            self.query_policy = SpeculativeQueryPolicy(config.block_tokens, config.select_budget,
                                                       config.gen_budget, self.draft_contract.query_tokens)
        if self._bounded_target and not config.block_tokens <= self.scheduler_config.max_num_batched_tokens <= config.gen_budget:
            raise ValueError("Bounded target batch budget must fit an aligned block and G headroom")
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
        if self._bounded_target:
            from .capacity import ReservedKVCacheManager
            self.kv_cache_manager = ReservedKVCacheManager(self.kv_cache_manager, self.vllm_config,
                                                           draft_contract=self.draft_contract)
        self.connector.scheduler_completion = self._complete_refresh
        self.transactions = {}
        self.table_updates = {}
        self._probe_submitted = False

    def begin_refresh(self, request, window_tokens, *, replay_tokens=0):
        if getattr(request, "num_in_flight_tokens", 0):
            raise RuntimeError("Wait for all target/speculation work before refreshing")
        if request.request_id in self.transactions or request.request_id in self.connector.pending:
            raise RuntimeError("Request already has a refresh transaction")
        prepared = []
        try:
            for group_id, manager in self.window_managers.items():
                prepared.append((group_id, manager.prepare_refresh(
                    request.request_id, request.num_computed_tokens, window_tokens,
                    replay_tokens=replay_tokens)))
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
        if getattr(self, "_bounded_target", False) and not request.is_finished():
            # No silent recompute/preemption of authoritative GDN history. New
            # requests wait until capacity is released by an existing request.
            return False
        return super()._request_blocks_can_be_freed(request)

    def _complete_refresh(self, request_id, command_id, selected_tokens=None):
        command, previous_step = self.transactions[request_id]
        if command.command_id != command_id:
            raise ValueError("Refresh command identity mismatch")
        # Validate every group before freeing any pages; no partial commit on a
        # stale ticket or unexpected native allocation change.
        for group_id, ticket in command.groups:
            self.window_managers[group_id].validate_refresh(ticket, selected_tokens)
        for group_id, ticket in command.groups:
            if not self.window_managers[group_id].commit_refresh(ticket, _Complete(), selected_tokens):
                raise RuntimeError("Acknowledged refresh could not commit")
        request = self.requests[request_id]
        request.next_decode_eligible_step = previous_step
        if not request.is_finished():
            self.table_updates[request_id] = tuple(
                [b.block_id for b in manager.req_to_blocks.get(request_id, [])]
                for manager in self.kv_cache_manager.coordinator.single_type_managers)
        del self.transactions[request_id]
        if getattr(self, "_bounded_target", False):
            report = {"request_id": request_id, "command_id": command_id,
                "history_begin": command.groups[0][1].history_end,
                "selected_tokens": selected_tokens,
                "replay_tokens": command.groups[0][1].replay_tokens,
                "groups": [{"group_id": group_id, "blocks": len(manager.req_to_blocks[request_id]),
                            "block_tokens": manager.block_size, "offset": manager.offsets[request_id]}
                           for group_id, manager in self.window_managers.items()]}
            with (Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "scheduler_refresh.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        if not getattr(self, "_bounded_target", False):
            return tokens
        offsets = {m.offsets.get(request.request_id, 0) for m in self.window_managers.values()}
        if len(offsets) != 1:
            raise RuntimeError("Full-attention groups disagree on committed history mapping")
        return self.query_policy.admit(num_computed_tokens, tokens, request.num_prompt_tokens, offsets.pop())

    def _build_kv_connector_meta(self, connector, output):
        if getattr(self, "_bounded_target", False):
            for request_id, query_tokens in output.num_scheduled_tokens.items():
                request = self.requests[request_id]
                offset = next(iter(self.window_managers.values())).offsets.get(request_id, 0)
                if self.query_policy.refresh(request.num_computed_tokens, request.num_prompt_tokens, offset):
                    self.begin_refresh(request, self.query_policy.select_budget, replay_tokens=query_tokens)
                    # Native V2 otherwise defers start_load_kv until after model
                    # forward. The query command must exist before this forward.
                    output.has_sync_kv_loads = True
        return super()._build_kv_connector_meta(connector, output)

    def schedule(self, throttle_prefills=False):
        armed = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "armed"
        if not getattr(self, "_bounded_target", False) and not self._probe_submitted and armed.is_file():
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
