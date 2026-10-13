"""Native scheduler subclass for transaction suspension and atomic commit.

Experimental bounded target mode clips admission and schedules semantic refresh.
Identity-refresh qualification remains separately available.
"""
import os
import hashlib
import inspect
import json
import time
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
        options = RuntimeOptions.from_vllm(self.vllm_config)
        if options.async_archive_retirement and not self._bounded_target:
            raise ValueError('Asynchronous archive retirement requires bounded execution')
        config = options.memory
        self.archive_admission = None
        if options.repository_gc:
            if not self._bounded_target:
                raise ValueError('Serving repository GC requires bounded execution')
            from .repository_gc import serving_gc_config
            serving_gc_config(config, self.scheduler_config.max_num_batched_tokens,
                              checkpoint_frames=int(bool(options.session_capacity)))
            from .archive_admission import ArchiveAdmission
            self.archive_admission = ArchiveAdmission(config.block_tokens, config.repository_blocks)
            self.connector.scheduler_archive_update = lambda sequence, capacity: self.archive_admission.update(
                sequence, capacity, {k for k, r in self.requests.items() if not r.is_finished()})
        self.prefill_cadence = options.prefill_cadence
        self.session_capture_decode = options.session_capture_decode
        self._mixed_steps = 0
        self._scheduled_queries = {"prefill": {}, "decode": {}}
        self._cadence_requested = 0
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
                draft_contract=self.draft_contract,
                max_authorities=self.scheduler_config.max_num_seqs if options.async_archive_retirement else None)
        self.connector.scheduler_completion = self._complete_refresh
        self.transactions = {}
        self.table_updates = {}
        self._probe_submitted = False
        if options.session_capacity:
            from .session_control import SessionLedger
            def prepare_import(request_id, history, window):
                prepared = []
                try:
                    for manager in self.window_managers.values():
                        manager.prepare_import(request_id, history, window)
                        prepared.append(manager)
                    if self.archive_admission is not None:
                        from .archive_admission import ArchiveCapacity
                        ready = self.connector.sessions.requests[request_id].ready_records
                        if not 0 <= ready <= config.repository_blocks:
                            raise ValueError('Imported checkpoint exceeds repository capacity')
                        self.archive_admission.snapshots[request_id] = ArchiveCapacity(history, ready)
                except Exception:
                    for manager in prepared:
                        manager.abort_import_preparation(request_id)
                    raise
            def abort_import(request_id):
                for manager in self.window_managers.values():
                    manager.abort_import_preparation(request_id)
                if self.archive_admission is not None:
                    self.archive_admission.forget(request_id)
            def commit_import(request_id, history):
                for manager in self.window_managers.values():
                    manager.validate_import(request_id, history)
                for manager in self.window_managers.values():
                    if not manager.commit_import(request_id, history, _Complete()):
                        raise RuntimeError('Acknowledged prompt import did not commit')
            self.connector.sessions = SessionLedger(options.session_capacity, config.block_tokens,
                                                    prepare_import, abort_import, commit_import,
                                                    terminal_capture=options.session_capture_terminal)

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
                    replay_tokens=replay_tokens, allow_import=bool(self.connector.sessions and
                        self.connector.sessions.import_scheduled(request.request_id)))))
        except Exception:
            for group_id, ticket in prepared:
                self.window_managers[group_id].abort_preparation(ticket)
            raise
        groups = tuple(prepared)
        command = RefreshCommand(request.request_id, groups[0][1].ticket_id, groups)
        self.transactions[request.request_id] = (command, request.next_decode_eligible_step)
        request.next_decode_eligible_step = 2**62
        self.connector.queue(command)

    def _write_lifecycle_snapshot(self):
        if not getattr(self, "_bounded_target", False) or os.getenv("KVMEM_LIFECYCLE_TRACE") != "1":
            return
        gate = self.kv_cache_manager
        owned = sum(not block.is_null for manager in gate.coordinator.single_type_managers
                    for blocks in manager.req_to_blocks.values() for block in blocks)
        report = {"at": time.perf_counter(), "requests": len(self.requests),
            "refresh_pending": len(self.transactions), "connector_pending": len(self.connector.pending),
            "reservations": list(gate.reservations), "reservation_rejections": gate.rejections,
            "authority_limit": gate.max_authorities, "authority_rejections": gate.authority_rejections,
            "owned_blocks": owned,
            "archive_capacity_deferrals": self.archive_admission.deferred if getattr(self, 'archive_admission', None) else 0,
            "free_blocks": gate.block_pool.get_num_free_blocks(),
            "total_blocks": gate.block_pool.num_gpu_blocks}
        sessions = self.connector.sessions
        if sessions is not None:
            report['sessions'] = dict(entries=len(sessions.entries), hits=sessions.hits,
                                      misses=sessions.misses, waits=sessions.waits)
        path = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"])
        temporary = path / "scheduler_resources.tmp"
        temporary.write_text(json.dumps(report))
        temporary.replace(path / "scheduler_resources.json")

    def update_from_output(self, *args, **kwargs):
        result = super().update_from_output(*args, **kwargs)
        # Include native deferred frees and connector completions from this
        # output. A snapshot at schedule() alone can precede the final free.
        self._write_lifecycle_snapshot()
        return result

    def _free_request(self, request, delay_free_blocks=False):
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if getattr(self, 'archive_admission', None) is not None:
            self.archive_admission.forget(request.request_id)
        if getattr(self, "_bounded_target", False) and os.getenv("KVMEM_LIFECYCLE_TRACE") == "1":
            report = {"request_id": request.request_id, "status": request.status.name,
                      "at": time.perf_counter(), "history_end": request.num_computed_tokens,
                      "in_flight_tokens": request.num_in_flight_tokens,
                      "refresh_pending": request.request_id in self.transactions,
                      "lifetime_reservation_retained": request.request_id in self.kv_cache_manager.reservations}
            with (Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "scheduler_finished.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
        return result

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
        tokens = self.query_policy.admit(num_computed_tokens, tokens, request.num_prompt_tokens, offsets.pop())
        if (self.session_capture_decode and self.connector.sessions is not None and
                request.request_id in self.connector.sessions.requests and
                num_computed_tokens >= request.num_prompt_tokens):
            # Any speculative prefix can be accepted. Leave a guaranteed sealed
            # boundary for the next CPU snapshot rather than jumping over it.
            tokens = min(tokens, self.query_policy.block_tokens -
                         num_computed_tokens % self.query_policy.block_tokens)
        gate = getattr(self, 'archive_admission', None)
        if gate is not None and tokens and not gate.allows(request.request_id, num_computed_tokens, tokens):
            return 0  # Native running-queue iteration continues with neighbours.
        return tokens

    def _build_kv_connector_meta(self, connector, output):
        if getattr(self, "_bounded_target", False):
            for request_id, query_tokens in output.num_scheduled_tokens.items():
                request = self.requests[request_id]
                if connector.sessions is not None:
                    connector.sessions.capture_before_final_prefill(request, query_tokens)
                    if self.session_capture_decode:
                        connector.sessions.capture_decode_boundary(request)
                if hasattr(self, '_scheduled_queries'):
                    stage = 'prefill' if request.num_computed_tokens < request.num_prompt_tokens else 'decode'
                    histogram = self._scheduled_queries[stage]
                    histogram[str(query_tokens)] = histogram.get(str(query_tokens), 0) + 1
                offset = next(iter(self.window_managers.values())).offsets.get(request_id, 0)
                imported_query = (connector.sessions is not None and connector.sessions.import_scheduled(request_id)
                                  and request.num_computed_tokens > self.query_policy.select_budget)
                if imported_query or self.query_policy.refresh(request.num_computed_tokens, request.num_prompt_tokens, offset):
                    self.begin_refresh(request, self.query_policy.select_budget, replay_tokens=query_tokens)
                    # Native V2 otherwise defers start_load_kv until after model
                    # forward. The query command must exist before this forward.
                    output.has_sync_kv_loads = True
        return super()._build_kv_connector_meta(connector, output)

    def _throttle_mixed_prefill(self):
        """Ask the native scheduler for decode-only steps, preserving its gates.

        Every cadence-th step releases prefills. Capacity-bound native scheduling
        can override the request; transactions and unavailable decode work never
        justify delaying prefill. No query sizes or request ordering are changed.
        """
        if not getattr(self, '_bounded_target', False) or getattr(self, 'prefill_cadence', 1) == 1:
            return False
        mixed = (any(r.is_prefill_chunk for r in self.running) or bool(self.waiting)) and any(
            not r.is_prefill_chunk and not r.is_finished() and not r.num_in_flight_tokens
            and r.request_id not in self.transactions and r.next_decode_eligible_step <= self.current_step
            for r in self.running)
        if not mixed or self.prefill_capacity_bound:
            self._mixed_steps = 0
            return False
        self._mixed_steps += 1
        throttle = self._mixed_steps % self.prefill_cadence != 0
        self._cadence_requested += int(throttle)
        return throttle

    def schedule(self, throttle_prefills=False):
        armed = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"]) / "armed"
        if not getattr(self, "_bounded_target", False) and not self._probe_submitted and armed.is_file():
            for request in self.running:
                if request.num_computed_tokens == request.num_prompt_tokens and request.num_output_tokens:
                    self.begin_refresh(request, request.num_computed_tokens)
                    self._probe_submitted = True
                    break
        output = super().schedule(throttle_prefills or self._throttle_mixed_prefill())
        if hasattr(self, '_scheduled_queries') and output.total_num_scheduled_tokens:
            # Actual admitted sizes, including native draft reservations and
            # alignment, are the evidence for a matched query-policy comparison.
            path = Path(os.environ['KVMEM_RUNNER_PROBE_DIR'])
            temporary = path / 'scheduled_queries.tmp'
            temporary.write_text(json.dumps({"queries": self._scheduled_queries,
                "prefill_cadence": self.prefill_cadence,
                "archive_capacity_deferrals": self.archive_admission.deferred if getattr(self, 'archive_admission', None) else 0,
                "throttle_requests": self._cadence_requested}))
            temporary.replace(path / 'scheduled_queries.json')
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
