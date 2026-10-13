"""Scheduler-owned target semantic refresh using native batched model execution.

Refresh transactions are eager, synchronous, text-only, TP=1, APC off; regular
decode may use FULL native graphs. All batch neighbours are restored after a
provisional query, while only selected requests change attention windows.
DFlash2 preserves independent native draft sliding pages. Exact terminal
checkpoints are handled separately; arbitrary session rollback is unsupported.
"""
from dataclasses import dataclass
import json
import os
import time
from types import MethodType

import torch

from .cache_groups import full_attention_groups
from .capture import capture_to
from .native_state import NativeRecurrentState
from .state import StateCheckpoints
from .window_assembly import NativeWindowAssembler
from .window_batch import WindowBatchAdapter


@dataclass
class QueryJob:
    command: object
    event: object = None
    selected_tokens: int | None = None

    def query(self):
        return self.event is not None and self.event.query()


def install_bounded_target(runner):
    draft_contract = None
    if runner.speculative_config is not None:
        if os.getenv("KVMEM_BOUNDED_DFLASH") != "1" and os.getenv("KVMEM_BOUNDED_MTP") != "1":
            raise ValueError("Bounded speculation requires explicit DFlash2 opt-in")
        from .draft_contract import DraftContract
        from .options import RuntimeOptions
        draft_contract = DraftContract.from_vllm(runner.vllm_config, RuntimeOptions.from_vllm(runner.vllm_config).memory)
    if (runner.vllm_config.max_concurrent_batches != 1 or
            runner.vllm_config.cache_config.enable_prefix_caching or
            runner.vllm_config.cache_config.mamba_cache_mode != "none" or
            os.getenv("KVMEM_WINDOW_METADATA") != "1" or not runner._kvmem_continuous):
        raise ValueError("Bounded execution requires synchronous APC-off archive/window execution")
    jobs = {}
    original = runner.model.forward

    def enqueue(command):
        if command.request_id in jobs:
            raise ValueError("Duplicate scheduled query refresh")
        job = QueryJob(command)
        jobs[command.request_id] = job
        return job

    runner._kvmem_query_refresh = enqueue

    def forward(self, *args, **kwargs):
        if not jobs:
            return original(*args, **kwargs)
        batch = runner._kvmem_last_batch
        started_at = time.perf_counter()
        lifecycle_trace = os.getenv("KVMEM_LIFECYCLE_TRACE") == "1"
        if lifecycle_trace:
            active = {"started_at": started_at, "requests": [
                {"request_id": request_id, "command_id": job.command.command_id,
                 "history_begin": job.command.groups[0][1].history_end,
                 "query_tokens": job.command.groups[0][1].replay_tokens}
                for request_id, job in jobs.items()]}
            temporary = runner._kvmem_probe_output / "bounded_active.tmp"
            temporary.write_text(json.dumps(active))
            temporary.replace(runner._kvmem_probe_output / "bounded_active.json")
        runtime = runner._kvmem_capture_runtime
        if runtime is None or not set(jobs).issubset(batch.req_ids):
            raise ValueError("Scheduled refresh is missing its archived history or query batch")
        draft_groups = draft_contract.validate_native_groups(runner) if draft_contract is not None else ()
        runtime.drain(wait=True)
        cfg = runtime.config
        extents = {}
        for request_id, job in jobs.items():
            i = batch.req_ids.index(request_id)
            expected_groups = full_attention_groups(runner.kv_cache_config)
            if tuple(group for group, _ in job.command.groups) != expected_groups:
                raise ValueError("Query command does not cover exactly the native attention groups")
            request_index = runner.req_states.req_id_to_index[request_id]
            for group_id, ticket in job.command.groups:
                expansion = runner.block_tables.blocks_per_kv_block[group_id]
                expected_pages = [block * expansion + page for block in ticket.original_block_ids
                                  for page in range(expansion)]
                count = int(runner.block_tables.num_blocks.np[group_id, request_index])
                actual_pages = runner.block_tables.block_tables[group_id].gpu[request_index, :count].cpu().tolist()
                if actual_pages != expected_pages:
                    raise ValueError("Worker physical pages do not match the scheduler refresh reservation")
            tickets = [ticket for _, ticket in job.command.groups]
            values = {(t.history_end, t.window_tokens, t.replay_tokens) for t in tickets}
            if len(values) != 1:
                raise ValueError("Refresh groups disagree on query extent")
            history, selected_cap, query = values.pop()
            authority = runtime.authorities[request_id]
            if (history != authority.history_end or history != int(batch.num_computed_tokens_np[i]) or
                    selected_cap != cfg.select_budget or
                    query != int(batch.num_scheduled_tokens[i])):
                raise ValueError("Scheduler query does not match accepted authority or native input batch")
            extents[request_id] = (i, history, query)

        needs_probe = any(history > cfg.select_budget for _, history, _ in extents.values())
        state_adapter = checkpoints = before = bundle = None
        captured = []
        if needs_probe:
            state_adapter = NativeRecurrentState(runner)
            bundle = state_adapter.borrow(batch.req_ids)
            checkpoints = StateCheckpoints("bounded-query", bundle.bytes, capacity=1)
            before = checkpoints.capture(0, bundle.tensors)
            with capture_to(captured.append):
                original(*args, **kwargs)
            if tuple(c.name for c in captured) != tuple(runner._kvmem_layers):
                raise RuntimeError("Native provisional batch missed attention layers")

        # The provisional query reads leased native GPU pages, not archived
        # bytes. Let its GPU execution overlap sealed-page packing/SHA/I/O;
        # publication still finishes before selection and any archive read.
        for request_id in jobs:
            authority = runtime.authorities[request_id]
            authority.publish_sealed(runtime._wait_archive_progress)

        if runner._kvmem_window_adapter is None:
            groups = full_attention_groups(runner.kv_cache_config)
            runner._kvmem_window_adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
        owners, reports = [], []
        for request_id, job in jobs.items():
            i, history, query = extents[request_id]
            authority = runtime.authorities[request_id]
            score_query = query
            if history <= cfg.select_budget:
                plan = authority.memory.select({})  # Exact all-history bootstrap.
            else:
                begin, end = batch.query_start_loc_np[i:i+2]
                if draft_contract is not None:
                    from .speculative_admission import SpeculativeQueryPolicy
                    score_query = SpeculativeQueryPolicy.authoritative_probe_tokens(
                        history, int(batch.prefill_len_np[i]), query)
                    end = begin + score_query
                queries = torch.stack([c.query[begin:end] for c in captured])
                plan = authority.select_for_queries(queries)
            selected = plan.total_window_tokens
            if not 0 < selected <= cfg.select_budget:
                raise RuntimeError("Original selector exceeded its reserved prefix")
            assembly_start = time.perf_counter()
            assembler = NativeWindowAssembler.from_runner(runner, request_id, runtime.names, cfg.block_tokens,
                                                          packed=runtime.options.packed_assembly,
                                                          direct_pinned_reads=runtime.options.direct_pinned_reads)
            assembly = assembler.assemble(authority, plan)
            assembly.event.synchronize()
            authority.memory.attach_completion(assembly)
            assert authority.memory.commit_selection()
            runner._kvmem_window_adapter.commit(request_id, history, selected)
            job.selected_tokens = selected
            owners.append((assembler, assembly, plan))
            reports.append({"request_id": request_id, "command_id": job.command.command_id,
                "history_begin": history, "selected_tokens": selected, "query_tokens": query,
                "rebuilt_tokens": assembly.rebuilt_tokens, "skipped_tokens": assembly.skipped_tokens,
                "batch_requests": list(batch.req_ids), "probe_executed": needs_probe,
                "query_score_tokens": score_query, "draft_groups": list(draft_groups),
                "assembly_seconds": time.perf_counter() - assembly_start,
                "assembly_host_prepare_seconds": assembly.host_prepare_seconds,
                "assembly_uploaded_bytes": assembly.uploaded_bytes,
                "assembly_read_stats": assembly.read_stats,
                "packed_assembly": runtime.options.packed_assembly,
                "direct_pinned_reads": runtime.options.direct_pinned_reads})

        if needs_probe:
            state_adapter.validate(bundle)
            checkpoints.restore(before, bundle.tensors)
        from vllm.forward_context import get_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        context = get_forward_context()
        block_tables, slots = runner.prepare_attn(batch)
        metadata = runner.model_state.prepare_attn(batch, context.cudagraph_runtime_mode,
            block_tables, slots, runner.attn_groups, runner.kv_cache_config)
        context.attn_metadata.clear()
        context.attn_metadata.update(metadata)
        context.slot_mapping.clear()
        context.slot_mapping.update(build_slot_mappings_by_layer(slots, runner.kv_cache_config))
        window = runner._kvmem_window_batch
        if window is not None:
            positions = window.positions
            if kwargs["positions"].ndim == 2:
                positions = positions.unsqueeze(0).expand(kwargs["positions"].shape[0], -1)
            kwargs = {**kwargs, "positions": positions}
        result = original(*args, **kwargs)
        completed = torch.cuda.Event()
        completed.record()
        # First integration retires scratch synchronously. Keep the connector
        # acknowledgment after every replayed neighbour and the query itself.
        completed.synchronize()
        if needs_probe:
            assert checkpoints.release(0)
        for assembler, _, _ in owners:
            assert assembler.close()
        finished_at = time.perf_counter()
        if lifecycle_trace:
            (runner._kvmem_probe_output / "bounded_active.json").unlink()
            for report in reports:
                report.update(refresh_started_at=started_at, refresh_finished_at=finished_at)
        with (runner._kvmem_probe_output / "bounded_refresh.jsonl").open("a") as stream:
            for report in reports:
                stream.write(json.dumps(report) + "\n")
        for job in jobs.values():
            job.event = completed
        jobs.clear()
        return result

    runner.model.forward = MethodType(forward, runner.model)
