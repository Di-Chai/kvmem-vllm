"""Scheduler-owned target semantic refresh using native batched model execution.

Initial integration is eager, synchronous, text-only, TP=1, APC off. All batch
neighbours are restored after a provisional query, while only selected requests
change attention windows. DFlash and persistent-session rollback are not enabled.
"""
from dataclasses import dataclass
import json
import os
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
    if (runner.speculative_config is not None or runner.vllm_config.max_concurrent_batches != 1 or
            runner.vllm_config.cache_config.enable_prefix_caching or
            runner.vllm_config.cache_config.mamba_cache_mode != "none" or
            os.getenv("KVMEM_WINDOW_METADATA") != "1" or not runner._kvmem_continuous):
        raise ValueError("Bounded target requires synchronous non-speculative APC-off archive/window execution")
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
        runtime = runner._kvmem_capture_runtime
        if runtime is None or not set(jobs).issubset(batch.req_ids):
            raise ValueError("Scheduled refresh is missing its archived history or query batch")
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
                    history % cfg.block_tokens or selected_cap != cfg.select_budget or
                    query != int(batch.num_scheduled_tokens[i])):
                raise ValueError("Scheduler query does not match accepted authority or native input batch")
            for archive_job, _ in authority.writes:
                archive_job.result()
            authority.poll()
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
            if tuple(c.name for c in captured) != runtime.names:
                raise RuntimeError("Native provisional batch missed attention layers")

        if runner._kvmem_window_adapter is None:
            groups = full_attention_groups(runner.kv_cache_config)
            runner._kvmem_window_adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
        owners, reports = [], []
        for request_id, job in jobs.items():
            i, history, query = extents[request_id]
            authority = runtime.authorities[request_id]
            if history <= cfg.select_budget:
                plan = authority.memory.select({})  # Exact all-history bootstrap.
            else:
                begin, end = batch.query_start_loc_np[i:i+2]
                queries = torch.stack([c.query[begin:end] for c in captured])
                plan = authority.select_for_queries(queries)
            selected = plan.total_window_tokens
            if not 0 < selected <= cfg.select_budget:
                raise RuntimeError("Original selector exceeded its reserved prefix")
            assembler = NativeWindowAssembler.from_runner(runner, request_id, runtime.names, cfg.block_tokens)
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
                "batch_requests": list(batch.req_ids), "probe_executed": needs_probe})

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
        with (runner._kvmem_probe_output / "bounded_refresh.jsonl").open("a") as stream:
            for report in reports:
                stream.write(json.dumps(report) + "\n")
        for job in jobs.values():
            job.event = completed
        jobs.clear()
        return result

    runner.model.forward = MethodType(forward, runner.model)
