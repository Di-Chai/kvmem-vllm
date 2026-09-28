"""Real K128K/G64K target compression qualification, not bounded serving.

The native scheduler retains its original allocation in this diagnostic. The
worker selects and rewrites a smaller dense prefix, restores recurrent state and
replays the query with new attention metadata. A second identical replay checks
state/output determinism before the one real sample/publication. Native page
trimming, asynchronous refresh and draft refresh remain separate work.
"""
import json
import faulthandler
import os
import signal
from types import MethodType

import torch

from .cache_groups import full_attention_groups
from .capture import capture_to
from .native_state import NativeRecurrentState
from .replay_probe import _clone, _equal
from .state import StateCheckpoints
from .window_assembly import NativeWindowAssembler
from .window_batch import WindowBatchAdapter


def install_compression_probe(runner):
    if os.getenv("KVMEM_WINDOW_METADATA") != "1":
        raise ValueError("Compression qualification requires the explicit dense-window metadata hook")
    if os.getenv("KVMEM_REBUILD_PROBE") == "1" or os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") == "1":
        raise ValueError("Compression qualification cannot share an identity rebuild transaction")
    if runner.speculative_config is not None or runner.vllm_config.max_concurrent_batches != 1:
        raise ValueError("Compression qualification requires synchronous target-only execution")
    cache = runner.vllm_config.cache_config
    if cache.enable_prefix_caching or cache.mamba_cache_mode != "none":
        raise ValueError("Compression qualification requires APC off and Mamba cache mode none")
    # Explicit diagnostic only: permit read-only stack dumps if a long native
    # step stalls, without attaching a privileged debugger to the worker.
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    model = runner.model
    original = model.forward
    reports = []

    def forward(self, *args, **kwargs):
        batch = getattr(runner, "_kvmem_last_batch", None)
        runtime = runner._kvmem_capture_runtime
        if batch is None or runtime is None or not batch.num_reqs:
            return original(*args, **kwargs)
        cfg = runtime.config
        # Refresh before a native batch would exceed K+G. Initial diagnostic is
        # one request at a full-block boundary; do not invent a partial checkpoint.
        offsets = (runner._kvmem_window_adapter.offsets
                   if runner._kvmem_window_adapter is not None else {})
        candidates = [i for i, req in enumerate(batch.req_ids)
                      if int(batch.num_computed_tokens_np[i]) - offsets.get(req, 0)
                      >= cfg.select_budget + cfg.gen_budget -
                      runner.vllm_config.scheduler_config.max_num_batched_tokens]
        if not candidates:
            return original(*args, **kwargs)
        if batch.num_reqs != 1 or candidates != [0]:
            raise ValueError("Long compression diagnostic currently requires a single request")
        request_id = batch.req_ids[0]
        runtime.drain(wait=True)
        authority = runtime.authorities[request_id]
        history = authority.history_end
        if history != int(batch.num_computed_tokens_np[0]) or history % cfg.block_tokens:
            raise ValueError("Query replay must begin at the exact archived block boundary")
        for job, _ in authority.writes:
            job.result()
        authority.poll()
        state_adapter = NativeRecurrentState(runner)
        bundle = state_adapter.borrow(batch.req_ids)
        checkpoints = StateCheckpoints("compressed-replay", bundle.bytes * 2, capacity=2)
        before = checkpoints.capture(0, bundle.tensors)
        probe_capture = []
        with capture_to(probe_capture.append):
            original(*args, **kwargs)
        if tuple(c.name for c in probe_capture) != runtime.names:
            raise RuntimeError("Query probe missed target attention layers")
        queries = torch.stack([c.query[:batch.num_tokens] for c in probe_capture])
        plan = authority.select_for_queries(queries)
        selected = plan.total_window_tokens
        if not 0 < selected <= cfg.select_budget < history:
            raise RuntimeError("Diagnostic did not produce a genuinely compressed selection")
        assembler = NativeWindowAssembler.from_runner(runner, request_id, runtime.names,
                                                      cfg.block_tokens)
        assembly = assembler.assemble(authority, plan)
        assembly.event.synchronize()
        authority.memory.attach_completion(assembly)
        assert authority.memory.commit_selection()
        state_adapter.validate(bundle)
        checkpoints.restore(before, bundle.tensors)
        if runner._kvmem_window_adapter is None:
            groups = full_attention_groups(runner.kv_cache_config)
            runner._kvmem_window_adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
        runner._kvmem_window_adapter.commit(request_id, history, selected)

        # Preserve references held by the outer native execute_model call. Dense
        # attention lengths/positions change; logical counters and GDN do not.
        from vllm.forward_context import get_forward_context
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        context = get_forward_context()
        block_tables, slots = runner.prepare_attn(batch)
        metadata = runner.model_state.prepare_attn(
            batch, context.cudagraph_runtime_mode, block_tables, slots,
            runner.attn_groups, runner.kv_cache_config)
        context.attn_metadata.clear()
        context.attn_metadata.update(metadata)
        context.slot_mapping.clear()
        context.slot_mapping.update(build_slot_mappings_by_layer(slots, runner.kv_cache_config))
        dense_positions = runner._kvmem_window_batch.positions
        if kwargs["positions"].ndim == 2:
            dense_positions = dense_positions.unsqueeze(0).expand(kwargs["positions"].shape[0], -1)
        replay_kwargs = {**kwargs, "positions": dense_positions}
        with capture_to(lambda _: None):
            replayed = _clone(original(*args, **replay_kwargs))
        after = checkpoints.capture(1, bundle.tensors)
        checkpoints.restore(before, bundle.tensors)
        result = original(*args, **replay_kwargs)
        output_equal = _equal(replayed, result)
        state_equal = all(_equal(after.tensors[name], value) for name, value in bundle.tensors.items())
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        assert checkpoints.release(0) and checkpoints.release(1)
        assert assembler.close()
        records_by_slot = {r.handle.slot: r for r in authority.records.values()}
        reports.append({"request_id": request_id, "history_begin": history,
            "selected_tokens": selected, "query_tokens": batch.num_tokens,
            "offset": history - selected, "rebuilt_tokens": assembly.rebuilt_tokens,
            "skipped_tokens": assembly.skipped_tokens, "assembly_chunks": assembly.chunks,
            "output_byte_equal": output_equal, "state_byte_equal": state_equal,
            "selection_budget": cfg.select_budget, "generation_budget": cfg.gen_budget,
            "selected_history_ranges": [[records_by_slot[r.block_id].history_begin,
                                          r.n_tokens, r.to_base] for r in plan.remaps]})
        (runner._kvmem_probe_output / "compression.json").write_text(json.dumps({
            "scope": "real target compression/replay; native allocation is not yet bounded",
            "refreshes": reports}, indent=2))
        if not output_equal or not state_equal:
            raise RuntimeError("Compressed query replay changed output or recurrent state bytes")
        return result

    model.forward = MethodType(forward, model)
