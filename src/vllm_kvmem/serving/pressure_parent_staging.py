"""Use one pressure transaction's existing all-layer staging for layer copies.

This does not retain or reuse native V: every non-skipped K/V byte still comes
from the original immutable archive. The original parent owns its private page
tables, bounded two-slot pool, DMA fences and the final transaction completion.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ParentCopyPlan:
    chunk_blocks: int
    row_bytes: int
    groups: tuple
    setup_seconds: float


def prepare_parent_pressure_copy(parent, controller, state, plan, chosen, names,
                                 skip_history_begins=()):
    """Validate and group every destination before the first GPU write.

Unsupported partial/unpublished records explicitly retain the original path.
Ownership, generation and inconsistent extents are errors, never fallbacks.
"""
    from .layerwise_pressure_efficiency import assembly_groups
    from ..core.memory import Phase
    from ..adapters.original_rope import enabled as original_rope
    import torch

    started = time.perf_counter()
    authority, rid = state['authority'], state['request_id']
    query, runner = controller.query, controller.runner
    runtime = getattr(runner, '_kvmem_capture_runtime', None)
    if (state.get('assembly_arena') != 'pressure' or state.get('pressure_stage') != 'prefill' or
            authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan or
            getattr(runtime, 'authorities', {}).get(rid) is not authority or
            getattr(parent, '_runner_binding', None) != (runner, rid) or
            parent.closed or parent.staging.owner is not parent):
        raise ValueError('Parent pressure staging requires its live suspended native transaction')
    if (original_rope() or not query.packed_assembly or not query.paired_reads or
            not query.batched_reads or getattr(controller, 'resident_values', False) or
            'pressure_value_sources' in state or authority.raw_element_bytes != 1 or
            authority.raw_dtype != torch.float8_e4m3fn or parent.raw_dtype != torch.float8_e4m3fn or
            authority.shape[1] != 32 or tuple(names[:len(controller.names)]) != tuple(controller.names) or
            len(names) != len(parent.layers) or len(chosen) != len(controller.names) or
            authority.shape[0] != len(names) or parent.block_tokens != 32):
        raise ValueError('Parent pressure staging requires the unchanged packed full-layer FP8 route')
    parent._validate_borrowed_layers(parent.layers, 32, None)
    row_bytes = 32*authority.shape[2]*authority.shape[3]
    expected_slot = len(parent.layers)*parent.chunk_blocks*row_bytes*2
    if parent.staging.slot_bytes != expected_slot or expected_slot <= 0:
        raise ValueError('Parent pressure staging differs from its original configured capacity')
    chunk_blocks = expected_slot//(2*row_bytes)
    if state['selected_tokens'] != plan.total_window_tokens:
        raise ValueError('Parent pressure copy changed the native selected extent')
    groups, partial = [], False
    for index in range(len(names)):
        records = chosen[index] if index < len(chosen) else chosen[0]
        skips = skip_history_begins if index == 0 or index >= len(chosen) else ()
        layer_groups, total = assembly_groups(records, chunk_blocks, skips)
        if total != state['selected_tokens']:
            raise ValueError('Parent pressure layer changed its selected-token extent')
        partial |= any(r.count != 32 or not r.archived for r in records)
        groups.append(tuple((cursor, tuple(group)) for cursor, group in layer_groups))
    if partial:
        state['pressure_parent_staging_fallback_reason'] = 'partial_or_unpublished_record'
        return None
    reader = state.get('prepared_pressure_reader')
    if reader is None:
        state['pressure_parent_staging_fallback_reason'] = 'prepared_reader_unavailable'
        return None
    if reader.authority is not authority or reader.plan is not plan:
        raise ValueError('Parent pressure reader belongs to a different transaction')
    reader.validate()
    state.pop('pressure_parent_staging_fallback_reason', None)
    return ParentCopyPlan(chunk_blocks, row_bytes, tuple(groups), time.perf_counter()-started)


def assemble_parent_pressure_layers(parent, controller, state, plan, chosen, names,
                                    skip_history_begins=()):
    """Return per-layer statistics, or None for a recorded unchanged fallback.

The caller remains inside its original prepared-reader and parent-assembly
contexts. This function never releases a pool or commits the transaction.
"""
    import torch
    from .layerwise_pressure_efficiency import packed_layer_chunk

    prepared = prepare_parent_pressure_copy(parent, controller, state, plan, chosen,
                                            names, skip_history_begins)
    if prepared is None:
        return None
    authority, staging = state['authority'], parent.staging
    reader = state['prepared_pressure_reader']
    total_raw = authority.shape[0]*prepared.row_bytes
    results = []
    try:
        for index, groups in enumerate(prepared.groups):
            layer = parent.layers[index]
            uploaded = chunks = 0
            fills = copy_seconds = upload_seconds = waits = dispatch_seconds = 0.
            native_chunks = 0
            native_times = {key:0. for key in ('source_prepare','copy','gil_release','gil_reacquire')}
            for cursor, group in groups:
                half = len(group)*prepared.row_bytes
                gpu, ready, slot = staging.destination(2*half)
                regions = ((index*prepared.row_bytes, prepared.row_bytes),
                           (total_raw+index*prepared.row_bytes, prepared.row_bytes))

                def fill(host):
                    nonlocal fills, copy_seconds, native_chunks
                    started = time.perf_counter()
                    _, stats = reader.read(group, regions, destination=host)
                    fills += time.perf_counter()-started
                    copy_seconds += stats.get('cache_copy_seconds', 0.)
                    native_chunks += int(stats.get('paired_native_gather', False))
                    for key in native_times:
                        native_times[key] += stats.get('paired_native_'+key+'_ns', 0)/1e9

                started = time.perf_counter()
                transfer = parent.transfers.upload_from_reader(2*half, gpu, fill, ready=ready)
                upload_seconds += time.perf_counter()-started
                if transfer is None:
                    started = time.perf_counter()
                    next(iter(parent.transfers.pending.values())).event.synchronize()
                    waits += time.perf_counter()-started
                    started = time.perf_counter()
                    transfer = parent.transfers.upload_from_reader(2*half, gpu, fill, ready=ready)
                    upload_seconds += time.perf_counter()-started
                if transfer is None:
                    raise RuntimeError('Parent pressure transfer made no progress')
                torch.cuda.current_stream(controller.runner.device).wait_event(transfer.event)
                started = time.perf_counter()
                if not packed_layer_chunk(state, layer, gpu, half, cursor, group,
                                          prepared.row_bytes, prepared.chunk_blocks):
                    raise RuntimeError('Validated full pressure blocks lost their packed geometry')
                dispatch_seconds += time.perf_counter()-started
                staging.consumed(slot)
                uploaded += 2*half
                chunks += 1
            results.append(dict(uploaded_bytes=uploaded, chunks=chunks, packed_chunks=chunks,
                chunk_blocks=prepared.chunk_blocks, staging_slot_bytes=staging.slot_bytes,
                paired_read_chunks=chunks, inline_read_chunks=chunks, prepared_read_chunks=chunks,
                native_paired_gather_chunks=native_chunks,
                parent_staging=True, parent_staging_setup_seconds=prepared.setup_seconds if index == 0 else 0.,
                parent_staging_cpu_fill_seconds=fills, parent_staging_cache_copy_seconds=copy_seconds,
                parent_staging_upload_host_seconds=upload_seconds,
                parent_staging_retry_wait_seconds=waits,
                parent_staging_split_dispatch_host_seconds=dispatch_seconds,
                **{'parent_staging_native_'+key+'_seconds':value for key,value in native_times.items()}))
        return tuple(results)
    finally:
        # packed_layer_chunk owns small GPU addressing tensors in state. Keep
        # them on the original parent until its final assembly fence drains.
        addressing = state.get('packed_layer_addressing')
        if addressing is not None:
            parent._metadata_owners.extend(addressing[1:])
