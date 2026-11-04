"""Execution-only optimizations for the existing per-layer pressure policy."""
from types import MethodType
import time


def layer_read_blocks(full_frame_blocks, archive_layers):
    """Use the existing full-frame byte allowance for one-layer reads.

    The archive reader and pinned staging already support up to 256 records.
    A single-layer chunk must not exceed the bytes of the configured all-layer
    chunk. This changes execution grouping only, never selection or KV capacity.
    """
    if (type(full_frame_blocks) is not int or not 1 <= full_frame_blocks <= 256 or
            type(archive_layers) is not int or archive_layers <= 0):
        raise ValueError('Invalid frozen assembly chunk or archive layer count')
    return min(256, full_frame_blocks * archive_layers)


def packed_layer_chunk(state, layer, payload, half, cursor, records, row_bytes, chunk_blocks):
    """Reuse split assembly's paged addressing for contiguous complete blocks.

    The reader already supplies contiguous raw K followed by byte-exact V for
    one layer. Treat that as a one-layer split archive. Partial blocks retain
    the existing token-addressed path; there is no change to the selected set.
    """
    if cursor % 32 or any(record.count != 32 for record in records):
        return False
    if not records or half != len(records)*row_bytes or cursor+len(records)*32 > state['selected_tokens']:
        raise ValueError('Packed layer chunk differs from its selected dense extent')
    import torch
    from .split_assembly import assemble_split_layer
    key = (payload.device, state['selected_tokens'], chunk_blocks)
    cached = state.get('packed_layer_addressing')
    if cached is None:
        bases = torch.arange((state['selected_tokens']+31)//32,device=payload.device,dtype=torch.int64)*32
        indices = torch.arange(chunk_blocks,device=payload.device,dtype=torch.int64)
        cached = state['packed_layer_addressing'] = key,bases,indices
    if cached[0] != key or len(records)>chunk_blocks:
        raise ValueError('Packed layer addressing belongs to a different extent or device')
    _,bases,indices = cached
    n = len(records);indices=indices[:n]
    assemble_split_layer(payload[:half],indices,payload[half:2*half],indices,
        bases[cursor//32:cursor//32+n],layer,layer_index=0,layers=1,block_tokens=32,
        raw_record_bytes=row_bytes,value_record_bytes=row_bytes,raw_dtype=torch.float8_e4m3fn)
    return True


def assembly_groups(records, chunk_blocks, skip_history_begins=()):
    """Keep dense destinations intact when the original plan skips a record."""
    if chunk_blocks <= 0:
        raise ValueError('Assembly chunk size must be positive')
    skip = set(skip_history_begins)
    groups, group, cursor, begin = [], [], 0, 0
    found = set()
    for record in records:
        if record.history_begin in skip:
            if group:
                groups.append((begin, group)); group = []
            found.add(record.history_begin)
        else:
            if not group:
                begin = cursor
            group.append(record)
            if len(group) == chunk_blocks:
                groups.append((begin, group)); group = []
        cursor += record.count
    if group:
        groups.append((begin, group))
    if found != skip:
        raise ValueError('Skipped source is absent from the layer selection')
    return groups, cursor


def packet_histories(frame, pending, starts, counts):
    """Reuse the validated shared geometry, replacing only original block IDs.

    Layerwise pressure requires identical dense extents and partial-tail
    geometry. Appended public tokens are shared; the immutable retained prefix
    is the only part with distinct identities. Partial retained records use
    the original extension/segmentation path.
    """
    from .action_attention import window_segments
    from .layerwise_pressure import extend_records
    end, count, _, local = pending.prefill
    if frame is None:
        raise ValueError('Fast metadata requires an established layer map')
    origin, layers = frame
    if origin > end - count:
        raise ValueError('Layer map is ahead of the accepted Query')
    result = []
    for records in layers:
        extent = sum(r[3] for r in records)
        if extent + end - count - origin != local:
            raise ValueError('Layer metadata changed its dense history extent')
        if any(r[3] != 32 for r in records):
            extended = extend_records(records, origin, end-count)
            ls, lc, history = window_segments(extended, end-count, local, count, 32)
            if list(ls) != list(starts) or list(lc) != list(counts):
                raise ValueError('Partial layer metadata changed segment geometry')
        else:
            n = len(records)
            if (list(starts[:n]) != [r[4] for r in records] or
                    list(counts[:n]) != [32]*n):
                raise ValueError('Layer metadata changed retained segment geometry')
            history = [r[2] for r in records] + list(pending.history[n:])
        result.append(history)
    if result[0] != list(pending.history):
        raise ValueError('Fast metadata changed the shared layer-zero identity')
    return result


def install_single_assembly(assembler, controller, job, runtime_names):
    """Replace a duplicated full-frame rebuild with one rebuild per layer.

    Native transaction planning, retirement, source pinning, error fencing and
    commit remain owned by NativeWindowAssembler/AsyncRefresh. Layer zero and
    MTP retain the original plan's skipped records and shared selection.
    Other target layers retain their original full-rebuild behavior.
    """
    names = tuple(runtime_names)
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Single pressure assembly changed archive layer order')
    rolling = job.command.pressure_stage == 'decode' and getattr(controller, 'rolling_decode', False)
    if job.command.pressure_stage != 'prefill' and not rolling:
        raise ValueError('Single pressure assembly is only qualified for prefill')

    def assemble(self, authority, plan):
        import torch
        from .memory import Phase
        from .window_assembly import AssemblyResult, _validate_destinations
        if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
            raise ValueError('Single pressure assembly needs the original active transaction')
        chosen, _ = job.layerwise_pressure_plan
        by_slot = {r.handle.slot:r for r in chosen[0]}
        remaps = list(plan.remaps)
        positions, cursor = {}, 0
        for record in chosen[0]:
            positions[record.handle.slot] = cursor
            cursor += record.count
        if ({r.block_id for r in remaps} != set(by_slot) or
                any(r.to_base != positions[r.block_id] for r in remaps)):
            raise ValueError('Layer-zero selection differs from the native assembly plan')
        _validate_destinations(remaps, by_slot, self.layers)
        # Query may have changed layer zero without changing the shared host
        # frame. In rolling mode rebuild every target and draft layer from its
        # immutable records, so native shared-map skip hints cannot leak in.
        skip = set() if rolling else {by_slot[r.block_id].history_begin for r in remaps if r.skip}
        state = dict(authority=authority, request_id=job.command.request_id,
                     selected_tokens=plan.total_window_tokens, assembly_arena='pressure')
        started = time.perf_counter(); uploaded = chunks = packed_chunks = paired_chunks = 0
        for index, name in enumerate(names):
            records = chosen[index] if index < len(chosen) else chosen[0]
            stats = controller.query.assemble_layer_pinned(state, index, records,
                layer_name=name, skip_history_begins=skip if index == 0 or index >= len(chosen) else ())
            uploaded += stats['uploaded_bytes']; chunks += stats['chunks']
            packed_chunks += stats.get('packed_chunks',0)
            paired_chunks += stats.get('paired_read_chunks',0)
        done = torch.cuda.Event(); done.record(torch.cuda.current_stream(controller.runner.device))
        job.layerwise_single_assembly_done = True
        rebuilt = sum(r.n_tokens for r in remaps if rolling or not r.skip)
        skipped = 0 if rolling else sum(r.n_tokens for r in remaps if r.skip)
        return AssemblyResult(done, (self, plan, authority), rebuilt, skipped, chunks,
            time.perf_counter()-started, uploaded,
            dict(layerwise_single_assembly=True, layerwise_assembled_layers=len(names),
                 layerwise_redundant_rebuilds=0,layerwise_packed_chunks=packed_chunks,
                 layerwise_paired_read_chunks=paired_chunks),
            partial_tokens=sum(r.n_tokens for r in remaps if not r.skip and r.n_tokens != 32))

    assembler._assemble = MethodType(assemble, assembler)
