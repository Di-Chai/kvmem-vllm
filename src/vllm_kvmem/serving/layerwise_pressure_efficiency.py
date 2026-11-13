"""Execution-only optimizations for the existing per-layer pressure policy."""
from types import MethodType
from contextlib import nullcontext
import os
import time
from ..core import settings as _settings


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


def fast_chunk_blocks(controller):
    """Largest physical prefill chunk, in blocks, the layer-0 shadow must stage."""
    options = getattr(getattr(controller.runner, '_kvmem_capture_runtime', None), 'options', None)
    tokens = (getattr(options, 'prefill_physical_max_chunks', 4) *
              getattr(options, 'prefill_physical_logical_tokens', 1024))
    return max(1, tokens // 32)


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


def isolate_partial_records():
    """Opt-in KVMEM_ASSEMBLY_ISOLATE_PARTIAL=1: a record the per-layer fast path cannot
    read (not a full, archived block) gets a chunk of its own, so the generic all-layer
    read covers only that record instead of the whole chunk around it. Same bytes and
    positions; only chunk boundaries move."""
    value = _settings.get('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_ASSEMBLY_ISOLATE_PARTIAL must be 0 or 1')
    return value == '1'


def assembly_groups(records, chunk_blocks, skip_history_begins=(), *, isolate=None):
    """Keep dense destinations intact when the original plan skips a record.

    ``isolate(record)`` true: that record forms its own group (dense cursors kept).
    """
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
        elif isolate is not None and isolate(record):
            if group:
                groups.append((begin, group)); group = []
            groups.append((cursor, [record]))
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


def native_plan_check(job, chosen, plan=None, layers=()):
    """Opt-in KVMEM_PRESSURE_NATIVE_PLAN_CHECK: (slots, counts, begins) of the
    layer-zero selection for the native plan check, or None for the original."""
    value = _settings.get('KVMEM_PRESSURE_NATIVE_PLAN_CHECK', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_PLAN_CHECK must be 0 or 1')
    from .. import _native
    if value != '1' or not callable(getattr(_native, 'check_pressure_plan', None)):
        return None
    plan_type = getattr(_native, 'Plan', None)
    if plan_type is None or type(plan) is not plan_type or not all(
            hasattr(layer, 'kernel_page_table') for layer in layers):
        return None      # only real native plans and layer views take the native check
    records = chosen[0]
    slots = getattr(job, 'layerwise_zero_slots', None)
    if slots is None or len(slots) != len(records):
        slots = [r.handle.slot for r in records]
    return list(slots), [r.count for r in records], [r.history_begin for r in records]


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
    parent_staging = _settings.get('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING', '0')
    if parent_staging not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING must be 0 or 1')
    # This candidate changes only prefill transport. A long answer can still
    # enter the existing rolling-decode path under the same process flag.
    parent_staging = parent_staging == '1' and job.command.pressure_stage == 'prefill'

    def assemble(self, authority, plan):
        import torch
        from ..core.memory import Phase
        from .window_assembly import AssemblyResult, _validate_destinations
        if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
            raise ValueError('Single pressure assembly needs the original active transaction')
        _v0 = time.perf_counter()
        chosen, _ = job.layerwise_pressure_plan
        # Rolling decode, or a session whose layer zero diverged from the native
        # frame after a probe-mode Query: rebuild layer zero and the draft too.
        full_rebuild = rolling or getattr(job, 'layerwise_layer0_diverged', False)
        native_check = native_plan_check(job, chosen, plan, self.layers)
        if native_check is not None:
            # Same checks, order and messages over the plan's C++ remaps.
            from ..adapters.original_rope import enabled as original_rope
            from .. import _native
            slots, counts, begins = native_check
            bounds = [(layer.kernel_page_table.numel() * layer.key.shape[1], layer.cos_sin.shape[0])
                      for layer in self.layers]
            skip_begins, rebuilt_tokens, skipped_tokens, partial_tokens = _native.check_pressure_plan(
                plan, slots, counts, begins, bounds, original_rope(), bool(full_rebuild))
            _v1 = _v2 = time.perf_counter()
            skip = set(skip_begins)
            remaps = None
        else:
            by_slot = {r.handle.slot:r for r in chosen[0]}
            remaps = list(plan.remaps)
            positions, cursor = {}, 0
            for record in chosen[0]:
                positions[record.handle.slot] = cursor
                cursor += record.count
            if ({r.block_id for r in remaps} != set(by_slot) or
                    any(r.to_base != positions[r.block_id] for r in remaps)):
                planned = {r.block_id for r in remaps}
                moved = [(by_slot[r.block_id].history_begin, r.to_base, positions[r.block_id])
                         for r in remaps if r.block_id in positions and r.to_base != positions[r.block_id]]
                raise ValueError('Layer-zero selection differs from the native assembly plan: '
                    f'plan_only={len(planned - set(by_slot))} chosen_only={len(set(by_slot) - planned)} '
                    f'moved={len(moved)} first_moved={moved[:3]} remaps={len(remaps)} chosen={len(by_slot)}')
            _v1 = time.perf_counter()
            _validate_destinations(remaps, by_slot, self.layers)
            _v2 = time.perf_counter()
            # Query may have changed layer zero without changing the shared host
            # frame. In rolling mode rebuild every target and draft layer from its
            # immutable records, so native shared-map skip hints cannot leak in.
            skip = set() if full_rebuild else {by_slot[r.block_id].history_begin for r in remaps if r.skip}
        state = dict(authority=authority, request_id=job.command.request_id,
                     selected_tokens=plan.total_window_tokens, assembly_arena='pressure',
                     pressure_stage=job.command.pressure_stage,
                     # Promised refresh: the next forward waits at layer 0's guard.
                     promised_layer0=getattr(job, 'promised_lease', None) is not None)
        if job.command.pressure_stage == 'prefill' and getattr(controller, 'reuse_layer_views', False):
            state['pressure_parent_assembler'] = self
        if job.command.pressure_stage == 'prefill' and getattr(controller, 'resident_values', False):
            source = job.layerwise_pressure_value_sources
            if source[:2] != (job.command.request_id, job.history):
                raise ValueError('Resident pressure V belongs to another accepted request')
            state['pressure_value_sources'] = source
            state['pressure_value_metadata_owners'] = self._metadata_owners
        started = time.perf_counter(); uploaded = chunks = packed_chunks = paired_chunks = 0
        inline_chunks = prepared_chunks = source_blocks = resident_v_tokens = 0
        native_paired_chunks = 0
        parent_results = None
        parent_timings = {}
        reader = None
        pipeline = getattr(controller, 'pipeline', None) if job.command.pressure_stage == 'prefill' else None
        # A promised (already acknowledged) lease binds its plan and commits
        # host maps first: the next forward may already be waiting on layer 0.
        early_lease = pipeline is not None and (_settings.get('KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE') == '1' or
                                                getattr(job, 'promised_lease', None) is not None)
        if getattr(job, 'promised_lease', None) is not None and not parent_staging:
            job.promise_commit_after_layer0 = True
        _r0 = time.perf_counter()
        # Opt-in: publish the prepared plan (ACK-able lease) before reading-plan
        # preparation for all layers; per-layer guards still order every use.
        lease = pipeline.begin(job, plan, names) if early_lease else None
        if early_lease:
            # Give the engine thread the GIL to dispatch the published plan before
            # reader preparation (long native calls) monopolizes this producer.
            # Pair with KVMEM_LAYERWISE_PRESSURE_IDLE_ACK_WAIT_MS so an idle
            # engine is already waiting at the schedule boundary.
            yield_ms = float(_settings.get('KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE_YIELD_MS', '0'))
            if not 0 <= yield_ms <= 50:
                raise ValueError('KVMEM_LAYERWISE_PRESSURE_EARLY_LEASE_YIELD_MS must be in [0, 50]')
            if yield_ms:
                deadline = time.perf_counter() + yield_ms / 1000
                # Wait for the whole ACK (job.done is set last by dispatch), not
                # only the host-map publication, so dispatch runs uncontended.
                while time.perf_counter() < deadline and not getattr(job, 'done', False):
                    time.sleep(0.0003)
        _r1 = time.perf_counter()
        if getattr(job, 'promised_lease', None) is not None:
            # Deferred read barrier of a promised refresh (async_refresh._assemble).
            authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        _rb = time.perf_counter()
        if (job.command.pressure_stage == 'prefill' and getattr(controller, 'prepared_reads', False) and
                getattr(controller.query, 'paired_reads', False)):
            from .pressure_reader import prepare_pressure_reader
            reader = prepare_pressure_reader(authority, plan, chosen)
        _r2 = time.perf_counter()
        from ..core.publication_trace import emit as _trace
        _trace('assembly_setup', rid=job.command.request_id, early_lease=early_lease,
               remap_checks=_v1-_v0, validate_destinations=_v2-_v1, pre_lease=_r0-_v2,
               lease_begin=_r1-_r0, seal_barrier=_rb-_r1, reader_prepare=_r2-_rb)
        with (reader if reader is not None else nullcontext()):
            if reader is not None:
                state['prepared_pressure_reader'] = reader
                source_blocks = reader.source_blocks
            if not early_lease:
                lease = pipeline.begin(job, plan, names) if pipeline is not None else None
            if parent_staging:
                if getattr(job, 'layerwise_select_layer', None) is not None:
                    raise ValueError('Parent staging needs every layer selection before uploads')
                from .pressure_parent_staging import assemble_parent_pressure_layers
                parent_results = assemble_parent_pressure_layers(
                    self, controller, state, plan, chosen, names, skip)
            select_next = getattr(job, 'layerwise_select_layer', None)
            from .pressure_reader import reader_extend
            extend_reader = select_next is not None and reader_extend()
            reader_extend_misses = 0
            fast_future = getattr(job, 'layer0_fast', None)
            fast_result = None
            fast_shadow = None
            if getattr(job, 'promised_lease', None) is not None and job.command.pressure_stage == 'prefill':
                from .layer0_fast import install
                fast_shadow = install(controller.runner)
            for index, name in enumerate(names):
                if (select_next is not None and len(chosen) <= index and
                        index < getattr(job, 'layerwise_layers_total', 0)):
                    # Layer-0-first: this layer's frozen-input selection runs
                    # just before its upload, while the GPU runs earlier layers.
                    select_next(index)
                    if reader is not None and extend_reader:
                        # Same sources and bytes; avoids the per-chunk read path.
                        extended = reader.extend(chosen[index])
                        reader_extend_misses += not extended
                records = chosen[index] if index < len(chosen) else chosen[0]
                if index == 0 and fast_future is not None:
                    fast_result = fast_future.result()
                if fast_result is not None and index < fast_result.published:
                    # KVMEM_LAYER0_FAST already assembled and published this layer
                    # from the GPU shadow; it must equal this unchanged selection.
                    from .layer0_fast import install as _layer0_fast
                    _layer0_fast(controller.runner).verify(fast_result, index, records,
                        skip if index == 0 else (), job.command.request_id, name)
                    stats = dict(uploaded_bytes=0, chunks=0)
                else:
                    stats = (parent_results[index] if parent_results is not None else
                        controller.query.assemble_layer_pinned(state, index, records,
                        layer_name=name, skip_history_begins=skip if index == 0 or index >= len(chosen) else ())
                        )
                    if lease is not None:
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(controller.runner.device))
                        lease.publish_layer(index, event)
                    if fast_shadow is not None and index < len(chosen):
                        # The fast path did not produce this layer: mirror the window it
                        # just committed while the plan (and reader) are still active.
                        fast_shadow.rebuild_layer(job.command.request_id, index, authority, records,
                                                  reader, len(chosen[0]), fast_chunk_blocks(controller))
                if lease is not None and index == 0 and getattr(job, 'promise_commit_after_layer0', False):
                    from .pressure_promise import commit
                    commit(pipeline, job, lease)
                uploaded += stats['uploaded_bytes']; chunks += stats['chunks']
                packed_chunks += stats.get('packed_chunks',0)
                paired_chunks += stats.get('paired_read_chunks',0)
                inline_chunks += stats.get('inline_read_chunks',0)
                prepared_chunks += stats.get('prepared_read_chunks',0)
                resident_v_tokens += stats.get('resident_v_reused_tokens',0)
                native_paired_chunks += stats.get('native_paired_gather_chunks',0)
                for key,value in stats.items():
                    if key.startswith('parent_staging_') and key.endswith('_seconds'):
                        parent_timings[key] = parent_timings.get(key,0.)+value
        done = torch.cuda.Event(); done.record(torch.cuda.current_stream(controller.runner.device))
        if lease is not None:
            lease.final_event = done
        job.layerwise_single_assembly_done = True
        deferred = getattr(job, 'layerwise_prepared_commit', None)
        if getattr(deferred, 'ready', True) is False:
            # Opt-in deferred maps: every layer upload is already submitted, so
            # building the 17-layer commit maps here never delays the plan/ACK.
            deferred.value()
        if remaps is None:
            rebuilt, skipped = rebuilt_tokens, skipped_tokens
        else:
            rebuilt = sum(r.n_tokens for r in remaps if full_rebuild or not r.skip)
            skipped = 0 if full_rebuild else sum(r.n_tokens for r in remaps if r.skip)
            partial_tokens = sum(r.n_tokens for r in remaps if not r.skip and r.n_tokens != 32)
        return AssemblyResult(done, (self, plan, authority), rebuilt, skipped, chunks,
            time.perf_counter()-started, uploaded,
            dict(layerwise_single_assembly=True, layerwise_assembled_layers=len(names),
                 layerwise_reused_layer_views='pressure_parent_assembler' in state,
                 layerwise_redundant_rebuilds=0,layerwise_packed_chunks=packed_chunks,
                 layerwise_paired_read_chunks=paired_chunks,
                  layerwise_inline_read_chunks=inline_chunks,
                  layerwise_prepared_read_chunks=prepared_chunks,
                  layerwise_prepared_source_blocks=source_blocks,
                  layerwise_resident_v_reused_tokens=resident_v_tokens,
                  layerwise_native_paired_gather_chunks=native_paired_chunks,
                  **(dict(layer0_fast_published=(fast_result.published if fast_result is not None else 0),
                          layer0_fast_reason=(getattr(fast_result, 'reason', '') if fast_result is not None
                                              else 'declined:' + str(getattr(job, 'layer0_fast_declined', 'not_launched'))),
                          layer0_fast_timings={k: round(1e3*v, 2) for k, v in getattr(fast_result, 'timings', ())})
                     if _settings.get('KVMEM_LAYER0_FAST') == '1' else {}),
                  **(dict(layerwise_reader_extended_blocks=getattr(reader, 'extended_blocks', 0),
                          layerwise_reader_extend_misses=reader_extend_misses)
                     if reader is not None and extend_reader else {}),
                  **(dict(reader.read_stats) if reader is not None and
                     getattr(reader,'paired_gather',False) else {}),
                  **(dict(layerwise_parent_staging_requested=True,
                     layerwise_parent_staging=parent_results is not None,
                     layerwise_parent_staging_fallback_reason=state.get('pressure_parent_staging_fallback_reason'),
                     layerwise_parent_chunk_blocks=(parent_results[0]['chunk_blocks'] if parent_results else None),
                     layerwise_parent_slot_bytes=self.staging.slot_bytes,
                     layerwise_parent_scratch_scope='Existing parent two-slot pool replaces child pool; live pressure scratch may be larger; no Active/Gen change',
                     **parent_timings) if parent_staging else {})),
            partial_tokens=partial_tokens)

    assembler._assemble = MethodType(assemble, assembler)
    # close() must remove this instance-bound override after its CUDA fence.
    # Otherwise assembler -> bound method -> assembler keeps the job, selected
    # records and tensor views alive until a process-wide cyclic-GC scan.
    assembler._layerwise_pressure_assembly_override = True
