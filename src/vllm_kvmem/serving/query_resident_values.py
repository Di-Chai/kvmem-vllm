"""Query-local mixed resident-V reuse; raw K always follows the archive path.

The initial global window is common to all target layers. A target layer is
untouched until its own fresh Query selects a replacement. Snapshot every reused
V source before any destination write, including cycles across transfer chunks.
No additional KV budget is allocated: the existing bounded V scratch is required.
"""
from dataclasses import dataclass, field

from ..core.memory import Phase


@dataclass
class QueryResidentSources:
    authority: object
    memory: object
    record_table: dict
    archive_namespace: str
    request_id: str
    request_generation: int
    history_end: int
    selected_tokens: int
    records: tuple
    sources: dict
    canonical_keys: frozenset
    value_proof: dict
    used_layers: set = field(default_factory=set)


def _identity(record):
    return record.handle.slot, record.handle.generation


def _query_scope(query, state):
    from ..adapters.original_rope import enabled as original_rope
    authority = state['authority']
    rid = state['request_id']
    runtime = getattr(query.runner, '_kvmem_capture_runtime', None)
    if (state.get('assembly_arena') == 'pressure' or
            authority.memory.phase is not Phase.READY or
            state.get('no_probe') or getattr(query, 'no_probe', False) or
            getattr(query, 'mtp_selection', 'global') != 'global' or
            'span' not in state or 'next_layer' not in state):
        raise ValueError('Resident V requires the public Query and its committed global window')
    # A restored P prefix deliberately retains its original archive namespace.
    # SessionWorker.publish_restore instead publishes this same authority under
    # the new native request ID. Prove that live binding; never rename the
    # archive or assume the two identities must match.
    if (getattr(runtime, 'authorities', {}).get(rid) is not authority or
            not isinstance(authority.memory.request_id, str) or not authority.memory.request_id):
        raise ValueError('Resident Query has no live native request-to-authority binding')
    if original_rope():
        raise ValueError('Query resident V is not qualified for original RoPE')
    generation = getattr(query.runner, '_kvmem_request_generations', {}).get(rid)
    if generation is None:
        raise ValueError('Resident Query has no live native request generation')
    if authority.history_end != state['history']:
        raise ValueError('Resident Query history changed')
    return authority, rid, generation


def _global_value_provenance(query, state, sources, global_assembly):
    """Certify only V actually read from archive by the committed global job.

The shared native working map describes layer zero before a global refresh;
its resident-copy source positions do not certify other layers' V. Even a
correct native-forward V can have a different capture/FP8 history. Neither
skipped nor previously resident values are eligible for this candidate.
"""
    import torch
    from .window_assembly import AssemblyResult, NativeWindowAssembler

    if not isinstance(global_assembly, tuple) or len(global_assembly) != 3:
        raise ValueError('Resident Query requires the actual committed global assembly proof')
    assembler, result, plan = global_assembly
    authority, rid = state['authority'], state['request_id']
    if (not isinstance(assembler, NativeWindowAssembler) or not isinstance(result, AssemblyResult) or
            not assembler.closed or getattr(assembler, '_runner_binding', None) != (query.runner, rid) or
            not result.event.query() or plan.total_window_tokens != state['selected_tokens'] or
            any(not any(owner is item for item in result.owners) for owner in (assembler, plan, authority))):
        raise ValueError('Resident Query global assembly is uncommitted or belongs to another owner')
    remaps = tuple(plan.remaps)
    by_slot = {slot: (key, source) for key, source in sources.items() for slot in (key[0],)}
    seen, proofs = set(), {}
    rebuilt = skipped = reused = 0
    for remap in remaps:
        pair = by_slot.get(remap.block_id)
        if pair is None or remap.block_id in seen:
            raise ValueError('Global assembly proof contains a foreign or duplicated record')
        key, (_, _, begin, count, destination) = pair
        if remap.n_tokens != count or remap.to_base != destination:
            raise ValueError('Global assembly remap differs from its committed native window')
        seen.add(remap.block_id)
        skipped += count if remap.skip else 0
        rebuilt += 0 if remap.skip else count
        reused += count if not remap.skip and remap.working_k_resident else 0
        proofs[key] = dict(history_begin=begin, count=count, dense_position=destination,
            skipped=bool(remap.skip), working_k_resident=bool(remap.working_k_resident),
            raw_region=(0, authority.raw_record_bytes),
            value_region=(authority.raw_record_bytes, authority.value_record_bytes))
    if (len(seen) != len(sources) or rebuilt != result.rebuilt_tokens or skipped != result.skipped_tokens or
            rebuilt + skipped != state['selected_tokens']):
        raise ValueError('Global assembly token accounting differs from its committed native window')
    full = all(source[3] == 32 for source in sources.values())
    supported = (full and assembler.reuse_resident_v and assembler.direct_pinned_reads and
        assembler.raw_cache is None and authority.raw_element_bytes == 1 and
        authority.raw_dtype == assembler.raw_dtype == torch.float8_e4m3fn and
        authority.shape == (len(assembler.layers), 32, *authority.shape[2:]) and
        authority.raw_record_bytes == authority.value_record_bytes ==
            authority.shape[0]*32*authority.shape[2]*authority.shape[3] and
        result.split_tokens == rebuilt and result.partial_tokens == 0)
    reason = None if supported else 'unsupported_global_value_provenance'
    if supported:
        expected_bytes = (2*rebuilt-reused)*authority.shape[0]*authority.shape[2]*authority.shape[3]
        if (result.resident_v_reused_tokens != reused or result.raw_cache_hit_tokens != 0 or
                result.uploaded_bytes != expected_bytes):
            raise ValueError('Global assembly byte accounting does not prove canonical archive V')
    canonical = frozenset(key for key, row in proofs.items()
        if supported and not row['skipped'] and not row['working_k_resident'])
    for key, row in proofs.items():
        row['canonical_archive_value'] = key in canonical
    return canonical, dict(scope='initial_global_archive_uploaded_only',
        route='full_fp8_split_no_raw_cache' if supported else None,
        fallback_reason=reason, canonical_blocks=len(canonical),
        uncertified_blocks=len(sources)-len(canonical), records=proofs)


def bind_query_resident_sources(query, state, working_records, *, global_assembly=None):
    """Call once immediately after the original global Query refresh commits.

working_records is the native store's actual ordered five-column working map:
(slot, generation, history_begin, count, dense_position). It is not reconstructed
from a set or assumed to share history order.
"""
    authority, rid, generation = _query_scope(query, state)
    if 'query_resident_sources' in state:
        raise ValueError('Resident Query source window was already bound')
    records = state['records']
    if not isinstance(records, tuple):
        raise ValueError('Resident Query requires its immutable record tuple')
    cursor, sources, history = 0, {}, set()
    for row in working_records:
        if len(row) != 5:
            raise ValueError('Invalid native Query source map')
        slot, owner_generation, begin, count, position = row
        key = slot, owner_generation
        record = authority.records.get(key)
        if (key in sources or begin in history or position != cursor or begin < 0 or
                not 0 < count <= 32 or begin+count > authority.history_end or
                position+count > state['selected_tokens'] or
                record is None or _identity(record) != key or
                record.history_begin != begin or record.count != count):
            raise ValueError('Resident Query source generation or dense extent changed')
        sources[key] = record, record.handle, begin, count, position
        history.add(begin)
        cursor += count
    if (cursor != state['selected_tokens'] or
            {slot for slot, _ in sources} != set(state['global_slots'])):
        raise ValueError('Resident Query source map does not cover its global selection')
    canonical, proof = _global_value_provenance(query, state, sources, global_assembly)
    binding = QueryResidentSources(authority, authority.memory, authority.records,
                                   authority.memory.request_id,
                                   rid, generation, authority.history_end,
                                   cursor, records, sources, canonical, proof)
    state['query_resident_sources'] = binding
    return binding


def plan_query_resident_values(query, state, layer_index, chosen):
    """Return original destination indices and immutable resident source bases.

Unsupported complete/partial geometry explicitly declines before GPU writes.
Stale ownership is an error, never a fallback to another source generation.
"""
    authority, rid, generation = _query_scope(query, state)
    binding = state.get('query_resident_sources')
    if (binding is None or binding.authority is not authority or
            binding.memory is not authority.memory or binding.record_table is not authority.records or
            binding.archive_namespace != authority.memory.request_id or
            binding.request_id != rid or
            binding.request_generation != generation or binding.history_end != authority.history_end or
            binding.selected_tokens != state['selected_tokens'] or binding.records is not state['records'] or
            type(layer_index) is not int or not 0 <= layer_index < len(query.names) or
            state['next_layer'] != layer_index or layer_index in binding.used_layers):
        raise ValueError('Resident Query source window or layer is stale')
    chosen = tuple(chosen)
    if sum(record.count for record in chosen) != binding.selected_tokens:
        raise ValueError('Resident Query selection changed its native causal length')
    seen, resident, missing, bases = set(), [], [], []
    known_resident = 0
    reason = None
    for index, record in enumerate(chosen):
        key = _identity(record)
        if (key in seen or authority.records.get(key) is not record or
                record.history_begin < 0 or record.history_begin+record.count > binding.history_end):
            raise ValueError('Resident Query selected an aliased or foreign generation')
        seen.add(key)
        if record.count != 32:
            reason = 'partial_selected_record'
        elif not record.archived and reason is None:
            reason = 'unpublished_selected_record'
        source = binding.sources.get(key)
        if source is None:
            missing.append(index)
        else:
            known_resident += 1
            old, handle, begin, count, position = source
            if (old is not record or record.handle is not handle or begin != record.history_begin or
                    count != record.count or position < 0 or position+count > binding.selected_tokens):
                raise ValueError('Resident Query record changed after its global installation')
            if count != 32 or position % 32:
                reason = reason or 'partial_source_window'
            if key in binding.canonical_keys:
                resident.append(index)
                bases.append(position)
            else:
                missing.append(index)
    if reason is None and not resident:
        reason = (binding.value_proof['fallback_reason'] or
                  ('no_canonical_resident_values' if known_resident else 'no_resident_values'))
    return chosen, tuple(resident), tuple(missing), tuple(bases), reason


def assemble_query_resident_values(query, state, layer_index, chosen, assembler,
                                   chunk_blocks, *, reader=None):
    """Use an already claimed one-layer assembler; its caller owns cleanup.

reader follows read_packed_records(authority, handles, destination=..., region=
... or regions=...). A future Query-scoped source cache can supply that callback
without changing the V copy plan or its byte counts. None means explicit legacy
fallback; state['query_resident_fallback_reason'] records why.
"""
    import torch
    from .packed_assembly import read_packed_records
    from .query_stage_profile import no_stage
    from .split_assembly import assemble_split_layer, gather_resident_values

    authority = state['authority']
    staging = assembler.staging
    if (not getattr(query, 'packed_assembly', False) or
            authority.raw_element_bytes != 1 or authority.shape[1] != 32 or
            len(assembler.layers) != 1 or staging.owner is not assembler or
            type(chunk_blocks) is not int or not 1 <= chunk_blocks <= 256):
        raise ValueError('Resident Query requires the original bounded packed FP8 assembler')
    chosen, resident, missing, source_bases, reason = plan_query_resident_values(
        query, state, layer_index, chosen)
    state['query_resident_sources'].used_layers.add(layer_index)
    if reason is not None:
        state['query_resident_fallback_reason'] = reason
        return None
    state.pop('query_resident_fallback_reason', None)
    options = query.runner._kvmem_capture_runtime.options
    if not options.reuse_resident_v:
        raise ValueError('Resident Query V scratch was not reserved by the original configuration')
    layer = assembler.layers[0]
    heads, dim = authority.shape[2:]
    row_bytes = 32*heads*dim
    total_raw = authority.shape[0]*row_bytes
    capacity = layer.kernel_page_table.numel()*layer.value.shape[1]
    if (max(base+32 for base in source_bases) > capacity or state['selected_tokens'] > capacity or
            state['selected_tokens'] > layer.cos_sin.shape[0] or
            len(resident)*row_bytes > staging.value_capacity_bytes or
            2*row_bytes*chunk_blocks > staging.slot_bytes):
        raise ValueError('Resident Query exceeds retained pages, RoPE or original staging capacity')
    reader = reader or read_packed_records
    profile = state.get('query_stage_profile')
    stage = profile.stage if profile is not None else no_stage
    owners = state.setdefault('query_resident_metadata_owners', [])
    metadata_start = len(assembler._metadata_owners)
    uploaded = chunks = reused_chunks = paired_chunks = 0
    inline_chunks = prepared_chunks = native_paired_chunks = 0
    read_stats = {}
    try:
        with stage('resident_v_snapshot', gpu=True, reused_tokens=32*len(resident)):
            bases = assembler._upload_metadata(source_bases)
            destination = staging.value_destination(len(resident)*row_bytes)
            values = gather_resident_values((layer,), bases, 32, destination, tiled=True)
        # Every old V source is now protected from arbitrary write permutations.
        # Grouping changes byte transport only; each original destination remains
        # its original index in chosen, and raw K keeps the same RoPE/FP8 kernel.
        resident_indices = {index: i for i,index in enumerate(resident)}
        for reuse, indices in ((True,resident),(False,missing)):
            if not indices:
                continue
            count = len(indices)
            with stage('assembly_groups_prepare', records=count):
                metadata = assembler._upload_metadata(
                    [index*32 for index in indices] +
                    [i % chunk_blocks for i in range(count)] +
                    [resident_indices[index] if reuse else i % chunk_blocks
                     for i,index in enumerate(indices)])
            for begin in range(0,count,chunk_blocks):
                current = indices[begin:begin+chunk_blocks]
                n = len(current)
                half = n*row_bytes
                size = half if reuse else 2*half
                gpu,ready,slot = staging.destination(size)
                handles = [chosen[index].handle for index in current]
                def fill(host):
                    nonlocal inline_chunks, prepared_chunks, native_paired_chunks
                    kwargs = ({'region': (layer_index*row_bytes,row_bytes)} if reuse else
                              {'regions': ((layer_index*row_bytes,row_bytes),
                                           (total_raw+layer_index*row_bytes,row_bytes))})
                    _, stats = reader(authority,handles,destination=host,**kwargs)
                    inline_chunks += int(stats.get('inline_cpu_region_read',False))
                    prepared_chunks += int(stats.get('prepared_cpu_region_read',False))
                    native_paired_chunks += int(stats.get('paired_native_gather',False))
                    for name,value in stats.items():
                        if type(value) in (int,float):
                            read_stats[name] = (max(read_stats.get(name,0),value)
                                if name == 'query_reader_source_blocks' else
                                read_stats.get(name,0)+value)
                transfer_profile = {'profile': profile} if profile is not None else {}
                with stage('reader_upload_total',transfer_bytes=size):
                    transfer = assembler.transfers.upload_from_reader(size,gpu,fill,ready=ready,**transfer_profile)
                if transfer is None:
                    with stage('pinned_slot_retry_wait'):
                        next(iter(assembler.transfers.pending.values())).event.synchronize()
                    with stage('reader_upload_total',transfer_bytes=size):
                        transfer = assembler.transfers.upload_from_reader(size,gpu,fill,ready=ready,**transfer_profile)
                if transfer is None:
                    raise RuntimeError('Resident Query pinned transfer made no progress')
                with stage('consumer_wait_for_h2d',gpu=True):
                    torch.cuda.current_stream(query.runner.device).wait_event(transfer.event)
                end = begin+n
                with stage('layout_and_rebuild',gpu=True,tokens=32*n):
                    assemble_split_layer(gpu[:half],metadata[count+begin:count+end],
                        values if reuse else gpu[half:2*half],metadata[2*count+begin:2*count+end],
                        metadata[begin:end],layer,layer_index=0,layers=1,block_tokens=32,
                        raw_record_bytes=row_bytes,value_record_bytes=row_bytes,raw_dtype=torch.float8_e4m3fn)
                staging.consumed(slot)
                uploaded += size
                chunks += 1
                reused_chunks += int(reuse)
                paired_chunks += int(not reuse)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(query.runner.device))
        staging.value_ready = done
        return dict(uploaded_bytes=uploaded,chunks=chunks,packed_chunks=chunks,
            chunk_blocks=chunk_blocks,staging_slot_bytes=staging.slot_bytes,
            paired_read_chunks=paired_chunks,inline_read_chunks=inline_chunks,
            prepared_read_chunks=prepared_chunks,native_paired_gather_chunks=native_paired_chunks,
            query_resident_v=True,assembly_read_stats=read_stats,
            resident_v_reused_tokens=32*len(resident),resident_v_missing_tokens=32*len(missing),
            resident_v_snapshot_bytes=len(resident)*row_bytes,
            resident_v_avoided_archive_bytes=len(resident)*row_bytes,
            resident_v_uncertified_tokens=sum(record.count for record in chosen
                if _identity(record) in state['query_resident_sources'].sources and
                _identity(record) not in state['query_resident_sources'].canonical_keys),
            query_resident_source_scope=state['query_resident_sources'].value_proof['scope'],
            resident_v_raw_only_chunks=reused_chunks,query_resident_fallback_reason=None)
    finally:
        # The Query's existing final CUDA fence retires these small host/device
        # addressing owners; assembler.close() must not free them while in flight.
        owners.extend(assembler._metadata_owners[metadata_start:])
