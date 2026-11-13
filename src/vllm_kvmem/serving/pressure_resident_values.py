"""Reuse unchanged native V bytes during a suspended prefill transaction.

Uses the existing bounded resident-V staging and gather/split CUDA kernels.
All old V sources are snapshotted before the first destination write, including
cycles crossing chunk boundaries. Keys retain the original archive/RoPE path.
Partial or nonresident layouts keep the original paired-read implementation.
"""


def resident_bases(chosen, mapping, local_tokens, block_tokens=32):
    by_begin={}
    cursor=0
    for _,_,begin,count,position in mapping:
        if (position!=cursor or count<=0 or count>block_tokens or begin in by_begin or
                begin<0 or position+count>local_tokens):
            raise ValueError('Resident pressure source map is not a unique dense accepted window')
        by_begin[begin]=(count,position)
        cursor+=count
    if cursor!=local_tokens:
        raise ValueError('Resident pressure map does not cover the accepted native window')
    if len({r.history_begin for r in chosen})!=len(chosen):
        raise ValueError('Repeated selected pressure source')
    bases=[]
    for r in chosen:
        source=by_begin.get(r.history_begin)
        if (r.count!=block_tokens or not r.archived or source is None or
                source[0]!=r.count or source[1]%block_tokens):
            return None
        bases.append(source[1])
    return bases


def assemble_resident_values(query,state,index,chosen,assembler,groups,total,row_bytes,chunk_blocks):
    import torch
    from ..core.memory import Phase
    from .packed_assembly import read_packed_records
    from .split_assembly import gather_resident_values,assemble_split_layer
    from ..adapters.original_rope import enabled as original_rope
    authority=state['authority'];staging=assembler.staging;layer=assembler.layers[0]
    rid,history,local,maps=state['pressure_value_sources']
    from .pressure_key_reuse import enabled as gpu_reuse
    reuse=gpu_reuse()
    # A pipelined refresh (opt-in GPU reuse only) publishes its host map at the
    # ACK; later layers then assemble under their per-layer guards while the
    # next forward may append accepted tokens past the old window.
    pipelined=reuse and authority.memory._layerwise_materialization is not None
    if (state.get('assembly_arena')!='pressure' or state.get('pressure_stage')!='prefill' or
            rid!=state['request_id'] or rid!=authority.memory.request_id or
            (authority.history_end<history if pipelined else authority.history_end!=history) or
            (not pipelined and authority.memory.phase is not Phase.ASSEMBLING) or
            staging.owner is not assembler):
        raise ValueError('Resident V reuse escaped its suspended prefill transaction')
    if original_rope():raise ValueError('Resident pressure V currently requires qualified compact positions')
    if not query.packed_assembly or not maps:
        return None
    mapping=maps[index if index<len(maps) else 0]  # Draft uses the original layer-zero selection.
    if reuse:
        from .pressure_key_reuse import resident_bases as reuse_bases
        sources=reuse_bases(chosen,mapping,local)      # native when available, same result
    else:
        sources=resident_bases(chosen,mapping,local)
    if sources is None:
        if reuse:
            from .pressure_key_reuse import record_stats
            record_stats(query.pressure, fallback=1)
        return None
    if total!=state['selected_tokens'] or total!=32*len(sources):
        raise ValueError('Resident pressure selection changed the native destination extent')
    options=query.runner._kvmem_capture_runtime.options
    if not options.reuse_resident_v:
        raise ValueError('Resident V reuse was not reserved by the original profile')
    capacity=layer.kernel_page_table.numel()*layer.value.shape[1]
    if max((p+32 for p in sources),default=0)>capacity or total>capacity:
        raise ValueError('Resident V source/destination exceeds retained native pages')
    if reuse:
        # Opt-in KVMEM_LAYERWISE_PRESSURE_GPU_REUSE: K is reused on the GPU too
        # (delta re-rotation, bounded by a per-block reuse count).
        from .pressure_key_reuse import assemble_reused
        owners=state['pressure_value_metadata_owners']
        metadata_start=len(assembler._metadata_owners)
        try:
            return assemble_reused(query,state,index,chosen,sources,assembler,row_bytes,chunk_blocks)
        finally:
            owners.extend(assembler._metadata_owners[metadata_start:])
    destination=staging.value_destination(len(sources)*row_bytes)
    # The existing parent assembler retains metadata through its final fence.
    owners=state['pressure_value_metadata_owners']
    metadata_start=len(assembler._metadata_owners)
    try:
        source_bases=assembler._upload_metadata(sources)
        values=gather_resident_values((layer,),source_bases,32,destination)
        bases=assembler._upload_metadata(list(range(0,total,32)))
        indices=assembler._upload_metadata(list(range(max(len(sources),chunk_blocks))))
        uploaded=chunks=inline=0
        from .gil_gate import chunk_checkpoint
        for chunk_index,(cursor,group) in enumerate(groups):
            if chunk_index:
                chunk_checkpoint(state.get('request_id'))   # opt-in KVMEM_GIL_GATE_CHUNKS
            n=len(group);size=n*row_bytes
            if cursor%32 or any(r.count!=32 for r in group):
                raise ValueError('Resident pressure chunk has changed geometry')
            gpu,ready,slot=staging.destination(size)
            def fill(host):
                nonlocal inline
                _,stats=read_packed_records(authority,[r.handle for r in group],destination=host,
                    region=(index*row_bytes,row_bytes),
                    inline_cpu=bool(getattr(authority.archive,'memory_only',False)))
                inline+=int(stats.get('inline_cpu_region_read',False))
            transfer=assembler.transfers.upload_from_reader(size,gpu,fill,ready=ready)
            if transfer is None:
                next(iter(assembler.transfers.pending.values())).event.synchronize()
                transfer=assembler.transfers.upload_from_reader(size,gpu,fill,ready=ready)
            if transfer is None:raise RuntimeError('Resident pressure pinned transfer made no progress')
            torch.cuda.current_stream(query.runner.device).wait_event(transfer.event)
            offset=cursor//32
            assemble_split_layer(gpu,indices[:n],values,indices[offset:offset+n],bases[offset:offset+n],
                layer,layer_index=0,layers=1,block_tokens=32,
                raw_record_bytes=row_bytes,value_record_bytes=row_bytes,raw_dtype=torch.float8_e4m3fn)
            staging.consumed(slot);uploaded+=size;chunks+=1
        done=torch.cuda.Event();done.record(torch.cuda.current_stream(query.runner.device))
        staging.value_ready=done
        return dict(uploaded_bytes=uploaded,chunks=chunks,packed_chunks=chunks,chunk_blocks=chunk_blocks,
            staging_slot_bytes=staging.slot_bytes,paired_read_chunks=0,inline_read_chunks=inline,
            prepared_read_chunks=0,resident_v_reused_tokens=uploaded//(row_bytes//32),
            resident_v_snapshot_bytes=len(sources)*row_bytes)
    finally:
        owners.extend(assembler._metadata_owners[metadata_start:])
