"""Build a frozen layerwise P view once, inside the original restore transaction.

The ordinary path first imports a common view for all layers, then replaces
target layers 1..N-1. This candidate writes each final layer directly. Native
selection, source leases, failure fencing and publication remain authoritative.
"""
import time
from contextlib import nullcontext
from types import MethodType

import torch

from .memory import Phase
from .layerwise_checkpoints import prepare_restore, _identity
from .window_assembly import AssemblyResult, _validate_destinations


def install_direct_restore(worker, request_id, checkpoint, assembler, plan):
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if controller is None or not getattr(controller.query, 'direct_prefix_restore', False):
        return False
    if (checkpoint.kind != 'P' or worker.bindings[request_id].pressure_restore or
            not worker.options.session_freeze_prompt or not controller.query.optimized):
        from .layerwise_checkpoints import sessions_enabled
        if sessions_enabled():
            return False        # session restores use restore_sync / restore_pressure_sync
        raise ValueError('Direct layerwise restore requires the frozen optimized P path')
    names = tuple(worker.runtime.names)
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Restore target/draft archive order changed')
    authority = checkpoint.authority
    if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
        raise ValueError('Direct restore needs the original active selection')
    by_slot = {r.handle.slot:r for r in authority.records.values()}
    planned = []
    for remap in sorted(plan.remaps, key=lambda r:r.to_base):
        record = by_slot[remap.block_id]
        if record.count != remap.n_tokens:
            raise ValueError('Restore plan changed original record lengths')
        planned.append((record.handle.slot,record.handle.generation,record.history_begin,
                        record.count,remap.to_base))
    chosen, frame, histories = prepare_restore(controller,request_id,authority,
        checkpoint.layerwise_pressure,worker.options,planned_records=planned)
    if (len(assembler.layers) != len(names) or
            authority.shape != (len(names),assembler.block_tokens,assembler.heads,assembler.dim) or
            authority.raw_dtype != assembler.raw_dtype or
            sum(r.count for r in chosen[0]) != plan.total_window_tokens):
        raise ValueError('Direct restore changed codec, physical layers or selected length')
    _validate_destinations(plan.remaps,by_slot,assembler.layers)
    # Only the _assemble body changes: NativeWindowAssembler.assemble still
    # owns all error fences, transaction failure, source release and close.
    def assemble_direct(parent, actual_authority, actual_plan):
        if actual_authority is not authority or actual_plan is not plan:
            raise ValueError('Direct restore crossed an authority/transaction')
        started = time.perf_counter()
        state = dict(authority=authority, request_id=request_id,
                     selected_tokens=plan.total_window_tokens, assembly_arena='layerwise')
        uploaded = chunks = 0
        for index, name in enumerate(names):
            records = chosen[index] if index < len(chosen) else chosen[0]
            stats = controller.query.assemble_layer_pinned(state,index,records,layer_name=name)
            uploaded += stats['uploaded_bytes']
            chunks += stats['chunks']
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(worker.runner.device))
        parent._direct_prefix_frame = (frame,histories,event)
        return AssemblyResult(event,(parent,authority,plan,checkpoint,chosen,frame,histories),
            plan.total_window_tokens,0,chunks,time.perf_counter()-started,uploaded,
            {'direct_layerwise_restore_layers':len(names)})
    assembler._assemble = MethodType(assemble_direct,assembler)
    return True


def finish_direct_restore(worker, request_id, checkpoint, assembler):
    controller = worker.runner._kvmem_layerwise_pressure
    authority = checkpoint.authority
    prepared = getattr(assembler, '_direct_prefix_frame', None)
    if prepared is None:
        raise ValueError('Direct restore has no unpublished frame')
    frame,histories,event = prepared
    if (authority.memory.phase is not Phase.READY or not event.query() or
            request_id in controller.maps or request_id in controller.histories or
            _identity(authority.memory._store.working_records()) != _identity(frame[1][0])):
        raise ValueError('Direct restore cannot publish an unfinished or foreign layer frame')
    controller.maps[request_id] = frame
    controller.histories[request_id] = histories
    # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: the restored per-layer windows come
    # from a held checkpoint; keep every layer's records out of retirement.
    from .layerwise_gc_holds import holds_of
    from .layerwise_gc_holds import wait_paused
    wait_paused(authority)
    holds_of(controller).protect(request_id, authority, frame[1])
    # The bound replacement closes over the original native plan. Leaving it
    # on parent makes parent -> bound method -> parent a cycle, retaining the
    # plan's checkpoint pins until an unrelated GC. An idle final owner can
    # then remain evicting indefinitely despite a completed CUDA fence.
    del assembler._assemble
    del assembler._direct_prefix_frame
    controller.query.emit('pressure_prefix_restored', request_id=request_id,
        actual_method='accumulated_attention_per_layer',fallback_reason=None,
        restore_backend='direct_all_layers',**checkpoint.layerwise_pressure.evidence())


def direct_session_restore():
    """Opt-in KVMEM_RESTORE_DIRECT_LAYERS=1: an asynchronous session swap-in
    (copy lane, KVMEM_LAYERWISE_SESSIONS) writes every layer's own window
    directly inside the background restore job, instead of a shared window
    into all layers followed by an engine-thread rebuild of layers 1..N-1
    (restore_sync). Each layer's bytes are read and uploaded once, and the
    engine thread only installs CPU state. Same records, packing, rounding,
    positions and order as restore_sync, so the restored pages are identical."""
    import os
    value = os.getenv('KVMEM_RESTORE_DIRECT_LAYERS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_RESTORE_DIRECT_LAYERS must be 0 or 1')
    return value == '1'


def prepared_restore_reads():
    """Opt-in KVMEM_RESTORE_PREPARED_READS=1 (with KVMEM_RESTORE_DIRECT_LAYERS): the
    direct swap-in reads through one prepared native source reader (as pressure
    refreshes do), inline and GIL-free, with parallel copies. Byte-identical."""
    import os
    value = os.getenv('KVMEM_RESTORE_PREPARED_READS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_RESTORE_PREPARED_READS must be 0 or 1')
    return value == '1'


def restore_gate_exempt():
    """Opt-in KVMEM_RESTORE_GATE_EXEMPT=1 (with KVMEM_RESTORE_DIRECT_LAYERS): the direct
    swap-in's per-layer assemblies are not parked by the GIL gate while the engine
    launches a step. Scheduling only: the same work in the same order."""
    import os
    value = os.getenv('KVMEM_RESTORE_GATE_EXEMPT', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_RESTORE_GATE_EXEMPT must be 0 or 1')
    return value == '1'


def restore_stage_profile():
    """Opt-in KVMEM_RESTORE_STAGE_PROFILE=1 (diagnostic): the direct swap-in's
    per-layer host stages and CUDA intervals, written to restore_stage_profile.jsonl
    by the engine thread once the restore completed."""
    import os
    value = os.getenv('KVMEM_RESTORE_STAGE_PROFILE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_RESTORE_STAGE_PROFILE must be 0 or 1')
    return value == '1'


def install_direct_session_restore(worker, request_id, checkpoint, assembler, plan):
    """Engine thread, before the background submit. Returns True when the
    restore job's assembler now writes each final layer directly."""
    if assembler is None or not direct_session_restore():
        return False
    from .layerwise_checkpoints import sessions_enabled
    from .tensor_parallel import active as tp_active
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if (controller is None or snapshot is None or not sessions_enabled() or tp_active() or
            worker.bindings[request_id].pressure_restore or
            getattr(worker.runner, '_kvmem_qsa_layers', None)):
        return False
    names = tuple(worker.runtime.names)
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Restore target/draft archive order changed')
    authority = checkpoint.authority
    if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
        raise ValueError('Direct restore needs the original active selection')
    by_slot = {r.handle.slot: r for r in authority.records.values()}
    planned = []
    for remap in sorted(plan.remaps, key=lambda r: r.to_base):
        record = by_slot[remap.block_id]
        if record.count != remap.n_tokens:
            raise ValueError('Restore plan changed original record lengths')
        planned.append((record.handle.slot, record.handle.generation, record.history_begin,
                        record.count, remap.to_base))
    chosen, frame, histories = prepare_restore(controller, request_id, authority, snapshot,
                                               worker.options, planned_records=planned)
    if (len(assembler.layers) != len(names) or
            authority.shape != (len(names), assembler.block_tokens, assembler.heads, assembler.dim) or
            authority.raw_dtype != assembler.raw_dtype or
            sum(r.count for r in chosen[0]) != plan.total_window_tokens):
        raise ValueError('Direct restore changed codec, physical layers or selected length')
    _validate_destinations(plan.remaps, by_slot, assembler.layers)
    device = worker.runner.device

    def assemble_direct(parent, actual_authority, actual_plan):
        # Background restore job, on its own stream. NativeWindowAssembler.assemble
        # still owns the transaction failure fence, source release and close.
        try:
            if actual_authority is not authority or actual_plan is not plan:
                raise ValueError('Direct restore crossed an authority/transaction')
            started = time.perf_counter()
            state = dict(authority=authority, request_id=request_id,
                         selected_tokens=plan.total_window_tokens, assembly_arena='refresh',
                         borrowed_staging=parent.staging, borrowed_layers=parent.layers,
                         borrowed_lease=getattr(parent, '_restore_lease', None),
                         restore_partial_operands={},
                         **({'restore_gate_exempt': True} if restore_gate_exempt() else {}))
            uploaded = chunks = 0
            if restore_stage_profile():
                from .query_stage_profile import QueryStageProfile
                output = getattr(worker.runner, '_kvmem_probe_output', None)
                profile = QueryStageProfile(output / 'restore_stage_profile.jsonl'
                                            if output is not None else None, request_id)
                state['query_stage_profile'] = parent._direct_session_profile = profile
            reader = None
            if prepared_restore_reads() and getattr(controller.query, 'paired_reads', False):
                # Opt-in: resolve, lease and export every layer's sources once
                # (the pressure refresh reader), then copy inline without the
                # per-chunk archive admission; the same bytes land in staging.
                from .pressure_reader import prepare_pressure_reader
                reader = prepare_pressure_reader(authority, plan, chosen)
            # The per-layer assemblies share this job's refresh scratch, which
            # only this producer thread uses; hand it over and take it back.
            parent.staging.release(parent)
            try:
                with (reader if reader is not None else nullcontext()):
                    if reader is not None:
                        state.update(prepared_pressure_reader=reader, restore_inline_reads=True,
                                     restore_parallel_reads=True)
                    for index, name in enumerate(names):
                        # The MTP draft follows layer zero, as restore_sync leaves it.
                        records = chosen[index] if index < len(chosen) else chosen[0]
                        stats = controller.query.assemble_layer_pinned(state, index, records,
                                                                      layer_name=name)
                        uploaded += stats['uploaded_bytes']
                        chunks += stats['chunks']
            finally:
                parent.staging.claim(parent)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            parent._direct_session_frame = (frame, histories, event)
            return AssemblyResult(event, (parent, authority, plan, checkpoint, chosen), plan.total_window_tokens, 0, chunks,
                time.perf_counter() - started, uploaded,
                {'direct_layerwise_restore_layers': len(names)})
        finally:
            # A bound method on its own instance is a cycle that would keep
            # the plan's checkpoint pins alive until an unrelated GC.
            parent.__dict__.pop('_assemble', None)
    assembler._assemble = MethodType(assemble_direct, assembler)
    return True


def finish_direct_session_restore(worker, request_id, checkpoint, assembler):
    """Engine thread, after the restore committed its selection: install the
    per-layer CPU state written by install_direct_session_restore's job.
    Returns False when the job did not use the direct path."""
    prepared = getattr(assembler, '_direct_session_frame', None) if assembler is not None else None
    if prepared is None:
        return False
    from .layerwise_checkpoints import install_restored, emit_prefix_evidence
    controller = worker.runner._kvmem_layerwise_pressure
    authority = checkpoint.authority
    frame, histories, event = prepared
    if (authority.memory.phase is not Phase.READY or not event.query() or
            request_id in controller.maps or request_id in controller.histories):
        raise ValueError('Direct restore cannot publish an unfinished or foreign layer frame')
    del assembler._direct_session_frame
    profile = assembler.__dict__.pop('_direct_session_profile', None)
    if profile is not None:
        profile.finish()        # diagnostic; every event completed before the commit
    install_restored(controller, request_id, authority, frame, histories, checkpoint.layerwise_pressure)
    emit_prefix_evidence(controller.query, 'pressure_prefix_restored', checkpoint.layerwise_pressure,
        request_id=request_id, actual_method='accumulated_attention_per_layer', fallback_reason=None,
        restore_backend='direct_all_layers_async')
    return True
