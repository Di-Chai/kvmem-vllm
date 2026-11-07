"""Build a frozen layerwise P view once, inside the original restore transaction.

The ordinary path first imports a common view for all layers, then replaces
target layers 1..N-1. This candidate writes each final layer directly. Native
selection, source leases, failure fencing and publication remain authoritative.
"""
import time
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
    # The bound replacement closes over the original native plan. Leaving it
    # on parent makes parent -> bound method -> parent a cycle, retaining the
    # plan's checkpoint pins until an unrelated GC. An idle final owner can
    # then remain evicting indefinitely despite a completed CUDA fence.
    del assembler._assemble
    del assembler._direct_prefix_frame
    controller.query.emit('pressure_prefix_restored', request_id=request_id,
        actual_method='accumulated_attention_per_layer',fallback_reason=None,
        restore_backend='direct_all_layers',**checkpoint.layerwise_pressure.evidence())
