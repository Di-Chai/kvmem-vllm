"""Per-owner logical cursors sharing one bounded physical DMA allocation.

No copied KV, new DMA slot, changed logical boundary or enlarged score bank.
The parent CaptureRuntime packet owns release; children only borrow views.
"""
import os
from types import SimpleNamespace

import torch


def owner_chunk_limit(owners, native_rows, logical_tokens, maximum):
    if any(type(v) is not int or v <= 0 for v in (owners,native_rows,logical_tokens,maximum)):
        raise ValueError('Positive integral physical cohort limits required')
    if logical_tokens not in (1024,2048):
        raise ValueError('Preserve a supported original logical chunk size')
    # Each original logical chunk samples at most one row per 32 tokens. The
    # native reference already uses this capped sampling rule. Reserve for
    # every live/waiting owner, even if publication skips some in this step.
    return min(maximum,native_rows//(owners*(logical_tokens//32)))


def owner_chunk_allocations(owners, native_rows, logical_tokens, maximum, *, rotation=0):
    """Distribute complete chunks within the unchanged native score-row bank.

    Equal flooring leaves two 1K chunks unused for six owners and 448 rows.
    Preserve every base reservation, distributing only the integral remainder.
    Rotation avoids permanently favoring the first running request. A blocked
    admission can waste its reserved rows, but cannot oversubscribe the bank.
    """
    base = owner_chunk_limit(owners, native_rows, logical_tokens, maximum)
    if type(rotation) is not int or rotation < 0:
        raise ValueError('Nonnegative integral owner rotation required')
    if base == 0:
        return (0,) * owners
    total = min(owners * maximum, native_rows // (logical_tokens // 32))
    result = [base] * owners
    for i in range(total - base * owners):
        result[(rotation + i) % owners] += 1
    return tuple(result)


def coalesce_with_decode():
    """Opt-in: logical owners may share a step with ordinary decode rows."""
    value=os.getenv('KVMEM_PREFILL_COALESCE_WITH_DECODE','0')
    if value not in ('0','1'):
        raise ValueError('KVMEM_PREFILL_COALESCE_WITH_DECODE must be 0 or 1')
    return value=='1'


def owners_prefilling(batch, owners):
    """All rows by default; only the logical owners' rows when mixing is enabled."""
    if not coalesce_with_decode():
        return all(batch.is_prefilling_np)
    ids=list(batch.req_ids)
    return all(bool(batch.is_prefilling_np[ids.index(rid)]) for rid in owners)


def capture_owner_plans(runtime, action_inputs, batch):
    groups=getattr(action_inputs,'logical_groups',{})
    slices=getattr(action_inputs,'logical_slices',{})
    if not groups and not slices:return {}
    if (not runtime.options.prefill_physical_multi_owner or
            set(groups)!=set(slices) or not set(groups).issubset(batch.req_ids) or
            not owners_prefilling(batch, groups)):
        raise RuntimeError('Invalid logical prefill owner cohort')
    result={}
    # Reuse the original complete single-owner plan/Query/coordinate validator.
    for rid in groups:
        i=batch.req_ids.index(rid)
        owner=SimpleNamespace(num_reqs=1,req_ids=(rid,),is_prefilling_np=(True,),
            num_computed_tokens_np=(batch.num_computed_tokens_np[i],),
            query_start_loc_np=tuple(batch.query_start_loc_np[i:i+2]))
        inputs=SimpleNamespace(logical_groups={rid:groups[rid]},logical_slices={rid:slices[rid]},
                               validation=getattr(action_inputs,'validation',None))
        result[rid]=runtime._logical_capture_inputs(inputs,owner)
    return result


def attach_owner_views(packet, plans):
    from .capture_runtime import CapturedPacket
    for i,rid in enumerate(packet.request_ids):
        if rid not in plans:continue
        group,parts=plans[rid]
        packet.logical_owners[i]=CapturedPacket((rid,),group.count,packet.download,
            confirmed=packet.confirmed,request_tokens=(group.count,),lane=packet.lane,
            source_generation=packet.source_generation,logical_group=group,logical_slices=parts,
            action_inputs={rid:packet.action_inputs[rid]})


def bind_owner_payload(runtime, parent, index):
    """Validate a fenced shared header; borrow a strided view for one owner."""
    child=parent.logical_owners[index]
    if child.logical_payload is not None:return child
    if parent.source_released or child.download is not parent.download:
        raise RuntimeError('Logical owner lost its shared DMA lease')
    # Caller has already queried/synchronized the original packet fence.
    n,t=len(parent.request_ids),parent.tokens
    data=parent.download.payload
    header_bytes=(4*n+1)*8
    raw_bytes=runtime.row_elements*t*runtime.raw_dtype.itemsize
    value_bytes=runtime.row_elements*t
    if data.numel()!=header_bytes+raw_bytes+value_bytes:
        raise ValueError('Shared logical packet has an invalid byte extent')
    header=data[:header_bytes].view(torch.int64).tolist()
    starts=header[:n+1]
    if starts[0]!=0 or starts[-1]!=t or any(b<a for a,b in zip(starts,starts[1:])):
        raise ValueError('Shared logical packet has invalid owner boundaries')
    begin,end=starts[index:index+2];group=child.logical_group
    if (end-begin!=group.count or parent.request_tokens[index]!=group.count or
            header[n+1+index]!=group.history_begin or header[2*n+1+index]!=group.local_begin or
            header[3*n+1+index]!=0):
        raise ValueError('Shared logical owner differs from its admitted prefix')
    shape=(len(runtime.layers),t,runtime.heads,runtime.dim)
    raw=data[header_bytes:header_bytes+raw_bytes].view(runtime.raw_dtype).reshape(shape)
    values=data[header_bytes+raw_bytes:].reshape(shape)
    child.logical_payload=raw[:,begin:end],values[:,begin:end]
    return child


def release_completed_owners(runtime, packet, indices):
    """Called under publication lock, before exposing completion to the engine."""
    try:
        for index in indices:
            child=packet.logical_owners.get(index)
            if child is None:continue
            if child.logical_cursor is None or not child.logical_cursor.done:
                raise RuntimeError('Incomplete logical owner reached publication ACK')
            child.logical_cursor.release()
            child.logical_payload=None;child.logical_slices=();child.action_inputs.clear()
            packet.action_inputs.pop(child.request_ids[0],None)
        complete=len(packet.published.union(indices))==len(packet.request_ids)
        if complete:
            if not runtime._download_pool(packet.lane).release(packet.download):
                raise RuntimeError('Completed logical cohort retained its DMA lease')
            packet.source_released=True
            for child in packet.logical_owners.values():child.source_released=True
        return complete
    except BaseException as error:
        # Keep the parent/source quarantined; no already-committed owner may be
        # replayed merely because another owner's publication/release failed.
        runtime._publication_error=error
        runtime._publication_condition.notify_all()
        raise
