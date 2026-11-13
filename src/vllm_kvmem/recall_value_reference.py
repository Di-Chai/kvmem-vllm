"""Archive adapter and immutable-state diagnostics for Value-based recall."""
import hashlib
import os
import time

import torch

from .recall_value_scoring import (REFERENCE_METHODS, aggregate_scores,
    aggregate_set_errors, iter_attention_moments)
from . import settings as _settings


def reference_enabled():
    return _settings.get('KVMEM_VALUE_RECALL_REFERENCE') == '1'


def prepare_reference(authority, records, queries, mask, *, cache_tiles=False,
                      reference_query_batch=4, reference_block_batch=None, packed=False,
                      region=False):
    """Own immutable archived operands; dequantize V with each target layer's scale."""
    from .recall_experiment import _signature
    layers = authority.scoring_layers
    if region and (not packed or not callable(getattr(authority, 'read_score_kv', None))):
        raise ValueError('Region Value recall requires native target-layer K/V gathers')
    read_bytes = 0
    read_started = time.perf_counter()
    region_stats = {}
    if region:
        # Final packed storage is the read destination. Avoid whole-record
        # materialization, draft-layer reads and two per-record clone passes.
        geometry = (len(records), layers, *authority.shape[1:])
        packed_k = torch.empty(geometry, dtype=authority.raw_dtype)
        packed_v = torch.empty(geometry, dtype=torch.uint8)
        missing = records
        for begin in range(0, len(records), 32):
            end = min(begin+32, len(records))
            _, counters = authority.read_score_kv([r.handle for r in records[begin:end]],
                destination=(packed_k[begin:end], packed_v[begin:end]))
            for name, value in counters.items():
                region_stats[name] = region_stats.get(name, 0)+value
        read_bytes = region_stats.get('key_read_bytes', 0)+region_stats.get('value_read_bytes', 0)
    else:
        cache = getattr(authority, '_value_recall_records', None)
        if cache is None:
            cache = authority._value_recall_records = {}
        live = {_signature(r) for r in authority.records.values()}
        for identity in tuple(cache):
            if identity not in live:
                del cache[identity]
        missing = [r for r in records if _signature(r) not in cache]
        for begin in range(0, len(missing), 16):
            group = missing[begin:begin+16]
            for record, (key, value) in zip(group, authority.read_records([r.handle for r in group])):
                if key.shape != value.shape or key.shape != (authority.shape[0], record.count, *authority.shape[2:]):
                    raise ValueError('Reference archived K/V geometry mismatch')
                read_bytes += key.numel()*key.element_size()+value.numel()*value.element_size()
                cache[_signature(record)] = (key[:layers].contiguous().clone(), value[:layers].contiguous().clone())
    archive_seconds = time.perf_counter() - read_started
    scales = getattr(authority, 'recall_value_scales', None)
    if scales is None:
        raise ValueError('Reference recall requires explicit target-layer Value scales')
    scales = torch.as_tensor(scales, device='cpu', dtype=torch.float32)
    if scales.shape != (layers,) or not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError('Invalid reference Value scales')
    owned = None if region else [cache[_signature(r)] for r in records]
    q = queries.to(dtype=torch.float16) if authority.mean_k_backend == 'qw3_cuda' else queries
    q = q.detach().to(device='cpu', dtype=torch.float32)

    stats = dict(archive_kv_read_bytes=read_bytes, new_operand_blocks=len(missing),
        archive_operand_seconds=archive_seconds, operand_pack_seconds=0.,
        operand_fetch_host_seconds=0., operand_transfer_bytes=0, operand_read_calls=0,
        operand_cache_hits=0, operand_backend='region_packed' if region else 'packed' if packed else 'legacy',
        archive_region_stats=region_stats,
        value_scales=scales.tolist(), value_encoding='fp8_e4m3fn_times_layer_scale',
        normalization='global_candidate_and_mandatory_token_softmax', key_frame='content',
        query_precision='fp16_to_fp32' if authority.mean_k_backend == 'qw3_cuda' else 'fp32',
        query_rows=q.shape[1], query_operand_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest())
    device = torch.device(_settings.get('KVMEM_RECALL_DEVICE', 'cpu'))
    if packed and not region:
        began = time.perf_counter()
        geometry = (len(records), layers, *authority.shape[1:])
        packed_k = torch.empty(geometry, dtype=owned[0][0].dtype)
        packed_v = torch.empty(geometry, dtype=owned[0][1].dtype)
        for i, (key, value) in enumerate(owned):
            if key.dtype != packed_k.dtype or value.dtype != packed_v.dtype:
                raise ValueError('Packed Value recall requires a uniform archive encoding')
            count = records[i].count
            packed_k[i, :, :count].copy_(key)
            packed_v[i, :, :count].copy_(value)
            if count < authority.shape[1]:
                packed_k[i, :, count:].zero_()
                packed_v[i, :, count:].zero_()
        stats['operand_pack_seconds'] = time.perf_counter() - began
    if packed:
        device_scales = scales.to(device)

    tile_cache = {}
    tile_owner = [None]
    def read_blocks(begin, end, layer, head):
        stats['operand_read_calls'] += 1
        if cache_tiles:
            if tile_owner[0] != (layer,head):
                tile_cache.clear();tile_owner[0]=(layer,head)
            if (begin,end) in tile_cache:
                stats['operand_cache_hits'] += 1
                return tile_cache[begin,end]
        began = time.perf_counter()
        if packed:
            # Pack raw bytes once, then gather/dequantize an entire block tile.
            # No token values, score math, reduction order or Query rows change.
            raw = packed_k[begin:end, layer, :, head].contiguous()
            value = packed_v[begin:end, layer, :, head].contiguous()
            stats['operand_transfer_bytes'] += (raw.numel()*raw.element_size() +
                                               value.numel()*value.element_size())
            k = raw.to(device).float()
            value = value.to(device)
            if value.dtype == torch.uint8:
                v = value.view(torch.float8_e4m3fn).float()*device_scales[layer]
            elif value.is_floating_point():
                v = value.float()
            else:
                raise ValueError('Unknown packed Value archive encoding')
            if cache_tiles:
                tile_cache[begin,end] = (k,v)
            stats['operand_fetch_host_seconds'] += time.perf_counter() - began
            return k, v
        size = authority.shape[1]
        k = torch.zeros((end-begin, size, authority.shape[-1]), dtype=torch.float32)
        v = torch.zeros_like(k)
        for j, (raw, values) in enumerate(owned[begin:end]):
            count = records[begin+j].count
            k[j, :count] = raw[layer, :, head].float()
            value = values[layer, :, head].contiguous()
            if value.dtype == torch.uint8:
                value = value.view(torch.float8_e4m3fn).float()*scales[layer]
            elif value.is_floating_point():
                value = value.float()  # Explicit floating test/reference inputs are already dequantized.
            else:
                raise ValueError('Unknown Value archive encoding')
            v[j, :count] = value
        if cache_tiles:
            tile_cache[begin,end]=(k,v)
        stats['operand_fetch_host_seconds'] += time.perf_counter() - began
        return k, v

    def iterator():
        return iter_attention_moments(q, read_blocks, [r.count for r in records], mask,
            kv_heads=authority.shape[2], device=_settings.get('KVMEM_RECALL_DEVICE', 'cpu'),
            query_batch=reference_query_batch,
            block_batch=(reference_block_batch or int(_settings.get('KVMEM_RECALL_BLOCK_BATCH', '16'))))

    iterator.read_blocks = read_blocks
    iterator.queries = q

    return iterator, stats


def score_value_records(authority, records, queries, mask, method):
    began = time.perf_counter()
    backend=_settings.get('KVMEM_VALUE_RECALL_BACKEND','reference')
    if backend not in ('reference','streaming','streaming_packed','streaming_region'):
        raise ValueError('Unknown Value recall backend; no fallback is allowed')
    streaming = backend.startswith('streaming')
    iterator, stats = prepare_reference(authority, records, queries, mask,
        cache_tiles=streaming, packed=backend in ('streaming_packed','streaming_region'),
        region=backend=='streaming_region')
    if streaming:
        from .recall_value_streaming import streaming_scores,streaming_value_norm
        operation=streaming_value_norm if method=='block_value_norm' else streaming_scores
        values,reduction=operation(iterator.queries,iterator.read_blocks,[r.count for r in records],mask,
            kv_heads=authority.shape[2],device=_settings.get('KVMEM_RECALL_DEVICE','cpu'),
            query_batch=int(_settings.get('KVMEM_RECALL_QUERY_BATCH','128')),
            block_batch=int(_settings.get('KVMEM_RECALL_BLOCK_BATCH','128')))
        scores={method:values} if method=='block_value_norm' else values
    else:
        scores, reduction = aggregate_scores(iterator(), layers=authority.scoring_layers, heads=queries.shape[2])
    stats.update(reduction, scoring_seconds=time.perf_counter()-began,
        implementation=reduction.get('implementation','reference_token_attention_value_moments_v1'),
        actual_backend=backend, score_sum=sum(scores[method]))
    return scores[method], stats


def compare_frozen_state(authority, records, queries, mask, pins, *, max_tokens, pressure, exact_scores=None):
    """Counterfactual native selections on one Q/K/V state, without committing any."""
    # The independent materialized-moment oracle can use larger tiles here:
    # only a few immutable qualification states are examined, never a scored run.
    iterator, stats = prepare_reference(authority, records, queries, mask,
        cache_tiles=True,reference_query_batch=32,reference_block_batch=128)
    began=time.perf_counter()
    scores, reduction = aggregate_scores(iterator(), layers=authority.scoring_layers, heads=queries.shape[2])
    stats['reference_scoring_seconds']=time.perf_counter()-began
    if _settings.get('KVMEM_VALUE_RECALL_BACKEND')=='streaming':
        from .recall_value_streaming import streaming_scores,streaming_value_norm
        began=time.perf_counter()
        optimized,backend_stats=streaming_scores(iterator.queries,iterator.read_blocks,
            [r.count for r in records],mask,kv_heads=authority.shape[2],
            device=_settings.get('KVMEM_RECALL_DEVICE','cpu'),
            query_batch=int(_settings.get('KVMEM_RECALL_QUERY_BATCH','128')),
            block_batch=int(_settings.get('KVMEM_RECALL_BLOCK_BATCH','128')))
        stats['optimized_scoring_seconds']=time.perf_counter()-began
        began=time.perf_counter()
        optimized['block_value_norm'],_=streaming_value_norm(iterator.queries,iterator.read_blocks,
            [r.count for r in records],mask,kv_heads=authority.shape[2],
            device=_settings.get('KVMEM_RECALL_DEVICE','cpu'),
            query_batch=int(_settings.get('KVMEM_RECALL_QUERY_BATCH','128')),
            block_batch=int(_settings.get('KVMEM_RECALL_BLOCK_BATCH','128')))
        stats['optimized_value_norm_seconds']=time.perf_counter()-began
        stats['optimized_score_max_abs_difference']={m:max(abs(a-b) for a,b in zip(scores[m],optimized[m])) for m in REFERENCE_METHODS}
        stats['optimized_implementation']=backend_stats['implementation']
    else:
        optimized=None
    if exact_scores is not None:
        stats['exact_mass_reference_max_abs_difference'] = max(abs(a-b) for a,b in zip(exact_scores, scores['exact_mass']))
        if stats['exact_mass_reference_max_abs_difference'] > 5e-5*max(1, queries.shape[1]):
            raise ValueError('Token-moment ExactMass differs from existing ExactMass scorer')
        scores['exact_mass'] = list(exact_scores)
    slots = [r.handle.slot for r in records]
    index = {slot:i for i,slot in enumerate(slots)}
    selections, reports = {}, {}
    for method in REFERENCE_METHODS:
        by_slot = dict(zip(slots, scores[method]))
        kwargs = dict(action=False, prefill_scores=None, max_tokens=max_tokens)
        if pressure:
            kwargs['pressure_scores'] = list(by_slot.items())
        chosen = set(authority.memory.preview_selection({} if pressure else by_slot, pins, **kwargs))
        if not set(pins) <= chosen or not chosen <= index.keys():
            raise ValueError('Counterfactual picker violated mandatory blocks/candidate domain')
        selections[method] = [index[s] for s in sorted(chosen)]
        ranks = sorted(range(len(records)), key=lambda i:(-scores[method][i], i))
        reports[method] = dict(block_scores=scores[method], ranked_block_ids=[slots[i] for i in ranks],
            selected_block_ids=sorted(chosen), selected_block_starts=[records[i].history_begin for i in selections[method]])
        if optimized is not None:
            fast_slots=dict(zip(slots,optimized[method]))
            fast_kwargs=dict(action=False,prefill_scores=None,max_tokens=max_tokens)
            if pressure:fast_kwargs['pressure_scores']=list(fast_slots.items())
            fast_chosen=set(authority.memory.preview_selection({} if pressure else fast_slots,pins,**fast_kwargs))
            reports[method]['optimized_selected_set_equal']=fast_chosen==chosen
            if method!='exact_mass' and fast_chosen!=chosen:
                raise ValueError('Optimized '+method+' changes the frozen-state selected set')
    errors = aggregate_set_errors(iterator(), selections, layers=authority.scoring_layers, heads=queries.shape[2])
    base = set(reports['exact_mass']['selected_block_ids'])
    for method, report in reports.items():
        chosen = set(report['selected_block_ids'])
        report.update(set_output_error=errors[method],
            selected_jaccard_vs_exact_mass=len(base & chosen)/len(base | chosen))
    return dict(stats, **reduction, methods=reports, mandatory_block_ids=list(pins),
        candidate_block_ids=slots, candidate_block_starts=[r.history_begin for r in records],
        candidate_counts=[r.count for r in records], normalization_mask=list(mask),
        error_scope='Full candidate content-frame attention versus renormalized entire selected set; not summed single-block errors.')
