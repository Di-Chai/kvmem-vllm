"""Opt-in, uniform block scoring for the fixed-policy recall experiment.

This module changes scores only. Native Query/pressure candidate domains,
mandatory blocks, budget accounting and assembly remain authoritative.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
import time

import torch


from .recall_value_scoring import VALUE_METHODS

METHODS = ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass', *VALUE_METHODS)
ACCUMULATED_ATTENTION = 'accumulated_attention'
_LOG_LOCK = threading.Lock()


def configured_method():
    method = os.getenv('KVMEM_RECALL_METHOD', '')
    if method and method not in METHODS:
        raise ValueError('Invalid KVMEM_RECALL_METHOD')
    if method in VALUE_METHODS and os.getenv('KVMEM_VALUE_RECALL_REFERENCE') != '1':
        raise ValueError('Value recall requires mandatory-inclusive reference normalization')
    if method and os.getenv('KVMEM_PRESSURE_QUERY_WINDOW', '128') != '128':
        raise ValueError('Recall comparison requires a fixed 128-token pressure Q window')
    return method


def configured_event_method(event_type):
    method = configured_method()
    stages = {stage: os.getenv('KVMEM_' + stage.upper() + '_PRESSURE_METHOD', 'same_method')
              for stage in ('prefill', 'decode')}
    if any(value not in ('same_method', ACCUMULATED_ATTENTION) for value in stages.values()):
        raise ValueError('Unknown pressure method; no fallback is allowed')
    stage = event_type.removesuffix('_pressure')
    return ACCUMULATED_ATTENTION if stages.get(stage) == ACCUMULATED_ATTENTION else method


def validate_options(options, method=None):
    if not (configured_method() if method is None else method):
        return
    forbidden = ('guided_query_tokens', 'query_history', 'query_fps_halo_blocks',
                 'query_mean_quota_fraction', 'query_fps_prebuild')
    if any(getattr(options, name, 0) for name in forbidden):
        raise ValueError('Recall experiment requires guided/history/halo/hybrid/prebuild disabled')
    if (getattr(options, 'graph_mode', 'off') != 'off' or
            getattr(options, 'graph_alpha', 0) != 0):
        raise ValueError('Recall experiment requires graph disabled')
    if getattr(options, 'query_score_mode', 'mean_k') != 'mean_k':
        raise ValueError('Recall experiment owns scoring; legacy query_score_mode must be mean_k')


def launch_environment(config, options):
    if config is None:
        return {}
    allowed = {'method', 'pressure_query_window', 'device', 'query_batch', 'block_batch',
               'include_mandatory_in_softmax', 'value_backend', 'prefill_pressure_method', 'decode_pressure_method',
               'concurrent_owners'}
    if not isinstance(config, dict) or set(config) - allowed or config.get('method') not in METHODS:
        raise ValueError('Invalid recall_experiment profile')
    if type(config.get('pressure_query_window')) is not int or config['pressure_query_window'] != 128:
        raise ValueError('recall_experiment requires pressure_query_window=128')
    owners = config.get('concurrent_owners', 1)
    if type(owners) is not int or owners not in (1, 2, 4, 8):
        raise ValueError('Recall requires explicitly declared concurrent owners in (1, 2, 4, 8)')
    device, queries, blocks = config.get('device', 'cpu'), config.get('query_batch', 4), config.get('block_batch', 32)
    if (device not in ('cpu', 'cuda') or type(queries) is not int or not 1 <= queries <= 128 or
            type(blocks) is not int or not 1 <= blocks <= (1024 if config.get('value_backend') in ('streaming_packed','streaming_region') else 128)):
        raise ValueError('Invalid recall_experiment scratch geometry/device')
    validate_options(options, method=config['method'])
    include_mandatory = config.get('include_mandatory_in_softmax', False)
    if type(include_mandatory) is not bool or (config['method'] in VALUE_METHODS and not include_mandatory):
        raise ValueError('Value recall requires candidate and mandatory blocks in attention normalization')
    backend = config.get('value_backend','reference')
    if backend not in ('reference','streaming','streaming_packed','streaming_region'):
        raise ValueError('Unknown Value recall backend')
    stages = {stage: config.get(stage + '_pressure_method', 'same_method') for stage in ('prefill', 'decode')}
    if any(value not in ('same_method', ACCUMULATED_ATTENTION) for value in stages.values()):
        raise ValueError('Unknown pressure method')
    if ACCUMULATED_ATTENTION in stages.values() and (not getattr(options, 'prefill_attention', False) or
            not options.memory.action_attention or getattr(options, 'pressure_policy', None) != 'attention'):
        raise ValueError('Accumulated prefill requires enabled prefill attention and attention pressure policy')
    if not options.request_planning or not options.async_refresh:
        raise ValueError('Recall experiment requires planned asynchronous reselection')
    if options.memory.block_tokens != 32:
        raise ValueError('This recall campaign freezes one 32-token physical block')
    if getattr(options, 'historical_gc', None) is not None:
        raise ValueError('Recall comparison requires complete history without historical GC')
    return {'KVMEM_RECALL_METHOD': config['method'], 'KVMEM_PRESSURE_QUERY_WINDOW': '128',
            'KVMEM_RECALL_CONCURRENT_OWNERS': str(owners),
            **{'KVMEM_' + stage.upper() + '_PRESSURE_METHOD': value for stage, value in stages.items()
               if value != 'same_method'},
            **({'KVMEM_VALUE_RECALL_REFERENCE': '1'} if include_mandatory else {}),
            **({'KVMEM_VALUE_RECALL_BACKEND': backend} if include_mandatory else {}),
            'KVMEM_RECALL_DEVICE': device, 'KVMEM_RECALL_QUERY_BATCH': str(queries),
            'KVMEM_RECALL_BLOCK_BATCH': str(blocks),
            'VLLM_FLOAT32_MATMUL_PRECISION': 'highest'}


def _signature(record):
    return (record.handle.slot, record.handle.generation, record.history_begin,
            record.count, id(record))


def _features(authority, records, method, *, device='cpu'):
    from .recall_scoring import build_block_features
    cache = getattr(authority, '_recall_feature_cache', None)
    if not cache or cache.get('method') != method:
        cache = dict(method=method, entries={})
        authority._recall_feature_cache = cache
    entries = cache['entries']
    live = {_signature(record) for record in authority.records.values()}
    for identity in tuple(entries):
        if identity not in live:
            del entries[identity]
    missing = [record for record in records if _signature(record) not in entries]
    layers = authority.scoring_layers
    stats = dict(feature_build_device=str(device), new_feature_blocks=len(missing),
                 key_read_seconds=0., feature_build_seconds=0., feature_copy_seconds=0.,
                 key_read_bytes=0, value_read_bytes=0, draft_key_read_bytes=0)
    for begin in range(0, len(missing), 16):
        group = missing[begin:begin + 16]
        read_started = time.perf_counter()
        if hasattr(authority, 'read_score_keys'):
            # The authority owns a bounded K-only gather: neither V nor MTP K
            # participates in scoring, and the returned slab owns its storage.
            slab, counters = authority.read_score_keys([record.handle for record in group])
            if slab.shape != (len(group), layers, *authority.shape[1:]):
                raise ValueError('Recall scoring Key slab geometry differs from accepted record')
            keys = slab.permute(1, 0, 2, 3, 4)
            for name in ('key_read_bytes', 'value_read_bytes', 'draft_key_read_bytes'):
                stats[name] += counters.get(name, 0)
        else:
            archived = authority.read_records([record.handle for record in group])
            keys = torch.zeros((layers, len(group), *authority.shape[1:]), dtype=authority.raw_dtype)
            for index, (record, (key, _)) in enumerate(zip(group, archived)):
                if key.shape != (authority.shape[0], record.count, *authority.shape[2:]):
                    raise ValueError('Recall index raw K geometry differs from accepted record')
                keys[:, index, :record.count] = key[:layers]
        stats['key_read_seconds'] += time.perf_counter()-read_started
        build_started = time.perf_counter()
        # Construction and scoring use the configured device, with the same
        # FP32 operations. Keep only a bounded construction tile on the GPU;
        # cached immutable features remain on CPU across pressure events.
        features = build_block_features(keys.to(device), [r.count for r in group], method)
        copied = time.perf_counter()
        features = features.to('cpu')
        stats['feature_build_seconds'] += copied-build_started
        stats['feature_copy_seconds'] += time.perf_counter()-copied
        for index, record in enumerate(group):
            entries[_signature(record)] = features.select_blocks([index])
    authority._recall_feature_last_stats = stats
    return [entries[_signature(record)] for record in records]


def _host_block_mass(logits, mask, *, row_batch=256):
    """Bound CPU scratch by rows; every row still spans all candidate blocks."""
    flat = logits.reshape(-1, logits.shape[-1])
    totals = torch.zeros(logits.shape[-1], dtype=torch.float64)
    for begin in range(0, flat.shape[0], row_batch):
        rows = flat[begin:begin + row_batch]
        if not bool(torch.isfinite(rows).all()):
            raise ValueError('FP32 recall logit overflow')
        rows.masked_fill_(~mask, -torch.inf)
        totals += rows.softmax(-1).sum(0, dtype=torch.float64)
    return totals


def score_records(authority, records, queries, score_mask, method):
    """Score a Query tile with one feature pass, preserving global softmax.

    All Query rows in a tile reuse one packed GPU Query and each block feature
    is transferred once. Reduced block logits return to CPU, where row tiles
    normalize over the *complete* candidate domain. GPU scratch therefore does
    not grow with the number of candidate blocks; CPU reduction scratch is also
    bounded without retaining a second full ExactMass feature copy.
    """
    from .recall_scoring import (BlockFeatures, block_logits, concat_block_features,
                                 prepare_queries)
    if (queries is None or queries.ndim != 4 or not records or
            len(score_mask) != len(records) or method not in METHODS):
        raise ValueError('Recall scoring requires Q and a nonempty valid candidate domain')
    if queries.shape[0] != authority.scoring_layers or queries.shape[1] < 1:
        raise ValueError('Recall Q layers/rows do not match the frozen scoring geometry')
    if not any(score_mask):
        return [0.] * len(records), dict(query_rows=queries.shape[1], score_sum=0.,
            normalization='empty_eligible_domain', no_eligible_candidates=True,
            scoring_seconds=0., feature_seconds=0.)
    if any(record.count < 1 for record in records):
        raise ValueError('Recall candidates must contain accepted tokens')
    if method in VALUE_METHODS:
        from .recall_value_reference import score_value_records
        return score_value_records(authority, records, queries, score_mask, method)
    device_name = os.getenv('KVMEM_RECALL_DEVICE', 'cpu')
    if device_name not in ('cpu', 'cuda'):
        raise ValueError('KVMEM_RECALL_DEVICE must be cpu or cuda')
    device = torch.device(device_name)
    if device_name == 'cuda' and (torch.get_float32_matmul_precision() != 'highest' or
                                  torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Recall scoring requires full FP32 matmul precision without TF32')
    # Retain the original CUDA MeanK/FPS input rounding for every arm.
    queries = queries.to(dtype=torch.float16) if authority.mean_k_backend == 'qw3_cuda' else queries
    queries = queries.to(device='cpu', dtype=torch.float32)
    if not bool(torch.isfinite(queries).all()):
        raise ValueError('Nonfinite Query in recall scoring')
    query_batch = int(os.getenv('KVMEM_RECALL_QUERY_BATCH', '4'))
    block_batch = int(os.getenv('KVMEM_RECALL_BLOCK_BATCH', '32'))
    if not 1 <= query_batch <= 128 or not 1 <= block_batch <= 128:
        raise ValueError('Recall scratch tiles exceed the qualified bounds')
    # The vectorized representative dimension must also fit the small 4090
    # scratch allowance. This cap affects compute tiling only, never candidates.
    representatives = {'mean_k': 1, 'fixed4': 4, 'fps4': 4,
                       'min_max': 2, 'exact_mass': authority.shape[1]}[method]
    tile_rows = min(query_batch, queries.shape[1])
    bytes_per_block = (queries.shape[0] * tile_rows * queries.shape[2] * representatives * 4)
    block_batch = min(block_batch, max(1, (96 * 1024**2) // bytes_per_block))
    began = time.perf_counter()
    if method == 'mean_k':
        # Use the original published MeanK index, including its precision.
        values = torch.stack([r.mean_key if r.mean_key is not None else
            (r.key_sum / r.count).to(torch.float16 if authority.mean_k_backend == 'qw3_cuda'
                                    else torch.float32) for r in records], dim=1)
        values = values[:authority.scoring_layers].float().unsqueeze(2)
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Nonfinite published MeanK index')
        features = BlockFeatures(method=method, values=values,
            valid=torch.ones(values.shape[:-1], dtype=torch.bool),
            counts=torch.tensor([r.count for r in records], dtype=torch.int64))
        def feature_tile(begin):
            return features.select_blocks(range(begin, min(begin + block_batch, len(records))))
    else:
        per_block = _features(authority, records, method, device=device)
        def feature_tile(begin):
            return concat_block_features(per_block[begin:begin + block_batch])
    built = time.perf_counter()
    mask = torch.tensor(score_mask, dtype=torch.bool)
    totals = torch.zeros(len(records), dtype=torch.float64)
    tile_pack_seconds = normalization_seconds = 0.
    feature_passes = score_tiles = feature_transfer_bytes = 0
    for begin in range(0, queries.shape[1], query_batch):
        query = queries[:, begin:begin + query_batch].to(device)
        prepared = prepare_queries(query, authority.shape[2],
            split_sign=method == 'min_max', validate=False)
        del query
        host_parts = []
        feature_passes += 1
        for offset in range(0, len(records), block_batch):
            packing_started = time.perf_counter()
            part = feature_tile(offset)
            tile_pack_seconds += time.perf_counter() - packing_started
            local = part.to(device)
            if device_name == 'cuda':
                feature_transfer_bytes += sum(t.numel() * t.element_size() for t in
                                               (part.values, part.valid, part.counts))
            logits = block_logits(prepared, local, validate=False)
            # Blocking D2H is intentional: CPU must not read an unfinished copy.
            host_parts.append(logits.to('cpu'))
            score_tiles += 1
            del logits, local, part
        del prepared
        host_logits = torch.cat(host_parts, dim=-1)
        host_parts.clear()
        normalization_started = time.perf_counter()
        totals += _host_block_mass(host_logits, mask)
        normalization_seconds += time.perf_counter() - normalization_started
        del host_logits
    totals /= queries.shape[0] * queries.shape[2]
    if not bool(torch.isfinite(totals).all()) or abs(float(totals.sum()) - queries.shape[1]) > 1e-3:
        raise ValueError('Recall score normalization failed')
    scores = totals.tolist()
    feature_stats = getattr(authority, '_recall_feature_last_stats', {}) if method != 'mean_k' else {}
    return scores, dict(scoring_device=device_name, query_batch=query_batch, block_batch=block_batch,
        **feature_stats,
        feature_seconds=built-began+tile_pack_seconds,
        scoring_seconds=time.perf_counter()-built-tile_pack_seconds,
        normalization_seconds=normalization_seconds, normalization_device='cpu',
        implementation='batched_gqa_host_softmax_v2', feature_passes=feature_passes,
        score_tiles=score_tiles, feature_transfer_bytes=feature_transfer_bytes,
        matmul_precision='fp32', query_rows=queries.shape[1], score_sum=sum(scores),
        query_reduction='sum', head_layer_reduction='mean', key_frame='content',
        query_precision='fp16_to_fp32' if authority.mean_k_backend == 'qw3_cuda' else 'fp32',
        normalization='global_candidate_token_softmax_block_sum' if method == 'exact_mass'
                      else 'global_candidate_block_softmax')


def _write_event(output, event):
    path = Path(output) / 'recall_events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(',', ':'), allow_nan=False)
    with _LOG_LOCK, path.open('a') as stream:
        stream.write(line + '\n')


def accumulated_prefill_scores(authority, records):
    """Reuse accepted PrefillHistory mass/exposure, never output scores or recency."""
    state = getattr(authority, 'action_history', None)
    prefill = getattr(state, 'prefill', None)
    if (prefill is None or prefill.samples <= 0 or prefill.chunks <= 0 or
            not 0 < prefill.end <= authority.history_end):
        raise ValueError('Accumulated prefill requires accepted prefill attention samples')
    history_scores = prefill.scores()
    scores = [history_scores.get(r.history_begin, 0.) for r in records]
    import math
    if not all(math.isfinite(v) and v >= 0 for v in scores) or not any(v > 0 for v in scores):
        raise ValueError('Accumulated prefill has no finite positive candidate attention; no fallback')
    return scores, dict(query_source='accepted_prefill_accumulation', query_begin=0,
        query_end=prefill.end, query_rows=prefill.samples, attention_source='prefill',
        prefill_attention_samples=prefill.samples, prefill_attention_chunks=prefill.chunks,
        prefill_attention_pages=sum(v > 0 for v in scores),
        prefill_score_normalization='decayed_mass_over_exposure',
        attention_normalization_domain='sampled_causal_active_windows',
        implementation='existing_accepted_prefill_history_v1', score_sum=sum(scores),
        scoring_seconds=0., feature_seconds=0.)


def accumulated_decode_scores(authority, records):
    """Reuse the native attention-pressure policy's accepted output/prefill seed."""
    from .prefill_attention import pressure_seed
    native = [(r.handle.slot, r.handle.generation, r.history_begin, r.count, 0) for r in records]
    by_slot, source = pressure_seed(authority, native)
    if source == 'prefill':
        scores, stats = accumulated_prefill_scores(authority, records)
        stats['decode_seed_source'] = 'prefill'
        return scores, stats
    if source != 'output':
        raise ValueError('Accumulated decode has no accepted attention; no recency fallback')
    state = authority.action_history
    scores = [value for _, value in by_slot]
    return scores, dict(query_source='accepted_output_accumulation', query_begin=0,
        query_end=authority.history_end, attention_source='output', decode_seed_source='output',
        output_attention_samples=state.samples, output_attention_rounds=state.rounds,
        action_scope=state.options.action_scope, score_sum=sum(scores),
        attention_normalization_domain='sampled_causal_active_windows',
        output_score_normalization='existing_action_decay_and_sample_mean',
        implementation='existing_accepted_output_history_v1', scoring_seconds=0., feature_seconds=0.)


def select_for_event(authority, queries, mandatory, *, max_tokens, event_type,
                     query_metadata, request_id, output):
    from .authority import record_key, query_score_frame
    configured = configured_method()
    method = configured_event_method(event_type)
    accumulated = method == ACCUMULATED_ATTENTION
    if not configured or event_type not in ('query', 'prefill_pressure', 'decode_pressure'):
        raise ValueError('Invalid recall experiment event')
    authority._require_open()
    validate_options(authority.graph_options)
    pressure = event_type != 'query'
    with authority.index_snapshot() as (all_records, _):
        pins = tuple(handle.slot for handle in mandatory)
        if pressure:
            working = tuple(authority.memory._store.working_records())
            by_identity = {record_key(record.handle): record for record in all_records}
            records = []
            for slot, generation, begin, count, _ in working:
                record = by_identity.get((slot, generation))
                if record is None or (record.history_begin, record.count) != (begin, count):
                    raise ValueError('Pressure candidates do not match accepted working records')
                records.append(record)
            frame_scope = nullcontext(None)
        else:
            records = all_records
            frame_scope = query_score_frame(authority.memory, records, pins, max_tokens)
        if not set(record_key(handle) for handle in mandatory).issubset(
                record_key(record.handle) for record in records):
            raise ValueError('Required block outside original recall candidate domain')
        if any(record.history_begin + record.count > authority.history_end for record in records):
            raise ValueError('Recall candidate extends beyond the accepted history')
        if (pressure and not accumulated and (query_metadata['query_end'] != authority.history_end or
                query_metadata['query_begin'] != max(0, authority.history_end - 128) or
                queries.shape[1] != min(128, authority.history_end) or
                query_metadata.get('query_rows', queries.shape[1]) != queries.shape[1])):
            raise ValueError('Pressure Q is not the complete accepted 128-token tail')
        if not pressure and (queries is None or queries.shape[1] !=
                query_metadata['query_end'] - query_metadata['query_begin']):
            raise ValueError('Public Query Q rows differ from its declared input span')
        with frame_scope as frame:
            mask = [True] * len(records) if pressure else tuple(frame.score_mask)
            from .recall_value_reference import reference_enabled
            if reference_enabled():
                # All accepted records are candidates or fixed keeps. Native
                # Query score masks exclude sink/recent; their attention must
                # nevertheless normalize all three Value-reference methods.
                mask = [True] * len(records)
            profile_path = Path(output) / 'value_query_optimization.jsonl'
            if (not accumulated and method in VALUE_METHODS and
                    os.getenv('KVMEM_VALUE_RECALL_PROFILE') == event_type and not profile_path.exists()):
                from .recall_value_profile import compare_query_backends
                backend_options = (dict(baseline_backend='streaming_packed', baseline_block_batch=1024,
                                        optimized_backend='streaming_region')
                    if os.getenv('KVMEM_VALUE_RECALL_BACKEND') == 'streaming_region' else {})
                report = compare_query_backends(authority, records, queries, mask, pins,
                                                max_tokens=max_tokens, **backend_options)
                report.update(event_type=event_type, request_id=request_id,
                    configured_method=configured, accepted_through=authority.history_end,
                    query_metadata=query_metadata)
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                with _LOG_LOCK, profile_path.open('a') as stream:
                    stream.write(json.dumps(report, allow_nan=False) + '\n')
                if not report['passed']:
                    raise ValueError('Real-state Query optimization failed correctness/efficiency validation')
            if accumulated:
                operation = accumulated_prefill_scores if event_type == 'prefill_pressure' else accumulated_decode_scores
                began = time.perf_counter()
                scores, stats = operation(authority, records)
                stats['scoring_seconds'] = time.perf_counter() - began
            else:
                scores, stats = score_records(authority, records, queries, mask, method)
            if reference_enabled() and not accumulated:
                stats.update(attention_normalization_domain='candidate_and_mandatory',
                    mandatory_in_normalization=True)
        if not accumulated and os.getenv('KVMEM_VALUE_RECALL_DIAGNOSTICS') == event_type:
            from .recall_value_reference import compare_frozen_state
            diagnostic = compare_frozen_state(authority, records, queries, mask, pins,
                max_tokens=max_tokens, pressure=pressure,
                exact_scores=scores if method == 'exact_mass' else None)
            diagnostic.update(event_type=event_type, configured_method=method,
                request_id=request_id, query_metadata=query_metadata,
                accepted_through=authority.history_end, budget=max_tokens)
            path = Path(output)/'value_recall_diagnostics.jsonl'
            with _LOG_LOCK, path.open('a') as stream:
                stream.write(json.dumps(diagnostic, allow_nan=False)+'\n')
        by_slot = {record.handle.slot: value for record, value in zip(records, scores)}
        kwargs = dict(action=False, prefill_scores=None, max_tokens=max_tokens)
        if pressure:
            kwargs['pressure_scores'] = list(by_slot.items())
            selection_scores = {}
        else:
            selection_scores = by_slot
        selected = set(authority.memory.preview_selection(selection_scores, pins, **kwargs))
        if not selected.issubset(by_slot):
            raise ValueError('Native picker escaped the frozen candidate set')
        selected_records = [r for r in records if r.handle.slot in selected]
        event = dict(version=1, sample_id=os.getenv('KVMEM_RECALL_SAMPLE_ID', request_id),
            request_id=request_id, authority_id=authority.memory.request_id,
            event_type=event_type, configured_method=configured, actual_method=method,
            fallback_reason=None, accepted_through=authority.history_end,
            candidate_block_count=len(records), score_mask_count=sum(mask),
            candidate_domain='working_records' if pressure else 'accepted_history',
            selected_block_ids=[r.handle.slot for r in selected_records],
            selected_block_generations=[r.handle.generation for r in selected_records],
            selected_block_starts=[r.history_begin for r in selected_records],
            selected_tokens=sum(r.count for r in selected_records),
            charged_tokens=len(selected_records) * authority.shape[1],
            budget=max_tokens, block_tokens=authority.shape[1], pressure_query_window=128)
        event.update(query_metadata)
        event.update(stats)
        if any(os.getenv('KVMEM_' + stage.upper() + '_PRESSURE_METHOD') == ACCUMULATED_ATTENTION
               for stage in ('prefill', 'decode')):
            event.update(configuration_kind='pressure_attention_query_recall', configured_event_method=method,
                prefill_pressure_method=os.getenv('KVMEM_PREFILL_PRESSURE_METHOD', 'same_method'),
                decode_pressure_method=os.getenv('KVMEM_DECODE_PRESSURE_METHOD', 'same_method'))
        _write_event(output, event)
        expected = tuple(record.handle for record in selected_records)
        plan = authority.memory.select(selection_scores, pins, **kwargs)

        def validate_selected():
            if any(authority.memory._store.handle(h.slot) != h for h in expected):
                raise RuntimeError('Recall selection contains a recycled generation')

        validate_selected()
        authority.memory._selection_validator = validate_selected
        authority.recall_selection_report = event
        return plan
