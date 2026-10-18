"""Run the existing V2 D smoke fixture and retain measured performance evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmark_graph_v2_performance import CANDIDATES
import graph_v2_protocol as protocol
import qualify_block_graph as q
import qualify_graph_v2 as qualifier


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def total(rows, field):
    values = [row[field] for row in rows if field in row]
    protocol.require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
        for value in values), 'Invalid measured counter: '+field)
    return dict(value=sum(values), observed_rows=len(values), total_rows=len(rows))


def cache_capacity_evidence(rows):
    """Capacity is a reusable-arena gauge, never bytes allocated per event."""
    measured = [row for row in rows if 'raw_cache_capacity_bytes' in row]
    total(measured, 'raw_cache_capacity_bytes')  # Validate observed numeric values.
    by_arena = None
    if measured and all(isinstance(row.get('raw_cache_arena'), str) and row['raw_cache_arena'] for row in measured):
        by_arena = {}
        for row in measured:
            name = row['raw_cache_arena']
            by_arena[name] = max(by_arena.get(name, 0), row['raw_cache_capacity_bytes'])
    return dict(observed_rows=len(measured), total_rows=len(rows),
        observed_max_bytes=max((row['raw_cache_capacity_bytes'] for row in measured), default=None),
        per_arena_max_bytes=by_arena, total_allocated_bytes=None,
        scope='Maximum observed capacity gauge, not a sum over refreshes or total worker allocation. '
              'Per-arena maxima require explicit arena identities; simultaneous total allocation is unmeasured.')


def interval_evidence(rows, begin_field, end_field):
    intervals = sorted((row[begin_field], row[end_field]) for row in rows
                       if begin_field in row and end_field in row)
    protocol.require(all(math.isfinite(a) and math.isfinite(b) and b >= a for a,b in intervals),
                     'Invalid observed maintenance interval')
    merged = []
    for begin, end in intervals:
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return dict(observed_rows=len(intervals), inclusive_sum_seconds=sum(b-a for a,b in intervals),
        union_seconds=sum(b-a for a,b in merged),
        scope='Host service intervals may overlap model work and each other; neither sum nor union is GPU idle.')


def seal_evidence(archive, enabled):
    counts = ('seal_batches', 'seal_records', 'seal_payload_bytes', 'seal_direct_records', 'seal_direct_bytes')
    timings = ('seal_pack_seconds', 'seal_hash_seconds', 'seal_cache_seconds',
               'seal_record_copy_seconds', 'seal_record_publish_seconds')
    fields = counts + timings
    for name in fields:
        if name in archive:
            value = archive[name]
            protocol.require(type(value) in ((int,) if name in counts else (int, float)) and
                math.isfinite(value) and value >= 0, 'Invalid successful seal counter: '+name)
    protocol.require(enabled or not archive.get('seal_direct_bytes', 0),
                     'Disabled direct sealing reported direct publication bytes')
    for direct, all_records in (('seal_direct_records', 'seal_records'), ('seal_direct_bytes', 'seal_payload_bytes')):
        if direct in archive and all_records in archive:
            protocol.require(archive[direct] <= archive[all_records], 'Direct seal counter exceeds total')
    return dict(configured=enabled, observed=any(name in archive for name in fields),
        measurement_complete=all(name in archive for name in fields),
        direct_path_observed=archive.get('seal_direct_bytes', 0) > 0,
        unobserved_fields=[name for name in fields if name not in archive],
        counters={name:archive.get(name) for name in counts},
        seconds={name:archive.get(name) for name in timings},
        scope='Final archive cumulative successful publications from this fresh worker; failed jobs excluded. '
              'Pack/hash/copy/publish work can overlap model execution and is not GPU idle.')


def metadata_evidence(capture, profile):
    kv = profile['kvmem']
    config_path = capture/'action_config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    for name in ('async_action_metadata', 'native_action_metadata'):
        if name in config:
            protocol.require(config[name] is kv.get(name, False), 'Actual Action metadata policy differs: '+name)
    snapshots = [row['metadata_upload_total'] for row in q.records(capture/'action_attention.jsonl')
                 if 'metadata_upload_total' in row]
    latest = max(snapshots, key=lambda row:(row.get('submissions', 0), row.get('completed', 0))) if snapshots else None
    if latest is not None:
        for name in ('submissions', 'completed', 'uploaded_bytes', 'packing_host_seconds', 'copy_submit_host_seconds'):
            protocol.require(type(latest.get(name)) in (int, float) and math.isfinite(latest[name]) and latest[name] >= 0,
                             'Invalid Action metadata cumulative metric: '+name)
    observed = latest is not None and latest['submissions'] > 0
    protocol.require(kv.get('async_action_metadata', False) or not observed,
                     'Disabled asynchronous Action metadata reported submissions')
    return dict(configured_async=kv.get('async_action_metadata', False),
        configured_native=kv.get('native_action_metadata', False), actual_config=config or None,
        snapshots=len(snapshots), async_path_observed=observed, latest_cumulative=latest,
        scope='One latest cumulative engine metadata upload snapshot, never the sum of per-owner snapshots. '
              'Host packing/submission time is not GPU transfer duration or GPU idle.')


def performance(capture, output, profile, candidate='raw-k-cache'):
    result = json.loads((output/'results.json').read_text())
    protocol.require(result.get('status') == 'passed' and
        all(row.get('status') == 'completed' for row in result['results']), 'Incomplete performance fixture')
    refresh = q.records(capture/'bounded_refresh.jsonl')
    restore = q.records(capture/'async_restore.jsonl')
    protocol.require(refresh and all(not row.get('cancelled') for row in refresh+restore),
                     'Missing refresh evidence or cancelled maintenance')
    for field in ('assembly_uploaded_bytes', 'assembly_host_prepare_seconds', 'raw_cache_hit_tokens'):
        protocol.require(total(refresh, field)['observed_rows'] == len(refresh),
                         'Missing actual refresh metric: '+field)
    cache_bytes = profile['kvmem']['gpu_raw_k_cache_bytes']
    hits = sum(row.get('raw_cache_hit_tokens', 0) for row in refresh+restore)
    protocol.require(cache_bytes or hits == 0, 'Disabled raw-K cache reported hits')
    read_stats = [row['assembly_read_stats'] for row in refresh if 'assembly_read_stats' in row]
    cases = {}
    for row in result['results']:
        directory = output/'cases'/row['id']
        response = json.loads((directory/'response.json').read_text())
        evidence = json.loads((directory/'evidence.json').read_text())
        protocol.require(type(row['completion_tokens']) is int and
            row['completion_tokens'] == len(response['output_ids']) > 0,
            'Performance output count differs from accepted public token IDs')
        protocol.require(type(row['input_tokens']) is int and row['input_tokens'] > 0 and
            all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                for key in ('seconds', 'ttft_seconds')) and row['ttft_seconds'] <= row['seconds'],
            'Invalid completed performance case counters')
        selected = [dict(phase='query', ranges=evidence['query']['selected_history_ranges'])]
        selected += [dict(phase='pressure', ranges=item['selected_history_ranges'])
                     for item in evidence['pressure']]
        cases[row['id']] = dict(input_sha256=row['input_sha256'], seed=row['seed'],
            output_ids_sha256=digest(response['output_ids']), selection_sha256=digest(selected),
            output_tokens=row['completion_tokens'], input_tokens=row['input_tokens'],
            evaluation=row['evaluation'], ttft_seconds=row['ttft_seconds'], seconds=row['seconds'])
    wall = result['wall_seconds']
    begin, end = result.get('measurement_started_at'), result.get('measurement_finished_at')
    protocol.require(len(cases) == len(result['results']) == 8 and
        all(type(value) in (int, float) and math.isfinite(value) for value in (wall, begin, end)) and
        wall > 0 and end > begin and math.isclose(end-begin, wall, rel_tol=1e-9, abs_tol=1e-5),
        'Performance qualification requires original smoke8 and its exact measured wall interval')
    archive = result.get('final_archive', {})
    if not archive and (capture/'archive.json').exists():
        archive = json.loads((capture/'archive.json').read_text())
    return dict(schema='kvmem-graph-v2-performance-3', candidate=candidate, graph_arm='D', cases=cases,
        seal=seal_evidence(archive, profile['kvmem'].get('cpu_direct_record_seal', False)),
        action_metadata=metadata_evidence(capture, profile),
        configured_total_raw_k_bytes=cache_bytes, configured_per_arena_cap_bytes=cache_bytes//2,
        cache_arenas=['main', 'refresh'],
        cache_budget_evidence='Actual worker configuration plus frozen window_assembly.py budget split; allocation is lazy.',
        raw_cache_capacity=cache_capacity_evidence(refresh+restore),
        actual_allocation_observed=any('raw_cache_capacity_bytes' in row for row in refresh+restore),
        raw_cache_hit_tokens=hits, raw_cache_exercised=hits > 0,
        refresh=dict(events=len(refresh), h2d_payload_bytes=total(refresh, 'assembly_uploaded_bytes'),
            host_prepare_seconds=total(refresh, 'assembly_host_prepare_seconds'),
            service=interval_evidence(refresh, 'refresh_started_at', 'refresh_finished_at')),
        restore=dict(events=len(restore), h2d_payload_bytes=total(restore, 'uploaded_bytes'),
            model_h2d_bytes=total(restore, 'model_bytes'), host_prepare_seconds=total(restore, 'host_prepare_seconds'),
            service=interval_evidence(restore, 'started', 'finished')),
        cpu_gather=dict(observed=bool(read_stats), cache_copy_seconds=total(read_stats, 'cache_copy_seconds'),
            cache_gather_tasks=total(read_stats, 'cache_gather_tasks'),
            scope='Dedicated gather counters only when emitted; async host_prepare includes staging and bounded-slot waits, not isolated gather.'),
        h2d_scope='Observed assembly payload bytes and separately reported model restore bytes; excludes uninstrumented metadata H2D.',
        wall_seconds=wall, useful_input_tokens_per_wall_second=sum(row['input_tokens'] for row in cases.values())/wall,
        accepted_output_tokens_per_wall_second=sum(row['output_tokens'] for row in cases.values())/wall,
        measurement_started_at=begin, measurement_finished_at=end,
        measurement_protocol='Cold eight-case cohort; no request warmup, no fixed 300-second window, no replacement workload.',
        rates_scope='Original useful input and accepted output divided by whole fixture wall time; neither is pure decode TPS.',
        ttft_median_seconds=statistics.median(row['ttft_seconds'] for row in cases.values()),
        actual_concurrency=result['actual_concurrency'], step_summary=result['step_summary'],
        quality=dict(macro_coverage=result['macro_coverage'], exact=result['exact']),
        gpu_idle_seconds=None, gpu_idle_inferred=False)


def validate_profile(profile, candidate):
    protocol.require(candidate in CANDIDATES, 'Unknown performance candidate')
    contract = CANDIDATES[candidate]
    kv = profile['kvmem']
    value = kv.get(contract['field'])
    protocol.require(any(type(value) is type(expected) and value == expected
        for expected in contract['variants'].values()) and
        kv.get('async_refresh') and kv.get('direct_pinned_reads'), 'Unexpected performance profile')
    if candidate == 'cpu-direct-record-seal':
        protocol.require(kv['gpu_raw_k_cache_bytes'] == 0 and kv.get('async_seal') is True and
            kv.get('archive_backend') == 'cpu', 'Direct sealing must be isolated from raw-K caching')
    elif contract['field'] == 'gpu_raw_k_cache_bytes':
        protocol.require(kv.get('cpu_direct_record_seal', False) is False, 'Raw-K pair changed sealing policy')
    else:
        protocol.require(kv['gpu_raw_k_cache_bytes'] == 0 and kv.get('cpu_direct_record_seal', False) is False,
                         'Action metadata must be isolated from raw-K and direct sealing')
    if candidate != 'async-action-metadata':
        protocol.require(kv.get('async_action_metadata', False) is False,
                         'Performance pair changed Action metadata policy')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('config', 'capture', 'output', 'fixture'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--arm', choices=('D',), required=True)
    args = parser.parse_args(argv)
    profile = json.loads(args.config.read_text())
    manifest = json.loads((args.output.parent/'manifest.json').read_text())
    candidate = manifest.get('candidate', 'raw-k-cache')
    validate_profile(profile, candidate)
    kv = profile['kvmem']
    launch = json.loads((args.capture.parent/'launch.json').read_text())['vllm_argv']
    worker = json.loads(launch[launch.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    protocol.require(all(worker.get(key, False) == kv.get(key, False) for key in
        ('gpu_raw_k_cache_bytes', 'async_refresh', 'direct_pinned_reads', 'cpu_direct_record_seal',
         'async_seal', 'async_action_metadata')),
        'Actual worker performance configuration differs')
    protocol.require(worker.get('native_action_metadata', False) is kv.get('native_action_metadata', False),
                     'Actual native Action metadata policy differs')
    try:
        qualifier.main(argv)
        metrics = performance(args.capture, args.output, profile, candidate)
        qpath = args.output/'performance.json'
        qualifier.save(qpath, metrics)
        result = json.loads((args.output/'results.json').read_text())
        result['performance_sha256'] = protocol.sha(qpath)
        qualifier.save(args.output/'results.json', result)
    except BaseException as error:
        if args.output.exists():
            qualifier.save(args.output/'client-failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
