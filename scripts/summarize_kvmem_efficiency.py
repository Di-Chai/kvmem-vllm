"""Summarize a fixed-token efficiency probe with separate clock domains.

CUDA-event intervals include launch/wait time, and nested host stages are
inclusive. Neither metric is GPU kernel time or proof of GPU idleness.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

try:
    from .gpu_efficiency_monitor import summarize as summarize_gpu
except ImportError:
    from gpu_efficiency_monitor import summarize as summarize_gpu


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines(keepends=True):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith('\n'):
                raise
    return result


def distribution(values):
    values = sorted(values)
    def percentile(fraction):
        if not values:
            return None
        index = (len(values)-1)*fraction
        lo = int(index)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(index-lo)
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values) if values else None,
                p50=percentile(.5), p95=percentile(.95), maximum=max(values, default=None))


def phase(row):
    prefill, decode = row.get('prefill_tokens', 0)>0, row.get('decode_requests', 0)>0
    return 'mixed' if prefill and decode else 'prefill' if prefill else 'decode' if decode else 'other'


def step_summary(values, start, end):
    included = [row for row in values if start <= row['started_at'] and row['submitted_at'] <= end]
    included.sort(key=lambda row:row['call_index'])
    groups = defaultdict(list)
    by_call = {r['call_index']:r for r in included}
    for row in included:
        groups[phase(row)].append(row)
    output = {}
    for name, group in groups.items():
        consecutive = [row for row in group if
            row.get('previous_call_index') == row['call_index']-1 and
            row['call_index']-1 in by_call]
        output[name] = dict(samples=len(group), prefill_tokens=sum(r.get('prefill_tokens',0) for r in group),
            decode_request_steps=sum(r.get('decode_requests',0) for r in group),
            metrics_ms={key:distribution([r[key] for r in group if r.get(key) is not None])
                for key in ('step_ms','model_ms','maintenance_ms','maintenance_host_ms','host_submit_ms')},
            consecutive_previous_end_to_model_ms=distribution([r['previous_end_to_model_ms'] for r in consecutive
                if r.get('previous_end_to_model_ms') is not None]),
            consecutive_host_between_submissions_ms=distribution([
                max(0.,r['started_at']-by_call[r['call_index']-1]['submitted_at'])*1000 for r in consecutive]),
            active_context_tokens=distribution([g['active_seq_len'] for r in group for g in r.get('batch_geometry',[])]))
    return dict(samples=len(included), boundary_or_outside_rows=len(values)-len(included),
        missing_call_indices=(included[-1]['call_index']-included[0]['call_index']+1-len(included)) if included else None,
        phases=output,
        scope='Metrics describe sampled public batches. Model CUDA-event time includes sampling/MTP/capture '
              'and launch gaps. Previous-end-to-model gaps require consecutive call_index and include maintenance; '
              'they must not be called GPU idle. No extrapolation across unsampled calls.'), included


def interval_union(intervals):
    total = 0.
    cursor = -math.inf
    for start,end in sorted(intervals):
        total += max(0.,end-max(start,cursor))
        cursor = max(cursor,end)
    return total


def host_summary(values, steps, start, end):
    included = [row for row in values if start<=row['started_at'] and row['finished_at']<=end]
    # Host span identities are unique only within each worker/thread.
    by_id = {(row['thread'],row['span']):row for row in included}
    by_sequence = {row['sequence']:row for row in steps}
    engine_threads = {row['thread'] for row in included if row['stage']=='runner.execute_model'}
    children = defaultdict(list)
    for row in included:
        children[(row['thread'],row.get('parent'))].append(row)
    groups = defaultdict(list)
    for row in included:
        ancestor = row
        while ancestor.get('parent') is not None:
            parent = by_id.get((ancestor['thread'],ancestor['parent']))
            if parent is None:
                break
            ancestor = parent
        step = by_sequence.get(ancestor.get('sequence'))
        category = 'background_or_between_steps'
        if ancestor['stage']=='runner.execute_model':
            # A maintenance-only runner call keeps the previous global sequence.
            # Require that the measured public step actually began in this call.
            category = (phase(step) if step and ancestor['started_at']<=step['started_at']<ancestor['finished_at']
                        else 'maintenance_only')
        elif (step and ancestor['thread'] in engine_threads and
              step['started_at']<=ancestor['started_at'] and ancestor['finished_at']<=step['submitted_at']):
            # Sampling/capture can be on the engine thread after execute_model
            # returns but before the public-step end event is recorded.
            category = phase(step)
        direct = children.get((row['thread'],row['span']),[])
        covered = interval_union([(max(row['started_at'],c['started_at']),
                                   min(row['finished_at'],c['finished_at'])) for c in direct])
        inclusive_cpu = row['thread_cpu_seconds']
        item = dict(wall=row['wall_seconds'], cpu=inclusive_cpu,
                    exclusive_wall=max(0.,row['wall_seconds']-covered),
                    exclusive_cpu=max(0.,inclusive_cpu-sum(c['thread_cpu_seconds'] for c in direct)))
        groups[(category,row['stage'])].append(item)
    stages = [dict(phase=category,stage=stage,calls=len(group),
        inclusive_wall_seconds=distribution([r['wall'] for r in group]),
        inclusive_thread_cpu_seconds=distribution([r['cpu'] for r in group]),
        exclusive_wall_seconds=distribution([r['exclusive_wall'] for r in group]),
        exclusive_thread_cpu_seconds=distribution([r['exclusive_cpu'] for r in group]),
        exclusive_wall_minus_thread_cpu_seconds=distribution([max(0.,r['exclusive_wall']-r['exclusive_cpu']) for r in group]))
        for (category,stage),group in groups.items()]
    stages.sort(key=lambda r:r['exclusive_wall_seconds']['total'],reverse=True)
    return dict(spans=len(included), boundary_or_outside_spans=len(values)-len(included), stages=stages,
        scope='Inclusive stages nest and cannot be summed. Exclusive wall subtracts direct same-thread child '
              'interval unions. Wall minus thread CPU includes waits/descheduling, not measured GPU idle. '
              'Background spans remain unclassified by model phase. Across threads, stage totals can overlap.')


def numeric_delta(before, after):
    output = {}
    for key,value in after.items():
        # CaptureRuntime._publication_metric explicitly initializes absent
        # counters to zero. This helper is used only for that cumulative map.
        old = before.get(key, 0)
        if isinstance(value,dict):
            child = numeric_delta(old if isinstance(old,dict) else {},value)
            if child:
                output[key]=child
        elif type(value) in (int,float) and type(old) in (int,float):
            output[key]=value-old
    return output


def analyze(probe, run):
    summary = read(probe/'summary.json')
    if not summary:
        raise ValueError('A completed probe summary is required')
    start,end = summary['started_monotonic'],summary['finished_monotonic']
    if not all(math.isfinite(x) for x in (start,end)) or end<=start:
        raise ValueError('Invalid monotonic probe interval')
    capture = probe/'capture'
    steps,included_steps = step_summary(rows(capture/'model_step_latency.jsonl'),start,end)
    host = host_summary(rows(capture/'host_stages.jsonl'),included_steps,start,end)
    gpu_rows = rows(run/'gpu-utilization.jsonl')
    gpu = summarize_gpu(gpu_rows,summary['started_at'],summary['finished_at']) if gpu_rows else dict(error='No GPU samples')
    before,after = read(probe/'before.json',{}),read(probe/'after-work.json',{})
    archive_before,archive_after = before.get('archive',{}),after.get('archive',{})
    progress_before,progress_after = before.get('progress',{}),after.get('progress',{})
    archive_keys = ('gc_wait_seconds','gc_wait_count','seal_pack_seconds','seal_hash_seconds','seal_cache_seconds',
                    'seal_record_copy_seconds','seal_record_publish_seconds','seal_payload_bytes','seal_direct_bytes')
    progress_keys = ('cpu_scatter_seconds','cpu_scatter_join_seconds','cpu_scatter_batches',
                     'batched_append_calls','batched_append_requests','full_graph_decode_calls','batches_submitted')
    counters = dict(archive={key:archive_after[key]-archive_before[key] for key in archive_keys
                            if key in archive_after and key in archive_before},
        progress={key:progress_after[key]-progress_before[key] for key in progress_keys
                  if key in progress_after and key in progress_before},
        publication=numeric_delta(progress_before.get('publication_metrics',{}),progress_after.get('publication_metrics',{})),
        gc=summary.get('gc_delta'),
        scope='Cumulative counter differences from before and after-work snapshots. Async export can lag the '
              'request boundary; stage timers and publication counters overlap. GC wait measures foreground '
              'capacity blocking, not background policy CPU time (not measured by these counters).')
    request_groups=defaultdict(list)
    for request in read(probe/'requests.json',[]):
        request_groups[f"turn-{request['turn']}/{request['part']}"].append(request['seconds'])
    return dict(probe=str(probe),run=str(run),label=summary['label'],passed=summary['passed'],
        wall_seconds=summary['wall_seconds'],completed_tokens_per_second=summary['completed_tokens_per_second'],
        fixture_sha256=summary['fixture_sha256'],
        window=dict(start_epoch=summary['started_at'],end_epoch=summary['finished_at'],
                    start_monotonic=start,end_monotonic=end),
        gpu=gpu,model_steps=steps,host_stages=host,counters=counters,native=summary.get('native'),
        request_seconds={key:distribution(value) for key,value in request_groups.items()},
        host_trace_status=read(run/'model/server/capture/host_stages.status.json'),
        scope='Fixed output component diagnostic, not Terminal-Bench quality or natural-generation throughput. '
              'GPU metrics use epoch time; model/host stages use probe monotonic bounds on the same Linux host. '
              'Only whole model/host observations inside the workload interval are included; cleanup is excluded.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.probe,args.run)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),passed=result['passed'],wall_seconds=result['wall_seconds'],
                         model_step_samples=result['model_steps']['samples'],host_spans=result['host_stages']['spans'])))


if __name__ == '__main__':
    main()
