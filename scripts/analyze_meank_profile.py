#!/usr/bin/env python3
"""Attribute a native Chrome trace without mistaking GPU annotations for CPU scopes."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path


GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
API_CATEGORIES = {'cuda_runtime', 'cuda_driver'}


def interval_union(ranges):
    total, end = 0.0, float('-inf')
    for begin, stop in sorted(ranges):
        total += max(0.0, stop - max(begin, end))
        end = max(end, stop)
    return total


def _interval(event, window):
    begin = max(float(event['ts']), window[0])
    end = min(float(event['ts']) + float(event['dur']), window[1])
    return (begin, end) if end > begin else None


def _totals(events, window, limit=None):
    groups = defaultdict(lambda: [0, 0.0])
    for event in events:
        span = _interval(event, window)
        if span:
            row = groups[event['name']]
            row[0] += 1
            row[1] += span[1] - span[0]
    rows = [dict(name=name, calls=row[0], summed_us=row[1]) for name, row in groups.items()]
    rows.sort(key=lambda row: row['summed_us'], reverse=True)
    return rows if limit is None else rows[:limit]


def analyze_trace(trace, phase='auto'):
    events = [e for e in trace.get('traceEvents', []) if e.get('ph') == 'X'
              and 'ts' in e and 'dur' in e]
    if any(float(e['dur']) < 0 for e in events):
        raise ValueError('Negative event duration')
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('kvmem::')]
    iterations = [e for e in events if e.get('cat') == 'user_annotation'
                  and e.get('name', '').startswith('execute_')]
    decode = sorted([e for e in iterations if 'context_0(' in e['name']
                     and 'generation_8(' in e['name']], key=lambda e: e['ts'])
    gpu = [e for e in events if e.get('cat') in GPU_CATEGORIES]
    if not scopes or not gpu:
        raise ValueError('Trace must contain CPU kvmem scopes and GPU activity')
    if phase == 'auto':
        phase = 'decode' if len(decode) >= 3 else 'prefill'
    complete_cycles = None
    limited = phase != 'decode' or len(decode) < 3
    if not limited:
        window = float(decode[1]['ts']), float(decode[-1]['ts'])
        complete_cycles = len(decode) - 2
        policy = 'second_C8_decode_iteration_start_to_last_start'
    else:
        relevant = scopes + iterations + gpu
        window = min(float(e['ts']) for e in relevant), max(float(e['ts']) + float(e['dur']) for e in relevant)
        policy = 'all_scoped_samples_and_GPU_activity; limited_sample'

    # Build proper CPU scope nesting on each host thread. GPU annotations carry
    # duplicate names and must never enter this tree or host-time calculations.
    by_thread = defaultdict(list)
    for scope in scopes:
        by_thread[scope['pid'], scope['tid']].append(scope)
    children = defaultdict(list)
    scope_index = {}
    for thread, rows in by_thread.items():
        rows.sort(key=lambda e: (e['ts'], -e['dur']))
        stack = []
        for row in rows:
            end = row['ts'] + row['dur']
            while stack and (row['ts'] >= stack[-1]['ts'] + stack[-1]['dur']
                             or end > stack[-1]['ts'] + stack[-1]['dur']):
                stack.pop()
            if stack:
                children[id(stack[-1])].append(row)
            stack.append(row)
        prefix_ends, latest_end = [], float('-inf')
        for row in rows:
            latest_end = max(latest_end, row['ts'] + row['dur'])
            prefix_ends.append(latest_end)
        scope_index[thread] = ([e['ts'] for e in rows], rows, prefix_ends)

    def stage_for(api):
        starts, rows, prefix_ends = scope_index.get((api['pid'], api['tid']), ((), (), ()))
        index = bisect_right(starts, api['ts']) - 1
        for i in range(index, -1, -1):
            if prefix_ends[i] <= api['ts']:
                break  # No earlier scope can contain this unscoped operation.
            scope = rows[i]
            if scope['ts'] <= api['ts'] < scope['ts'] + scope['dur']:
                return scope['name']
        return 'unscoped'

    cpu_rows = defaultdict(lambda: dict(calls=0, inclusive_us=0.0, exclusive_us=0.0))
    for scope in scopes:
        span = _interval(scope, window)
        if span is None:
            continue
        row = cpu_rows[scope['name']]
        row['calls'] += 1
        duration = span[1] - span[0]
        child_spans = [p for child in children[id(scope)] if (p := _interval(child, span))]
        row['inclusive_us'] += duration
        row['exclusive_us'] += max(0.0, duration - interval_union(child_spans))

    apis = [e for e in events if e.get('cat') in API_CATEGORIES]
    correlations = defaultdict(list)
    for api in apis:
        correlation = api.get('args', {}).get('correlation')
        if correlation is not None:
            correlations[correlation].append(api)
    for rows in correlations.values():
        rows.sort(key=lambda e: e['ts'])
    stage_cache = {}
    by_device, by_stage, by_category = defaultdict(list), defaultdict(list), defaultdict(list)
    mapped = unmapped = unscoped = clipped = 0
    stage_events = defaultdict(int)
    for event in gpu:
        span = _interval(event, window)
        if span is None:
            continue
        if span != (float(event['ts']), float(event['ts']) + float(event['dur'])):
            clipped += 1
        device = event.get('args', {}).get('device', event['pid'])
        by_device[device].append(span)
        by_category[device, event['cat']].append(span)
        candidates = correlations.get(event.get('args', {}).get('correlation'), ())
        # Graph kernels legitimately share the one cudaGraphLaunch correlation.
        launch = next((api for api in reversed(candidates) if api['ts'] <= event['ts']), None)
        if launch is None:
            stage = 'unmapped_launch'
            unmapped += 1
        else:
            mapped += 1
            if id(launch) not in stage_cache:
                stage_cache[id(launch)] = stage_for(launch)
            stage = stage_cache[id(launch)]
            unscoped += stage == 'unscoped'
        by_stage[device, stage].append(span)
        stage_events[device, stage] += 1

    cpu = [dict(stage=stage, **row) for stage, row in cpu_rows.items()]
    cpu.sort(key=lambda row: row['exclusive_us'], reverse=True)
    # Attribute CPU torch operators separately from launch scopes. Nested aten
    # operators (to -> _to_copy -> copy_) must not triple-count the same work.
    operators = defaultdict(list)
    for event in events:
        if event.get('cat') == 'cpu_op':
            operators[event['pid'], event['tid']].append(event)
    operator_rows = defaultdict(lambda: dict(calls=0, inclusive_us=0.0, exclusive_us=0.0))
    for rows in operators.values():
        rows.sort(key=lambda e: (e['ts'], -e['dur']))
        stack, direct_children = [], defaultdict(list)
        for row in rows:
            end = row['ts'] + row['dur']
            while stack and (row['ts'] >= stack[-1]['ts'] + stack[-1]['dur']
                             or end > stack[-1]['ts'] + stack[-1]['dur']):
                stack.pop()
            if stack:
                direct_children[id(stack[-1])].append(row)
            stack.append(row)
        for event in rows:
            span = _interval(event, window)
            if span is None:
                continue
            stage = stage_for(event)
            if stage == 'unscoped':
                continue
            row = operator_rows[stage, event['name']]
            row['calls'] += 1
            row['inclusive_us'] += span[1] - span[0]
            nested = [p for child in direct_children[id(event)] if (p := _interval(child, span))]
            row['exclusive_us'] += max(0.0, span[1] - span[0] - interval_union(nested))
    cpu_operators = [dict(stage=stage, operator=name, **row)
                     for (stage, name), row in operator_rows.items()]
    cpu_operators.sort(key=lambda row: row['exclusive_us'], reverse=True)
    gpu_stages = [dict(device=device, stage=stage, events=stage_events[device, stage],
                       summed_us=sum(b-a for a, b in spans), interval_union_us=interval_union(spans))
                  for (device, stage), spans in by_stage.items()]
    gpu_stages.sort(key=lambda row: row['summed_us'], reverse=True)
    duration = window[1] - window[0]
    return dict(
        phase=phase, performance_eligible=False,
        window=dict(start_us=window[0], end_us=window[1], duration_us=duration,
                    policy=policy, complete_cycles=complete_cycles, limited_sample=limited,
                    observed_iteration_scopes=len(iterations), observed_C8_decode_scopes=len(decode)),
        gpu_timelines=[dict(device=device, interval_union_us=interval_union(spans),
                            summed_activity_us=sum(b-a for a, b in spans),
                            uncovered_interval_us=max(0.0, duration-interval_union(spans)))
                       for device, spans in by_device.items()],
        gpu_categories=[dict(device=device, category=category, events=len(spans),
                             summed_us=sum(b-a for a, b in spans), interval_union_us=interval_union(spans))
                        for (device, category), spans in by_category.items()],
        gpu_attribution=dict(mapped_launch_events=mapped, unmapped_launch_events=unmapped,
                             mapped_but_unscoped_events=unscoped, boundary_clipped_events=clipped),
        gpu_stages=gpu_stages, cpu_stages=cpu,
        cpu_operators=cpu_operators,
        top_kernels=_totals([e for e in gpu if e['cat'] == 'kernel'], window, 30),
        memcpy_events=_totals([e for e in gpu if e['cat'] == 'gpu_memcpy'], window),
        cuda_api_waits=_totals([e for e in apis if any(s in e['name'].lower()
                                  for s in ('synchronize', 'waitevent', 'eventquery'))], window),
        cuda_api_totals=_totals(apis, window, 25),
        profiler_overhead=_totals([e for e in events if e.get('cat') == 'overhead'], window),
        limitations=[
            'Instrumented sample is not a throughput benchmark; profiler overhead changes timing.',
            'GPU union includes traced kernel/memcpy/memset only; uncovered intervals do not prove hardware idle.',
            'GPU stages use the innermost CPU kvmem launch scope, including shared CUDA graph launch correlations.',
            'CPU exclusive time excludes nested kvmem scopes only; it includes CUDA API waits and unlabelled work.',
            'CPU operator exclusive time excludes nested cpu_op intervals; operator rows are instrumented host costs, not uninstrumented savings.',
            'Stage GPU unions can overlap each other across streams and must not be summed as wall time.',
            'CUDA wait API duration is host waiting; event queries and stream wait insertion are not all blocking.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--phase', choices=('auto', 'decode', 'prefill'), default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt') as stream:
        result = analyze_trace(json.load(stream), args.phase)
    result['trace'] = str(args.trace.resolve())
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
