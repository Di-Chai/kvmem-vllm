"""Attribute measured engine-thread restore intervals without calling them GPU idle."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def union_seconds(intervals, begin, end):
    clipped = sorted((max(a, begin), min(b, end)) for a, b in intervals
                     if a < end and b > begin)
    total, cursor = 0., begin
    for a, b in clipped:
        total += max(0., b - max(cursor, a))
        cursor = max(cursor, b)
    return total


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


def distribution(values):
    return dict(count=len(values), sum=sum(values), mean=statistics.mean(values) if values else None,
                p50=percentile(values, .5), p95=percentile(values, .95), maximum=max(values, default=None))


def validate_row(row):
    stages = row['timing']['stages']
    totals = [stage for stage in stages if stage['name'] == 'restore_main_thread_total']
    if len(totals) != 1:
        raise ValueError('Each restore observation needs exactly one outer interval')
    total = totals[0]
    for stage in stages:
        a, b = stage['started_at'], stage['finished_at']
        if not all(math.isfinite(x) for x in (a, b, stage['wall_seconds'], stage['thread_cpu_seconds'])):
            raise ValueError('Non-finite restore timing')
        if a > b or stage['thread_cpu_seconds'] < 0 or abs(stage['wall_seconds'] - (b - a)) > 1e-6:
            raise ValueError('Invalid restore clock interval')
        if a < total['started_at'] or b > total['finished_at']:
            raise ValueError('Restore substage lies outside its outer interval')
    return total


def analyze(observations, steps, begin, end):
    if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
        raise ValueError('Require a finite positive observation window')
    included, boundary, outside = [], 0, 0
    groups = defaultdict(list)
    for row in observations:
        total = validate_row(row)
        a, b = total['started_at'], total['finished_at']
        if b <= begin or a >= end:
            outside += 1
            continue
        if a < begin or b > end:
            boundary += 1
            continue
        item = dict(row, total=total)
        intervals = [(s['started_at'], s['finished_at']) for s in row['timing']['stages']
                     if s['name'] != 'restore_main_thread_total']
        item['unattributed_host_seconds'] = max(0., b - a - union_seconds(intervals, a, b))
        included.append(item)
        for stage in row['timing']['stages']:
            groups[(row['phase'], row['lane'], stage['name'])].append(stage)
    if not included:
        raise ValueError('No complete restore observation in the requested window')
    stage_summary = [dict(phase=key[0], lane=key[1], stage=key[2],
        wall_seconds=distribution([r['wall_seconds'] for r in values]),
        thread_cpu_seconds=distribution([r['thread_cpu_seconds'] for r in values]),
        wall_minus_thread_cpu_seconds=distribution([r['wall_seconds'] - r['thread_cpu_seconds'] for r in values]))
        for key, values in sorted(groups.items())]
    public = []
    for step in steps:
        a, b = step['started_at'], step.get('model_submitted_at')
        if b is None or a < begin or b > end:
            continue
        matching = [row for row in included
                    if row['total']['started_at'] < b and row['total']['finished_at'] > a]
        if not matching:
            continue
        measured = union_seconds([(r['total']['started_at'], r['total']['finished_at']) for r in matching], a, b)
        public.append(dict(sequence=step['sequence'], started_at=a, model_submitted_at=b,
            decode_requests=step.get('decode_requests'), prefill_tokens=step.get('prefill_tokens'),
            decode_context_bucket=step.get('decode_context_bucket'),
            maintenance_host_seconds=b-a, restore_host_overlap_seconds=measured,
            other_maintenance_seconds=max(0., b-a-measured),
            requests=[dict(request_id=r['request_id'], phase=r['phase'], lane=r['lane'],
                           restore_host_seconds=r['total']['wall_seconds']) for r in matching]))
    longest = sorted(included, key=lambda r:r['total']['wall_seconds'], reverse=True)[:20]
    return dict(window=dict(begin=begin, end=end), observations=len(included),
        boundary_observations_excluded=boundary, outside_observations=outside,
        stages=stage_summary,
        unattributed_host_seconds=distribution([r['unattributed_host_seconds'] for r in included]),
        longest_restores=longest,
        overlapping_public_steps=public,
        slowest_public_steps=sorted(public, key=lambda r:r['maintenance_host_seconds'], reverse=True)[:20],
        scope='Diagnostic only. All stage durations are inclusive and must not be summed. '
              'Union of outer restore intervals is intersected with each public maintenance interval. '
              'Its remainder includes other work, waits and diagnostic logging. Wall minus thread CPU '
              'does not identify GIL waits, native synchronization or GPU idle. Empty lane polls are not logged. '
              'Whole observations crossing formal boundaries are excluded rather than clipped into stage statistics.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = args.arm / 'server/capture'
    window = json.loads((args.arm / 'saturated/window.json').read_text())
    result = analyze(rows(capture / 'restore_host_stages.jsonl'), rows(capture / 'model_step_latency.jsonl'),
                     window['begin'], window['end'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(observations=result['observations'], output=str(args.output))))


if __name__ == '__main__':
    main()
