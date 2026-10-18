#!/usr/bin/env python3
"""Offline inclusive Action capture timings; never infer idle from nested sums."""
import argparse
import json
import math
from pathlib import Path


def analyze(rows, *, start=None, end=None, time_basis='scope'):
    if time_basis not in ('scope', 'batch'):
        raise ValueError('time_basis must be scope or batch')
    if any(value is not None and not math.isfinite(value) for value in (start, end)):
        raise ValueError('Window bounds must be finite')
    if start is not None and end is not None and start >= end:
        raise ValueError('Window start must precede end')
    rows = list(rows)
    batches = {}
    for row in rows:
        if row.get('event') == 'batch':
            if row['batch_id'] in batches:
                raise ValueError(f"Duplicate batch {row['batch_id']}")
            batches[row['batch_id']] = row
    totals, host_seen, gpu_seen = {}, set(), set()
    summaries = [row for row in rows if row.get('event') == 'summary']
    unmapped = selected_drops = 0

    def include(row):
        nonlocal unmapped
        if time_basis == 'scope':
            timestamp = row.get('started_at')
        else:
            timestamp = batches.get(row['batch_id'], {}).get('started_at')
        if timestamp is None:
            # A capture skip can leave a GPU interval without its batch row.
            # Do not guess its requested window from later reclamation time.
            unmapped += 1
            return False
        if not math.isfinite(timestamp):
            raise ValueError('Invalid original stage/batch timestamp')
        return (start is None or start <= timestamp) and (end is None or timestamp < end)

    def group(row):
        key = row.get('kind', 'batch'), row.get('layer'), row['stage']
        return totals.setdefault(key, dict(kind=key[0], layer=key[1], stage=key[2],
            host_calls=0, host_wall_seconds=0., host_thread_cpu_seconds=0.,
            gpu_calls=0, gpu_elapsed_ms=0., transfer_calls=0, transfer_bytes=0,
            gpu_event_drops=0))

    for batch in batches.values():
        for row in batch['stages']:
            identity = row['batch_id'], row['scope_id']
            if identity in host_seen:
                raise ValueError(f'Duplicate host scope {identity}')
            host_seen.add(identity)
            if not include(row):
                continue
            total = group(row)
            total['host_calls'] += 1
            total['host_wall_seconds'] += row['host_wall_seconds']
            total['host_thread_cpu_seconds'] += row['host_thread_cpu_seconds']
            if 'transfer_bytes' in row:
                total['transfer_calls'] += 1
                total['transfer_bytes'] += row['transfer_bytes']
            if row.get('gpu_event_dropped'):
                total['gpu_event_drops'] += 1
                selected_drops += 1

    for row in rows:
        if row.get('event') != 'gpu_stage':
            continue
        identity = row['batch_id'], row['scope_id']
        if identity in gpu_seen:
            raise ValueError(f'Duplicate completed GPU scope {identity}')
        gpu_seen.add(identity)
        if include(row):
            total = group(row)
            total['gpu_calls'] += 1
            total['gpu_elapsed_ms'] += row['gpu_elapsed_ms']
            # GPU rows copy host metadata; never count host spans twice.

    latest = summaries[-1] if summaries else {}
    return dict(
        window=dict(start=start, end=end, time_basis=time_basis,
                    interval='start inclusive, end exclusive; original monotonic timestamps'),
        stages=sorted(totals.values(), key=lambda row: (-row['host_wall_seconds'],
            row['kind'], row['layer'] or '', row['stage'])),
        selected_host_gpu_event_drops=selected_drops,
        scopes_without_requested_timestamp=unmapped,
        completed_gpu_scopes_without_host_record=len(gpu_seen-host_seen),
        lifetime=dict(counters=latest.get('counters'),
            pending_gpu_event_pairs=latest.get('pending_gpu_event_pairs'),
            max_gpu_event_pairs=latest.get('max_gpu_event_pairs'),
            sample_interval=latest.get('sample_interval'), summary_rows=len(summaries),
            scope='Last cumulative summary for this file; not a window delta. '
                  'Pending events remain unmeasured. Missing summary is null.'),
        interpretation='Stage timings are inclusive. Nested scopes overlap; do not sum stages '
            'to infer total runtime or GPU idle. CUDA intervals may include stream/host enqueue '
            'gaps. GPU rows are selected by original scope/batch start, never by emission or '
            'reclamation time. Transfer bytes are requested payload. Diagnostic overhead and '
            'sample/event drops prohibit direct throughput comparison with an uninstrumented run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--time-basis', choices=('scope', 'batch'), default='scope')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.input.open() as stream:
        result = analyze((json.loads(line) for line in stream if line.strip()),
                         start=args.start, end=args.end, time_basis=args.time_basis)
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
