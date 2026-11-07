"""Read-only accounting of bounded final-Query diagnostic stage records."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    requests = []
    for record in records:
        if any(record[key] for key in ('unresolved_gpu_events', 'dropped_rows', 'dropped_gpu_events')):
            raise ValueError('Incomplete Query stage observation')
        rows = record['stages']
        timed = [row for row in rows if 'sequence' in row]
        by_id = {row['sequence']: row for row in timed}
        if len(by_id) != len(timed):
            raise ValueError('Duplicate stage sequence')
        totals = defaultdict(lambda: dict(calls=0, host_inclusive_seconds=0.,
            host_exclusive_seconds=0., thread_cpu_inclusive_seconds=0., gpu_event_ms=0.))
        archive = defaultdict(float)
        for row in rows:
            if row['stage'] == 'archive_report':
                for key, value in row['stats'].items():
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        archive[key] += value
        for row in timed:
            if 'error_type' in row or row['finished_at'] < row['started_at']:
                raise ValueError('Failed or negative diagnostic stage')
            children = sorted((child['started_at'], child['finished_at']) for child in timed
                              if child['parent'] == row['sequence'])
            covered = 0.
            previous = row['started_at']
            for start, end in children:
                if start < row['started_at'] or end > row['finished_at']:
                    raise ValueError('Stage nesting crossed its parent')
                covered += max(0., end - max(previous, start))
                previous = max(previous, end)
            total = totals[row['stage']]
            total['calls'] += 1
            total['host_inclusive_seconds'] += row['wall_seconds']
            total['host_exclusive_seconds'] += max(0., row['wall_seconds'] - covered)
            total['thread_cpu_inclusive_seconds'] += row['thread_cpu_seconds']
            total['gpu_event_ms'] += row.get('gpu_elapsed_ms', 0.)
        requests.append(dict(request_id=record['request_id'], stages=dict(totals),
            archive_report_sums=dict(archive), first_observed_at=min(r['started_at'] for r in timed),
            last_observed_at=max(r['finished_at'] for r in timed)))
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        requests=requests, caveats=[
            'Diagnostic runs are for attribution; performance requires separate runs with the probe disabled.',
            'Exclusive host time only removes child spans from this collector, not descheduling or GPU waits.',
            'CUDA event intervals include stream contention and host launch gaps; they are not pure kernel time.',
            'Archive copy/queue statistics overlap the surrounding host read spans; do not add them.',
            'No difference from a shared-selection control is implied by these layerwise-only records.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.capture/'query_stage_profile.jsonl')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for request in report['requests']:
        print(request['request_id'])
        for name, values in sorted(request['stages'].items(),
                                   key=lambda item: -item[1]['host_exclusive_seconds'])[:10]:
            print(name, round(values['host_inclusive_seconds'], 4),
                  round(values['host_exclusive_seconds'], 4), round(values['gpu_event_ms'], 3))
