"""Read-only accepted-authority sampling during an existing serving workload.

Stable intervals exclude request changes, newly scheduled prefills and active
refreshes. One preceding interval is also excluded after a prefill to avoid
counting its trailing accepted capture as decode. Zero-progress intervals remain
in the denominator. This is approximate archive-commit throughput, not SSE arrival
throughput, model quality or a matched performance acceptance result.
"""
import argparse
import json
from pathlib import Path
import time


def read_sample(capture):
    # Scheduler counters bracket the other reads. Reject a prefill racing these
    # reads; ordinary decode may advance without making the sample unusable.
    first = json.loads((capture/'scheduled_queries.json').read_text())['queries']
    progress = json.loads((capture/'progress.json').read_text())
    scheduler = json.loads((capture/'scheduler_resources.json').read_text())
    last = json.loads((capture/'scheduled_queries.json').read_text())['queries']
    return {'at': time.monotonic(), 'unix': time.time(), 'progress': progress, 'scheduler': scheduler,
            'prefill': sum(last['prefill'].values()), 'decode': sum(last['decode'].values()),
            'prefill_raced': first['prefill'] != last['prefill']}


def steady_interval(older, before, after, concurrency):
    samples = (older, before, after)
    if any(s.get('error') or s['prefill_raced'] for s in samples):
        return None
    owners = [s['progress']['accepted_history'] for s in samples]
    if (any(len(o) != concurrency for o in owners) or
            any(set(o) != set(owners[0]) for o in owners[1:]) or
            len({s['prefill'] for s in samples}) != 1):
        return None
    if any(s['scheduler']['requests'] != concurrency or s['scheduler']['refresh_pending'] or
           s['scheduler']['connector_pending'] or s['progress']['retiring_requests'] for s in samples):
        return None
    elapsed = after['at'] - before['at']
    deltas = {key: owners[2][key] - owners[1][key] for key in owners[1]}
    if elapsed <= 0 or any(v < 0 for v in deltas.values()) or after['decode'] < before['decode']:
        return None
    return {'start': before['at'], 'end': after['at'], 'seconds': elapsed,
            'accepted_input_tokens': sum(deltas.values()), 'per_request_tokens': deltas,
            'scheduled_verifier_queries': after['decode'] - before['decode']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=180)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--concurrency', type=int, default=2)
    args = parser.parse_args()
    if args.samples < 3 or not .25 <= args.interval <= 10 or args.concurrency < 1:
        parser.error('Need >=3 samples, 0.25–10 second interval and positive concurrency')
    args.output.mkdir(parents=True, exist_ok=False)
    samples, intervals = [], []
    started = time.monotonic()
    with (args.output/'samples.jsonl').open('x') as stream:
        for i in range(args.samples):
            deadline = started+i*args.interval
            if deadline > time.monotonic():
                time.sleep(deadline-time.monotonic())
            try:
                sample = read_sample(args.capture)
            except (OSError, ValueError, KeyError) as error:
                sample = {'at': time.monotonic(), 'error': repr(error)}
            samples.append(sample)
            stream.write(json.dumps(sample)+'\n'); stream.flush()
            if len(samples) >= 3:
                interval = steady_interval(*samples[-3:], args.concurrency)
                if interval is not None:
                    intervals.append(interval)
    seconds = sum(x['seconds'] for x in intervals)
    tokens = sum(x['accepted_input_tokens'] for x in intervals)
    report = {'scope': __doc__, 'capture': str(args.capture.resolve()), 'concurrency': args.concurrency,
        'wall_seconds': samples[-1]['at']-samples[0]['at'], 'sample_count': len(samples),
        'sample_errors': sum('error' in s for s in samples), 'steady_seconds': seconds,
        'accepted_input_tokens': tokens, 'accepted_input_tokens_per_second': tokens/seconds if seconds else None,
        'intervals': intervals, 'performance_eligible': False,
        'limitations': 'Snapshots update periodically and across separate files. Endpoint lag and short in-flight work can skew estimates; use native request timing and completed matched benchmarks for acceptance.'}
    (args.output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('scope', 'intervals')}))


if __name__ == '__main__':
    main()
