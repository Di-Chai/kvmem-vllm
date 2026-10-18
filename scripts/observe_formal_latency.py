"""Read existing Prometheus histograms; never modify an inference process.

The native inter-token metric records EngineCoreOutput intervals. With MTP an
output can contain several tokens; this is not a per-token delivery histogram.
Snapshots must be bounded by the actual bridge events before claiming complete
formal-run coverage. Histogram quantiles are reported as bucket bounds.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request

HISTOGRAM = 'vllm:inter_token_latency_seconds'


def parse_metrics(body):
    values = {}
    buckets = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0]
        if name == HISTOGRAM + '_bucket':
            labels, number = line.rsplit(' ', 1)
            match = re.search(r'(?:\{|,)le="([^"]+)"', labels)
            if match is None or match[1] in buckets:
                raise ValueError('Require one engine/model and unique buckets')
            buckets[match[1]] = float(number)
        elif name in (HISTOGRAM + '_sum', HISTOGRAM + '_count',
                      'vllm:generation_tokens_total', 'vllm:num_preemptions_total'):
            if name in values:
                raise ValueError('Require one engine/model')
            values[name] = float(line.rsplit(' ', 1)[1])
    required = {HISTOGRAM + '_sum', HISTOGRAM + '_count',
                'vllm:generation_tokens_total', 'vllm:num_preemptions_total'}
    if set(values) != required or '+Inf' not in buckets:
        raise ValueError('Incomplete native latency metrics')
    if any(not math.isfinite(v) or v < 0 for v in [*values.values(), *buckets.values()]):
        raise ValueError('Invalid native counter value')
    if buckets['+Inf'] != values[HISTOGRAM + '_count']:
        raise ValueError('Histogram count mismatch')
    counts = [buckets[k] for k in sorted(buckets, key=float)]
    if any(a > b for a, b in zip(counts, counts[1:])):
        raise ValueError('Non-monotonic cumulative buckets')
    return dict(values=values, buckets=buckets)


def delta_histogram(before, after):
    if set(before['buckets']) != set(after['buckets']):
        raise ValueError('Histogram boundaries changed')
    buckets = {k: after['buckets'][k]-before['buckets'][k] for k in before['buckets']}
    values = {k: after['values'][k]-before['values'][k] for k in before['values']}
    if any(v < 0 for v in [*buckets.values(), *values.values()]):
        raise ValueError('Counter reset or reversed snapshots')
    ordered = sorted(buckets, key=float)
    if any(buckets[a] > buckets[b] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Invalid histogram difference')
    count = values[HISTOGRAM+'_count']
    bounds = {}
    for label, quantile in (('p50', .5), ('p95', .95), ('p99', .99)):
        lower = 0.
        bounds[label] = None
        if count:
            for boundary in ordered:
                if buckets[boundary] >= quantile*count:
                    bounds[label] = dict(lower_seconds=lower,
                        upper_seconds=None if boundary == '+Inf' else float(boundary))
                    break
                lower = float(boundary)
    return dict(events=count, mean_seconds=values[HISTOGRAM+'_sum']/count if count else None,
                quantile_bucket_bounds=bounds, counter_deltas=values, bucket_deltas=buckets)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def observe(root, interval):
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    previous = {}
    while True:
        state = read(root/'status.json')
        terminal = state.get('status') in ('failed', 'interrupted', 'completed_pending_review')
        for arm in ('A0', 'P1'):
            # Both arms reuse the same HTTP port. Never attribute the next
            # server's counters to the preceding completed arm.
            if state.get('stage') != 'deepswe-'+arm:
                continue
            directory = root/('deepswe-'+arm)
            run = read(directory/'run.json')
            markers = [directory/name for name in ('atomic-query-preflight.json',
                'short-tail-preflight.json', 'generation-preflight.json', 'cleanup-preflight.json')]
            if not run or not all(read(p).get('passed') is True for p in markers):
                continue
            if time.monotonic()-previous.get(arm, -1e30) < interval and not terminal:
                continue
            profile = read(directory/'profile.json')
            started = time.time()
            sample = dict(started_at=started, started_monotonic=time.monotonic(),
                run_phase_before=run['status'], observer_sha256=own_hash,
                interval_seconds=interval, preflight_marker_mtime=max(p.stat().st_mtime for p in markers))
            try:
                with client.open(f"http://127.0.0.1:{profile['port']}/metrics", timeout=5) as response:
                    sample['metrics'] = parse_metrics(response.read().decode())
                sample['run_phase_after'] = read(directory/'run.json').get('status')
            except (OSError, ValueError) as error:
                sample['error'] = repr(error)
            sample['finished_at'] = time.time()
            sample['finished_monotonic'] = time.monotonic()
            with (directory/'latency-metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            previous[arm] = time.monotonic()
        if terminal:
            return
        time.sleep(.25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('Keep read-only monitoring to at most one scrape per second')
    observe(args.root.resolve(), args.interval)
