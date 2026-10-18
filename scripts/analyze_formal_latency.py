"""Validate formal-run histogram coverage and report native output intervals.

This is postprocessing only. It does not alter the frozen server or workload.
Native output intervals can cover multiple MTP tokens, unlike request mean TPOT.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from observe_formal_latency import delta_histogram, read
from analyze_agent_scheduling import distribution


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


def analyze(run, events, samples):
    starts = [r for r in events if r['event'] == 'start']
    finishes = [r for r in events if r['event'] == 'finish']
    valid = [s for s in samples if 'metrics' in s and
             s['preflight_marker_mtime'] <= s['started_at'] <= s['finished_at']]
    provenance = {(s['observer_sha256'], s['interval_seconds']) for s in valid}
    if len(provenance) > 1:
        raise ValueError('Observer or scrape interval changed within the run')
    report = dict(complete_formal_coverage=False, formal_requests_started=len(starts),
        formal_requests_finished=len(finishes), valid_snapshots=len(valid),
        failed_snapshots=sum('error' in s for s in samples),
        scope='Native EngineCoreOutput event intervals, possibly several MTP tokens per event. '
              'Quantiles are histogram bucket bounds. Request mean TPOT is a separate metric.')
    if not starts:
        return report | dict(reason='No formal requests yet')
    first = min(r.get('request_started_at', r['at']) for r in starts)
    baseline = [s for s in valid if s['finished_at'] <= first]
    if not baseline:
        return report | dict(reason='No clean baseline after preflight and before the first formal request')
    before = max(baseline, key=lambda s: s['finished_at'])
    after = max(valid, key=lambda s: s['finished_at'])
    terminals = [r['at'] for r in events if r['event'] in
                 ('finish', 'cancelled', 'error', 'history_released')]
    last = max(terminals, default=first)
    pending = Counter()
    unmatched_terminals = 0
    for event in events:
        if event['event'] not in ('start', 'finish', 'error', 'cancelled'):
            continue
        key = (event['task'], event['turn'])
        if event['event'] == 'start':
            pending[key] += 1
        elif pending[key]:
            pending[key] -= 1
        else:
            unmatched_terminals += 1
    tasks = run.get('tasks', {})
    complete = (run['status'] == 'completed' and len(tasks) == 16 and
                all(t['state'] == 'completed' for t in tasks.values()) and
                not any(pending.values()) and not unmatched_terminals and
                after['started_at'] >= last and after['started_at'] > before['finished_at'])
    native = delta_histogram(before['metrics'], after['metrics'])
    measured = [r for r in finishes if r.get('metrics')]
    ttft = [r['metrics']['time_to_first_token_ms']/1000 for r in measured
            if r['metrics'].get('time_to_first_token_ms') is not None]
    tpots = [r['metrics']['generation_time_ms']/1000/(r['usage']['completion_tokens']-1)
             for r in measured if r['usage']['completion_tokens'] > 1 and
             r['metrics'].get('generation_time_ms') is not None]
    return report | dict(complete_formal_coverage=complete,
        reason='Complete formal interval covered' if complete else 'Partial execution or missing final boundary',
        window=dict(first_formal_request_at=first, last_formal_terminal_at=last,
            baseline_started_at=before['started_at'], baseline_finished_at=before['finished_at'],
            final_started_at=after['started_at'], final_finished_at=after['finished_at'],
            baseline_finished_monotonic=before.get('finished_monotonic'),
            final_started_monotonic=after.get('started_monotonic')),
        observer_sha256=before['observer_sha256'], interval_seconds=before['interval_seconds'],
        pending_model_requests=sum(pending.values()), unmatched_terminal_events=unmatched_terminals,
        native_output_intervals=native,
        request_metrics=dict(requests_with_native_metrics=len(measured),
            time_to_first_token_seconds=distribution(ttft),
            mean_decode_seconds_per_output_token=distribution(tpots),
            scope='Distributions over completed formal requests; TPOT quantiles describe request means, '
                  'not individual token/event tail latency.'),
        error_events=sum(r['event'] == 'error' for r in events),
        cancelled_events=sum(r['event'] == 'cancelled' for r in events))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(read(args.root/'run.json'), rows(args.root/'bridge/requests.jsonl'),
                     rows(args.root/'latency-metrics.jsonl'))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('native_output_intervals', 'request_metrics')}))
