"""Post-run audit of the finite prefill ABBA; never launches or retries inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from audit_layerwise_gilopt import compare, save_new


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p90=values[min(len(values)-1, int(len(values)*.9))])


def phases(root):
    capture = root/'run/server/capture'
    release, = [r for r in lines(capture/'prefill_phase_gate.jsonl')
                if r['event'] == 'history_prefill_released']
    rows = [e for p in (root/'run/answers/cases').glob('*/events.json') for e in read(p)]
    pressure = {(e['request_id'], e['command_id']) for e in rows
                if e['selection'] == 'pressure' and e['refresh_finished_at'] <= release['at']}
    pressure_rows = [e for e in rows if (e['request_id'], e['command_id']) in pressure]
    assert len(pressure_rows) == len(pressure)
    uploads = [e['assembly_uploaded_bytes'] for e in pressure_rows]
    resident = [e.get('assembly_read_stats', {}).get('layerwise_resident_v_reused_tokens', 0)
                for e in pressure_rows]
    groups = defaultdict(dict)
    for e in lines(capture/'async_refresh.jsonl'):
        key = e['request_id'], e['command_id']
        if key in pressure:
            if e['stage'] in groups[key]:
                raise ValueError('Repeated pressure stage')
            groups[key][e['stage']] = e
    assert len(groups) == len(pressure) == read(root/'phase-measurement.json')['prefill_pressure']
    boundaries = [('suspended', 'ready'), ('assembly_queued', 'assembly_started'),
                  ('assembly_started', 'assembly_submitted'), ('assembly_submitted', 'ready')]
    durations = {}
    for start, end in boundaries:
        values = [g[end]['at'] - g[start]['at'] for g in groups.values()]
        assert min(values) >= 0
        durations[start+'_to_'+end] = stats(values)
    steps = [s for s in lines(capture/'model_step_latency.jsonl')
             if s['started_at'] < release['at']]
    assert all(s['decode_requests'] == 0 for s in steps)
    traced_tokens = sum(s['prefill_tokens'] for s in steps)
    return dict(pressure_seconds=durations, historical_public_steps=len(steps),
                pressure_uploaded_bytes=sum(uploads),
                mean_pressure_uploaded_bytes=statistics.mean(uploads),
                resident_v_layer_tokens=sum(resident),
                pressure_events_with_resident_v=sum(v > 0 for v in resident),
                traced_public_prefill_tokens=traced_tokens,
                traced_token_fraction=traced_tokens/release['history_tokens'],
                model_ms_sum=sum(s['model_ms'] for s in steps),
                maintenance_ms_sum=sum(s['maintenance_ms'] for s in steps),
                scope='Per-event durations overlap across owners and must not be added as wall time. '
                'assembly_started follows layer selection; queued interval includes selection. '
                'submitted-to-ready includes completion observation, not just GPU copy. '
                'Existing CUDA-event model/maintenance sums are supporting diagnostics, not throughput. '
                'Early initialization batches can precede timing registration; report coverage explicitly. '
                'Model event intervals include host submission gaps, sampling, MTP and capture, '
                'and are not isolated attention kernel time.')


def analyze(root, output):
    summary = read(root/'comparison.json')
    assert read(root/'status.json')['status'] == 'completed_measurement'
    output.mkdir(parents=True, exist_ok=False)
    reports = summary['reports']
    detail = {r['name']: phases(root/r['name']) for r in reports}
    matches = {}
    # A/A and B/B measure intrinsic multi-owner reproducibility; do not rerun
    # either implementation to hunt for a matching sampled response.
    for before, after in [('01-A', '04-A'), ('02-B', '03-B'), ('01-A', '02-B')]:
        report = compare(root/before, root/after)
        name = before+'_vs_'+after
        save_new(output/(name+'.json'), report)
        matches[name] = dict(cases=report['cases'], passed=report['passed'],
                             equal_counts=report['equal_counts'])
    rates = {kind: [r['tokens_per_second'] for r in reports if r['kind'] == kind]
             for kind in ('baseline', 'candidate')}
    result = dict(pooled_rates=summary['pooled_rates'],
        observed_change=summary['observed_change'],
        per_run_rates=rates,
        candidate_runs_faster_than_both_baselines=min(rates['candidate']) > max(rates['baseline']),
        phases=detail, exact_equivalence=matches,
        serial_semantic_qualification=summary['serial_semantic_qualification'],
        continuous_serving_acceptance=False,
        scope='Two observations per arm in a predeclared ABBA, not a statistical confidence interval. '
              'Phased complete-input historical prefill only. Full answers preserved, no external scoring.')
    save_new(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'phases'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve())
