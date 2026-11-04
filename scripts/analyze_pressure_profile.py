"""Attribute recorded refresh intervals without adding concurrent owner waits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * .95
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return dict(count=len(values), total=sum(values), mean=statistics.fmean(values),
                median=statistics.median(values), p95=values[lo] + (at-lo)*(values[hi]-values[lo]),
                maximum=values[-1])


def refresh_intervals(events, reports):
    jobs = defaultdict(dict)
    for event in events:
        key = event['request_id'], event['command_id']
        stage = event['stage']
        if stage in ('suspended', 'assembly_queued', 'assembly_started',
                     'selection_prepared', 'assembly_submitted', 'ready'):
            if stage in jobs[key]:
                raise ValueError('Duplicate refresh stage')
            jobs[key][stage] = event
    rows = []
    for report in reports:
        key = report['request_id'], report['command_id']
        stages = jobs[key]
        names = ['suspended', 'assembly_queued', 'assembly_started']
        if 'selection_prepared' in stages:
            names.append('selection_prepared')
        names += ['assembly_submitted', 'ready']
        if any(name not in stages for name in names):
            raise ValueError('Completed refresh lacks its lifecycle stages')
        times = [stages[name]['at'] for name in names]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Refresh lifecycle is not ordered')
        spans = {a + '_to_' + b: t1-t0 for a,b,t0,t1 in zip(names,names[1:],times,times[1:])}
        rows.append(dict(request_id=key[0], command_id=key[1], phase=report['selection'],
            start=times[0], end=times[-1], seconds=times[-1]-times[0], stages_seconds=spans,
            publication_seconds=stages.get('selection_prepared', {}).get('publication_seconds'),
            selection_seconds=stages.get('selection_prepared', {}).get('selection_seconds'),
            uploaded_bytes=report['assembly_uploaded_bytes'],
            assembly_host_seconds=report['assembly_seconds'],
            assembly_read_stats=report.get('assembly_read_stats')))
    return rows


def held_concurrency(rows):
    """A timeline of held refresh jobs; never a measurement of GPU idleness."""
    changes = defaultdict(int)
    for row in rows:
        changes[row['start']] += 1
        changes[row['end']] -= 1
    seconds = defaultdict(float)
    active, previous = 0, None
    for at, delta in sorted(changes.items()):
        if previous is not None:
            seconds[active] += at - previous
        active += delta
        previous = at
    if active:
        raise ValueError('Unbalanced held-job timeline')
    return dict(sorted(seconds.items()))


def layerwise_query_intervals(events, refreshes):
    """Keep initial global refresh and causal per-layer selection distinct."""
    owners=defaultdict(list)
    for event in events:
        if event['event'] in ('armed','layer_selected','completed'):
            owners[event['request_id']].append(event)
    result=[]
    for rid,rows in owners.items():
        armed=[e for e in rows if e['event']=='armed']
        done=[e for e in rows if e['event']=='completed']
        layers=[e for e in rows if e['event']=='layer_selected']
        if len(armed)!=1 or len(done)!=1 or [e['layer'] for e in layers]!=list(range(armed[0]['layers'])):
            raise ValueError('Incomplete or duplicate final Query timing evidence')
        key='monotonic_at' if all('monotonic_at' in e for e in rows) else 'at'
        start,end=armed[0][key],done[0][key]
        if end<start:raise ValueError('Final Query timing moved backwards')
        refresh=[r for r in refreshes if r['request_id']==rid and r['phase']=='query']
        if len(refresh)!=1:raise ValueError('Final Query timing needs its unique initial refresh')
        full=None
        if key=='monotonic_at':
            if not refresh[0]['start']<=start<=end or refresh[0]['end']>end:
                raise ValueError('Final Query/refresh clock boundaries disagree')
            full=end-refresh[0]['start']
        result.append(dict(request_id=rid,clock=key,selection_chain_seconds=end-start,
            refresh_to_last_layer_assembly_seconds=full,
            scoring_host_seconds=sum(e['scoring_seconds'] for e in layers),
            assembly_submit_host_seconds=sum(e['assembly_host_seconds'] for e in layers),
            initial_refresh_seconds=refresh[0]['seconds'],layers=len(layers)))
    return dict(owners=result,selection_chain_seconds=distribution([r['selection_chain_seconds'] for r in result]),
        refresh_to_last_layer_assembly_seconds=distribution([r['refresh_to_last_layer_assembly_seconds']
            for r in result if r['refresh_to_last_layer_assembly_seconds'] is not None]),
        scope='Initial async Query refresh is global preselection/assembly. Selection chain runs from arming '
              'through final-layer KV assembly, including scheduling and intervening model layers; it ends '
              'before the final attention computation. Host score/assembly times are nested, not additive '
              'wall losses. Legacy wall-clock events cannot be joined to monotonic refresh timestamps.')


def analyze(capture):
    def records(name):
        path = capture / name
        return [json.loads(line) for line in path.open()] if path.exists() else []
    rows = refresh_intervals(records('async_refresh.jsonl'), records('bounded_refresh.jsonl'))
    phases = {}
    for phase in sorted({row['phase'] for row in rows}):
        selected = [row for row in rows if row['phase'] == phase]
        owners = defaultdict(float)
        stages = defaultdict(list)
        for row in selected:
            owners[row['request_id']] += row['seconds']
            for name, value in row['stages_seconds'].items():
                stages[name].append(value)
        phases[phase] = dict(count=len(selected), latency_seconds=distribution([r['seconds'] for r in selected]),
            per_owner_total_seconds=distribution(list(owners.values())),
            stages_seconds={k:distribution(v) for k,v in stages.items()},
            uploaded_bytes=sum(r['uploaded_bytes'] for r in selected),
            held_jobs_by_concurrency_seconds=held_concurrency(selected))
    result = dict(phases=phases, refreshes=rows,
        layerwise_query=layerwise_query_intervals(records('layerwise_query.jsonl'),rows),
        scope='Per-owner suspended-to-ready latency. Stage intervals are disjoint within a refresh. '
              'Different owners overlap, including with public forwards. Do not sum owners, '
              'nested Action scopes, or CPU archive counters as wall-time losses. '
              'Held-job concurrency is not GPU-idle attribution. Query here ends at initial async refresh; '
              'the separate layerwise_query field covers later causal layer selection.')
    steps = records('model_step_latency.jsonl')
    prefill = [r for r in steps if r['prefill_tokens'] and not r['decode_requests']]
    if prefill:
        tokens = sum(r['prefill_tokens'] for r in prefill)
        seconds = sum(r['model_ms'] for r in prefill) / 1000
        result['prefill_steps'] = dict(count=len(prefill), executed_tokens=tokens, model_interval_seconds=seconds,
            model_interval_tokens_per_second=tokens/seconds,
            mean_batch_tokens=tokens/len(prefill),
            scope='Model interval includes capture, scoring, sampling and submission gaps; not pure kernels.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.capture), indent=2) + '\n')
