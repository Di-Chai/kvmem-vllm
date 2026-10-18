"""Read-only decomposition of a frozen agent scheduling run.

Reports sampled model time separately from accepted client output throughput.
Restore pre-submit time includes preparation and queueing; host gather overlaps
the restore service interval and must never be added to it a second time.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import time


def records(directory, name):
    paths=list(directory.rglob(name))
    if not paths:return []
    if len(paths)!=1:raise ValueError('Require exactly one native worker log: '+name)
    values=[]
    for line in paths[0].read_text().splitlines(keepends=True):
        try:values.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith('\n'):raise
    return values


def percentile(values, quantile):
    values=sorted(values)
    if not values:return None
    at=(len(values)-1)*quantile;lo=int(at);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def distribution(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
                maximum=max(values,default=None))


def prefill_origin(row):
    """Keep ordinary ingestion separate from atomic Query and unknown work."""
    total = row['prefill_tokens']
    names = ('ordinary_prefill_tokens', 'atomic_prefill_tokens', 'unknown_prefill_tokens')
    if not all(name in row for name in names):
        return 'unknown' if total else 'none'
    ordinary, atomic, unknown = (row[name] for name in names)
    if (any(type(n) is not int or n < 0 for n in (total, ordinary, atomic, unknown))
            or ordinary + atomic + unknown != total):
        raise ValueError('Prefill origin counters do not partition public tokens')
    if unknown:
        return 'unknown'
    if ordinary and atomic:
        return 'ordinary_and_atomic'
    return 'ordinary' if ordinary else 'atomic' if atomic else 'none'


def step_groups(rows):
    groups=defaultdict(list)
    origins=defaultdict(list)
    for row in rows:
        d,p=row['decode_requests'],row['prefill_tokens']
        mode='mixed' if d and p else ('prefill_only' if p else ('decode_only' if d else 'maintenance_only'))
        groups[mode].append(row)
        origins[mode,prefill_origin(row)].append(row)
    summary={}
    for name,selected in groups.items():
        entry=dict(samples=len(selected),
            **{key:distribution([r[key] for r in selected if key in r]) for key in
                ('model_ms','maintenance_ms','step_ms','host_submit_ms')})
        if name=='prefill_only':
            timed=[r for r in selected if r.get('model_ms',0)>0]
            entry.update(sampled_input_tokens=sum(r['prefill_tokens'] for r in timed),
                tokens_per_sampled_model_second=(sum(r['prefill_tokens'] for r in timed)/
                    sum(r['model_ms'] for r in timed)*1000 if timed else None),
                tokens_per_sampled_step_second=(sum(r['prefill_tokens'] for r in timed)/
                    sum(r['step_ms'] for r in timed)*1000 if timed else None))
        summary[name]=entry
    # A0/P1 comparisons must match batch composition and dense context. Older
    # captures lack context buckets: keep them explicitly unknown, never zero.
    bins=defaultdict(list)
    for row in rows:
        key=(row['decode_requests'],row.get('prefill_requests'),row['prefill_tokens'],
             row.get('decode_context_bucket'),row.get('prefill_context_bucket'),prefill_origin(row))
        if row.get('model_ms') is not None:bins[key].append(row['model_ms'])
    origin_groups=[]
    for (mode,origin),selected in sorted(origins.items()):
        origin_groups.append(dict(mode=mode,prefill_origin=origin,samples=len(selected),
            sampled_model_seconds=sum(r.get('model_ms',0) for r in selected)/1000,
            **{key:distribution([r[key] for r in selected if key in r]) for key in
               ('decode_requests','prefill_requests','prefill_tokens','ordinary_prefill_tokens',
                'atomic_prefill_tokens','unknown_prefill_tokens','model_ms','step_ms',
                'maintenance_ms','maintenance_host_ms')}))
    return dict(groups=summary,prefill_origin_groups=origin_groups,
        matched_composition_bins=[dict(decode_requests=k[0],prefill_requests=k[1],
        prefill_tokens=k[2],decode_context_bucket=k[3],prefill_context_bucket=k[4],
        prefill_origin=k[5],
        model_ms=distribution(v)) for k,v in sorted(bins.items(),key=lambda pair:repr(pair[0]))],
        scope='Sampled public model steps only; model_ms includes MTP/sampling/capture. '
              'No extrapolated whole-run prefill throughput; private Query replay is excluded. '
              'Public atomic Query replay and unknown work are separate from ordinary prefill. '
              'Summed sampled model durations are not exclusive wall time or GPU utilization.')


def restore_groups(restores, stages):
    submitted=defaultdict(list)
    for row in stages:
        if row['stage']=='restore_submitted':submitted[row['request_id']].append(row)
    groups=defaultdict(list);mapped=[]
    for row in restores:
        candidates=submitted.get(row['request_id'],[])
        if len(candidates)!=1:raise ValueError('Missing/duplicate exact restore submission: '+row['request_id'])
        at=candidates[0]['at']
        if not row['started']<=at<=row['finished']:
            raise ValueError('Restore submission outside its measured interval')
        mode='resident' if row.get('resident_active_tokens',0) else 'cpu'
        result=dict(request_id=row['request_id'],mode=mode,cancelled=row['cancelled'],
            history_tokens=row['history_tokens'],active_tokens=row['active_tokens'],
            pre_submit_seconds=at-row['started'],service_seconds=row['finished']-at,
            total_seconds=row['finished']-row['started'],host_prepare_seconds=row['host_prepare_seconds'],
            uploaded_bytes=row['uploaded_bytes'],model_bytes=row['model_bytes'])
        mapped.append(result)
        groups[mode].append(result)
    summary={}
    for mode,rows in groups.items():
        summary[mode]=dict(restores=len(rows),cancelled=sum(r['cancelled'] for r in rows),
            uploaded_bytes=sum(r['uploaded_bytes'] for r in rows),model_bytes=sum(r['model_bytes'] for r in rows),
            **{key:distribution([r[key] for r in rows]) for key in
                ('pre_submit_seconds','service_seconds','total_seconds','host_prepare_seconds')})
    return dict(groups=summary,requests=mapped,
        scope='Pre-submit includes waiting and synchronous preparation. Service includes background gather, '
              'transfers and fence observation; host_prepare is an overlapping component, not an extra duration. '
              'Upload bytes / service time is not a DMA bandwidth measurement.')


def client_metrics(traces):
    if not traces:return None
    if any(not r['timeline'] or sum(n for _,n in r['timeline'])!=len(r['output_ids']) for r in traces):
        raise ValueError('Client output timeline does not cover accepted tokens')
    begin=min(r['begin'] for r in traces);end=max(r['end'] for r in traces)
    output=sum(len(r['output_ids']) for r in traces)
    gaps=[b[0]-a[0] for r in traces for a,b in zip(r['timeline'],r['timeline'][1:])]
    return dict(requests=len(traces),accepted_output_tokens=output,seconds=end-begin,
        accepted_output_tokens_per_second=output/(end-begin),output_event_gap_seconds=distribution(gaps),
        ttft_seconds=distribution([r['timeline'][0][0]-r['begin'] for r in traces]),
        request_seconds=distribution([r['end']-r['begin'] for r in traces]),
        scope='Output events may contain multiple accepted MTP tokens; event gaps are not per-token ITL. '
              'The full span includes prefill, restoration and inter-turn tool waits.')


def policy_waits(rows,begin,end,request_ids=None):
    if not rows:return None
    totals=defaultdict(float)
    for i,row in enumerate(rows):
        left=max(begin,row['at']);right=min(end,rows[i+1]['at'] if i+1<len(rows) else end)
        if right<=left:continue
        for candidate in row['candidates']:
            if request_ids is not None and candidate['request_id'] not in request_ids:continue
            if candidate['reason']:totals[candidate['reason']]+=right-left
    return dict(request_seconds=dict(totals),
        scope='Scheduler state-change snapshots integrated over the window; concurrent request waits overlap. '
              'These are observed reasons, not exclusive wall-time phases or a sum of GPU stall time.')


def policy_cost_coverage(rows,begin,end,request_ids=None):
    """Show whether measured-cost ordering had evidence to act on this load."""
    if not rows:return None
    counts=defaultdict(int);phases=defaultdict(lambda:defaultdict(int));waits=[]
    for row in rows:
        if not begin<=row['at']<end:continue
        ready=[]
        for c in row['candidates']:
            if request_ids is not None and c['request_id'] not in request_ids:continue
            # Decode's zero remaining cost is not evidence for fill ordering.
            if c['reason'] is not None or not (c['phase']=='prefill' or c['phase'].endswith('_admission')):continue
            known=c['estimated_ms'] is not None
            phases[c['phase']]['known' if known else 'unknown']+=1
            counts['ready_candidate_observations']+=1
            counts['ready_candidates_with_estimates']+=known
            ready.append(c);waits.append(c['waiting_seconds'])
        if len(ready)>1:
            counts['competing_ready_snapshots']+=1
            counts['competing_all_costs_known_snapshots']+=all(c['estimated_ms'] is not None for c in ready)
        for rid,cost in row.get('maintenance_costs',{}).items():
            if request_ids is not None and rid not in request_ids:continue
            counts['maintenance_cost_observations']+=1
            counts['maintenance_costs_known']+=cost is not None
    keys=('ready_candidate_observations','ready_candidates_with_estimates','competing_ready_snapshots',
          'competing_all_costs_known_snapshots','maintenance_cost_observations','maintenance_costs_known')
    return dict(**{k:counts[k] for k in keys},by_phase={k:dict(v) for k,v in phases.items()},
        ready_wait_seconds=distribution(waits),
        scope='State-change snapshot counts, not per-step or wall-time percentages. Decode zero-cost '
              'entries are excluded. Known costs permit ordering but do not prove it changed a choice; '
              'aging, native eligibility and available credits can still determine the outcome.')


def restore_overlap_bins(steps, restores, stages):
    """Correlate step starts with pending restore services, without causal claims."""
    submitted={r['request_id']:r['at'] for r in stages if r['stage']=='restore_submitted'}
    intervals=[(submitted[r['request_id']],r['finished']) for r in restores if r['request_id'] in submitted]
    starts=sorted(b for b,_ in intervals);ends=sorted(e for _,e in intervals)
    groups=defaultdict(list)
    for row in steps:
        if row['prefill_tokens'] or not row['decode_requests'] or row.get('model_ms') is None:continue
        at=row['started_at']
        busy=bisect_right(starts,at)>bisect_right(ends,at)
        groups[row['decode_requests'],row.get('decode_context_bucket'),busy].append(row['model_ms'])
    return dict(bins=[dict(decode_requests=d,decode_context_bucket=context,
        restore_service_pending_at_step_start=busy,model_ms=distribution(values))
        for (d,context,busy),values in sorted(groups.items(),key=lambda pair:repr(pair[0]))],
        scope='Correlates host step-start timestamps with restore submission/completion intervals. '
              'Not kernel-level overlap or causal interference; context, MTP acceptance and tool CPU work '
              'can confound these comparisons. Missing context buckets remain unknown.')


def analyze(server, *, traces=None, begin=None, end=None):
    steps=records(server,'model_step_latency.jsonl')
    restores=records(server,'async_restore.jsonl')
    stages=records(server,'async_refresh.jsonl')
    decisions=records(server,'offline_decisions.jsonl')
    expected=policy_ids=None
    if traces is not None:
        begin=min(r['begin'] for r in traces);end=max(r['end'] for r in traces)
        response_ids={r['response_id'] for r in traces}
        if len(response_ids)!=len(traces):raise ValueError('Duplicate client response ID')
        expected=set()
        policy_ids=set()
        for response in response_ids:
            pattern=re.compile(re.escape(response)+r'-0-[0-9a-f]{8}')
            matches={r['request_id'] for r in restores if pattern.fullmatch(r['request_id'])}
            if len(matches)>1:raise ValueError('Ambiguous native request ID')
            expected.update(matches)
            policy_ids.update(c['request_id'] for row in decisions for c in row['candidates']
                              if pattern.fullmatch(c['request_id']))
    times=[r['started_at'] for r in steps]+[r['started'] for r in restores]
    begin=min(times,default=0.) if begin is None else begin
    end=time.perf_counter() if end is None else end
    if end<=begin:raise ValueError('Empty measurement window')
    steps=[r for r in steps if begin<=r['started_at']<end]
    overlap=restore_overlap_bins(steps,restores,stages)
    restores=[r for r in restores if (r['request_id'] in expected if expected is not None else begin<=r['started']<end)]
    report=dict(window=dict(begin=begin,end=end,seconds=end-begin),clients=client_metrics(traces),
        model=step_groups(steps),restores=restore_groups(restores,stages),
        policy_waits=policy_waits(decisions,begin,end,policy_ids),
        policy_cost_coverage=policy_cost_coverage(decisions,begin,end,policy_ids),
        restore_overlap=overlap,
        scope='Read-only observations; not a correctness gate or proof of causal speedup. '
              'Compare identical oracle settings; oracle-enabled lifecycle work is excluded from performance.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',type=Path,required=True)
    parser.add_argument('--traces',type=Path,help='Exact case traces.json; excludes warmup/cleanup request IDs')
    parser.add_argument('--last-seconds',type=float,help='Live monotonic window; incompatible with --traces')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.traces and args.last_seconds:parser.error('Choose a case or a live window')
    end=time.perf_counter();begin=end-args.last_seconds if args.last_seconds else None
    result=analyze(args.server,traces=json.loads(args.traces.read_text()) if args.traces else None,begin=begin,end=end)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('model','restores','restore_overlap')}))
