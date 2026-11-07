"""Read-only attribution of publication work overlapping measured GPU gaps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from analyze_prefill_phases import exclusive
from analyze_publication_bottleneck import host_analysis


def summarize(root):
    data=json.loads((root/'bottleneck-analysis.json').read_text())
    db=sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    ranges=[dict(r) for r in db.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) label "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE n.end>n.start AND coalesce(n.text,s.value) LIKE 'prefill_phase::%'")]
    publishers={r['globalTid'] for r in ranges if r['label']=='prefill_phase::logical_packet.publish'}
    parts={t:exclusive([dict(start=r['start'],end=r['end'],label=r['label'].removeprefix('prefill_phase::'))
                        for r in ranges if r['globalTid']==t]) for t in publishers}
    gaps=[]
    for gap in data['gpu']['largest_gpu_gaps']:
        costs=defaultdict(float)
        for spans in parts.values():
            for a,b,label in spans:
                costs[label]+=max(0,min(b,gap['end_ns'])-max(a,gap['begin_ns']))/1e6
        costs={k:v for k,v in costs.items() if v>0}
        if gap['ms']<10 or sum(costs.values())<gap['ms']/2:
            continue
        gaps.append({k:v for k,v in gap.items() if k!='main_scopes'} | dict(
            publisher_exclusive_ms=dict(sorted(costs.items(),key=lambda x:-x[1])),
            publisher_attributed_ms=sum(costs.values())))
    packets=[]
    rows=[r for p in (root/'probe').glob('publication-*.json') for r in json.loads(p.read_text())['rows']]
    for packet in rows:
        if packet['label']!='logical_packet.publish':
            continue
        subset=[r for r in rows if r['tid']==packet['tid'] and
                packet['start_ns']<=r['start_ns'] and r['end_ns']<=packet['end_ns']]
        costs=host_analysis(subset)['labels']
        packets.append(dict(history_begin=packet['history_begin'], tokens=packet['tokens'],
            wall_ms=(packet['end_ns']-packet['start_ns'])/1e6,
            cpu_ms=packet['cpu_ns']/1e6, costs=costs))
    db.close()
    return dict(diagnostic_only=True,performance_eligible=False,
        exact_same_shape_pressure_query_output=data['equivalence']['passed'],
        captured_window=json.loads((root/'probe/pressure-range.json').read_text()),
        publication_gaps=gaps, complete_host_packets=packets,
        total_profiled_gpu=data['gpu']['total'],
        limitations=[
            'Nsight and host wrappers add overhead; these durations are diagnostic, not throughput.',
            'Each gap uses exclusive intervals from the publication thread; other threads are not added.',
            'Main-thread GIL overlap is a separate overlapping quantity, not an additive cause.',
            'Whole-window host aggregation can include callbacks crossing profiler stop; use clipped NVTX or contained complete packets.',
            'Only the recorded steady window is attributed; do not extrapolate its absolute timings to all samples.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.root)
    with (args.root/'stall-attribution.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(passed=result['exact_same_shape_pressure_query_output'],
        gaps=[dict(ms=g['ms'],main_gil_ms=g['main_gil_ms'],
                   costs=list(g['publisher_exclusive_ms'].items())[:8]) for g in result['publication_gaps']],
        packets=[{k:v for k,v in p.items() if k!='costs'} for p in result['complete_host_packets']]),indent=2))
