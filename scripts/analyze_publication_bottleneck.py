"""Join bounded GPU idle intervals to nested publication host measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_prefill_cpu_commits import analyze as gpu_analysis


def host_analysis(rows):
    """Exclusive costs only subtract direct children on the same thread."""
    annotated = []
    totals = defaultdict(lambda: dict(calls=0, wall_ms=0., cpu_ms=0.,
                                      exclusive_wall_ms=0., exclusive_cpu_ms=0.))
    by_thread = defaultdict(list)
    for row in rows:
        by_thread[row['tid']].append(dict(row, children=[]))
    for tid, thread_rows in by_thread.items():
        stack = []
        for row in sorted(thread_rows, key=lambda r: (r['start_ns'], -r['end_ns'])):
            while stack and stack[-1]['end_ns'] <= row['start_ns']:
                stack.pop()
            if stack:
                assert row['end_ns'] <= stack[-1]['end_ns'], 'Non-nested host intervals'
                stack[-1]['children'].append(row)
            stack.append(row)
        for row in thread_rows:
            wall = row['end_ns']-row['start_ns']
            own_wall = wall-sum(c['end_ns']-c['start_ns'] for c in row['children'])
            own_cpu = row['cpu_ns']-sum(c['cpu_ns'] for c in row['children'])
            assert own_wall >= 0 and own_cpu >= 0
            summary = totals[tid, row['label']]
            summary['calls'] += 1
            summary['wall_ms'] += wall/1e6
            summary['cpu_ms'] += row['cpu_ns']/1e6
            summary['exclusive_wall_ms'] += own_wall/1e6
            summary['exclusive_cpu_ms'] += own_cpu/1e6
            if row['label'] in ('logical_packet.publish', 'logical_transaction.advance'):
                # Direct children are a partition of the timed call, with the
                # residual retaining Python glue and instrumentation overhead.
                children = defaultdict(float)
                for c in row['children']:
                    children[c['label']] += (c['end_ns']-c['start_ns'])/1e6
                annotated.append({k:v for k,v in row.items() if k!='children'} | dict(
                    wall_ms=wall/1e6, cpu_ms=row['cpu_ns']/1e6,
                    residual_ms=own_wall/1e6, direct_children_ms=dict(children)))
    return dict(labels=sorted([dict(tid=t,label=n,**v) for (t,n),v in totals.items()],
                              key=lambda r:-r['exclusive_wall_ms']),
                transactions=sorted(annotated,key=lambda r:r['start_ns']))


def analyze(root):
    rows=[]
    for path in (root/'probe').glob('publication-*.json'):
        rows.extend(json.loads(path.read_text())['rows'])
    assert rows, 'Missing diagnostic host timings'
    host=host_analysis(rows)
    gpu=gpu_analysis(root)
    return dict(diagnostic_only=True, performance_eligible=False, host=host, gpu=gpu,
                equivalence=json.loads((root/'equivalence.json').read_text()),
                limitations=gpu['limitations']+[
                    'Host timing wrappers add overhead; compare only attribution, not throughput.',
                    'Exclusive host time is relative to instrumented calls, not all Python/native functions.',
                    'A bounded window can contain incomplete outer packets; only completed intervals are analyzed.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root)
    with (args.root/'bottleneck-analysis.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(total=result['gpu']['total'],
        host_top=result['host']['labels'][:18], passed=result['equivalence']['passed']),indent=2))
