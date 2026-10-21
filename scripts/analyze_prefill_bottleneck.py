"""Summarize a diagnostic Nsight window or validated single-request experiments.

Nsight stack samples are observations, not exclusive CPU time. GPU activity is
the union of recorded kernels/copies/memsets, not hardware utilization. Completed
uninstrumented requests provide the independent performance measurements.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from compare_pressure_runs import summarize
from compare_gen_budget_runs import resources


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def nsight(path, pid):
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        merged = []
        tables = ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY',
                  'CUPTI_ACTIVITY_KIND_MEMSET')
        intervals = []
        for table in tables:
            intervals.extend((a, b) for a, b, p in db.execute(
                f'SELECT start,end,globalPid FROM {table}') if (p >> 24) & 0xffffff == pid)
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        starts = [x[0] for x in merged]
        elapsed = (merged[-1][1]-merged[0][0])/1e9
        active = sum(b-a for a, b in merged)/1e9
        kernels = Counter()
        calls = Counter()
        for start, end, name, p in db.execute(
                'SELECT start,end,shortName,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            if (p >> 24) & 0xffffff == pid:
                kernels[strings[name]] += (end-start)/1e9
                calls[strings[name]] += 1
        samples = Counter()
        no_activity = Counter()
        count = 0
        for start, tid, payload in db.execute(
                'SELECT start,globalTid,jsonText FROM NVTX_EVENTS '
                'WHERE eventType=34 AND jsonText IS NOT NULL'):
            if (tid >> 24) & 0xffffff != pid or tid & 0xffffff != pid:
                continue
            if not merged[0][0] <= start <= merged[-1][1]:
                continue
            frames = json.loads(payload)['Python Periodic Sampling']['Frames']
            count += 1
            at = bisect_right(starts, start)-1
            gpu = at >= 0 and start < merged[at][1]
            for frame in frames:
                file = strings[frame['File']]
                if '/vllm_kvmem/' in file:
                    key = f"{Path(file).name}:{frame['Line']}:{strings[frame['Function']]}"
                    samples[key] += 1
                    if not gpu:
                        no_activity[key] += 1
                    break
        return dict(pid=pid, recorded_span_seconds=elapsed,
            recorded_gpu_union_seconds=active, recorded_gpu_union_fraction=active/elapsed,
            kernel_duration_sum_seconds=sum(kernels.values()),
            kernels=[dict(name=k, seconds=v, calls=calls[k]) for k,v in kernels.most_common(20)],
            main_thread_samples=count,
            deepest_project_frames=[dict(frame=k, samples=v, without_recorded_gpu=no_activity[k])
                                    for k,v in samples.most_common(25)],
            scope=__doc__)
    finally:
        db.close()


def completed_arm(root):
    summary = summarize(root, expected_cases=1)
    if read(root/'run/cleanup.json') != dict(remaining_owned_groups={}, gpu_processes=[]):
        raise ValueError('Incomplete cleanup')
    completed = read(root/'run/answers/completed.json')
    result = completed['results'][0]
    case = root/'run/answers/cases'/result['id']
    evidence, response = read(case/'evidence.json'), read(case/'response.json')
    capture = root/'run/server/capture'
    work = read(capture/'public_work.json')
    if not work['all_submitted_completed'] or work['submitted'] != work['completed']:
        raise ValueError('Missing final work counters')
    action = evidence['terminal']['action_attention']
    score_keys = ('samples', 'rounds', 'positive_pages', 'score_sum', 'scope',
                  'prefill_samples', 'prefill_chunks', 'prefill_end', 'prefill_pages',
                  'candidate_samples_total', 'scoring_backend_counts', 'block_graph')
    selections = [{k: e[k] for k in ('selection', 'history_begin', 'selected_history_ranges',
                                     'mandatory_fit', 'selected_tokens')}
                  for e in [*evidence['pressure'], evidence['query']]]
    # Graph records contain provenance owner IDs; exclude only those identities.
    graph = [{k:v for k,v in g.items() if k not in (
                'authority_id', 'request_id', 'scoring_seconds', 'graph_selection_seconds')}
             for g in evidence['graph']]
    archive, progress = read(capture/'archive.json'), read(capture/'progress.json')
    summary.update(ttft_input_tokens_per_second=result['input_tokens']/result['ttft_seconds'],
        resources=resources(root), work_snapshots=work['sequence'], final_work=work['completed'],
        semantic_digests=dict(output_ids=digest(response['output_ids']),
            selections=digest(selections), selection_count=len(selections), graph=digest(graph),
            action=digest({k:action[k] for k in score_keys}),
            terminal=digest({k:evidence['terminal'][k] for k in (
                'accepted_tokens','rejected_tokens','terminal_prefix','terminal_discarded_tokens')})),
        graph_statistics=graph,
        archive_work={k:v for k,v in archive.items() if k.startswith('seal_')},
        publication_work=progress['publication_metrics'],
        execution_audit=read(root/'run/execution-audit.json'))
    return summary


def graph_comparison(before, after):
    differences = []
    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                visit(a[key], b[key], path+'/'+str(key))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path+'/'+str(i))
        elif a != b:
            numerical = isinstance(a, float) and isinstance(b, float)
            differences.append(dict(path=path, before=a, after=b,
                absolute_difference=abs(a-b) if numerical else None,
                within_tolerance=numerical and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)))
    visit(before, after, '')
    return dict(exact_match=not differences,
        within_tolerance=all(d['within_tolerance'] for d in differences), differences=differences,
        scope='Graph summary statistics, not an export of every edge. Floating summaries '
              'use rtol=1e-6 and atol=1e-6; discrete fields and selected page ranges must be exact.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsight', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--arms', nargs='*', type=Path, default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = dict(scope=__doc__)
    if args.nsight:
        result['nsight'] = nsight(args.nsight, args.pid)
    if args.arms:
        arms = {p.name:completed_arm(p) for p in args.arms}
        first = next(iter(arms.values()))
        if any(a['profile'] != first['profile'] for a in arms.values()):
            raise ValueError('Different model or memory/scoring configurations')
        inputs = lambda a: {k:v['input_sha256'] for k,v in a['cases'].items()}
        if any(inputs(a) != inputs(first) for a in arms.values()):
            raise ValueError('Different request inputs')
        fixtures = [read(p/'dataset-summary.json')['fixture_sha256'] for p in args.arms]
        if len(set(fixtures)) != 1:
            raise ValueError('Different fixture/sampling specification')
        graph_checks = {name:graph_comparison(first['graph_statistics'], a['graph_statistics'])
                        for name, a in arms.items()}
        result.update(arms=arms, graph_checks=graph_checks, semantic_match=all(
            all(v == first['semantic_digests'][k] for k,v in a['semantic_digests'].items()
                if k != 'graph') and a['final_work'] == first['final_work'] and
            graph_checks[name]['within_tolerance'] for name, a in arms.items()))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
