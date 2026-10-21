"""Read-only host-phase attribution for the bounded Nsight diagnostic.

This is a causal-location aid, not a throughput benchmark. CPU scopes may be
nested and background work overlaps both GPU activity and foreground scopes.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from analyze_prefill_nsight import duration, merged
from compare_prefill_nsight import compare, intersection, load
from prefill_phase_probe import PREFIX


def exclusive(scopes):
    """Partition thread-local nested ranges into nonoverlapping deepest scopes."""
    events = defaultdict(list)
    for index, scope in enumerate(scopes):
        events[scope['start']].append((True, index))
        events[scope['end']].append((False, index))
    active = {}
    result = []
    previous = None
    for point, changes in sorted(events.items()):
        if previous is not None and point > previous and active:
            current = max(active.values(), key=lambda r:(r['start'], -r['end']))
            result.append((previous, point, current['label']))
        for entering, index in sorted(changes,
                key=lambda change:(change[0], -scopes[change[1]]['end'] if change[0] else 0)):
            if entering:
                current = scopes[index]
                if any(other['end'] < current['end'] for other in active.values()):
                    raise ValueError('Diagnostic NVTX scopes cross rather than nest')
                active[index] = current
            else:
                active.pop(index)
        previous = point
    return result


def analyze(root):
    summary, shapes, apis, activities = load(root)
    union = merged((r['start'],r['end']) for r in activities)
    if not union:
        raise ValueError('No observed GPU activity')
    window = [(union[0][0], union[-1][1])]
    gaps = [(left[1],right[0]) for left,right in zip(union,union[1:])]
    main = Counter(api['globalTid'] for api in apis if 'Launch' in api['name']).most_common(1)[0][0]
    connection = sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    strings = dict(connection.execute('SELECT id,value FROM StringIds'))
    scopes, incomplete = defaultdict(list), []
    for index, row in enumerate(connection.execute('SELECT * FROM NVTX_EVENTS')):
        name = row['text'] or strings.get(row['textId'], '')
        if not name.startswith(PREFIX):
            continue
        if row['end'] is None or row['end'] <= row['start']:
            incomplete.append(dict(label=name[len(PREFIX):], global_tid=row['globalTid']))
            continue
        scopes[row['globalTid']].append(dict(start=row['start'], end=row['end'],
            label=name[len(PREFIX):], index=index))
    if not scopes:
        raise ValueError('No completed diagnostic phase ranges; do not infer zero phase cost')
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    gil = []
    if {'OSRT_API','OSRT_CALLCHAINS'} <= tables:
        chains = {r[0] for r in connection.execute('SELECT DISTINCT c.id FROM OSRT_CALLCHAINS c '
            'JOIN StringIds s ON c.symbol=s.id WHERE s.value LIKE ?', ('%take_gil%',))}
        gil = [(r['start'],r['end']) for r in connection.execute(
            'SELECT * FROM OSRT_API WHERE globalTid=?', (main,))
            if r['callchainId'] in chains and strings.get(r['nameId']) == 'pthread_cond_timedwait']
    connection.close()
    partitions = {thread:exclusive(values) for thread,values in scopes.items()}
    starts = {thread:[row[0] for row in values] for thread,values in partitions.items()}

    def phase(thread, at):
        values = partitions.get(thread, [])
        index = bisect_right(starts.get(thread, []), at)-1
        return values[index][2] if index >= 0 and at < values[index][1] else 'unscoped'

    launched, launched_apis = defaultdict(list), Counter()
    for api in apis:
        if any(token in api['name'] for token in ('Launch', 'Memcpy', 'Memset')):
            launched_apis[api['globalTid'],phase(api['globalTid'],api['start'])] += 1
    for row in activities:
        api = row['api']
        key = (api['globalTid'],phase(api['globalTid'],api['start'])) if api else (None,'unknown_launch')
        launched[key].append((row['start'],row['end']))
    by_gpu_start = defaultdict(list)
    for row in activities:
        by_gpu_start[row['start']].append(row)
    unsubmitted, details = [], []
    for begin,end in gaps:
        following = [row for row in by_gpu_start[end] if row['api']]
        next_activity = min(following, key=lambda row:row['api']['start']) if following else None
        api = next_activity['api'] if next_activity else None
        until = max(begin, min(end,api['start'])) if api else begin
        if until > begin:
            unsubmitted.append((begin,until))
        details.append(dict(begin_ns=begin, end_ns=end, gap_ms=(end-begin)/1e6,
            next_launch_not_entered_ms=(until-begin)/1e6,
            next_launch_phase=phase(api['globalTid'],api['start']) if api else 'unknown',
            next_category=next_activity['category'] if next_activity else 'unknown',
            next_step=next_activity['step'] if next_activity else -1))
    keys = set(launched) | {(thread,row['label']) for thread,values in scopes.items() for row in values}
    result = []
    for thread,label in sorted(keys, key=lambda row:(row[0] or 0,row[1])):
        inclusive = [(r['start'],r['end']) for r in scopes.get(thread, []) if r['label']==label]
        own = [(a,b) for a,b,name in partitions.get(thread, []) if name==label]
        result.append(dict(label=label, global_tid=thread, main_launch_thread=thread==main,
            calls=sum(r['label']==label for r in scopes.get(thread, [])),
            host_inclusive_in_gpu_window_ms=intersection(inclusive,window)/1e6,
            host_exclusive_in_gpu_window_ms=intersection(own,window)/1e6,
            coincident_gpu_gaps_ms=intersection(own,gaps)/1e6,
            coincident_unsubmitted_gaps_ms=intersection(own,unsubmitted)/1e6,
            coincident_main_confirmed_gil_wait_ms=intersection(own,gil)/1e6,
            cuda_launch_api_count=launched_apis[thread,label],
            linked_gpu_activity_count=len(launched[thread,label]),
            linked_gpu_activity_union_ms=duration(launched[thread,label])/1e6))
    main_covered = [(a,b) for a,b,_ in partitions.get(main, [])]
    prepare_fences = []
    for index,shape in enumerate(shapes):
        begin,end = shape['start_ns'],shape['end_ns']
        preparing = [r for r in scopes.get(main, []) if r['label']=='action.prepare'
                     and begin <= r['start'] < end]
        if len(preparing) != 1:
            continue
        row = preparing[0]
        interval = [(row['start'],row['end'])]
        sync = [api for api in apis if api['globalTid']==main and
                row['start'] <= api['start'] < row['end'] and 'Synchronize' in api['name']]
        previous = [r for r in activities if 0 <= r['step'] < index]
        previous_phases = defaultdict(list)
        for activity in previous:
            api = activity['api']
            label = phase(api['globalTid'],api['start']) if api else 'unknown_launch'
            previous_phases[label].append((activity['start'],activity['end']))
        uploads = [r for r in activities if r['kind']=='memcpy' and r.get('copyKind')==2 and
            r['api'] is not None and r['step']==index and
            phase(r['api']['globalTid'],r['api']['start'])=='action.prepare']
        arm = [(r['start'],r['end']) for r in scopes.get(main, []) if
            r['label']=='native_stats.arm' and row['start'] <= r['start'] < row['end']]
        prepare_fences.append(dict(step=index,
            host_start_after_execute_ms=(row['start']-begin)/1e6,
            host_end_after_execute_ms=(row['end']-begin)/1e6,
            host_total_ms=(row['end']-row['start'])/1e6,
            host_residual_excluding_cuda_sync_ms=((row['end']-row['start'])-
                intersection(interval,[(api['start'],api['end']) for api in sync]))/1e6,
            native_arm_host_ms=duration(arm)/1e6,
            previous_gpu_end_after_execute_ms=(max(r['end'] for r in previous)-begin)/1e6 if previous else None,
            previous_gpu_busy_inside_prepare_ms=intersection(interval,
                [(r['start'],r['end']) for r in previous])/1e6,
            previous_gpu_inside_prepare_by_launch_phase_ms={label:intersection(interval,spans)/1e6
                for label,spans in previous_phases.items() if intersection(interval,spans)},
            synchronizes=[dict(name=api['name'],phase=phase(main,api['start']),
                start_after_execute_ms=(api['start']-begin)/1e6,
                end_after_execute_ms=(api['end']-begin)/1e6,
                duration_ms=(api['end']-api['start'])/1e6) for api in sync],
            metadata_d2h=[dict(bytes=r.get('bytes'),stream=r.get('streamId'),
                start_after_execute_ms=(r['start']-begin)/1e6,
                end_after_execute_ms=(r['end']-begin)/1e6) for r in uploads]))
    return dict(diagnostic_only=True, source=str(root), actual_geometry_verified=True,
        main_launch_global_tid=main, gpu_window_ms=duration(window)/1e6,
        gpu_activity_union_ms=duration(union)/1e6, gpu_gaps_ms=duration(gaps)/1e6,
        unsubmitted_gaps_ms=duration(unsubmitted)/1e6,
        unsubmitted_gaps_outside_main_named_phases_ms=(duration(unsubmitted)-intersection(unsubmitted,main_covered))/1e6,
        phase_rows=sorted(result, key=lambda r:(not r['main_launch_thread'],-r['coincident_unsubmitted_gaps_ms'])),
        largest_gaps=sorted(details,key=lambda r:r['gap_ms'],reverse=True)[:30],
        action_prepare_fences=prepare_fences,
        incomplete_phase_ranges=incomplete,
        limitations=['Additional NVTX scopes perturb this run; no throughput or speedup conclusion.',
            'Deepest phase host intervals are disjoint only within one thread; background overlaps are not additive or proof of GIL ownership.',
            'GPU activities are attributed by CUDA launch correlation and may outlive their CPU scope.',
            'Unsubmitted gaps end when the next observed GPU activity API begins; this is a conservative host timing bound, not proof of a specific cause.',
            'Only completed observed ranges are attributed. Installed wrappers and old function aliases may differ; unnamed time remains unscoped.',
            'Confirmed GIL waits require sampled OSRT take_gil backtraces; other waits are not inferred.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(phases=analyze(args.candidate), comparisons={})
    for name,path in [('native_reference',args.native),('previous_kvmem_reference',args.baseline)]:
        if path is not None:
            # The comparator's historical keys are native=reference and
            # kvmem=candidate. Its template/shape audit remains mandatory.
            result['comparisons'][name] = compare(path,args.candidate)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
