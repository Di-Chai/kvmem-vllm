"""Matched-step attribution of two bounded Nsight diagnostics, never throughput."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3

from analyze_prefill_nsight import duration, kernel_category, merged, process, text_name


def intersection(left, right):
    a, b = merged(left), merged(right)
    i = j = result = 0
    while i < len(a) and j < len(b):
        result += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def canonical_fa2(name):
    name = name.replace('KVMemAttention', 'AttentionVariant')
    return re.sub(r'flashinfer::DefaultAttention<[^>]+>', 'AttentionVariant', name)


def load(root):
    summary = json.loads((root / 'timeline-summary.json').read_text())
    if not summary.get('actual_geometry_verified'):
        raise ValueError('Actual active shapes must pass the diagnostic audit first')
    shapes = summary['profiled_worker_shapes']
    c = sqlite3.connect('file:' + str((root / 'trace.sqlite').resolve()) + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(c.execute('SELECT id,value FROM StringIds'))
    def rows(table):
        return [dict(r) for r in c.execute('SELECT * FROM "'+table+'"')] if table in tables else []
    apis, links = [], defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        for r in rows(table):
            r['name'] = text_name(r, strings)
            apis.append(r)
            if any(x in r['name'] for x in ('Launch', 'Memcpy', 'Memset')):
                links[process(r), r.get('correlationId')].append(r)
    starts = [s['start_ns'] for s in shapes]
    data = []
    for table, kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),
                        ('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),
                        ('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        for r in rows(table):
            if r.get('end', 0) <= r.get('start', 0):
                continue
            r['name'] = text_name(r, strings) if kind == 'kernel' else kind
            r['category'] = kernel_category(r['name']) if kind == 'kernel' else kind
            r['kind'] = kind
            candidates = [x for x in links[process(r),r.get('correlationId')] if x['start'] <= r['start']]
            r['api'] = min(candidates, key=lambda x:x['start']) if candidates else None
            r['step'] = bisect_right(starts,r['api']['start'])-1 if r['api'] else -1
            index = r['step']
            r['phase'] = ('unknown' if index < 0 else 'execute_model_submission' if
                          r['api']['start'] <= shapes[index]['end_ns'] else 'after_execute_model_submission')
            data.append(r)
    c.close()
    return summary, shapes, apis, data


def summarize(root):
    summary, shapes, apis, data = load(root)
    union = merged((r['start'],r['end']) for r in data)
    gaps = [(a[1],b[0]) for a,b in zip(union,union[1:])]
    by_start = defaultdict(list)
    for row in data:
        by_start[row['start']].append(row)
    gap_details = []
    for begin,end in gaps:
        following = by_start[end]
        correlated = [r for r in following if r['api']]
        next_row = min(correlated,key=lambda r:r['api']['start']) if correlated else None
        gap_details.append(dict(begin_ns=begin,end_ns=end,ms=(end-begin)/1e6,
            next_step=next_row['step'] if next_row else -1,
            next_submission_phase=next_row['phase'] if next_row else 'unknown',
            next_category=next_row['category'] if next_row else 'unknown',
            next_api=next_row['api']['name'] if next_row else None,
            launch_not_entered_ms=max(0,min(end,next_row['api']['start'])-begin)/1e6 if next_row else 0))
    phases = defaultdict(lambda:dict(gap_ms=0.,launch_not_entered_ms=0.,count=0))
    for gap in gap_details:
        p=phases[gap['next_submission_phase']]
        p['gap_ms']+=gap['ms'];p['launch_not_entered_ms']+=gap['launch_not_entered_ms'];p['count']+=1
    steps, fa2 = [], []
    for i, shape in enumerate(shapes):
        own = [r for r in data if r['step']==i]
        attention = sorted((r for r in own if 'BatchPrefillWithPagedKVCacheKernel' in r['name']), key=lambda r:r['start'])
        if len(attention) != 17:
            raise ValueError(f'Expected 17 full-attention calls in step {i}, got {len(attention)}')
        fa2.append(attention)
        end = max(r['end'] for r in own)
        step_gaps = [g for g in gap_details if g['next_step']==i]
        selected=[r for r in attention if 'KVMemAttention' in r['name']]
        initial_span=[(shape['start_ns'],min(shape['end_ns'],shape['start_ns']+30_000_000))]
        previous_work=[(r['start'],r['end']) for r in data if 0 <= r['step'] < i]
        steps.append(dict(query_tokens=shape['query_tokens'],active_seq_len=shape['active_seq_len'],
            host_execute_submission_ms=(shape['end_ns']-shape['start_ns'])/1e6,
            last_gpu_after_host_execute_end_ms=(end-shape['end_ns'])/1e6,
            activity_union_ms=duration((r['start'],r['end']) for r in own)/1e6,
            fa2_ms=sum(r['end']-r['start'] for r in attention)/1e6,
            instrumented_fa2_ms=sum(r['end']-r['start'] for r in selected)/1e6,
            instrumented_indices=[n for n,r in enumerate(attention) if 'KVMemAttention' in r['name']],
            pre_next_activity_gaps_ms=sum(g['ms'] for g in step_gaps),
            next_launch_not_entered_ms=sum(g['launch_not_entered_ms'] for g in step_gaps),
            gaps_during_first_30ms_of_execute_ms=intersection(gaps,[(shape['start_ns'],
                min(shape['end_ns'],shape['start_ns']+30_000_000))])/1e6,
            previous_step_gpu_in_first_30ms_of_execute_ms=intersection(previous_work,initial_span)/1e6,
            gaps_during_remaining_execute_ms=intersection(gaps,[(min(shape['end_ns'],
                shape['start_ns']+30_000_000),shape['end_ns'])])/1e6,
            after_execute_activity_count=sum(r['phase']=='after_execute_model_submission' for r in own),
            after_execute_activity_summed_ms=sum(r['end']-r['start'] for r in own
                if r['phase']=='after_execute_model_submission')/1e6))
    synchronizes = [(r['start'],r['end']) for r in apis if 'Synchronize' in r['name']]
    begin,end=union[0][0],union[-1][1]
    sync_union = merged((max(begin,a),min(end,b)) for a,b in synchronizes)
    execute_spans=[(s['start_ns'],s['end_ns']) for s in shapes]
    main_thread=Counter(r['globalTid'] for r in apis if 'Launch' in r['name']).most_common(1)[0][0]
    connection=sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro',uri=True)
    osrt=defaultdict(list)
    for start,stop,name in connection.execute('SELECT o.start,o.end,s.value FROM OSRT_API o '
            'JOIN StringIds s ON s.id=o.nameId WHERE o.globalTid=?',(main_thread,)):
        osrt[name].append((start,stop))
    gil_chains={r[0] for r in connection.execute('SELECT DISTINCT c.id FROM OSRT_CALLCHAINS c '
        'JOIN StringIds s ON s.id=c.symbol WHERE s.value LIKE ?',('%take_gil%',))}
    gil_waits=[(a,b) for a,b,chain in connection.execute('SELECT o.start,o.end,o.callchainId '
        'FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId WHERE o.globalTid=? AND '
        's.value=?',(main_thread,'pthread_cond_timedwait')) if chain in gil_chains]
    connection.close()
    return dict(source=str(root), diagnostic_only=True, steps=steps,
        gpu_window=summary['devices']['0'], gaps_by_next_submission_phase=dict(phases),
        largest_gaps=sorted(gap_details,key=lambda r:r['ms'],reverse=True)[:30],
        sync_calls_in_observed_gpu_window=dict(union_ms=duration(sync_union)/1e6,
            overlapping_gpu_work_ms=intersection(sync_union,union)/1e6,
            overlapping_gpu_gaps_ms=intersection(sync_union,gaps)/1e6),
        gaps_inside_cpu_execute_ms=intersection(gaps,execute_spans)/1e6,
        gaps_outside_cpu_execute_ms=(duration(gaps)-intersection(gaps,execute_spans))/1e6,
        main_launch_thread_osrt=sorted((dict(name=k,count=len(v),union_ms=duration(v)/1e6,
            overlapping_gpu_gaps_ms=intersection(v,gaps)/1e6) for k,v in osrt.items()),
            key=lambda r:r['overlapping_gpu_gaps_ms'],reverse=True)[:12],
        main_thread_confirmed_gil_waits=dict(count=len(gil_waits),union_ms=duration(gil_waits)/1e6,
            overlapping_gpu_gaps_ms=intersection(gil_waits,gaps)/1e6,
            note='Only OSRT backtraces explicitly containing take_gil; unsampled waits remain unclassified.'),
        activity_without_known_launch=sum(r['api'] is None for r in data)),fa2


def compare(native, kvmem):
    results={};banks={}
    for name,root in [('native',native),('kvmem',kvmem)]:
        results[name],banks[name]=summarize(root)
    shape=lambda r:[(s['query_tokens'],s['active_seq_len']) for s in r['steps']]
    if shape(results['native']) != shape(results['kvmem']):
        raise ValueError('Profiled actual shapes differ')
    paired=[]
    for step,(native_rows,kvmem_rows) in enumerate(zip(banks['native'],banks['kvmem'],strict=True)):
        selected=[];plain=[]
        for layer,(a,b) in enumerate(zip(native_rows,kvmem_rows,strict=True)):
            if canonical_fa2(a['name']) != canonical_fa2(b['name']):
                raise ValueError('Different FA2 template geometry')
            geometry=('gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory')
            if any(a[k]!=b[k] for k in geometry):
                raise ValueError('Different FA2 launch geometry')
            row=dict(index=layer,native_ms=(a['end']-a['start'])/1e6,kvmem_ms=(b['end']-b['start'])/1e6,
                native_registers=a['registersPerThread'],kvmem_registers=b['registersPerThread'])
            (selected if 'KVMemAttention' in b['name'] else plain).append(row)
        paired.append(dict(active_seq_len=results['native']['steps'][step]['active_seq_len'],
            selected_layers=selected,
            selected_layer_delta_ms=sum(r['kvmem_ms']-r['native_ms'] for r in selected),
            other_layer_delta_ms=sum(r['kvmem_ms']-r['native_ms'] for r in plain)))
    for result in results.values():
        result['gpu_window'].pop('top_kernels',None)
        result['gpu_window'].pop('largest_gaps',None)
    return dict(diagnostic_only=True, shapes_identical=True, arms=results, matched_fa2=paired,
        limitations=['Single instrumented window per arm; no throughput or stable speedup claim.',
            'Step assignment uses CPU launch correlation, not GPU timestamp clipping; post-execute is a submission phase, not exclusively archive.',
            'Event synchronize duration mostly overlaps GPU execution and is not an additive penalty.',
            'Unsubmitted-next-activity gaps establish host submission timing, not the specific Python/GIL or archive cause.',
            'A gap inside the next CPU forward can be caused by loss of the previous GPU tail that formerly hid preparation; gap location alone does not rule out an earlier fence.',
            'Matched FA2 differences also contain clocks/cache/noise; compare unmodified layers as a control.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--kvmem',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(compare(args.native,args.kvmem),indent=2)+'\n')
