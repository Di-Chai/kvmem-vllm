"""Attribute measured CPU commits and confirmed GIL waits, without causal overclaim."""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3

from analyze_prefill_nsight import duration,merged,kernel_category,process,text_name
from analyze_prefill_phases import exclusive
from compare_prefill_nsight import intersection


def analyze(root):
    db=sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    def rows(table):return [dict(r) for r in db.execute('SELECT * FROM '+table)] if table in tables else []
    apis=[];links=defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        for r in rows(table):
            r['name']=text_name(r,strings);apis.append(r)
            if any(t in r['name'] for t in ('Launch','Memcpy','Memset')):
                links[process(r),r.get('correlationId')].append(r)
    gpu=[]
    for table,kind in (('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')):
        for r in rows(table):
            if r['end']<=r['start']:continue
            r['name']=text_name(r,strings) if kind=='kernel' else kind
            r['category']=kernel_category(r['name']) if kind=='kernel' else kind
            eligible=[a for a in links[process(r),r.get('correlationId')] if a['start']<=r['start']]
            r['api']=min(eligible,key=lambda a:a['start']) if eligible else None
            gpu.append(r)
    assert gpu,'No GPU trace; cannot infer zero GPU work'
    busy=merged((r['start'],r['end']) for r in gpu)
    window=[(busy[0][0],busy[-1][1])]
    gaps=[(a[1],b[0]) for a,b in zip(busy,busy[1:])]
    main=Counter(r['globalTid'] for r in apis if 'Launch' in r['name']).most_common(1)[0][0]
    scopes=defaultdict(list);markers=defaultdict(dict);incomplete=[]
    for r in rows('NVTX_EVENTS'):
        name=r['text'] or strings.get(r['textId'],'')
        if name.startswith('prefill_phase::'):
            if r['end'] is None or r['end']<=r['start']:
                incomplete.append(name);continue
            scopes[r['globalTid']].append(dict(start=r['start'],end=r['end'],label=name.removeprefix('prefill_phase::')))
        elif name.startswith('prefill_pressure::'):
            _,stage,command=name.split('::',2);markers[command][stage]=r['start']
    parts={t:exclusive(rs) for t,rs in scopes.items()}
    starts={t:[a for a,b,label in rs] for t,rs in parts.items()}
    def label(t,at):
        i=bisect_right(starts.get(t,[]),at)-1
        return parts[t][i][2] if i>=0 and at<parts[t][i][1] else 'unscoped'
    chains=set()
    if 'OSRT_CALLCHAINS' in tables:
        chains={r[0] for r in db.execute('SELECT DISTINCT c.id FROM OSRT_CALLCHAINS c JOIN StringIds s ON c.symbol=s.id WHERE s.value LIKE ?',('%take_gil%',))}
    gil=defaultdict(list);waits=defaultdict(list)
    for r in rows('OSRT_API'):
        name=strings.get(r['nameId'],'');span=(r['start'],r['end'])
        waits[r['globalTid'],name].append(span)
        if r['callchainId'] in chains and name=='pthread_cond_timedwait':gil[r['globalTid']].append(span)
    db.close()
    def clip(xs,windows):
        return [(max(a,c),min(b,d)) for a,b in xs for c,d in windows if b>c and a<d]
    pressure=[(m['suspended'],m['ready']) for m in markers.values() if 'suspended' in m and 'ready' in m]
    def metrics(windows):
        return dict(wall_ms=duration(windows)/1e6,gpu_busy_ms=intersection(busy,windows)/1e6,
            gpu_idle_ms=intersection(gaps,windows)/1e6,
            main_confirmed_gil_ms=intersection(gil[main],windows)/1e6,
            main_confirmed_gil_gpu_idle_ms=intersection(clip(gil[main],windows),gaps)/1e6)
    gc_generations = {}
    gc_all = []
    for generation in range(3):
        events = [(r['start'], r['end']) for rs in scopes.values() for r in rs
                  if r['label'] == f'python.gc.generation{generation}']
        observed = clip(events, window)
        gc_all.extend(observed)
        gc_generations[str(generation)] = dict(count=len(observed),
            max_ms=max((b-a for a,b in observed), default=0)/1e6, **metrics(observed))
    gc_stats = dict(generations=gc_generations, **metrics(merged(gc_all)))
    gap_distribution = {
        str(threshold): dict(count=sum(b-a >= threshold*1e6 for a,b in gaps),
                            total_ms=sum(b-a for a,b in gaps if b-a >= threshold*1e6)/1e6)
        for threshold in (1, 5, 20, 50, 100)}
    per_scope=[]
    for thread,rs in scopes.items():
        for name in sorted({r['label'] for r in rs}):
            inc=[(r['start'],r['end']) for r in rs if r['label']==name]
            own=[(a,b) for a,b,n in parts[thread] if n==name]
            per_scope.append(dict(thread=thread,tid=thread%2**24,label=name,calls=len(inc),main=thread==main,
                inclusive_ms=intersection(inc,window)/1e6,exclusive_ms=intersection(own,window)/1e6,
                exclusive_gpu_idle_ms=intersection(own,gaps)/1e6,
                inclusive_gpu_idle_ms=intersection(inc,gaps)/1e6,
                confirmed_gil_ms=intersection(clip(inc,window),gil[thread])/1e6,
                pressure_exclusive_gpu_idle_ms=intersection(clip(own,pressure),gaps)/1e6))
    publisher=[t for t,rs in scopes.items() if any(r['label']=='layerwise_pressure.prepare_accept' for r in rs)]
    publisher_python=[(a,b) for t in publisher for a,b,n in parts[t] if n in
        ('layerwise_pressure.prepare_accept','cpu_commit.buffered_accept')]
    publisher_native=[(a,b) for t in publisher for a,b,n in parts[t] if n=='cpu_commit.native_committed']
    categories=defaultdict(list)
    for r in gpu:categories[r['category']].append((r['start'],r['end']))
    after=defaultdict(list)
    for r in gpu:after[r['start']].append(r)
    gap_rows=[]
    # Only the largest 25 gaps are returned below. Do the expensive CPU-scope
    # intersections only for those gaps, rather than for every microsecond
    # separation between GPU kernels in a multiowner trace. Global metrics
    # above still use the full interval union.
    for a,b in sorted(gaps,key=lambda span:span[1]-span[0],reverse=True)[:25]:
        options=[r for r in after[b] if r['api']]
        following=min(options,key=lambda r:r['api']['start']) if options else None
        api=following['api'] if following else None
        gap_rows.append(dict(begin_ns=a,end_ns=b,ms=(b-a)/1e6,
            next_launch_unentered_ms=max(0,min(b,api['start'])-a)/1e6 if api else None,
            next_scope=label(api['globalTid'],api['start']) if api else None,
            next_thread=api['globalTid'] if api else None,next_kernel=following['name'] if following else None,
            main_gil_ms=intersection(gil[main],[(a,b)])/1e6,
            publisher_python_ms=intersection(publisher_python,[(a,b)])/1e6,
            publisher_native_ms=intersection(publisher_native,[(a,b)])/1e6,
            main_scopes=[dict(label=n,ms=(min(e,b)-max(s,a))/1e6) for s,e,n in parts.get(main,[]) if s<b and e>a]))
    cpu=defaultdict(lambda:dict(calls=0,wall_ms=0.,thread_cpu_ms=0.))
    for p in (root/'probe').glob('cpu-commits-*.json'):
        for r in json.loads(p.read_text())['rows']:
            x=cpu[r['tid'],r['label']];x['calls']+=1;x['wall_ms']+=(r['end_ns']-r['start_ns'])/1e6;x['thread_cpu_ms']+=r['cpu_ns']/1e6
    pressure_rows=[]
    for command,m in markers.items():
        if 'ready' not in m or 'suspended' not in m:continue
        stages=sorted(m,key=m.get)
        pressure_rows.append(dict(command=command,**metrics([(m['suspended'],m['ready'])]),
            stages=[dict(stage=a+'->'+b,**metrics([(m[a],m[b])])) for a,b in zip(stages,stages[1:])]))
    # The assembly_started marker is emitted AFTER layerwise prepare(). Do not
    # mistake that preamble for executor queue delay.
    preparation=[]
    for command,m in markers.items():
        if not {'assembly_queued','assembly_started'}<=m.keys():continue
        q,s=m['assembly_queued'],m['assembly_started']
        # A queued owner can wait while a DIFFERENT owner assembles. Associate
        # only the worker range containing this command's started marker.
        assemblies=[r for rs in scopes.values() for r in rs
                    if r['label']=='refresh._assemble' and q<=r['start']<=s<r['end']]
        entry=m.get('assembly_worker_started')
        if entry is None and assemblies:
            entry=max(r['start'] for r in assemblies)
        calls=[(r['start'],r['end']) for rs in scopes.values() for r in rs
               if entry is not None and r['label']=='layerwise_pressure.prepare'
               and entry<=r['start']<s and r['end']<=s]
        preparation.append(dict(command=command,queued_to_started_ms=(s-q)/1e6,
            queued_to_worker_entry_ms=(entry-q)/1e6 if entry is not None else None,
            layerwise_preparation_ms=duration(calls)/1e6,
            main_confirmed_gil_during_preparation_ms=intersection(calls,gil[main])/1e6))
    return dict(diagnostic_only=True,performance_eligible=False,main_thread=main,publisher_threads=publisher,
        total=metrics(window),pressure=metrics(pressure),pressure_events=pressure_rows,
        python_gc=gc_stats,gpu_gaps_at_least_ms=gap_distribution,
        preparation_before_started_marker=preparation,
        gpu_categories_nonadditive_ms={k:duration(v)/1e6 for k,v in categories.items()},
        confirmed_main_gil_overlaps_publisher_python_ms=intersection(clip(gil[main],window),publisher_python)/1e6,
        confirmed_main_gil_overlaps_publisher_native_ms=intersection(clip(gil[main],window),publisher_native)/1e6,
        osrt_take_gil_callchain_count=len(chains),
        cpu_commit_timing=[dict(tid=t,label=n,**v) for (t,n),v in cpu.items()],
        scopes=sorted(per_scope,key=lambda r:-r['exclusive_gpu_idle_ms']),
        confirmed_gil_waits=[dict(thread=t,tid=t%2**24,ms=intersection(rs,window)/1e6,gpu_idle_ms=intersection(rs,gaps)/1e6) for t,rs in gil.items()],
        osrt_waits=[dict(thread=t,name=n,ms=intersection(rs,window)/1e6) for (t,n),rs in waits.items() if intersection(rs,window)>1e6],
        largest_gpu_gaps=sorted(gap_rows,key=lambda r:-r['ms'])[:25],
        incomplete_ranges=incomplete,limitations=[
            'Profiled run includes instrumentation overhead; no unprofiled throughput claim.',
            'CPU scopes are nonadditive across threads; background elapsed work is not foreground stall.',
            'Native scope includes argument conversion and GIL reacquisition. Thread CPU is not GIL hold time.',
            'Confirmed GIL waits need OSRT take_gil backtraces; overlap cannot identify the GIL owner or prove sole causality.',
            'Only observed completed ranges and recorded OSRT calls are attributed. No wait record does not prove zero waits.',
            'GPU category unions overlap; fused attention statistics cannot be separated by kernel name.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();result=analyze(a.root)
    with (a.root/'cpu-commit-analysis.json').open('x') as out:json.dump(result,out,indent=2)
    print(json.dumps({k:result[k] for k in ('total','pressure','cpu_commit_timing','confirmed_gil_waits')},indent=2))
