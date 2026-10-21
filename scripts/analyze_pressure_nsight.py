"""Read-only single-pressure CPU/CUDA attribution, never a throughput result."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from analyze_prefill_nsight import merged, duration
from analyze_prefill_phases import exclusive
from compare_prefill_nsight import load, intersection

def analyze(root):
 summary,shapes,apis,data=load(root)
 verified=summary['pressure_verified']; marks=verified['marker_times_ns']; origin=marks['suspended'][0]; end=marks['ready'][1]
 db=sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro',uri=True); db.row_factory=sqlite3.Row
 strings=dict(db.execute('select id,value from StringIds'))
 scopes=defaultdict(list)
 for i,r in enumerate(db.execute('select * from NVTX_EVENTS')):
  name=r['text'] or strings.get(r['textId'],'')
  if name.startswith('prefill_phase::') and r['end'] is not None and r['end']>r['start']:
   scopes[r['globalTid']].append(dict(start=r['start'],end=r['end'],label=name[15:],index=i))
 partitions={thread:exclusive(rows) for thread,rows in scopes.items()}
 def select(label):
  return [(r['start'],r['end']) for rows in scopes.values() for r in rows if r['label']==label and r['start']<end and r['end']>origin]
 def clip(xs,a=origin,b=end):return [(max(s,a),min(e,b)) for s,e in xs if s<b and e>a]
 allgpu=merged((r['start'],r['end']) for r in data)
 gpu=clip(allgpu)
 gap=[];at=origin
 for s,e in gpu:
  if s>at:gap.append((at,s))
  at=max(at,e)
 if at<end:gap.append((at,end))
 main=Counter(api['globalTid'] for api in apis if 'Launch' in api['name']).most_common(1)[0][0]
 assembler=next(t for t,rs in scopes.items() if any(r['label']=='assembly._assemble_split' and r['start']<end and r['end']>origin for r in rs))
 publishers=sorted(t for t,rs in scopes.items() if any(r['label']=='action.prepare_accept' and r['start']<end and r['end']>origin for r in rs))
 def label(t,point):
  for s,e,n in partitions.get(t,[]):
   if s<=point<e:return n
  return 'unscoped'
 gilchains={r[0] for r in db.execute('select distinct c.id from OSRT_CALLCHAINS c join StringIds s on c.symbol=s.id where s.value like ?',('%take_gil%',))}
 gil=defaultdict(list); waits=defaultdict(list)
 for r in db.execute('select * from OSRT_API where start<? and end>?',(end,origin)):
  waits[r['globalTid'],strings.get(r['nameId'],'')].append((r['start'],r['end']))
  if r['callchainId'] in gilchains and strings.get(r['nameId'])=='pthread_cond_timedwait':gil[r['globalTid']].append((r['start'],r['end']))
 def window(a,b):
  gs=clip(allgpu,a,b)
  return dict(begin_ms=(a-origin)/1e6,end_ms=(b-origin)/1e6,wall_ms=(b-a)/1e6,gpu_union_ms=duration(gs)/1e6,idle_ms=(b-a-duration(gs))/1e6,
   main_confirmed_gil_ms=intersection(gil[main],[(a,b)])/1e6,assembly_confirmed_gil_ms=intersection(gil[assembler],[(a,b)])/1e6,
   main_confirmed_gil_gpu_idle_ms=intersection(clip(gil[main],a,b),gap)/1e6,
   assembly_confirmed_gil_gpu_idle_ms=intersection(clip(gil[assembler],a,b),gap)/1e6)
 stages=verified['stages'];stage_windows=[dict(phase=a+'->'+b,**window(marks[a][0],marks[b][0])) for a,b in zip(stages,stages[1:])]
 split=select('assembly._assemble_split')[0]
 metadata=sorted(select('assembly._upload_metadata'))
 prep_end=next(s for s,e in metadata if s>=split[0]); preparation=window(split[0],prep_end)
 perthread=[]
 for thread,rs in scopes.items():
  labels=sorted({r['label'] for r in rs if r['start']<end and r['end']>origin})
  for n in labels:
   inc=clip([(r['start'],r['end']) for r in rs if r['label']==n]);own=clip([(s,e) for s,e,lab in partitions[thread] if lab==n])
   perthread.append(dict(thread=thread,label=n,count=len(inc),inclusive_ms=duration(inc)/1e6,exclusive_ms=duration(own)/1e6,
    inclusive_gpu_idle_overlap_ms=intersection(inc,gap)/1e6,
    exclusive_gpu_idle_overlap_ms=intersection(own,gap)/1e6,confirmed_gil_overlap_ms=intersection(inc,gil[thread])/1e6))
 gaps=[]
 for a,b in gap:
  future=[r for r in data if r['start']>=b and r['api'] is not None]
  following=min(future,key=lambda r:(r['start'],r['api']['start'])) if future else None
  api=following['api'] if following else None
  cut=max(a,min(b,api['start'])) if api else a
  gaps.append(dict(begin_ms=(a-origin)/1e6,end_ms=(b-origin)/1e6,ms=(b-a)/1e6,
    next_launch_not_entered_ms=(cut-a)/1e6,next_api=None if api is None else api['name'],
    next_thread=None if api is None else api['globalTid'],next_host_scope=None if api is None else label(api['globalTid'],api['start']),
    next_kernel=None if following is None else ('split_rebuild' if '_assemble_split' in following['name'] else following['category']),
    assembly_exclusive_during_gap=[dict(label=n,ms=(min(e,b)-max(s,a))/1e6) for s,e,n in partitions[assembler] if s<b and e>a]))
 read_scopes=sorted(select('archive.read_packed_records')); launch_scopes=sorted(select('rebuild.assemble_split_layer'))
 copies=[r for r in data if r['kind']=='memcpy' and r.get('copyKind')==1 and r.get('bytes',0)>1048576 and origin<=r['start']<end]
 rebuilt=[r for r in data if r['kind']=='kernel' and '_assemble_split' in r['name'] and origin<=r['start']<end]
 chunks=[]
 for i,(s,e) in enumerate(read_scopes):
  nexts=read_scopes[i+1][0] if i+1<len(read_scopes) else split[1]
  launches=[(a,b) for a,b in launch_scopes if e<=a<nexts]
  owncopy=[r for r in copies if r['api'] and e<=r['api']['start']<nexts]
  ownkernels=[r for r in rebuilt if r['api'] and e<=r['api']['start']<nexts]
  ownactivities=owncopy+ownkernels
  gpu_end=max(r['end'] for r in ownactivities)
  chunks.append(dict(index=i,read=window(s,e),cpu_rebuild_calls=len(launches),cpu_rebuild_scopes_ms=duration(launches)/1e6,
   h2d_bytes=sum(r['bytes'] for r in owncopy),h2d_gpu_ms=duration((r['start'],r['end']) for r in owncopy)/1e6,
   rebuild_kernels=len(ownkernels),rebuild_gpu_ms=duration((r['start'],r['end']) for r in ownkernels)/1e6,
   last_gpu_end_ms=(gpu_end-origin)/1e6,next_read_after_gpu_end_ms=(nexts-gpu_end)/1e6 if i+1<len(read_scopes) else None,
   cpu_launch_scopes_gpu_overlap_ms=intersection(launches,allgpu)/1e6))
 # Work already launched before the suspension: MTP/capture tail belongs to
 # the preceding public forward, not to reconstruction.
 old=[r for r in data if r['api'] and r['api']['start']<origin and r['end']>origin]
 oldgroups=defaultdict(list)
 for r in old:oldgroups[r['category']].append((r['start'],r['end']))
 # Next public dispatch and embedding launch after ready; the 32-token replay
 # is logged as a public scheduled batch and is not a private Query forward.
 nextshape=next(s for s in shapes if s['scheduled_tokens'] and s['start_ns']>=end)
 nextapis=[r for r in data if r['api'] and r['api']['start']>=nextshape['start_ns']]
 nextgpu=min(nextapis,key=lambda r:r['start'])
 native_scopes=[]
 for n in ('action.prepare_accept','graph.prepare','host_native.scatter_history_batch','host_native.finalize_mean_batch','host_native.pack_record_pairs','archive._seal_views','archive._record_digests','authority.publish_sealed','selection.select_pressure','assembly.assemble'):
  native_scopes.append(dict(label=n,windows=[window(s,e) for s,e in select(n)]))
 polling=[]
 for first,last in zip(stages,stages[1:]):
  a,b=marks[first][0],marks[last][0]
  for n in ('publication.publication_ready','publication.advance_publication','refresh.pump','refresh.step'):
   ss=clip(select(n),a,b)
   polling.append(dict(phase=first+'->'+last,label=n,count=len(ss),
    union_ms=duration(ss)/1e6,gpu_idle_overlap_ms=intersection(ss,gap)/1e6))
 main_exclusive=[]
 for name in sorted({n for s,e,n in partitions[main] if s<end and e>origin}):
  spans=clip([(s,e) for s,e,n in partitions[main] if n==name])
  main_exclusive.append(dict(label=name,host_elapsed_ms=duration(spans)/1e6,
   gpu_idle_overlap_ms=intersection(spans,gap)/1e6,
   assembler_gil_wait_overlap_ms=intersection(spans,gil[assembler])/1e6,
   publisher_gil_wait_overlap_ms=intersection(spans,[v for t in publishers for v in gil[t]])/1e6))
 block_names=('clock_nanosleep','pthread_cond_timedwait','sem_wait','pthread_mutex_lock')
 blocking=[span for (thread,name),spans in waits.items() if thread==main and name in block_names for span in spans]
 tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
 db.close()
 return dict(source=str(root.resolve()),diagnostic_only=True,sql_sha256=hashlib.sha256((root/'trace.sqlite').read_bytes()).hexdigest(),
  caution='Warm single-pressure scoped trace; elapsed_seconds includes profiler API overhead. CPU scope, GPU occupancy, DMA, native waits overlap; no extrapolated throughput benefit.',
  command_id=verified['command_id'],threads=dict(main=main,assembly=assembler,publication=publishers),lifecycle=window(origin,end),
  nonoverlapping_stage_windows=stage_windows,assembly_initial_preparation=preparation,
  preceding_forward_tail=dict(union_after_suspend_ms=duration(clip([(r['start'],r['end']) for r in old]))/1e6,
   last_end_ms=max((r['end']-origin for r in old),default=0)/1e6,
   categories={k:duration(clip(v))/1e6 for k,v in oldgroups.items()}),
  important_scopes_nonadditive=native_scopes,scope_attribution=perthread,chunks=chunks,
  largest_gpu_gaps=sorted(gaps,key=lambda x:x['ms'],reverse=True)[:30],
  launch_not_entered_ms=sum(r['next_launch_not_entered_ms'] for r in gaps),
  next_public=dict(query_tokens=nextshape['query_tokens'],active_seq_len=nextshape['active_seq_len'],
    cpu_execute_after_ready_ms=(nextshape['start_ns']-end)/1e6,first_gpu_after_ready_ms=(nextgpu['start']-end)/1e6),
  total_kernel_rebuild_gpu_ms=duration((r['start'],r['end']) for r in rebuilt)/1e6,
  total_bulk_h2d_gpu_ms=duration((r['start'],r['end']) for r in copies)/1e6,
  total_bulk_h2d_bytes=sum(r['bytes'] for r in copies),
  read_gpu_overlap_ms=intersection(read_scopes,allgpu)/1e6,
  polling_calls_nonadditive=polling,
  main_thread_exclusive_host_scopes=main_exclusive,
  main_thread_recorded_blocking_union_ms=duration(clip(blocking))/1e6,
  confirmed_gil_waits=[dict(thread=thread,count=len(gil[thread]),union_ms=duration(clip(gil[thread]))/1e6,
   gpu_idle_overlap_ms=intersection(gil[thread],gap)/1e6,
   max_recorded_wait_ms=max((e-s for s,e in clip(gil[thread])),default=0)/1e6)
   for thread in (main,assembler,*publishers)],
  evidence_limits=['CPU ranges overlap and do not establish GIL ownership.',
   'Confirmed GIL waits require an OSRT callchain containing take_gil; unrecorded waits remain unknown.',
   'Frequent pump/readiness calls establish polling, not CPU busy time; NVTX instrumentation adds per-call overhead.',
   'Stage and scope/GPU idle intersections are not additive across threads or nested ranges.'],
  cuda_api_in_lifecycle=[dict(name=n,count=len(rows),union_ms=duration(clip([(r['start'],r['end']) for r in rows]))/1e6)
   for n,rows in ((n,[r for r in apis if r['name']==n and r['start']<end and r['end']>origin]) for n in sorted({r['name'] for r in apis})) if rows],
  osrt_by_thread=[dict(thread=t,name=n,union_ms=duration(clip(v))/1e6,overlapping_gpu_idle_ms=intersection(v,gap)/1e6) for (t,n),v in waits.items() if t in (main,assembler,*publishers)],
  gpu=verified['gpu'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(analyze(a.root),indent=2)+'\n')
