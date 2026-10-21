"""Read-only attribution of the two audited 14K prefill Nsight intervals.

Kernel sums, all-activity union and gaps are distinct quantities. This diagnostic
does not measure throughput or identify CPU archive work from generic CUDA names.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import compare_prefill_nsight as comparison
from analyze_prefill_nsight import duration, merged

EXPECTED = [(14336, 16384), (14336, 30720)]
LAUNCH_GEOMETRY = ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'dynamicSharedMemory')
FA2 = 'BatchPrefillWithPagedKVCacheKernel'
NORM = '_fused_qk_rmsnorm_rope_gate_kernel'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(data):
    return dict(count=len(data), summed_ms=sum(r['end']-r['start'] for r in data)/1e6,
                union_ms=duration((r['start'], r['end']) for r in data)/1e6,
                bytes=sum(r.get('bytes', 0) for r in data))


def category(row, copies):
    if row['kind'] == 'memcpy':
        return 'memcpy:'+copies.get(row.get('copyKind'), str(row.get('copyKind')))
    if row['kind'] != 'kernel':
        return row['kind']
    name = row['name']
    if FA2 in name:
        return 'fa2_with_page_stats' if 'KVMemAttention' in name else 'fa2_without_page_stats'
    if NORM in name:
        return 'norm_rope_including_capture_if_enabled'
    return row['category']


def arm(root):
    summary, shapes, apis, data = comparison.load(root)
    require([(s['query_tokens'], s['active_seq_len']) for s in shapes] == EXPECTED,
            'Require exactly two audited 14336-query shapes at active 16384 and 30720')
    require(all(s.get('actual_geometry_verified') and s['context_requests'] == 1 and
                not s['generation_requests'] and not s['generation_tokens'] for s in shapes),
            'Unverified, concurrent or decode worker geometry')
    require({r.get('deviceId', 0) for r in data} == {0}, 'Unexpected CUDA device')
    with sqlite3.connect('file:'+str((root/'trace.sqlite').resolve())+'?mode=ro', uri=True) as db:
        copies = dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
    assigned = [r for r in data if 0 <= r['step'] < 2]
    unassigned = [r for r in data if not 0 <= r['step'] < 2]
    require(assigned, 'No CUDA activities assigned through launch correlation')
    begin, end = min(r['start'] for r in assigned), max(r['end'] for r in assigned)
    clipped = [(max(begin,r['start']),min(end,r['end'])) for r in data]
    union = merged(clipped)
    require(union and union[0][0] == begin and union[-1][1] == end, 'Bad CUDA observation boundary')
    gaps = [(a[1],b[0]) for a,b in zip(union,union[1:])]
    categories, names = defaultdict(list), defaultdict(list)
    for row in assigned:
        categories[category(row,copies)].append(row)
        if row['kind'] == 'kernel':
            names[row['name']].append(row)
    following = defaultdict(list)
    for row in data:
        following[row['start']].append(row)
    gap_details = []
    for start, stop in gaps:
        next_rows = following[stop]
        known = [r for r in next_rows if r['api'] is not None]
        next_row = min(known,key=lambda r:r['api']['start']) if known else None
        gap_details.append(dict(begin_ns=start,end_ns=stop,ms=(stop-start)/1e6,
            next_step=next_row['step'] if next_row else None,
            next_names=[r['name'] for r in next_rows],
            next_phase=next_row['phase'] if next_row else None,
            next_launch_entered_ns=next_row['api']['start'] if next_row else None,
            next_launch_not_entered_ms=max(0,min(stop,next_row['api']['start'])-start)/1e6
                                      if next_row else None))
    by_phase = defaultdict(lambda: dict(count=0,gap_ms=0.,next_launch_not_entered_ms=0.))
    for gap in gap_details:
        bucket=by_phase[gap['next_phase'] or 'unknown']
        bucket['count']+=1;bucket['gap_ms']+=gap['ms']
        bucket['next_launch_not_entered_ms']+=gap['next_launch_not_entered_ms'] or 0.
    banks, step_reports = [], []
    for i, shape in enumerate(shapes):
        own = [r for r in assigned if r['step']==i]
        fa2 = sorted((r for r in own if FA2 in r['name']),key=lambda r:r['api']['start'])
        norm = sorted((r for r in own if NORM in r['name']),key=lambda r:r['api']['start'])
        require(len(fa2)==17, f'Step {i}: expected 16 target + 1 MTP prefill FA2 calls')
        paired = [r for r in norm if r['api']['start'] < fa2[-1]['api']['start']]
        require(len(paired)==17, f'Step {i}: norm/RoPE count does not match 17 FA2 calls')
        for n,(nr,fa) in enumerate(zip(paired,fa2,strict=True)):
            require(nr['api']['start'] < fa['api']['start'] and
                    (not n or fa2[n-1]['api']['start'] < nr['api']['start']),
                    'Norm/RoPE ordering cannot identify a unique FA2-associated projection')
            cta = re.search(r'KernelTraits<\(flashinfer::MaskMode\)\d+, \(unsigned int\)(\d+),',fa['name'])
            require(cta is not None and int(cta[1])==64,'Expected actual CTA64 FA2 template')
        tail = [r for r in norm if r['api']['start'] > fa2[-1]['api']['start']]
        sections = defaultdict(list)
        for r in own:
            sections[category(r,copies)].append(r)
        step_reports.append(dict(index=i,query_tokens=shape['query_tokens'],
            active_seq_len=shape['active_seq_len'],cpu_execute_start_ns=shape['start_ns'],
            cpu_execute_end_ns=shape['end_ns'],assigned_gpu=statistics(own),
            fa2=statistics(fa2),fa2_paired_norm_rope=statistics(paired),
            remaining_norm_rope_tail=statistics(tail),
            categories={k:statistics(v) for k,v in sections.items()},
            pre_next_activity_gap_ms=sum(g['ms'] for g in gap_details if g['next_step']==i)))
        banks.append((fa2,paired))
    sync = [(max(begin,r['start']),min(end,r['end'])) for r in apis if 'Synchronize' in r['name']]
    scopes = dict(begin_ns=begin,end_ns=end,span_ms=(end-begin)/1e6,
        all_gpu_activity_union_ms=duration(union)/1e6,gaps_ms=duration(gaps)/1e6,
        kernel_summed_ms=sum(r['end']-r['start'] for r in assigned if r['kind']=='kernel')/1e6,
        kernel_union_ms=duration((r['start'],r['end']) for r in assigned if r['kind']=='kernel')/1e6)
    require(abs(scopes['span_ms']-scopes['all_gpu_activity_union_ms']-scopes['gaps_ms'])<1e-6,
            'GPU union/gap partition failed')
    return dict(source=str(root.resolve()),steps=step_reports,matched_gpu_window=scopes,
        categories={k:statistics(v) for k,v in categories.items()},
        kernel_names=sorted((dict(name=k,**statistics(v)) for k,v in names.items()),
                            key=lambda r:r['summed_ms'],reverse=True),
        gaps_by_next_submission_phase=dict(by_phase),
        largest_gaps=sorted(gap_details,key=lambda r:r['ms'],reverse=True)[:30],
        synchronizes=dict(union_ms=duration(sync)/1e6,
            overlapping_activity_ms=comparison.intersection(sync,union)/1e6,
            overlapping_gaps_ms=comparison.intersection(sync,gaps)/1e6),
        unassigned_activity_count=len(data)-len(assigned),
        unassigned_activity_overlap_ms=duration((max(begin,r['start']),min(end,r['end']))
            for r in unassigned)/1e6,
        all_activity_count=len(data),
        timeline_summary_observed_span_ms=summary['devices']['0']['observed_span_ms']),banks


def compare(native,kvmem):
    reports, banks = {}, {}
    for name,path in [('native',native),('kvmem',kvmem)]:
        reports[name],banks[name]=arm(path)
    plan_a=json.loads((native/'plan.json').read_text());plan_b=json.loads((kvmem/'plan.json').read_text())
    for key in ['input_sha256','runtime_proof_sha256']:
        require(plan_a[key]==plan_b[key],'Native/KVMem source differs: '+key)
    paired=[]
    for step,(a,b) in enumerate(zip(banks['native'],banks['kvmem'],strict=True)):
        layers=[]
        for index,(left,right,nleft,nright) in enumerate(zip(a[0],b[0],a[1],b[1],strict=True)):
            require(comparison.canonical_fa2(left['name'])==comparison.canonical_fa2(right['name']),
                    'FA2 template differs beyond KVMem statistics variant')
            require(all(left[k]==right[k] for k in LAUNCH_GEOMETRY),'FA2 actual launch geometry differs')
            layers.append(dict(index=index,selected='KVMemAttention' in right['name'],
                fa2_native_ms=(left['end']-left['start'])/1e6,
                fa2_kvmem_ms=(right['end']-right['start'])/1e6,
                norm_native_ms=(nleft['end']-nleft['start'])/1e6,
                norm_kvmem_ms=(nright['end']-nright['start'])/1e6,
                native_registers=left['registersPerThread'],kvmem_registers=right['registersPerThread'],
                norm_native_launch={k:nleft.get(k) for k in LAUNCH_GEOMETRY},
                norm_kvmem_launch={k:nright.get(k) for k in LAUNCH_GEOMETRY}))
        require([r['index'] for r in layers if r['selected']]==[0,7,15],
                'Expected only target layers 3/31/63 to collect FA2 page statistics')
        paired.append(dict(index=step,query_tokens=EXPECTED[step][0],active_seq_len=EXPECTED[step][1],
            layers=layers,
            selected_fa2_delta_ms=sum(r['fa2_kvmem_ms']-r['fa2_native_ms'] for r in layers if r['selected']),
            other_fa2_delta_ms=sum(r['fa2_kvmem_ms']-r['fa2_native_ms'] for r in layers if not r['selected']),
            paired_norm_rope_delta_ms=sum(r['norm_kvmem_ms']-r['norm_native_ms'] for r in layers)))
    sources=[Path(__file__),Path(comparison.__file__),Path(__file__).parent/'analyze_prefill_nsight.py']
    sources += [path/name for path in (native,kvmem) for name in ('trace.sqlite','timeline-summary.json','plan.json')]
    return dict(diagnostic_only=True,shape_and_runtime_audit_passed=True,arms=reports,matched_layers=paired,
        source_sha256={str(path.resolve()):digest(path) for path in sources},
        limitations=[
            'One profiled two-step window per arm; wall times are not throughput or stable speedup estimates.',
            'The 17 FA2-associated norm/RoPE kernels include 16 target layers and one MTP prefill layer.',
            'Matching kernel names alone do not separate norm/RoPE arithmetic from captured tensor writes.',
            'Kernel sums/categories can overlap. Only all-activity union plus gaps partitions its enclosing span.',
            'Per-step activity intervals can overlap and must not be summed as elapsed wall time.',
            'Memcpy direction and sizes do not identify archive versus scores; CPU archive tasks are not traced by CUDA names.',
            'Launch-not-entered gaps identify submission timing, not a specific Python, GIL or publication cause.',
            'Selected FA2 deltas include clocks/cache/noise; unchanged layers are the control.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--kvmem',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output path')
    result=compare(args.native,args.kvmem)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
