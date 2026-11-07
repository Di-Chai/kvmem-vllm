"""Frozen pressure-execution comparison with confirmed historical-prefill timing.

Reuses the full-input Query driver and its exact-selection/output gates. Query
execution stays tiled + lazy reader, with resident V off. Measurements read
existing completed CUDA-event rows; no new timing hooks, waits, or model work.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import difflib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


HELPER_NAME='_query_stage_driver.py'
helper=Path(__file__).with_name(HELPER_NAME)
if not helper.exists():
    helper=Path(__file__).with_name('benchmark_query_stage_efficiency.py')
spec=importlib.util.spec_from_file_location('pressure_query_driver',helper)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
BASE_VERIFY,BASE_SUMMARIZE=base.verify,base.summarize
BASE_PATCH,BASE_FREEZE_NATIVE=base.patch_runtime,base.freeze_native
REPO=base.REPO
REFERENCE=REPO/'artifacts/query-canonical-transport-20261106-r9/02-reader'
QUERY_ENV={
    'KVMEM_QUERY_STAGE_PROFILE':'0',
    'KVMEM_LAYERWISE_QUERY_RECORD_INDEX':'1',
    'KVMEM_LAYERWISE_QUERY_PAIRED_GATHER':'0',
    'KVMEM_LAYERWISE_QUERY_SCORE_BACKEND':'tiled',
    'KVMEM_LAYERWISE_QUERY_LAZY_READER':'1',
    'KVMEM_LAYERWISE_QUERY_RESIDENT_V':'0',
}
PRESSURE_FLAGS=frozenset('KVMEM_LAYERWISE_PRESSURE_'+name for name in (
    'PARENT_STAGING','BATCH_SELECTION','PAIRED_GATHER','PAIRED_GATHER_WORKERS',
    'RESIDENT_VALUES','PREPARED_READS',
    'REUSE_LAYER_VIEWS','FAST_METADATA','POST_FORWARD_REAP','SUBMIT_ON_PREPARE',
    'SINGLE_ASSEMBLY','SELECTION_YIELD','BATCH_ACCEPT','PIPELINE',
    'PIPELINE_HANDOFF','PREPARED_COMMIT','PIPELINE_PREFORWARD','PIPELINE_PROGRESS')) | frozenset((
    'KVMEM_WORKER_STATIC_GC','KVMEM_NATIVE_PREFILL_STATIC_QUAD','KVMEM_LAYERWISE_PRESSURE_LANES'))
STORE_SOURCES=frozenset(('third_party/qw3/include/qw3/kvmem_store.hpp',
                         'third_party/qw3/src/kvmem_store.cpp'))
SCOPE=('Completed CUDA-event interval from the first measured historical public '
       'step start through the last historical step end. Includes pressure, CPU '
       'launch gaps and intervening GPU-stream waits. Excludes untimed initial '
       'tokens, final Query, decode and reused-prefix logical tokens. Host envelope '
       'through Query-refresh dispatch is reported separately; neither is kernel-only time.')


def variant_order(variants,order):
    if not order or any(name not in variants for name in order):
        raise ValueError('Every pressure arm requires declared execution flags')
    for flags in variants.values():
        if any(k not in PRESSURE_FLAGS or v not in (
                ('1','2') if k in ('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS',
                                   'KVMEM_LAYERWISE_PRESSURE_LANES') else ('0','1'))
                for k,v in flags.items()):
            raise ValueError('Only whitelisted pressure execution values may differ')
        if (flags.get('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS','1')=='2' and
                flags.get('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER')!='1'):
            raise ValueError('Two gather workers require explicit paired gather')
    return [dict(name=f'{i+1:02d}-{name}',variant=name,environment={**QUERY_ENV,**variants[name]})
            for i,name in enumerate(order)]


@contextmanager
def preparation_hooks():
    previous=base.variant_order,base.patch_runtime,base.freeze_native
    base.variant_order=variant_order
    base.patch_runtime=patch_runtime
    base.freeze_native=freeze_native
    try:yield
    finally:base.variant_order,base.patch_runtime,base.freeze_native=previous


def patch_runtime(runtime,source,overlays,patch_path):
    """Only the named Store API source pair extends the existing allowlist."""
    extra=set(overlays)&STORE_SOURCES
    if extra and extra!=STORE_SOURCES:
        raise ValueError('Declare both exact Store header and implementation sources')
    patch=[]
    extra_changed=[]
    for name in sorted(extra):
        target,origin=runtime/name,source/name
        if not target.is_file() or not origin.is_file():
            raise ValueError('Store source must exist in the frozen runtime and overlay')
        if target.read_bytes()!=origin.read_bytes():extra_changed.append(name)
        patch.extend(difflib.unified_diff(target.read_text().splitlines(keepends=True),
            origin.read_text().splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
    changed=BASE_PATCH(runtime,source,[n for n in overlays if n not in STORE_SOURCES],patch_path)
    if patch:
        addition=''.join(patch)
        env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        env['GIT_CEILING_DIRECTORIES']=str(runtime.resolve().parent)
        for options in (['--check'],[]):
            subprocess.run(['git','apply',*options,'-'],input=addition,cwd=runtime,
                env=env,text=True,capture_output=True,check=True)
        patch_path.write_text(patch_path.read_text()+addition)
        changed+=extra_changed
    return sorted(changed)


def freeze_native(runtime,binary,changed):
    native_changed=set(changed)&({'csrc/bindings.cpp'}|STORE_SOURCES)
    if native_changed and binary is None:
        raise ValueError('Changed native Store/binding sources require an explicitly supplied binary')
    report=BASE_FREEZE_NATIVE(runtime,binary,changed)
    if report is not None:
        report['declared_source_sha256'].update({name:base.sha(runtime/name)
            for name in sorted(STORE_SOURCES) if (runtime/name).is_file()})
    return report


def prepare(root, *, source_arm=REFERENCE, fixture_path=None, overlay_source=REPO,
            overlays=(), case_ids=base.DEFAULT_IDS, autotune_source=None,
            variants=None,order=('baseline','candidate','candidate','baseline'),
            port=19460,native_binary=None,prefill_state_binary=None):
    root,source_arm=Path(root).resolve(),Path(source_arm).resolve()
    fixture_path=Path(fixture_path) if fixture_path is not None else source_arm/'fixture.json'
    autotune_source=(Path(autotune_source) if autotune_source is not None else
                     source_arm.parent/'autotune-seed')
    variants=variants or {'baseline':{'KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING':'0'},
                         'candidate':{'KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING':'1'}}
    variant_order(variants,order)
    with preparation_hooks():
        base.prepare(root,source_arm=source_arm,fixture_path=fixture_path,
            overlay_source=overlay_source,overlays=overlays,case_ids=case_ids,
            autotune_source=autotune_source,variants=variants,order=order,port=port,
            native_binary=native_binary,prefill_state_binary=prefill_state_binary)
    # This directory is still being prepared and has never been launched.
    # Bundle the reused driver so future actions never import a changed workspace.
    BASE_VERIFY(root)
    shutil.copy2(helper,root/HELPER_NAME)
    shutil.copy2(__file__,root/'launch.py')
    plan=base.read(root/'plan.json')
    plan['files'].update({HELPER_NAME:base.sha(root/HELPER_NAME),'launch.py':base.sha(root/'launch.py')})
    plan.update(driver='pressure-stage-v1',query_environment=QUERY_ENV,
        pressure_flag_allowlist=sorted(PRESSURE_FLAGS),prefill_measurement_scope=SCOPE,
        measurement_instrumentation_added=False,
        scope='Original full inputs and P-prefix reuse; serial clients, original eight-slot server. '
              'Pressure execution variants only; exact full-answer and selection gates. '+SCOPE)
    base.save(root/'plan.json',plan,replace=True)
    base.save(root/'prepared-proof.json',dict(plan_sha256=base.sha(root/'plan.json')),replace=True)
    return verify(root)


def verify(root):
    plan=BASE_VERIFY(root)
    if plan.get('driver')!='pressure-stage-v1' or plan.get('query_environment')!=QUERY_ENV:
        raise ValueError('Frozen pressure driver or qualified Query configuration changed')
    variants={}
    order=[]
    for trial in plan['trials']:
        flags=trial['environment']
        if any(flags.get(k)!=v for k,v in QUERY_ENV.items()):
            raise ValueError('Qualified Query flags must be identical in every pressure arm')
        pressure={k:v for k,v in flags.items() if k not in QUERY_ENV}
        name=trial['variant']
        if name in variants and variants[name]!=pressure:
            raise ValueError('Repeated pressure variant changed its flags')
        variants[name]=pressure
        order.append(name)
    if variant_order(variants,order)!=plan['trials']:
        raise ValueError('Pressure trial order or flags changed')
    return plan


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def finite(value,name):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid measured '+name)
    return value


def history_interval(result,steps,refresh):
    """Only consecutive, end-event-confirmed rows may define the GPU interval."""
    rid=result['attention']['request_id']
    command=result['attention']['command_id']
    own=[r for r in refresh if r['request_id']==rid]
    ends=[r['at'] for r in own if r['command_id']==command and r['stage']=='suspended']
    if len(ends)!=1:
        raise ValueError('Need exactly one final Query dispatch boundary')
    end=ends[0]
    history=result['attention']['history_begin']
    measured=[r for r in steps if r['started_at']<end and
              any(g['request_id']==rid for g in r['batch_geometry'])]
    if result['prefix_reused']:
        if measured:
            raise ValueError('Warm prefix unexpectedly executed historical model steps')
        return dict(id=result['id'],included=False,reason='warm_prefix_reused',history_tokens=history,
                    executed_prefill_tokens=0)
    if not measured or any(r['stage']=='query' and r['at']<end for r in own):
        raise ValueError('Missing history or Query execution entered the measured interval')
    measured.sort(key=lambda r:r['started_at'])
    if any(r['decode_requests'] or len(r['batch_geometry'])!=1 or
           r['batch_geometry'][0]['request_id']!=rid or not r['batch_geometry'][0]['prefill']
           for r in measured):
        raise ValueError('Pure history interval requires one owner and no decode')
    first=measured[0]
    begin=first['started_at']
    initial=first['batch_geometry'][0]['logical_seq_len']-first['batch_geometry'][0]['query_tokens']
    if initial and first is not steps[0]:
        raise ValueError('A later cold request is missing its initial measured history steps')
    cursor=initial
    tokens=0
    gpu_ms=finite(first['step_ms'],'step duration')
    for i,row in enumerate(measured):
        g=row['batch_geometry'][0]
        count=g['query_tokens']
        if (type(count) is not int or count<=0 or row['prefill_tokens']!=count or
                g['logical_seq_len']!=cursor+count or row['submitted_at']>end):
            raise ValueError('Historical public steps have a hole, replay or invalid token extent')
        if i:
            previous=measured[i-1]
            if (row['previous_sequence']!=previous['sequence'] or
                    row['previous_call_index']!=previous['call_index'] or
                    row['call_index']!=previous['call_index']+1):
                raise ValueError('Dropped or interleaved timing rows cannot define a full interval')
            gpu_ms+=finite(row['previous_end_to_model_ms'],'completed inter-step gap')
            gpu_ms+=finite(row['model_ms'],'model duration')
        cursor+=count
        tokens+=count
    if initial<0 or cursor!=history or tokens!=history-initial or gpu_ms<=0 or end<=begin:
        raise ValueError('Measured completed history does not reach the frozen Query boundary')
    # Events on the same CUDA timeline need not align to the host clock. Never
    # add individual kernel durations to infer the CPU wall interval.
    commands=defaultdict(dict)
    for row in own:
        if row['command_id']!=command and begin<=row['at']<=end:
            stage=row['stage']
            if stage in ('pipeline_handoff_prime','pipeline_handoff_attempt'):
                # Multiple readiness/admission polls do not define a unique
                # state-transition timestamp. Retain their original logs;
                # pressure latency pairs below use actual transitions only.
                continue
            if stage in commands[row['command_id']]:
                raise ValueError('Repeated pressure stage makes latency attribution ambiguous')
            commands[row['command_id']][stage]=row
    timings={}
    for a,b in [('suspended','ready'),('assembly_queued','assembly_worker_started'),
                ('assembly_worker_started','assembly_started'),('assembly_started','selection_prepared'),
                ('selection_prepared','assembly_submitted'),('assembly_submitted','ready')]:
        values=[finite(stages[b]['at']-stages[a]['at'],a+' to '+b)
                for stages in commands.values() if a in stages and b in stages]
        if values:timings[a+'_to_'+b]=dict(count=len(values),mean_seconds=statistics.mean(values),total_seconds=sum(values))
    return dict(id=result['id'],included=True,history_tokens=history,executed_prefill_tokens=tokens,
        unmeasured_initial_tokens=initial,measured_steps=len(measured),first_call_index=first['call_index'],
        last_call_index=measured[-1]['call_index'],begin_first_timed_step=begin,end_query_dispatch=end,
        last_history_step_submitted_at=measured[-1]['submitted_at'],
        confirmed_gpu_interval_seconds=gpu_ms/1000,host_envelope_seconds=end-begin,
        confirmed_gpu_interval_tokens_per_second=tokens/(gpu_ms/1000),
        host_envelope_tokens_per_second=tokens/(end-begin),pressure_events=len(commands),
        pressure_timings=timings,decode_requests_in_phase=0,query_forwards_in_phase=0,
        confirmation='StepTiming emits rows only after each end CUDA event query succeeds; '
                     'all previous-event sequence links are consecutive.',scope=SCOPE)


def measure(arm,case_ids):
    capture=arm/'run/server/capture'
    steps=lines(capture/'model_step_latency.jsonl')
    if not steps or steps[0]['sequence']!=1 or steps[0]['call_index']!=1:
        raise ValueError('Completed timing capture is missing its first observed public step')
    refresh=lines(capture/'async_refresh.jsonl')
    cases=[history_interval(base.read(arm/'run/answers/cases'/cid/'result.json'),steps,refresh) for cid in case_ids]
    work=base.read(capture/'public_work.json')
    cold=[r for r in cases if r['included']]
    if not cold:
        raise ValueError('No measured cold histories')
    last_submitted=max(r['last_history_step_submitted_at'] for r in cold)
    fence=work.get('completed_fence')
    if not work.get('counter_id') or not isinstance(fence,dict):
        raise ValueError('Public work lacks a completed GPU fence/counter snapshot')
    recorded_started=finite(fence.get('recorded_started_at'),'completed fence recording start')
    recorded=finite(fence.get('recorded_at'),'completed fence recording end')
    confirmed=finite(fence.get('completion_upper_bound'),'completed fence confirmation')
    if recorded_started<last_submitted or recorded<recorded_started or confirmed<recorded:
        raise ValueError('Completed public fence does not cover the final cold history step')
    completed,submitted=work['completed'],work['submitted']
    unconfirmed={}
    for key,value in completed.items():
        if key=='batch_composition':continue
        latest=submitted.get(key)
        if type(value) is not int or type(latest) is not int or value<0 or latest<value:
            raise ValueError('Public counter completion exceeds its submitted work')
        unconfirmed[key]=latest-value
    if (type(completed.get('public_prefill_tokens')) is not int or
            type(completed.get('public_batches')) is not int or
            work['all_submitted_completed']!=(unconfirmed['public_batches']==0) or
            work.get('not_yet_confirmed',unconfirmed)!=unconfirmed):
        raise ValueError('Public completion flags or deltas disagree with their counters')
    pending=work.get('pending_fence')
    if pending is not None:
        when=finite(pending.get('recorded_at'),'pending public fence recording')
        if (when<recorded or not unconfirmed['public_batches'] or
                type(pending.get('sequence')) is not int or
                pending['sequence']<=fence.get('sequence',-1)):
            raise ValueError('Pending public fence is not later unconfirmed work')
    if completed['public_prefill_tokens']<sum(r['history_tokens'] for r in cold):
        raise ValueError('Confirmed public counters cannot cover measured cold history')
    tokens=sum(r['executed_prefill_tokens'] for r in cold)
    gpu_seconds=sum(r['confirmed_gpu_interval_seconds'] for r in cold)
    host_seconds=sum(r['host_envelope_seconds'] for r in cold)
    return dict(cases=cases,executed_prefill_tokens=tokens,confirmed_gpu_interval_seconds=gpu_seconds,
        confirmed_gpu_interval_tokens_per_second=tokens/gpu_seconds,host_envelope_seconds=host_seconds,
        host_envelope_tokens_per_second=tokens/host_seconds,cold_cases=len(cold),
        excluded_warm_cases=len(cases)-len(cold),counter_id=work['counter_id'],
        final_public_completed=work['completed'],final_completed_fence=work['completed_fence'],
        final_public_submitted=work['submitted'],final_pending_fence=pending,
        final_unconfirmed_public_work=unconfirmed,
        all_final_public_work_confirmed=work['all_submitted_completed'],
        completed_fence_covers_cold_history=True,last_cold_step_submitted_at=last_submitted,
        pending_work_scope=('Later work outside measured cold histories; retained without extrapolation.'
                            if any(unconfirmed.values()) else 'No unconfirmed final public work.'),
        source_sha256={name:base.sha(capture/name) for name in
            ('model_step_latency.jsonl','async_refresh.jsonl','public_work.json')},scope=SCOPE)


def summarize(root):
    plan=verify(root)
    report=BASE_SUMMARIZE(root)
    measurements={trial['name']:measure(root/trial['name'],plan['case_ids']) for trial in plan['trials']}
    first=measurements[plan['trials'][0]['name']]
    geometry=lambda m:[(r['id'],r['included'],r['executed_prefill_tokens'],r.get('unmeasured_initial_tokens')) for r in m['cases']]
    if any(geometry(m)!=geometry(first) for m in measurements.values()):
        raise ValueError('Pressure variants measured different executed history spans')
    for m in measurements.values():
        m['confirmed_interval_throughput_change_percent']=100*(m['confirmed_gpu_interval_tokens_per_second']/first['confirmed_gpu_interval_tokens_per_second']-1)
        m['host_envelope_throughput_change_percent']=100*(m['host_envelope_tokens_per_second']/first['host_envelope_tokens_per_second']-1)
    output=dict(full_input_equivalent=report['full_input_equivalent'],measurements=measurements,
        measurement_valid=True,correctness_qualified=report['full_input_equivalent'],
        stable_speedup_claim=False,
        caution='Per-case and pooled rates cover only measured cold histories; warm prefix reuse is excluded. '
                'Repeated baseline/candidate observations are needed to distinguish a stable gain.',scope=SCOPE)
    base.save(root/'prefill-comparison.json',output)
    return {**report,'prefill':output}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','run','arm','analyze'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm')
    parser.add_argument('--source-arm',type=Path,default=REFERENCE)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--overlay-source',type=Path,default=REPO)
    parser.add_argument('--overlays-json',type=Path)
    parser.add_argument('--native-binary',type=Path)
    parser.add_argument('--prefill-state-binary',type=Path)
    parser.add_argument('--autotune-source',type=Path)
    parser.add_argument('--variants-json',type=Path)
    parser.add_argument('--order',default='baseline,candidate,candidate,baseline')
    parser.add_argument('--port',type=int,default=19460)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':
        prepare(root,source_arm=args.source_arm,fixture_path=args.fixture,overlay_source=args.overlay_source,
            overlays=base.read(args.overlays_json) if args.overlays_json else (),native_binary=args.native_binary,
            prefill_state_binary=args.prefill_state_binary,
            autotune_source=args.autotune_source,variants=base.read(args.variants_json) if args.variants_json else None,
            order=tuple(args.order.split(',')),port=args.port)
    elif args.action=='verify':verify(root)
    elif args.action=='analyze':summarize(root)
    else:
        base.verify,base.summarize=verify,summarize
        if args.action=='run':base.run(root)
        else:base.arm(root,args.arm)


if __name__=='__main__':main()
