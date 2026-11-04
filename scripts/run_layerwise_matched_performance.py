"""Same-input throughput controls after strict layerwise execution qualification.

Shared controls deliberately change the selection algorithm. They measure its
cost; only the layerwise candidate is required to match its unoptimized version.
This finite comparison creates no scheduled monitor and never replaces answers.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify

REPO = Path(__file__).resolve().parents[1]
MODES = ('shared_global', 'shared_layerwise_query', 'layerwise_optimized')


def controls(fixture, profile, mode, index):
    if mode not in MODES:
        raise ValueError('Unknown matched performance arm')
    fixture, profile = deepcopy(fixture), deepcopy(profile)
    fixture.update(layerwise_workers=8, qualification_only=True, performance_comparison=True,
        performance_query_mode='shared_global' if mode == 'shared_global' else 'sequential',
        performance_arm=mode, cohort='matched-eight-owner-performance',
        layerwise_prefill_pressure=mode == 'layerwise_optimized')
    if mode != 'layerwise_optimized':
        fixture.update(layerwise_pressure_single_assembly=False, layerwise_pressure_fast_metadata=False)
        fixture['layerwise_buffered_history'] = False
    if mode == 'shared_global':
        fixture['layerwise_packed_assembly'] = False
        fixture['layerwise_prepared_index'] = False
        fixture['layerwise_batched_reads'] = False
        fixture['layerwise_paired_reads'] = False
    profile.update(max_num_seqs=8, port=19080+index, served_model_name='matched-performance-'+mode)
    return fixture, profile


def assert_matched(arms):
    """Whitelist execution endpoints only; catch generation and hidden budget drift."""
    base_fixture, base_profile = arms[0]
    allowed_fixture = {'performance_query_mode', 'performance_arm', 'layerwise_prefill_pressure',
                       'layerwise_pressure_single_assembly', 'layerwise_pressure_fast_metadata'}
    allowed_fixture.add('layerwise_packed_assembly')
    allowed_fixture.add('layerwise_prepared_index')
    allowed_fixture.add('layerwise_batched_reads')
    allowed_fixture.add('layerwise_paired_reads')
    allowed_fixture.add('layerwise_buffered_history')
    clean = lambda value, omit: {k:v for k,v in value.items() if k not in omit}
    for fixture, profile in arms:
        if (clean(profile, {'port', 'served_model_name'}) != clean(base_profile, {'port', 'served_model_name'}) or
                clean(fixture, allowed_fixture) != clean(base_fixture, allowed_fixture)):
            raise ValueError('Matched controls differ beyond the declared recall policy')
        kv = profile['kvmem']
        if (kv['select_budget'], kv['gen_budget'], kv['block_tokens'], kv['sink_blocks'],
                kv['recent_blocks'], fixture['layerwise_workers'], profile['max_num_seqs']) != (36864,16384,32,41,0,8,8):
            raise ValueError('Original capacity or eight-owner comparison changed')


def paired_backend_audit(fixture, evidence):
    """Require the qualified CPU backend to execute in the timed full inputs."""
    if not fixture.get('layerwise_paired_reads',False):return None
    layers=evidence['layers']
    query=[e.get('assembly_backend',{}).get('paired_read_chunks',0) for e in layers]
    pressure=[e.get('assembly_read_stats',{}).get('layerwise_paired_read_chunks',0)
              for e in evidence['refresh'] if e['selection']=='pressure']
    if [e['layer'] for e in layers]!=list(range(16)) or not all(n>0 for n in query):
        raise ValueError('Timed Query did not execute the qualified paired archive backend')
    if fixture['layerwise_prefill_pressure'] and (not pressure or not all(n>0 for n in pressure)):
        raise ValueError('Timed pressure did not execute the qualified paired archive backend')
    return dict(passed=True,query_chunks=query,pressure_chunks=pressure,
                independent_pressure=fixture['layerwise_prefill_pressure'])


def buffered_backend_audit(fixture,evidence):
    if not fixture.get('layerwise_buffered_history'):return None
    events=[e for e in evidence['events'] if e['event']=='pressure_selected']
    if not fixture['layerwise_prefill_pressure'] or not events or any(
            [l['layer'] for l in e['layers']]!=list(range(16)) or any(
                l.get('attention_state_backend')!='native_buffered' or l.get('buffered_commits',0)<=0
                for l in e['layers']) for e in events):
        raise ValueError('Timed pressure did not execute the qualified native history backend')
    return dict(passed=True,pressure_events=len(events))


def archive_backend_audit(profile,stats,region_min_bytes=None,*,require_region_execution=True):
    expected=profile['kvmem'].get('cpu_seal_hash_workers',1)
    result=None
    if expected!=1:
        if (stats['cpu_seal_hash_workers']!=expected or stats['seal_hash_parallel_batches_completed']<=0 or
                stats['seal_hash_parallel_shards_completed']<=stats['seal_hash_parallel_batches_completed'] or
                stats['archive_io_pending_bytes'] or stats['pending_downloads']):
            raise ValueError('Configured parallel archive hashing did not execute or drain')
        result=dict(passed=True,workers=expected,batches=stats['seal_hash_parallel_batches_completed'],
                    shards=stats['seal_hash_parallel_shards_completed'])
    if region_min_bytes is not None:
        if (region_min_bytes!=32*1024**2 or stats['cpu_region_gather_parallel_min_bytes']!=region_min_bytes or
                (require_region_execution and (stats['cpu_region_gather_inline_calls']<=0 or
                                               stats['cpu_region_gather_inline_bytes']<=0)) or
                stats['archive_io_pending_bytes'] or stats['pending_downloads']):
            raise ValueError('Configured inline layer-region gathering did not execute or drain')
        if any(type(stats['cpu_region_gather_'+key]) is not int or stats['cpu_region_gather_'+key]<0
               for key in ('inline_calls','inline_bytes','parallel_calls','parallel_bytes')):
            raise ValueError('Invalid actual region gathering counters')
        result=dict(result or dict(passed=True),region_gather={k:stats['cpu_region_gather_'+k] for k in
            ('parallel_min_bytes','inline_calls','inline_bytes','parallel_calls','parallel_bytes')})
        if not require_region_execution:
            result['region_gather']['execution_required']=False
    if profile['kvmem'].get('cpu_direct_record_seal',False):
        if (stats['seal_direct_records']!=stats['seal_records'] or stats['seal_records']<=0 or
                stats['seal_direct_bytes']!=stats['seal_payload_bytes'] or stats['seal_payload_bytes']<=0 or
                stats['seal_record_copy_seconds']!=0 or stats['archive_io_pending_bytes'] or
                stats['pending_downloads']):
            raise ValueError('Direct CPU publication did not execute without a second record copy or drain')
        result=dict(result or dict(passed=True),direct_seal=dict(records=stats['seal_direct_records'],
            bytes=stats['seal_direct_bytes'],second_record_copy_seconds=stats['seal_record_copy_seconds']))
    if profile['kvmem'].get('cpu_seal_serial_allocate',False):
        if (not profile['kvmem'].get('cpu_direct_record_seal',False) or
                profile['kvmem'].get('cpu_seal_pack_workers',1)!=1 or
                stats['cpu_seal_serial_allocate'] is not True or
                stats['cpu_seal_pack_workers']!=1 or
                stats['cpu_seal_serial_copy_workers']!=8 or
                stats['seal_serial_allocate_batches_completed']!=stats['seal_batches'] or
                stats['seal_batches']<=0 or
                not 0<stats['seal_pack_parallel_batches_completed']<=stats['seal_batches'] or
                not 2*stats['seal_pack_parallel_batches_completed']<=stats['seal_pack_parallel_shards_completed']<=8*stats['seal_pack_parallel_batches_completed']):
            raise ValueError('Private CPU packing did not execute the qualified serial-allocation/parallel-copy backend')
        result=dict(result,private_pack=dict(batches=stats['seal_serial_allocate_batches_completed'],
            parallel_batches=stats['seal_pack_parallel_batches_completed'],
            parallel_shards=stats['seal_pack_parallel_shards_completed'],copy_workers=8))
    return result


def prepare(root, source):
    verify(source)
    if root.exists():
        raise ValueError('Use a fresh version; preserve all previous measurements')
    report = p.read(source/'equivalence.json')
    if not report['passed'] or report['cases'] != 8:
        raise ValueError('Full-input strict equivalence must pass before performance acceptance')
    fixture, profile = p.read(source/'fixture.json'), p.read(source/'profile.json')
    if len(fixture['rows']) != 8 or not fixture['layerwise_prefill_pressure']:
        raise ValueError('Expected the eight original full-input layerwise qualification cases')
    if profile['kvmem'].get('action_native_scores','off') != 'off':
        if not p.read(source/'native-reuse-audit.json')['passed']:
            raise ValueError('Native scores require an audited no-fallback qualification')
    # Repeat the stronger current offline candidate/method proof, even when the
    # source queue's frozen comparator predates it. Never overwrite its report.
    from run_layerwise_pressure_efficiency import signature, full_prefix_candidate_proof
    baseline = Path(report['baseline']); verify(baseline)
    scope_checks = []
    for row in fixture['rows']:
        cases = [r/'formal/sequential/cases'/row['id'] for r in (baseline, source)]
        evidence = [p.read(c/'evidence.json') for c in cases]
        responses = [p.read(c/'response.json') for c in cases]
        if (signature(evidence[0]) != signature(evidence[1]) or
                any(responses[0][key] != responses[1][key] for key in ('output_ids','text'))):
            raise ValueError('Current strict pressure/Query/output audit failed')
        for case in cases:
            result = p.read(case/'result.json')
            if any(result[key+'_sha256'] != p.sha(case/(key+'.json')) for key in ('response','evidence')):
                raise ValueError('First answer or evidence changed')
        scope_checks.append(dict(id=row['id'], proofs=[full_prefix_candidate_proof(e,p.read(r/'profile.json'))
            for e,r in zip(evidence,(baseline,source))]))
    variants = [controls(fixture, profile, mode, i) for i,mode in enumerate(MODES)]
    assert_matched(variants)
    root.mkdir(parents=True)
    stages = []
    for mode, (frozen, config) in zip(MODES, variants):
        arm = root/mode; arm.mkdir(); runtime = arm/'runtime-v1'
        shutil.copytree(source/'runtime-v1', runtime, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('inputs','upstream'):
            shutil.copytree(source/name,arm/name)
        for name in ('run_concurrent_layerwise_ruler.py','gpu_efficiency_monitor.py','analyze_pressure_profile.py'):
            shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
        # Observation only: share the native refresh clock so final per-layer
        # assembly is not mistaken for the earlier global preselection finish.
        query_file=runtime/'src/vllm_kvmem/layerwise_query.py'
        text=query_file.read_text()
        old='record = dict(event=event, mode=self.mode, at=time.time(), **fields)'
        new='record = dict(event=event, mode=self.mode, at=time.time(), monotonic_at=time.perf_counter(), **fields)'
        if text.count(old)==1:query_file.write_text(text.replace(old,new))
        elif text.count(new)!=1:raise ValueError('Unknown layerwise observation boundary')
        p.save(arm/'fixture.json',frozen);p.save(arm/'profile.json',config)
        p.save(arm/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
            if f.is_file() and '__pycache__' not in f.parts})
        p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,
            profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
            runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
        verify(arm)
        p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
        stages.append(dict(name=mode,root=str(arm),prepared_sha256=p.sha(arm/'prepared.json')))
    shutil.copy2(__file__,root/'launch.py')
    p.save(root/'plan.json',dict(source=str(source),source_equivalence_sha256=p.sha(source/'equivalence.json'),
        launch_sha256=p.sha(root/'launch.py'),
        stages=stages,source_scope_checks=scope_checks,scheduled_monitor=False,
        scope='Identical input, model, precision, sampling, capacity, chunks, instrumentation and eight-owner concurrency. '
              'The recall algorithm is the explicit difference. Baseline outputs need not match layerwise outputs.',
        target_layerwise_to_shared_throughput_ratio=.90))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))


def run(root):
    """Exclusive finite comparison; callers may launch only after GPU release."""
    import pynvml
    from run_recall_comparison import termination_guard
    plan=p.read(root/'plan.json');source=Path(plan['source'])
    if (p.sha(root/'launch.py')!=plan['launch_sha256'] or
            p.sha(source/'equivalence.json')!=plan['source_equivalence_sha256']):
        raise ValueError('Qualified source or frozen comparison changed')
    pointer=root.parent/'value-recall-active.json'
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=p.read(pointer)
        if current.get('queue_root') not in (str(source.parent),str(root)):
            raise RuntimeError('Another experiment owns GPU scheduling')
        if p.read(source.parent/'status.json')['status'] not in ('completed','failed'):
            raise RuntimeError('Qualification queue is still active; do not interrupt it')
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        if pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            raise RuntimeError('GPU has an owner; do not launch another model')
        for name in ('process.json','child.json'):
            stat=Path('/proc')/str(p.read(source.parent/name)['pid'])/'stat'
            if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X'):
                raise RuntimeError('Qualification parent or child has not exited')
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        p.save(pointer,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
            process=str(root/'process.json'),experiment_type='matched-layerwise-performance',at=time.time()))
        for stage in plan['stages']:
            arm=Path(stage['root']);verify(arm)
            if p.sha(arm/'prepared.json')!=stage['prepared_sha256']:
                raise ValueError('Frozen performance stage changed')
            p.save(root/'status.json',dict(status='running',stage=stage['name'],active_root=str(arm),at=time.time()))
            command=[sys.executable,'-B',str(arm/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),
                     '--root',str(arm)]
            with (arm/'driver.log').open('x') as log:
                child=subprocess.Popen(command,cwd=arm/'runtime-v1',stdout=log,stderr=subprocess.STDOUT)
            p.save(root/'child.json',dict(pid=child.pid,stage=stage['name'],command=command,at=time.time()))
            if child.wait()!=0:
                raise RuntimeError('Technical performance failure; preserve answers and timing')
        analyze(root)
        p.save(root/'status.json',dict(status='completed',stages=len(plan['stages']),at=time.time()))


def analyze(root):
    """Report wall throughput and overlapping phase latency without adding them."""
    from analyze_pressure_profile import analyze as phases
    from gpu_efficiency_monitor import summarize
    plan = p.read(root/'plan.json'); reports = {}
    arms = []
    for stage in plan['stages']:
        arm = Path(stage['root']); verify(arm)
        if p.sha(arm/'prepared.json') != stage['prepared_sha256']:
            raise ValueError('Performance stage changed')
        if p.read(arm/'status.json')['status'] != 'completed':
            raise ValueError('Do not accept incomplete performance measurements')
        fixture, profile = p.read(arm/'fixture.json'), p.read(arm/'profile.json')
        arms.append((fixture,profile))
        attempts = sorted((arm/'formal/sequential/server-attempts').iterdir())
        if len(attempts) != 1:
            raise ValueError('A technical retry needs explicit attribution, not merged timings')
        capture = attempts[0]/'server/capture'
        archive_backend = archive_backend_audit(profile,p.read(capture/'archive.json'),
                                               fixture.get('cpu_region_gather_min_bytes'),
                                               require_region_execution=stage['name']!='shared_global')
        native_audit = None
        if profile['kvmem'].get('action_native_scores','off') != 'off':
            native = p.read(capture/'native_attention_stats.json')
            counts = native['layer_phase_counts']
            expected = 16 if fixture['layerwise_prefill_pressure'] else 3
            if (len(native['producer_layers'])!=expected or native['disabled_reason'] or
                    native['validation_failures'] or not counts.get('reuse:prefill',0) or
                    any(v for k,v in counts.items() if k.startswith('fallback:prefill'))):
                raise ValueError('Performance measurement silently changed native attention execution')
            native_audit = dict(passed=True,producer_layers=native['producer_layers'],counts=counts)
        wall = p.read(arm/'concurrency-performance.json')
        paired_backend = [dict(id=row['id'],audit=paired_backend_audit(fixture,
            p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')))
            for row in fixture['rows']] if fixture.get('layerwise_paired_reads') else None
        buffered_backend = [dict(id=row['id'],audit=buffered_backend_audit(fixture,
            p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')))
            for row in fixture['rows']] if fixture.get('layerwise_buffered_history') else None
        records = [json.loads(line) for line in (attempts[0]/'gpu.jsonl').read_text().splitlines() if line]
        gpu = summarize(records,wall['started_at'],wall['completed_at'])
        if gpu['external_pids'] or gpu['attribution_coverage_fraction'] < .95:
            raise ValueError('GPU utilization cannot be attributed to this experiment')
        refresh = [json.loads(line) for line in (capture/'bounded_refresh.jsonl').read_text().splitlines() if line]
        queries = [r for r in refresh if r['selection']=='query']
        if len(queries) != len(fixture['rows']):
            raise ValueError('Missing final Query boundaries')
        historical_tokens = sum(q['history_begin'] for q in queries)
        envelope = max(q['refresh_started_at'] for q in queries)-wall['started_monotonic']
        if envelope <= 0:
            raise ValueError('Prefill clock domains or boundaries are invalid')
        reports[stage['name']] = dict(
            inference={k:wall[k] for k in ('workers','cases','inference_wall_seconds','original_input_tokens',
                'input_tokens_per_second','cases_per_hour','scope')},
            history_prefill_envelope=dict(input_tokens=historical_tokens,seconds=envelope,
                tokens_per_second=historical_tokens/envelope,
                scope='Inference dispatch to the last final-Query refresh start; includes pressure and any '
                      'overlapping earlier owners Query/decode. Excludes final Query tail. Not pure GPU compute.'),
            gpu=gpu,phases=phases(capture),native_reuse=native_audit,paired_backend=paired_backend,
            buffered_backend=buffered_backend,archive_backend=archive_backend)
    assert_matched(arms)
    optimized = reports['layerwise_optimized']
    ratios = {name:{metric:optimized[metric]['tokens_per_second']/value[metric]['tokens_per_second']
        for metric in ('history_prefill_envelope',)} for name,value in reports.items() if name!='layerwise_optimized'}
    for name in ratios:
        ratios[name]['end_to_end'] = optimized['inference']['input_tokens_per_second']/reports[name]['inference']['input_tokens_per_second']
    p.save(root/'performance-comparison.json',dict(reports=reports,ratios=ratios,
        goal_met=ratios['shared_layerwise_query']['history_prefill_envelope']>=plan['target_layerwise_to_shared_throughput_ratio'],
        scope='Performance evidence only; full-input exact consistency remains independently required. '
              'Shared controls may generate different output lengths under identical sampling.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','analyze'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path)
    args = parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source.resolve())
        elif args.action=='run':run(root)
        else:analyze(root)
    except BaseException as error:
        if root.exists() and args.action=='run':
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
