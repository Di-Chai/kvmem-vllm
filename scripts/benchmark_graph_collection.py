"""Independent cold smoke pair measuring graph collection with attention unchanged."""
import argparse
from copy import deepcopy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as bench
import benchmark_graph_v2 as shared
import benchmark_graph_v2_performance as perf
import graph_v2_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'kvmem-graph-collection-cold-1'
VARIANTS = ('off', 'collect')
HELPERS = shared.HELPERS + ('benchmark_graph_v2_performance.py', 'qualify_graph_v2_performance.py',
                           'benchmark_graph_collection.py', 'qualify_graph_collection.py')
require, sha, write = protocol.require, protocol.sha, bench.write


def native_identity():
    distribution = importlib.metadata.distribution('vllm')
    names = ('vllm/v1/core/kv_cache_utils.py', 'vllm/v1/kv_cache_interface.py')
    return dict(version=distribution.version, files={str(distribution.locate_file(name).resolve()):
        sha(distribution.locate_file(name)) for name in names}, python=sys.version,
        executable=str(Path(sys.executable).resolve()),
        scope='Installed vLLM version and cache-layout source; complete KVMem Python/native runtime is separately frozen.')


def profiles(base, port):
    common = protocol.profiles(base, port)['A']
    common['kvmem'].update(gpu_raw_k_cache_bytes=0, cpu_direct_record_seal=False,
        async_action_metadata=False, native_action_metadata=False, action_profiling=False)
    result = {name:deepcopy(common) for name in VARIANTS}
    for name, profile in result.items(): profile['kvmem']['graph_mode'] = name
    validate_profiles(result)
    return result


def validate_profile(profile, variant):
    require(variant in VARIANTS and profile['kvmem'].get('graph_mode') == variant, 'Unknown collection variant')
    normalized = deepcopy(profile)
    normalized['kvmem']['graph_mode'] = 'collect'
    protocol.validate_profile(normalized, 'A')
    kv = profile['kvmem']
    require(kv.get('gpu_raw_k_cache_bytes') == 0 and type(kv['gpu_raw_k_cache_bytes']) is int and
        all(kv.get(name) is False for name in ('cpu_direct_record_seal','async_action_metadata',
                                             'native_action_metadata','action_profiling')),
        'Collection pair must keep performance candidates and profiling disabled')
    require(kv.get('async_refresh') is True and kv.get('async_restore') is True and
        kv.get('async_seal') is True and kv.get('direct_pinned_reads') is True,
        'Collection pair keeps the common asynchronous baseline')


def validate_profiles(configs):
    require(set(configs) == set(VARIANTS), 'Both off and collect profiles are required')
    common = []
    for name in VARIANTS:
        validate_profile(configs[name], name)
        value = deepcopy(configs[name]); value['kvmem'].pop('graph_mode'); common.append(value)
    require(common[0] == common[1], 'Only graph_mode may differ between collection profiles')


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root/'status.json', dict(status='preparing',at=time.time()))
    try:
        configs = profiles(json.loads(args.profile.read_text()),args.port)
        identity = shared.model_identity(configs['off'])
        write(root/'niah-A.json',configs['collect'])
        fixture = shared.freeze_niah(SimpleNamespace(reference=args.reference,scope='smoke8'),root,configs['collect'])
        write(root/'fixture.json',fixture)
        hashes, jobs = None, []
        from vllm_kvmem.serve import build_launch
        native = native_identity()
        for name in VARIANTS:
            write(root/(name+'.json'),configs[name])
            target = root/name
            bench.prepare(SimpleNamespace(output=target,fixture=root/'fixture.json',arm=name,
                profile=root/(name+'.json'),qualifier=ROOT/'scripts/qualify_graph_collection.py',
                synchronous=False,port=args.port,client_timeout_seconds=7200))
            for helper in HELPERS: shutil.copyfile(ROOT/'scripts'/helper,target/'runtime/scripts'/helper)
            current = {str(path.relative_to(target/'runtime')):sha(path) for path in
                       (target/'runtime').rglob('*') if path.is_file()}
            require(any(k.startswith('src/vllm_kvmem/_native') and k.endswith('.so') for k in current),
                    'Current native score-frame extension is absent')
            require(hashes is None or current == hashes,'Runtime changed while freezing the pair')
            hashes = current
            expected = build_launch(configs[name],target,target/'server')
            write(target/'expected-launch.json',{key:expected[key] for key in ('profile','vllm_argv')})
            manifest = json.loads((target/'manifest.json').read_text())
            manifest.update(runtime_sha256=current,schema=SCHEMA,native_identity=native,
                expected_launch_sha256=sha(target/'expected-launch.json'))
            write(target/'manifest.json',manifest)
            jobs.append(dict(id=name,manifest_sha256=sha(target/'manifest.json')))
        frozen = root/'frozen'; frozen.mkdir()
        for name in HELPERS:
            shutil.copyfile(ROOT/'scripts'/name,frozen/name)
            require(sha(frozen/name)==hashes['scripts/'+name],'Helper changed during freeze: '+name)
        patch = subprocess.check_output(['git','diff','HEAD','--','src','csrc','third_party','scripts'],cwd=ROOT)
        (root/'source.diff').write_bytes(patch)
        status = subprocess.check_output(['git','status','--porcelain','--','src','csrc','third_party','scripts'],cwd=ROOT,text=True)
        maximum = max(row['served_input_tokens'] for row in fixture['rows'])+8192+16384
        credit = ((maximum+31)//32)*32*17*4*256*2
        require(8*credit <= fixture['cpu_payload_credits_bytes'],'Eight complete histories do not fit CPU credits')
        write(root/'manifest.json',dict(schema=SCHEMA,variants=list(VARIANTS),jobs=jobs,created_at=time.time(),
            source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_dirty=bool(status.strip()),source_worktree_status=status,source_diff_sha256=sha(root/'source.diff'),
            runtime_sha256=hashes,frozen_sha256={p.name:sha(p) for p in frozen.iterdir()},
            fixture_sha256=sha(root/'fixture.json'),model_identity=identity,native_identity=native,
            input_sha256={**{str(p.relative_to(root)):sha(p) for p in (root/'dataset').rglob('*') if p.is_file()},
                'niah-A.json':sha(root/'niah-A.json'),**{name+'.json':sha(root/(name+'.json')) for name in VARIANTS}},
            start_cpu_available_bytes=8*credit+24*1024**3,minimum_cpu_available_bytes=8*1024**3,minimum_gpu_free_mib=4096,
            changed_profile_field='kvmem.graph_mode',pressure_policy='attention',attention_capture=True,
            protocol='One cold original eight-case256K cohort per fresh service, off then collect; no warmup,300s window, replacement or retries.'))
        write(root/'status.json',dict(status='prepared',at=time.time()))
        print(json.dumps(dict(prepared=str(root),variants=VARIANTS,gpu_started=False)),flush=True)
    except BaseException as error:
        write(root/'status.json',dict(status='failed',error=repr(error),at=time.time())); raise


def validate(root):
    m = json.loads((root/'manifest.json').read_text())
    require(m.get('schema')==SCHEMA and m.get('variants')==list(VARIANTS) and
        [j['id'] for j in m['jobs']]==list(VARIANTS),'Collection schema/order changed')
    require(m['native_identity']==native_identity(),'Installed native implementation or interpreter changed')
    require(sha(root/'source.diff')==m['source_diff_sha256'] and sha(root/'fixture.json')==m['fixture_sha256'],
            'Frozen source/fixture changed')
    for name,expected in m['frozen_sha256'].items(): require(sha(root/'frozen'/name)==expected,'Frozen controller changed')
    for name,expected in m['input_sha256'].items(): require(sha(root/name)==expected,'Frozen input/profile changed')
    configs = {}
    for job in m['jobs']:
        target = root/job['id']
        require(sha(target/'manifest.json')==job['manifest_sha256'],'Frozen job manifest changed')
        child = json.loads((target/'manifest.json').read_text())
        require(child.get('schema')==SCHEMA and child['arm']==job['id'] and child['runtime_sha256']==m['runtime_sha256'] and
            child['native_identity']==m['native_identity'],'Job runtime/native contract differs')
        require(sha(target/'profile.json')==child['profile_sha256']==sha(root/(job['id']+'.json')) and
            sha(target/'fixture.json')==child['fixture_sha256']==m['fixture_sha256'] and
            sha(target/'expected-launch.json')==child['expected_launch_sha256'],'Job inputs changed')
        for name,expected in child['runtime_sha256'].items(): require(sha(target/'runtime'/name)==expected,'Frozen runtime changed')
        configs[job['id']]=json.loads((target/'profile.json').read_text())
    validate_profiles(configs)
    for group in ('config_sha256','tokenizer_sha256'):
        for name,expected in m['model_identity'][group].items(): require(sha(name)==expected,'Model/tokenizer identity changed')
    for name,expected in m['model_identity']['weight_stat_identity'].items():
        stat=Path(name).stat()
        require(dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)==expected,'Model weights changed')
    return m


def comparison(root):
    metrics = {name:json.loads((root/name/'qualification/collection.json').read_text()) for name in VARIANTS}
    fixtures=[]
    for name in VARIANTS:
        directory=root/name/'qualification'
        result=json.loads((directory/'results.json').read_text())
        require(result.get('status')=='passed' and result.get('all_owned_sessions_released') is True and
            result.get('collection_sha256')==sha(directory/'collection.json'),'Missing passed, released result or changed measurements')
        fixtures.append(result['fixture_sha256'])
    require(fixtures[0]==fixtures[1],'Pair fixture identities differ')
    off, collect = (metrics[name] for name in VARIANTS)
    require(all(row.get('schema')==SCHEMA and row.get('variant')==name for name,row in metrics.items()),'Wrong collection result schema')
    require(off['cases'].keys()==collect['cases'].keys() and len(off['cases'])==8,'Pair case identities differ')
    for name in off['cases']:
        require(all(off['cases'][name][field]==collect['cases'][name][field] for field in
            ('input_sha256','input_tokens','seed','query_span')),'Pair input/Query/sampling identity differs')
    conditions = dict(identical_accepted_outputs=all(off['cases'][k]['output_ids_sha256']==collect['cases'][k]['output_ids_sha256'] for k in off['cases']),
        identical_query_and_pressure_selections=all(off['cases'][k]['selection_sha256']==collect['cases'][k]['selection_sha256'] for k in off['cases']),
        identical_attention_samples=all(off['cases'][k]['attention_samples']==collect['cases'][k]['attention_samples'] for k in off['cases']),
        collection_exercised=collect['collection_exercised'],off_has_no_graph_state=off['off_has_no_graph_state'],
        actual_native_c8_observed=all(item['actual_concurrency']['executed_request_peak']==8 for item in metrics.values()))
    blockers = [k for k,v in conditions.items() if not v]
    require(all(type(row.get('wall_seconds')) in (int,float) and row['wall_seconds']>0 for row in metrics.values()),
            'Invalid measured collection wall interval')
    ratio = collect['wall_seconds']/off['wall_seconds']
    for name in VARIANTS: metrics[name]['nvml']=json.loads((root/name/'nvml-summary.json').read_text())
    return dict(schema=SCHEMA,variants=metrics,conditions=conditions,inconclusive_reasons=blockers,
        verdict='inconclusive' if blockers else 'comparable_cold_collection_smoke',
        raw_wall_ratio_collect_over_off=ratio,
        observed_collection_extra_wall_seconds=collect['wall_seconds']-off['wall_seconds'] if not blockers else None,
        observed_collection_wall_fraction=ratio-1 if not blockers else None,
        performance_gain_established=False,gpu_idle_inferred=False,
        scope='Single cold pair; collection includes its parser,graph acceptance,snapshots and diagnostics. No continuous256K or300s performance claim.')


def run(args):
    root = args.output.resolve()
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m = validate(root)
        require(Path(__file__).resolve()==root/'frozen/benchmark_graph_collection.py','Use the frozen collection controller')
        require(json.loads((root/'status.json').read_text())['status']=='prepared' and all(
            json.loads((root/name/'status.json').read_text())['status']=='prepared' for name in VARIANTS),
            'Started,failed,interrupted or completed campaigns cannot be retried')
        write(root/'status.json',dict(status='running',at=time.time()))
        try:
            with bench.cleanup_on_termination() as begin_cleanup:
                for name in VARIANTS:
                    require(shared.cpu_available()>=m['start_cpu_available_bytes'],'Insufficient host headroom')
                    target,process,samples = root/name,None,[]
                    try:
                        with (target/'controller.log').open('x') as log,(target/'resources.jsonl').open('x') as output:
                            process = subprocess.Popen([sys.executable,'-B',str(target/'runtime/scripts/benchmark_graph_collection.py'),
                                'arm-run','--output',str(target)],stdout=log,stderr=subprocess.STDOUT,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                            while process.poll() is None:
                                sample=perf.sample_resources(); samples.append(sample)
                                output.write(json.dumps(sample)+'\n'); output.flush()
                                require(sample['cpu_available_bytes']>=m['minimum_cpu_available_bytes'] and
                                    sample['gpu_free_mib']>=m['minimum_gpu_free_mib'],'Runtime memory floor reached')
                                time.sleep(1)
                        require(process.returncode==0,'Collection variant failed; preserve first answers')
                    finally:
                        if process is not None: require(not bench.stop_owned_groups((process,)),'Owned controller did not stop')
                    metrics=json.loads((target/'qualification/collection.json').read_text())
                    window=[r for r in samples if metrics['measurement_started_at']<=r['at']<=metrics['measurement_finished_at']]
                    write(target/'nvml-summary.json',perf.resource_summary(window))
                begin_cleanup()
            write(root/'comparison.json',comparison(root))
            write(root/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            write(root/'status.json',dict(status='failed',error=repr(error),at=time.time())); raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','validate','run','arm-run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=ROOT/'artifacts/graph-overhead-20261018/gpu-r2/after/profile.json')
    parser.add_argument('--reference',type=Path,default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--port',type=int,default=18688)
    args=parser.parse_args(argv)
    if args.action=='prepare': prepare(args)
    elif args.action=='validate':
        validate(args.output.resolve()); print(json.dumps(dict(valid=True,gpu_started=False)))
    elif args.action=='run': run(args)
    else:
        require(Path(__file__).resolve()==args.output.resolve()/'runtime/scripts/benchmark_graph_collection.py','Use frozen arm runner')
        bench.run(args)


if __name__=='__main__': main()
