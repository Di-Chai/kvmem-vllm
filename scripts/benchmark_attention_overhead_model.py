"""Four independent, alternating full C8 comparisons; native QK reuse is off."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import psutil

import benchmark_block_graph as owner
from benchmark_graph_cost import cpu_available,read,save,sha

ROOT=Path(__file__).resolve().parents[1]
SWITCHES=dict(decode='action_native_decode_commit',pool='action_download_pool',
    fused='action_fused_scoring',propagation='graph_native_propagation')
COUNTERS=('vllm:spec_decode_num_drafts_total','vllm:spec_decode_num_draft_tokens_total',
          'vllm:spec_decode_num_accepted_tokens_total')


def prepare(args):
    from graph_v2_protocol import validate_profile
    from vllm_kvmem.serve import build_launch
    from vllm_kvmem.options import RuntimeOptions
    reference=args.reference.resolve();base=read(reference/'profile.json');old=read(reference/'manifest.json')
    if read(reference/'status.json')['status']!='passed':raise ValueError('Require original passed full C8 reference')
    for key,name in [('profile_sha256','profile.json'),('fixture_sha256','fixture.json')]:
        if sha(reference/name)!=old[key]:raise ValueError('Reference changed')
    validate_profile(base,'D','niah')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    jobs={};order=[]
    for component,flag in SWITCHES.items():
        for repetition in (1,2):
            for mode in ('reference','optimized'):
                name=f'{component}-{mode}-{repetition}';order.append(name)
                profile=deepcopy(base)
                profile.update(port=args.port,served_model_name='kvmem-attention-overhead')
                profile['kvmem'].update({key:False for key in SWITCHES.values()})
                profile['kvmem'].update(action_commit_max_bytes=16777216,
                    action_download_max_bytes=67108864,action_download_max_slots=128)
                # The independent commits also work without the experimental
                # native-QK route being installed in this checkout.
                if 'action_native_scores' in RuntimeOptions.__dataclass_fields__:
                    profile['kvmem']['action_native_scores']='off'
                else:
                    profile['kvmem'].pop('action_native_scores',None)
                profile['kvmem'][flag]=mode=='optimized'
                validate_profile(profile,'D','niah')
                path=root/(name+'-profile.json');save(path,profile);job=root/name
                build_launch(profile,job,job/'launch-preflight')
                owner.prepare(argparse.Namespace(output=job,fixture=reference/'fixture.json',arm='D',
                    profile=path,qualifier=ROOT/'scripts/qualify_graph_v2_decode.py',port=args.port,
                    synchronous=False,protocol=None,client_timeout_seconds=2400))
                for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,job/'runtime/scripts'/p.name)
                manifest=read(job/'manifest.json')
                manifest.update(overhead_component=component,overhead_mode=mode,overhead_switch=flag,
                    repetition=repetition,native_qk_mode='off',reference_manifest_sha256=sha(reference/'manifest.json'),
                    runtime_sha256={str(p.relative_to(job/'runtime')):sha(p) for p in (job/'runtime').rglob('*') if p.is_file()})
                save(job/'manifest.json',manifest);jobs[name]=sha(job/'manifest.json')
    frozen=root/'frozen';frozen.mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,frozen/p.name)
    save(root/'manifest.json',dict(order=order,jobs=jobs,switches=SWITCHES,reference=str(reference),
        frozen_sha256={p.name:sha(p) for p in frozen.iterdir()},created_at=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        cpp_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('csrc','third_party/qw3')
            for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.cuh')},
        start_cpu_available_bytes=105620439040,minimum_cpu_available_bytes=8589934592,minimum_gpu_free_mib=4096,
        scope='Four independent A/B comparisons with two fresh cohorts per arm. Original V2 D multi/alpha.2/graph-pressure '
              'policy and full256K+16K C8 workload are fixed; exactly one overhead switch differs in each comparison. '
              'Native FlashInfer/XQA QK reuse is disabled throughout.'))
    save(root/'status.json',dict(status='prepared',at=time.time()))


def metrics(port):
    begin=time.perf_counter()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    values={key:0. for key in COUNTERS}
    with opener.open(f'http://127.0.0.1:{port}/metrics',timeout=2) as response:
        for line in response.read().decode().splitlines():
            if line.startswith('#'):continue
            key=line.split('{',1)[0].split(' ',1)[0]
            if key in values:values[key]+=float(line.rsplit(' ',1)[1])
    return dict(started_at=begin,finished_at=time.perf_counter(),counters=values)


def audit_switch(job):
    manifest=read(job/'manifest.json');capture=job/'server/capture'
    if (capture/'native_attention_stats.json').exists():raise ValueError('Unexpected native QK reuse')
    config=read(capture/'action_config.json')
    actions=[json.loads(line) for line in (capture/'action_attention.jsonl').read_text().splitlines()]
    native=manifest['overhead_mode']=='optimized';component=manifest['overhead_component']
    if config.get('native_decode_commit',False)!=(native and component=='decode'):raise ValueError('Decode mode differs')
    if config.get('download_pool',False)!=(native and component=='pool'):raise ValueError('Download mode differs')
    if config.get('fused_scoring',False)!=(native and component=='fused'):raise ValueError('Scoring mode differs')
    if component=='decode' and native and not max(r['decode_commit_native_calls'] for r in actions):
        raise ValueError('Native decode acceptance did not execute')
    if component=='pool' and native:
        for r in actions:
            pool=r['download_pool_total']
            if (not pool['reuses'] or pool['exhaustions'] or pool['quarantined_slots'] or
                    pool['allocated_pinned_bytes']>pool['max_pinned_bytes'] or pool['event_count']>pool['max_events']):
                raise ValueError('Pool was unused, exceeded capacity or failed')
    if component=='fused' and native and not max(r['scoring_backend_counts'].get('fused',0) for r in actions):
        raise ValueError('Fused bypass scorer did not execute')
    graph=[json.loads(line) for line in (capture/'block_graph.jsonl').read_text().splitlines()]
    expected=native and component=='propagation'
    for phase in ('query','pressure'):
        rows=[r for r in graph if r['phase']==phase]
        if not rows or any((r.get('backend')=='cpp_v2')!=expected for r in rows):
            raise ValueError('Actual propagation backend differs: '+phase)
    return dict(config=config,actions=actions,graph_backend='cpp_v2' if expected else 'python_reference')


def run(args):
    from audit_attention_overhead_run import audit
    root=args.output.resolve();manifest=read(root/'manifest.json')
    if read(root/'status.json')['status']!='prepared':raise ValueError('Use a new attempt; never auto-repeat failures')
    for name,digest in manifest['frozen_sha256'].items():
        if sha(root/'frozen'/name)!=digest:raise ValueError('Frozen coordinator changed')
    process=None
    try:
        for name in manifest['order']:
            job=root/name
            if sha(job/'manifest.json')!=manifest['jobs'][name]:raise ValueError('Frozen job changed')
            while owner.gpu_processes() or cpu_available()<manifest['start_cpu_available_bytes']:
                save(root/'status.json',dict(status='waiting_for_resources',job=name,at=time.time()));time.sleep(5)
            command=[sys.executable,'-B',str(job/'runtime/scripts/benchmark_block_graph.py'),'run',
                '--output',str(job),'--fixture',str(job/'fixture.json'),'--arm','multi']
            # The generic runner CLI accepts graph-mode labels. Its frozen
            # manifest, not this launcher label, supplies arm D to the strict
            # V2 qualification client.
            began=time.time();first_testing=None;last_metric=0.
            with owner.cleanup_on_termination() as cleanup:
                try:
                    with (root/(name+'.log')).open('x') as log,(root/(name+'-resources.jsonl')).open('x') as resources,(root/(name+'-mtp-counters.jsonl')).open('x') as counter_file:
                        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        save(root/'status.json',dict(status='running',job=name,pid=process.pid,at=time.time()))
                        while process.poll() is None:
                            used,free,util=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
                            available=cpu_available()
                            try:state=read(job/'status.json')
                            except (FileNotFoundError,json.JSONDecodeError):state=dict(status='transition')
                            if state['status']=='testing' and first_testing is None:first_testing=time.time()
                            row=dict(at=time.time(),cpu_available_bytes=available,gpu_used_mib=used,gpu_free_mib=free,gpu_utilization=util,phase=state['status'])
                            if time.time()-last_metric>=5:
                                last_metric=time.time();rss=pss=0
                                try:children=psutil.Process(process.pid).children(recursive=True)
                                except psutil.Error:children=[]
                                for child in children:
                                    try:
                                        memory=child.memory_full_info();rss+=memory.rss;pss+=getattr(memory,'pss',memory.rss)
                                    except psutil.Error:pass
                                row.update(owned_process_rss_bytes=rss,owned_process_pss_bytes=pss)
                                if state['status']=='testing':
                                    try:counter_file.write(json.dumps(metrics(read(job/'profile.json')['port']))+'\n');counter_file.flush()
                                    except (OSError,ValueError):pass
                            resources.write(json.dumps(row)+'\n');resources.flush()
                            if available<manifest['minimum_cpu_available_bytes'] or free<manifest['minimum_gpu_free_mib']:
                                raise RuntimeError('Resource guard reached')
                            time.sleep(1)
                        returncode=process.returncode
                finally:
                    cleanup()
                    if owner.stop_owned_groups((process,),grace_seconds=60):raise RuntimeError('Owned processes remained')
                    process=None
            proof=audit(job)
            proof.update(controller_returncode=returncode,engine_startup_seconds=None if first_testing is None else first_testing-began,
                overhead=audit_switch(job))
            save(root/(name+'-executed-audit.json'),proof);print(name+' complete',flush=True)
        save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_attempt_preserved',error=str(error),at=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);parser.add_argument('--port',type=int,default=18836)
    args=parser.parse_args();(prepare if args.action=='prepare' else run)(args)


if __name__=='__main__':main()
