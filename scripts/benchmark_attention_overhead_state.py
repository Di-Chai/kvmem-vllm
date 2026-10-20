"""Frozen two-owner, three-turn correctness for each independent switch."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import time
import benchmark_block_graph as owner
from benchmark_graph_cost import read,save,sha
from benchmark_attention_overhead_model import SWITCHES

ROOT=Path(__file__).resolve().parents[1]


def prepare(args):
    from vllm_kvmem.options import RuntimeOptions
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    reference=args.reference.resolve();base=read(reference/'profile.json')
    if read(reference/'status.json')['status']!='passed':raise ValueError('Require passed original state reference')
    jobs={};order=['reference',*SWITCHES,'all']
    for name in order:
        profile=deepcopy(base);profile.update(port=args.port,served_model_name='kvmem-overhead-state')
        kv=profile['kvmem']
        for key in ('graph_locality_weight','graph_attention_weight','action_pressure'):kv.pop(key,None)
        kv.update(graph_mode='multi',graph_scoring_version='v2',graph_alpha=.2,pressure_policy='graph',
            graph_trace=False,action_fraction=0.,action_commit_max_bytes=16777216,
            action_download_max_bytes=67108864,action_download_max_slots=128)
        if 'action_native_scores' in RuntimeOptions.__dataclass_fields__:
            kv['action_native_scores']='off'
        else:
            kv.pop('action_native_scores',None)
        kv.update({flag:name in (component,'all') for component,flag in SWITCHES.items()})
        path=root/(name+'-profile.json');save(path,profile);job=root/name
        owner.prepare(argparse.Namespace(output=job,fixture=reference/'qualification/fixture.json',arm='multi',
            profile=path,qualifier=ROOT/'scripts/qualify_block_graph.py',port=args.port,synchronous=False,
            protocol=None,client_timeout_seconds=1200))
        for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,job/'runtime/scripts'/p.name)
        manifest=read(job/'manifest.json')
        manifest.update(overhead_component=name,runtime_sha256={str(p.relative_to(job/'runtime')):sha(p)
            for p in (job/'runtime').rglob('*') if p.is_file()})
        save(job/'manifest.json',manifest);jobs[name]=sha(job/'manifest.json')
    frozen=root/'frozen';frozen.mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,frozen/p.name)
    save(root/'manifest.json',dict(order=order,jobs=jobs,frozen_sha256={p.name:sha(p) for p in frozen.iterdir()},
        reference=str(reference),scope='Correctness only. Same exact two-user first-turn inputs, tool continuation and '
        'edited suffix fixture. Each switch alone, plus all four. V2 multi/alpha.2, graph pressure, native QK reuse off. '
        'Require actual target/MTP/recurrent byte restores, accepted sample oracle, mandatory preservation and release.'))
    save(root/'status.json',dict(status='prepared',at=time.time()))


def check(job):
    import json
    profile=read(job/'profile.json')['kvmem'];result=read(job/'qualification/results.json')
    capture=job/'server/capture';config=read(capture/'action_config.json')
    if result['status']!='passed' or not result['all_owned_sessions_released'] or not result['graph_checks']['complete']:
        raise ValueError('Real state qualification or required coverage incomplete')
    if result['quality']['correct']!=result['quality']['total']:raise ValueError('Narrow synthetic answer check failed')
    for key,name in [('native_decode_commit','action_native_decode_commit'),('download_pool','action_download_pool'),('fused_scoring','action_fused_scoring')]:
        if config[key]!=profile[name]:raise ValueError('Actual state switch differs: '+name)
    actions=[json.loads(line) for line in (capture/'action_attention.jsonl').read_text().splitlines()]
    graphs=[json.loads(line) for line in (capture/'block_graph.jsonl').read_text().splitlines()]
    if profile['action_native_decode_commit'] and not max(r['decode_commit_native_calls'] for r in actions):raise ValueError('Native accepted rows not exercised')
    if profile['action_download_pool'] and any(r['download_pool_total']['exhaustions'] or not r['download_pool_total']['reuses'] for r in actions):raise ValueError('Pool was unused or exhausted')
    if profile['action_fused_scoring'] and not max(r['scoring_backend_counts'].get('fused',0) for r in actions):raise ValueError('Fused scorer unused')
    if profile['graph_native_propagation']:
        for phase in ('query','pressure'):
            if not any(r.get('backend')=='cpp_v2' and r['phase']==phase for r in graphs):raise ValueError('Native propagation unused: '+phase)
    target=[r for case in result['evidence'] for r in case.get('target_restore',[])]
    recurrent=[r for case in result['evidence'] for r in case.get('recurrent_restore',[])]
    if len(target)<4 or len(recurrent)<4 or not all(r['byte_equal'] and r['mtp_attention_layers']>0 for r in target) or not all(r['recurrent_byte_equal'] for r in recurrent):raise ValueError('Missing target/MTP/recurrent byte restore proof')
    for case in result['evidence']:
        query=case['query'];covered={i for begin,count in query['selected_history_ranges'] for i in range(begin//32,(begin+count-1)//32+1)}
        if not set(query['mandatory_fit']['hard_blocks'])<=covered:raise ValueError('Mandatory history omitted')
    if (capture/'native_attention_stats.json').exists():raise ValueError('Unexpected native QK route')
    return dict(status='passed',quality=result['quality'],graph_checks=result['graph_checks'],
        target_mtp_byte_restores=len(target),recurrent_byte_restores=len(recurrent),
        action_config=config,actions=actions,result_sha256=sha(job/'qualification/results.json'),cleanup=read(job/'cleanup.json'))


def run(args):
    root=args.output.resolve();manifest=read(root/'manifest.json')
    if read(root/'status.json')['status']!='prepared':raise ValueError('No automatic retry of started attempts')
    for name,digest in manifest['frozen_sha256'].items():
        if sha(root/'frozen'/name)!=digest:raise ValueError('Frozen controller changed')
    results={};process=None
    try:
        for name in manifest['order']:
            job=root/name
            if sha(job/'manifest.json')!=manifest['jobs'][name]:raise ValueError('Frozen manifest changed')
            while owner.gpu_processes():save(root/'status.json',dict(status='waiting_for_gpu',job=name,at=time.time()));time.sleep(2)
            command=[sys.executable,'-B',str(job/'runtime/scripts/benchmark_block_graph.py'),'run',
                '--output',str(job),'--fixture',str(job/'fixture.json'),'--arm','multi']
            with owner.cleanup_on_termination() as cleanup:
                try:
                    with (root/(name+'.log')).open('x') as log:
                        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        save(root/'status.json',dict(status='running',job=name,pid=process.pid,at=time.time()))
                        code=process.wait()
                finally:
                    cleanup()
                    if owner.stop_owned_groups((process,),grace_seconds=60):raise RuntimeError('Owned state processes remained')
                    process=None
            if code:raise RuntimeError('State case failed; retain its first attempt: '+name)
            results[name]=check(job);save(root/(name+'-proof.json'),results[name]);print(name+' passed',flush=True)
        save(root/'summary.json',dict(status='passed',jobs=results));save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_attempt_preserved',error=str(error),at=time.time()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path);parser.add_argument('--port',type=int,default=18838)
    args=parser.parse_args();(prepare if args.action=='prepare' else run)(args)


if __name__=='__main__':main()
