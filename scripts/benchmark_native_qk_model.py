"""Freeze and alternate original full C8 runs, changing only native QK reuse.

Uses the original qualifier and independent executed-window audit. Every attempt
gets a new immutable runtime; failures are retained and never automatically retried.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import benchmark_block_graph as owner
from benchmark_graph_cost import cpu_available, read, save, sha

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    from qualify_graph_cost import profiles
    from vllm_kvmem.serve import build_launch
    reference = args.reference.resolve()
    old = read(reference/'manifest.json')
    if read(reference/'status.json')['status'] != 'passed':
        raise ValueError('Require the original passed C8 reference')
    for key,name in (('profile_sha256','profile.json'),('fixture_sha256','fixture.json')):
        if sha(reference/name) != old[key]:
            raise ValueError('Reference changed: '+name)
    base = profiles(read(reference/'profile.json'))['attention']
    root = args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    order = ['reuse-1','bypass-1','reuse-2','bypass-2']
    jobs = {}
    for name in order:
        profile = deepcopy(base)
        profile.update(port=args.port,served_model_name='kvmem-native-qk-cost')
        profile['kvmem'].update(action_native_scores='reuse' if name.startswith('reuse') else 'off',
            action_native_max_rows=256,action_native_max_bytes=2147483648)
        profile_path = root/(name+'-profile.json');save(profile_path,profile)
        job = root/name
        build_launch(profile,job,job/'launch-preflight')
        owner.prepare(argparse.Namespace(output=job,fixture=reference/'fixture.json',arm='attention',
            profile=profile_path,qualifier=ROOT/'scripts/qualify_graph_cost.py',port=args.port,
            synchronous=False,protocol=None,client_timeout_seconds=2400))
        # Freeze the same current helpers as the serving package. The original
        # qualifier imports several helpers beyond the owner harness's default.
        for path in (ROOT/'scripts').glob('*.py'):
            shutil.copyfile(path,job/'runtime/scripts'/path.name)
        manifest = read(job/'manifest.json')
        manifest.update(native_qk_mode=profile['kvmem']['action_native_scores'],
            reference_manifest_sha256=sha(reference/'manifest.json'),base_commit='f639acf',
            runtime_sha256={str(p.relative_to(job/'runtime')):sha(p)
                            for p in (job/'runtime').rglob('*') if p.is_file()})
        save(job/'manifest.json',manifest)
        jobs[name] = sha(job/'manifest.json')
    frozen = root/'frozen';frozen.mkdir()
    for path in (ROOT/'scripts').glob('*.py'):
        shutil.copyfile(path,frozen/path.name)
    save(root/'manifest.json',dict(order=order,jobs=jobs,reference=str(reference),
        frozen_sha256={p.name:sha(p) for p in frozen.iterdir()},
        allowed_profile_differences=['kvmem.action_native_scores'],
        start_cpu_available_bytes=105620439040,minimum_cpu_available_bytes=8589934592,
        minimum_gpu_free_mib=4096,created_at=time.time(),
        scope='Two fresh C8 cohorts per arm, alternating reuse/off; original recency pressure, '
              'Mean-K Query policy, sampling and graph policy remain identical.'))
    save(root/'status.json',dict(status='prepared',at=time.time()))


def run(args):
    from audit_graph_cost_run import audit
    root = args.output.resolve();manifest = read(root/'manifest.json')
    if read(root/'status.json')['status'] != 'prepared':
        raise ValueError('Use a new attempt; started/failed runs are not repeated')
    for name,digest in manifest['frozen_sha256'].items():
        if sha(root/'frozen'/name) != digest:
            raise ValueError('Frozen coordinator changed')
    process = None
    try:
        for name in manifest['order']:
            job = root/name
            if sha(job/'manifest.json') != manifest['jobs'][name]:
                raise ValueError('Frozen job changed')
            while owner.gpu_processes() or cpu_available() < manifest['start_cpu_available_bytes']:
                save(root/'status.json',dict(status='waiting_for_resources',job=name,at=time.time()))
                time.sleep(5)
            command = [sys.executable,'-B',str(job/'runtime/scripts/benchmark_block_graph.py'),
                'run','--output',str(job),'--fixture',str(job/'fixture.json'),'--arm','attention']
            began = time.time();first_testing = None
            with owner.cleanup_on_termination() as cleanup:
                try:
                    with (root/(name+'.log')).open('x') as log, (root/(name+'-resources.jsonl')).open('x') as resources:
                        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        save(root/'status.json',dict(status='running',job=name,pid=process.pid,at=time.time()))
                        while process.poll() is None:
                            used,free,util = map(float,subprocess.check_output(['nvidia-smi',
                                '--query-gpu=memory.used,memory.free,utilization.gpu',
                                '--format=csv,noheader,nounits'],text=True).strip().split(','))
                            memory = cpu_available()
                            try:
                                state = read(job/'status.json')
                            except (FileNotFoundError,json.JSONDecodeError):
                                state = dict(status='transition')
                            if state['status']=='testing' and first_testing is None:
                                first_testing=time.time()
                            resources.write(json.dumps(dict(at=time.time(),cpu_available_bytes=memory,
                                gpu_used_mib=used,gpu_free_mib=free,gpu_utilization=util,phase=state['status']))+'\n')
                            resources.flush()
                            if memory < manifest['minimum_cpu_available_bytes'] or free < manifest['minimum_gpu_free_mib']:
                                raise RuntimeError('Resource guard reached')
                            time.sleep(1)
                        returncode = process.returncode
                finally:
                    cleanup()
                    # The child owns separate server/client process groups and
                    # needs up to 30+15 seconds for its own bounded cleanup.
                    if owner.stop_owned_groups((process,),grace_seconds=60):
                        raise RuntimeError('Owned process cleanup failed')
                    process = None
            # Preserve the strict client's outcome, independently checking its
            # complete executed window exactly as the original cost experiment.
            proof = audit(job)
            proof.update(controller_returncode=returncode,engine_startup_seconds=(
                None if first_testing is None else first_testing-began))
            capture = job/'server/capture'
            mode = read(job/'manifest.json')['native_qk_mode']
            native = read(capture/'native_attention_stats.json') if (capture/'native_attention_stats.json').exists() else None
            if mode == 'reuse' and (native is None or native['requested_mode'] != 'reuse' or
                    not native['layer_phase_counts'].get('reuse:prefill') or
                    not native['layer_phase_counts'].get('reuse:decode')):
                raise ValueError('Native scoring was not exercised in both phases')
            proof['native_stats'] = native
            save(root/(name+'-executed-audit.json'),proof)
            print(name+' complete',flush=True)
        save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_attempt_preserved',error=str(error),at=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--port',type=int,default=18726)
    args = parser.parse_args()
    (prepare if args.action=='prepare' else run)(args)


if __name__ == '__main__':
    main()
