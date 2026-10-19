"""Freeze and run a first three-arm attention/collection overhead screen.

Original C8 input, output, MTP and K/G geometry remain fixed. Common recency
pressure keeps collected statistics out of selection. No automatic retries.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import benchmark_block_graph as owner

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'graph-cost-screen-1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare(args):
    from qualify_graph_cost import VARIANTS, profiles
    old = args.reference.resolve()
    prior = read(old/'manifest.json')
    if read(old/'status.json')['status'] != 'passed':
        raise ValueError('Reference C8 must have passed')
    for field,name in (('profile_sha256','profile.json'),('fixture_sha256','fixture.json')):
        if sha(old/name) != prior[field]: raise ValueError('Reference '+name+' changed')
    for name, digest in prior['runtime_sha256'].items():
        if sha(old/'runtime'/name) != digest:
            raise ValueError('Reference runtime changed: '+name)
    configs = profiles(read(old/'profile.json'))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    from vllm_kvmem.serve import build_launch
    manifests = {}
    for variant in VARIANTS:
        job = root/variant
        job.mkdir()
        shutil.copytree(old/'runtime', job/'runtime')
        shutil.copyfile(ROOT/'scripts/qualify_graph_cost.py', job/'runtime/scripts/qualify_graph_cost.py')
        profile = configs[variant]
        profile.update(port=args.port, served_model_name='kvmem-cost-screen')
        build_launch(profile, job, job/'launch-preflight')
        save(job/'profile.json', profile)
        shutil.copyfile(old/'fixture.json', job/'fixture.json')
        runtime = {str(p.relative_to(job/'runtime')):sha(p) for p in (job/'runtime').rglob('*') if p.is_file()}
        # Reuse the exact qualified source, not a hidden executable patch.
        for name, digest in runtime.items():
            if name.startswith('src/') and sha(ROOT/name) != digest:
                raise ValueError('Reference serving core differs from current source: '+name)
        manifest = dict(prior, runtime_sha256=runtime, arm=variant,
            qualification_script='qualify_graph_cost.py', fixture=str(job/'fixture.json'),
            original_fixture=str(old/'fixture.json'), profile_sha256=sha(job/'profile.json'),
            created_at=time.time(), reference_manifest_sha256=sha(old/'manifest.json'),
            cost_schema=SCHEMA,
            scope='First overhead screen; all variants use common recency pressure and unpropagated Mean-K. '
                'Only attention capture and graph collection differ; selection/output drift must be reported.')
        manifest['reference_harness_commit'] = manifest.pop('harness_commit', None)
        manifest['cost_client_sha256'] = runtime['scripts/qualify_graph_cost.py']
        save(job/'manifest.json', manifest)
        save(job/'status.json', dict(status='prepared', at=time.time()))
        manifests[variant] = sha(job/'manifest.json')
    snapshot = root/'frozen'
    snapshot.mkdir()
    for name in ('benchmark_graph_cost.py', 'benchmark_block_graph.py', 'qualify_graph_cost.py',
                 'audit_graph_cost_run.py', 'analyze_graph_cost.py'):
        shutil.copyfile(ROOT/'scripts'/name, snapshot/name)
    # The frozen coordinator imports the qualifier, which imports these modules.
    for source in (old/'runtime/scripts').glob('*.py'):
        if not (snapshot/source.name).exists(): shutil.copyfile(source, snapshot/source.name)
    save(root/'manifest.json', dict(schema=SCHEMA, created_at=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reference=str(old), reference_manifest_sha256=sha(old/'manifest.json'),
        jobs=manifests, frozen_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        common_pressure_policy='recency', allowed_profile_differences=[
            'kvmem.action_attention', 'kvmem.prefill_attention', 'kvmem.graph_mode'],
        start_cpu_available_bytes=105620439040, minimum_cpu_available_bytes=8589934592,
        minimum_gpu_free_mib=4096, repeats=1, order=list(VARIANTS),
        limits='Single cold screen, no repeated paired causal throughput claim if selection/output/MTP drift. '
            'Original full256K input/16K accepted output and strict native C8 gate retained.'))
    save(root/'status.json', dict(status='prepared', at=time.time()))


def cpu_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def run(args):
    root = args.output.resolve()
    manifest = read(root/'manifest.json')
    if read(root/'status.json')['status'] != 'prepared':
        raise ValueError('Use a new attempt; never rerun started or failed jobs')
    if len(os.sched_getaffinity(0)) < 2 or os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'):
        raise ValueError('GPU controller must retain inference CPU affinity and visible GPU')
    for name,digest in manifest['frozen_sha256'].items():
        if sha(root/'frozen'/name) != digest: raise ValueError('Frozen helper changed')
    # Load the exact frozen serving package only after coordinator startup.
    # A scheduled-query histogram can publish late; preserve its original
    # outcome and independently inspect every actual overlapping model step.
    sys.path.insert(0, str(root/'disabled/runtime/src'))
    from audit_graph_cost_run import audit
    environment = {k:v for k,v in os.environ.items() if k not in ('DEEPSEEK_API_KEY','DEEPSEEK_BASE_URL')}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    process = None
    try:
        for variant in manifest['order']:
            job = root/variant
            if sha(job/'manifest.json') != manifest['jobs'][variant]: raise ValueError('Job manifest changed')
            while owner.gpu_processes() or cpu_available() < manifest['start_cpu_available_bytes']:
                save(root/'status.json',dict(status='waiting_for_resources',variant=variant,at=time.time()))
                time.sleep(10)
            command = [sys.executable,'-B',str(job/'runtime/scripts/benchmark_block_graph.py'),'run',
                '--output',str(job),'--fixture',str(job/'fixture.json'),'--arm','multi']
            with owner.cleanup_on_termination() as cleanup:
                try:
                    with (root/(variant+'.log')).open('x') as log, (root/(variant+'-resources.jsonl')).open('x') as resources:
                        process = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        save(root/'status.json',dict(status='running',variant=variant,pid=process.pid,at=time.time()))
                        while process.poll() is None:
                            free,util,power = map(float,subprocess.check_output(['nvidia-smi',
                                '--query-gpu=memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(','))
                            memory = cpu_available()
                            resources.write(json.dumps(dict(at=time.time(),cpu_available_bytes=memory,
                                gpu_free_mib=free,gpu_utilization=util,gpu_power_w=power))+'\n');resources.flush()
                            if memory < manifest['minimum_cpu_available_bytes'] or free < manifest['minimum_gpu_free_mib']:
                                raise RuntimeError('Resource guard reached')
                            time.sleep(1)
                        client_returncode = process.returncode
                finally:
                    cleanup()
                    if owner.stop_owned_groups((process,)): raise RuntimeError('Owned process cleanup failed')
                    process = None
            result = audit(job)
            result['frozen_client_returncode'] = client_returncode
            save(root/(variant+'-executed-audit.json'), result)
            clean = read(job/'cleanup.json')
            if clean['remaining_owned_process_groups'] or clean['remaining_gpu_pids']:
                raise RuntimeError('Owned/GPU resources remain')
        save(root/'status.json',dict(status='runs_completed_independent_execution_audits_passed',at=time.time(),
            legacy_snapshot_gate_outcomes='Preserved individually, not relabeled'))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_attempt_preserved',error=str(error),at=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--port',type=int,default=18720)
    args = parser.parse_args()
    if args.action == 'prepare' and args.reference is None:
        parser.error('prepare requires --reference pointing to a passed C8 attempt')
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__': main()
