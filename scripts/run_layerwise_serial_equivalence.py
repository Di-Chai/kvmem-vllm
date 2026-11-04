"""Control batch composition before accepting execution-only recall changes.

The same eight complete inputs run in fixed order with one active owner. A
second unchanged baseline must reproduce every pressure/Query choice and full
output token before candidates proceed. Eight-owner throughput is a separate
measurement; none of these answers replace formal benchmark answers.
"""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify

REPO=Path(__file__).resolve().parents[1]
STAGES=(('baseline','baseline'),('repeat_baseline','baseline'),('native_scores','native_scores'),
        ('single_assembly','single_assembly'),('fast_metadata','fast_metadata'),('compiled','compiled'))


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')


def prepare(root, source):
    if root.exists():raise ValueError('Preserve frozen attempts; use a fresh version')
    root.mkdir(parents=True)
    plans=[]
    for index,(name,origin) in enumerate(STAGES):
        previous=source/origin;verify(previous)
        arm=root/name;runtime=arm/'runtime-v1';arm.mkdir()
        shutil.copytree(previous/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(previous/'inputs',arm/'inputs')
        shutil.copytree(previous/'upstream',arm/'upstream')
        shutil.copy2(REPO/'scripts/run_concurrent_layerwise_ruler.py',runtime/'scripts/run_concurrent_layerwise_ruler.py')
        shutil.copy2(REPO/'src/vllm_kvmem/runner.py',runtime/'src/vllm_kvmem/runner.py')
        # Keep the exact pressure/Query implementation of each source stage.
        # Only the verified first-public-forward bootstrap is added here.
        fixture=deepcopy(p.read(previous/'fixture.json'))
        fixture.update(layerwise_workers=1,cohort='serial-execution-equivalence-full-input8',
            diagnostic_scope='Fixed owner order; repeated unchanged baseline; every pressure/Query choice and full output token.')
        profile=deepcopy(p.read(previous/'profile.json'))
        profile.update(max_num_seqs=1,port=19070+index,served_model_name='serial-equivalence-'+name)
        p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
        proof={str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
               if f.is_file() and '__pycache__' not in f.parts}
        p.save(arm/'runtime-proof.json',proof)
        p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=len(fixture['rows']),
            profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
            runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
        verify(arm)
        p.save(arm/'status.json',dict(status='prepared',expected=len(fixture['rows']),completed=0))
        plans.append(dict(name=name,root=str(arm),source=str(previous),
                          prepared_sha256=p.sha(arm/'prepared.json')))
    shutil.copy2(__file__,root/'launch.py')
    shutil.copy2(REPO/'scripts/run_layerwise_pressure_efficiency.py',root/'comparison-code.py')
    p.save(root/'plan.json',dict(source_queue=str(source),stages=plans,
        launch_sha256=p.sha(root/'launch.py'),comparison_sha256=p.sha(root/'comparison-code.py'),
        source_plan_sha256=p.sha(source/'plan.json'),fixed_sample_order=p.read(root/'baseline/fixture.json')['qualification_ids'],
        controls='Original full inputs, budgets, generation and 512-token logical chunks; one owner in every correctness arm.',
        acceptance='Repeated baseline and every candidate must have exact pressure/Query blocks and full output tokens/text.',
        no_formal_answer_replacement=True,no_automatic_retry=True,scheduled_monitor=False))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))


def run(root):
    import importlib.util
    import pynvml
    plan=p.read(root/'plan.json');source=Path(plan['source_queue'])
    if p.sha(root/'launch.py')!=plan['launch_sha256'] or p.sha(root/'comparison-code.py')!=plan['comparison_sha256']:
        raise ValueError('Frozen serial comparison changed')
    if p.sha(source/'plan.json')!=plan['source_plan_sha256']:
        raise ValueError('Source queue controls changed')
    spec=importlib.util.spec_from_file_location('frozen_serial_comparator',root/'comparison-code.py')
    comparator=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparator)
    active=root.parent/'value-recall-active.json'
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        while True:
            pointer=p.read(active)
            if pointer.get('queue_root') not in (str(source),str(root)):
                raise RuntimeError('Another campaign owns GPU access')
            state=p.read(source/'status.json')
            if state['status'] in ('completed','failed'):
                # A completed measurement with a failed equivalence gate is
                # evidence, not an unanswered technical attempt to retry.
                for name in ('baseline','native_scores'):
                    if p.read(source/name/'status.json')['status']!='completed':
                        raise RuntimeError('Source attempt has unfinished technical work; preserve it')
                process=p.read(source/'process.json')
                child=p.read(source/'child.json')
                pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
                if not pynvml.nvmlDeviceGetComputeRunningProcesses(handle) and not any(
                        alive(pid) for pid in (process['pid'],child['pid'])):break
            p.save(root/'status.json',dict(status='waiting_for_eight_owner_comparison',at=time.time()))
            time.sleep(30)
        p.save(active,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
            process=str(root/'process.json'),experiment_type='layerwise-serial-execution-equivalence',at=time.time()))
        reports=[]
        for stage in plan['stages']:
            arm=Path(stage['root']);verify(arm)
            if p.sha(arm/'prepared.json')!=stage['prepared_sha256']:raise ValueError('Stage controls changed')
            p.save(root/'status.json',dict(status='running',stage=stage['name'],active_root=str(arm),at=time.time()))
            command=[sys.executable,'-B',str(arm/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
            with (arm/'driver.log').open('x') as log:
                child=subprocess.Popen(command,cwd=arm/'runtime-v1',stdout=log,stderr=subprocess.STDOUT)
            p.save(root/'child.json',dict(pid=child.pid,stage=stage['name'],command=command,at=time.time()))
            if child.wait()!=0:raise RuntimeError('Technical failure; preserve all first answers: '+stage['name'])
            if stage['name']=='baseline':continue
            report=comparator.compare(arm,root/'baseline');reports.append(report)
            if stage['name']!='repeat_baseline':
                stats=p.read(arm/'formal/sequential/server-attempts/0001/server/capture/native_attention_stats.json')
                counts=stats['layer_phase_counts']
                good=(len(stats['producer_layers'])==16 and not stats['disabled_reason'] and
                    not stats['validation_failures'] and counts.get('reuse:prefill',0)>0 and
                    not any(v for k,v in counts.items() if k.startswith('fallback:prefill')))
                p.save(arm/'native-reuse-audit.json',dict(passed=good,statistics=stats))
            else:good=True
            p.save(root/'comparison.json',dict(passed=all(r['passed'] for r in reports) and good,reports=reports))
            if not report['passed'] or not good:
                raise RuntimeError('Exact full-input equivalence or native reuse failed: '+stage['name'])
        p.save(root/'status.json',dict(status='completed',stages=len(plan['stages']),at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source.resolve())
        else:run(root)
    except BaseException as error:
        if root.exists() and args.action=='run':p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
