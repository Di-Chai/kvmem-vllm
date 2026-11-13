"""Qualify native inline small-region gathering on the original full inputs."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify

REPO=Path(__file__).resolve().parents[1]
TESTS=('test_memory_gather.py','test_paired_archive_reads.py','test_storage_regions.py',
       'test_native_memory_storage.py','test_cpu_seal_hash.py')


def component(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    if len(cases)!=89 or any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):
        raise ValueError('All 89 native/paired/gather/lease/hash checks must pass')


def prepare(root,source,previous,tests):
    from run_layerwise_query_efficiency import source_audit
    from vllm_kvmem import _native; from vllm_kvmem.core import memory_storage
    source_audit(source);component(tests)
    if root.exists():raise ValueError('Use a fresh qualification root')
    if memory_storage.CPU_REGION_GATHER_PARALLEL_MIN_BYTES!=32*1024**2:
        raise ValueError('Unexpected tested region gather threshold')
    native=Path(_native.__file__).resolve();relative=Path('src/vllm_kvmem')/native.name
    if p.sha(native)!=p.sha(source/'runtime-v1'/relative):
        raise ValueError('CPU component tests used a different native copy binary')
    root.mkdir(parents=True);arm=root/'candidate';arm.mkdir()
    for name in ('runtime-v1','inputs','upstream'):
        shutil.copytree(source/name,arm/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    changed='src/vllm_kvmem/memory_storage.py'
    shutil.copy2(REPO/changed,arm/'runtime-v1'/changed)
    fixture=deepcopy(p.read(source/'fixture.json'));profile=deepcopy(p.read(source/'profile.json'))
    if fixture.get('cpu_region_gather_min_bytes') is not None:
        raise ValueError('Expected the qualified original gather dispatch baseline')
    fixture.update(cohort='native-inline-region-full-input8',cpu_region_gather_min_bytes=32*1024**2)
    profile.update(port=19109,served_model_name='layerwise-region-inline')
    p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
    proof=p.read(source/'runtime-proof.json');proof[changed]=p.sha(arm/'runtime-v1'/changed)
    p.save(arm/'runtime-proof.json',proof)
    p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,profile_sha256=p.sha(arm/'profile.json'),
        fixture_sha256=p.sha(arm/'fixture.json'),runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
    p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
    shutil.copy2(__file__,root/'launch.py');shutil.copy2(tests,root/'cpu-tests.xml')
    files={str(source/n):p.sha(source/n) for n in ('prepared.json','fixture.json','profile.json','runtime-proof.json','equivalence.json')}
    for row in fixture['rows']:
        case=source/'formal/sequential/cases'/row['id']
        files.update({str(case/n):p.sha(case/n) for n in ('request.json','response.json','evidence.json','result.json')})
    helpers={str(path):p.sha(path) for path in (REPO/'scripts').glob('*.py')}
    component_inputs={str(REPO/'tests'/name):p.sha(REPO/'tests'/name) for name in TESTS}
    component_inputs.update({str(REPO/changed):p.sha(REPO/changed),str(native):p.sha(native),
                             str(Path(sys.executable)):p.sha(Path(sys.executable))})
    p.save(root/'plan.json',dict(kind='cpu_region_gather',source=str(source),previous=str(previous),source_files=files,
        helpers=helpers,component_inputs=component_inputs,prepared_sha256=p.sha(arm/'prepared.json'),
        launch_sha256=p.sha(root/'launch.py'),tests_sha256=p.sha(root/'cpu-tests.xml'),changed_engine_files=[changed],
        scheduled_monitor=False,no_automatic_retry=True,
        scope='Only CPU region copy scheduling/counters change. Regions below 32 MiB use the existing native '
              'inline copy; full records keep their 8 MiB parallel threshold. Inputs, GPU arithmetic, selection, '
              'Active/Gen budgets, CPU staging capacity, hash workers and generation stay unchanged.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()));frozen(root)


def frozen(root):
    plan=p.read(root/'plan.json');component(root/'cpu-tests.xml');verify(root/'candidate')
    if (p.sha(root/'launch.py')!=plan['launch_sha256'] or p.sha(root/'cpu-tests.xml')!=plan['tests_sha256'] or
            p.sha(root/'candidate/prepared.json')!=plan['prepared_sha256']):
        raise ValueError('Frozen region qualification changed')
    for name,digest in {**plan['source_files'],**plan['helpers'],**plan['component_inputs']}.items():
        if p.sha(Path(name))!=digest:raise ValueError('Frozen region qualification dependency changed: '+name)
    old=p.read(Path(plan['source'])/'runtime-proof.json');new=p.read(root/'candidate/runtime-proof.json')
    if set(old)!=set(new) or [name for name in old if old[name]!=new[name]]!=plan['changed_engine_files']:
        raise ValueError('Unexpected engine change beyond the CPU copy dispatcher')
    return plan


def qualified(root):
    from run_layerwise_query_efficiency import source_audit
    from run_layerwise_matched_performance import archive_backend_audit
    frozen(root);arm=root/'candidate';acceptance=p.read(root/'acceptance.json')
    if not acceptance['passed'] or acceptance['cases']!=8:
        raise ValueError('All eight original full-input cases must pass')
    source_audit(arm)
    archive_backend_audit(p.read(arm/'profile.json'),
        p.read(arm/'formal/sequential/server-attempts/0001/server/capture/archive.json'),32*1024**2)
    return arm


def run(root):
    from run_recall_comparison import gpu_lock,termination_guard,stop_owned
    from run_layerwise_matched_followup import identity
    from run_layerwise_pressure_efficiency import compare
    from run_layerwise_matched_performance import paired_backend_audit,buffered_backend_audit,archive_backend_audit
    import benchmark_block_graph as lifecycle
    plan=frozen(root);previous=Path(plan['previous']);arm=root/'candidate';runtime=arm/'runtime-v1'
    pointer=previous.parent/'value-recall-active.json'
    with termination_guard():
        with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
            if (p.read(pointer)['queue_root']!=str(previous) or p.read(previous/'status.json')['status']!='completed' or
                    identity(p.read(previous/'process.json')['pid']) is not None or lifecycle.gpu_processes()):
                raise RuntimeError('Previous measurement has not released GPU ownership')
            if p.read(previous/'performance-comparison.json')['goal_met']:
                raise RuntimeError('Efficiency already accepted; continue original Memory protocol')
            if (root/'process.json').exists():raise ValueError('Preserve the first qualification attempt')
            p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
            p.save(pointer,dict(queue_root=str(root),root=str(arm),status=str(root/'status.json'),
                process=str(root/'process.json'),experiment_type='native-inline-region-qualification',at=time.time()))
        environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src')+os.pathsep+str(runtime/'scripts'),
            CUDA_HOME='/usr/local/cuda',PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        p.save(root/'status.json',dict(status='running',stage='full_input8',active_root=str(arm),at=time.time()))
        command=[sys.executable,'-B',str(runtime/'scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
        with (arm/'driver.log').open('x') as log:
            child=subprocess.Popen(command,cwd=runtime,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        p.save(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
        try:code=child.wait()
        except BaseException:
            stop_owned(child);raise
        if code:raise RuntimeError('Preserve all first answers after technical failure')
        frozen(root);equality=compare(arm,Path(plan['source']))
        capture=arm/'formal/sequential/server-attempts/0001/server/capture'
        native=p.read(capture/'native_attention_stats.json');counts=native['layer_phase_counts']
        good=(len(native['producer_layers'])==16 and not native['disabled_reason'] and not native['validation_failures'] and
            counts.get('reuse:prefill',0)>0 and not any(v for k,v in counts.items() if k.startswith('fallback:prefill')))
        p.save(arm/'native-reuse-audit.json',dict(passed=bool(good),statistics=native))
        fixture=p.read(arm/'fixture.json')
        for row in fixture['rows']:
            evidence=p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')
            paired_backend_audit(fixture,evidence);buffered_backend_audit(fixture,evidence)
        archive=archive_backend_audit(p.read(arm/'profile.json'),p.read(capture/'archive.json'),32*1024**2)
        good=bool(good and equality['passed'])
        p.save(root/'acceptance.json',dict(passed=good,cases=8,archive_backend=archive,efficiency_accepted=False))
        if not good:raise RuntimeError('CPU region gathering changed exact full-input results')
        p.save(root/'status.json',dict(status='completed',cases=8,correctness_passed=True,efficiency_accepted=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    for name in ('source','previous','tests'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source.resolve(),args.previous.resolve(),args.tests.resolve())
        else:run(root)
    except BaseException as error:
        if args.action=='run' and root.exists():p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
