"""Qualify existing serial allocation and parallel CPU record copies on unchanged full inputs."""
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


def component(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    if len(cases)!=120 or any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):
        raise ValueError('Existing private-copy component tests did not all pass')


def archive_backend(profile,stats,fixture):
    from run_layerwise_matched_performance import archive_backend_audit
    if not profile['kvmem'].get('cpu_seal_serial_allocate'):
        raise ValueError('Private CPU packing is disabled')
    return archive_backend_audit(profile,stats,fixture.get('cpu_region_gather_min_bytes'))


def prepare(root,source,previous,tests):
    from run_layerwise_query_efficiency import source_audit
    source_audit(source);component(tests)
    if root.exists():raise ValueError('Use a new full-input qualification root')
    fixture=deepcopy(p.read(source/'fixture.json'));profile=deepcopy(p.read(source/'profile.json'))
    if (not profile['kvmem'].get('cpu_direct_record_seal',False) or
            profile['kvmem'].get('cpu_seal_serial_allocate',False) or
            profile['kvmem'].get('cpu_seal_pack_workers',1)!=1):
        raise ValueError('Expected qualified direct CPU records with original serial packing')
    root.mkdir(parents=True);arm=root/'candidate';arm.mkdir()
    for name in ('runtime-v1','inputs','upstream'):
        shutil.copytree(source/name,arm/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    fixture['cohort']='existing-private-pack-full-input8'
    profile.update(port=19111,served_model_name='layerwise-private-pack')
    profile['kvmem']['cpu_seal_serial_allocate']=True
    p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
    shutil.copy2(source/'runtime-proof.json',arm/'runtime-proof.json')
    p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,
        profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
        runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
    p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
    shutil.copy2(__file__,root/'launch.py');shutil.copy2(tests,root/'cpu-tests.xml')
    files={str(source/n):p.sha(source/n) for n in ('prepared.json','fixture.json','profile.json','runtime-proof.json','equivalence.json')}
    for row in fixture['rows']:
        case=source/'formal/sequential/cases'/row['id']
        files.update({str(case/n):p.sha(case/n) for n in ('request.json','response.json','evidence.json','result.json')})
    repo=Path(__file__).resolve().parents[1]
    helpers={str(path):p.sha(path) for path in (repo/'scripts').glob('*.py')}
    component_inputs={str(repo/'tests'/n):p.sha(repo/'tests'/n) for n in
        ('test_cpu_seal_hash.py','test_cpu_direct_record_seal_options.py','test_cpu_direct_record_seal.py','test_cpu_batched_record_seal.py',
         'test_cpu_private_record_pack.py','test_cpu_private_record_pack_options.py')}
    component_inputs[str(source/'runtime-proof.json')]=p.sha(source/'runtime-proof.json')
    component_inputs[str(Path(sys.executable))]=p.sha(Path(sys.executable))
    p.save(root/'plan.json',dict(kind='cpu_private_pack',source=str(source),previous=str(previous),
        source_files=files,helpers=helpers,prepared_sha256=p.sha(arm/'prepared.json'),
        launch_sha256=p.sha(root/'launch.py'),tests_sha256=p.sha(root/'cpu-tests.xml'),
        component_inputs=component_inputs,
        scheduled_monitor=False,no_automatic_retry=True,
        scope='Only cpu_seal_serial_allocate changes from false to true. Engine bytes, gather/hash pool capacity, '
              'all selection rules, model, inputs, generation and KV budgets stay unchanged.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()));frozen(root)


def frozen(root):
    plan=p.read(root/'plan.json');component(root/'cpu-tests.xml');verify(root/'candidate')
    if (p.sha(root/'launch.py')!=plan['launch_sha256'] or p.sha(root/'cpu-tests.xml')!=plan['tests_sha256'] or
            p.sha(root/'candidate/prepared.json')!=plan['prepared_sha256']):
        raise ValueError('Frozen private-copy qualification changed')
    for name,digest in {**plan['source_files'],**plan['helpers'],**plan['component_inputs']}.items():
        if p.sha(Path(name))!=digest:raise ValueError('Qualified source changed: '+name)
    source=Path(plan['source']);arm=root/'candidate'
    if p.read(source/'runtime-proof.json')!=p.read(arm/'runtime-proof.json'):
        raise ValueError('Existing private-copy engine bytes changed')
    expected=deepcopy(p.read(source/'profile.json'));actual=p.read(arm/'profile.json')
    expected.update(port=actual['port'],served_model_name=actual['served_model_name'])
    expected['kvmem']['cpu_seal_serial_allocate']=True
    if expected!=actual:raise ValueError('Controls changed beyond private CPU packing')
    expected_fixture=p.read(source/'fixture.json');expected_fixture['cohort']='existing-private-pack-full-input8'
    if expected_fixture!=p.read(arm/'fixture.json'):
        raise ValueError('Full-input fixture changed beyond its descriptive cohort name')
    return plan


def qualified(root):
    plan=frozen(root);arm=root/'candidate';acceptance=p.read(root/'acceptance.json')
    if not acceptance['passed'] or acceptance['cases']!=8:
        raise ValueError('Private CPU packing requires all eight exact full-input cases')
    from run_layerwise_query_efficiency import source_audit
    source_audit(arm)
    archive_backend(p.read(arm/'profile.json'),p.read(arm/'formal/sequential/server-attempts/0001/server/capture/archive.json'),p.read(arm/'fixture.json'))
    return arm


def run(root):
    from run_recall_comparison import gpu_lock,termination_guard,stop_owned
    from run_layerwise_matched_followup import identity
    from run_layerwise_pressure_efficiency import compare
    from run_layerwise_matched_performance import paired_backend_audit,buffered_backend_audit
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
            if (root/'process.json').exists():raise ValueError('Preserve the first private-copy qualification attempt')
            p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
            p.save(pointer,dict(queue_root=str(root),root=str(arm),status=str(root/'status.json'),
                process=str(root/'process.json'),experiment_type='private-archive-copy-qualification',at=time.time()))
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime/'src')+os.pathsep+str(runtime/'scripts'),
            CUDA_HOME='/usr/local/cuda',PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        p.save(root/'status.json',dict(status='running',stage='full_input8',active_root=str(arm),at=time.time()))
        command=[sys.executable,'-B',str(runtime/'scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
        with (arm/'driver.log').open('x') as log:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
        backend=archive_backend(p.read(arm/'profile.json'),p.read(capture/'archive.json'),fixture);good=bool(good and equality['passed'])
        p.save(root/'acceptance.json',dict(passed=good,cases=8,archive_backend=backend,efficiency_accepted=False))
        if not good:raise RuntimeError('Private CPU packing changed exact full-input results')
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
        process=root/'process.json'
        if args.action=='run' and process.exists() and p.read(process).get('pid')==os.getpid():
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
