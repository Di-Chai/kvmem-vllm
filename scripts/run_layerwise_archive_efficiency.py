"""Frozen paired archive-read qualification; no first-answer retries."""
import argparse
from copy import deepcopy
import importlib.util
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
MODULES=('storage.py','layerwise_query.py','layerwise_pressure_efficiency.py')
CPU_TESTS=('test_paired_archive_reads.py','test_archive_submission_gate.py',
           'test_native_memory_storage.py','test_storage_regions.py','test_memory_gather.py',
           'test_packed_assembly.py')
GPU_TEST='test_layerwise_paired_reads_gpu.py'


def replace_once(path,before,after):
    text=path.read_text()
    if text.count(before)!=1:raise ValueError('Patch anchor changed: '+str(path))
    path.write_text(text.replace(before,after))


def prepare(root,source,previous,cpu_evidence):
    from run_layerwise_query_efficiency import source_audit
    checks=source_audit(source)
    if root.exists():raise ValueError('Use a fresh frozen qualification directory')
    root.mkdir(parents=True);arm=root/'candidate';arm.mkdir();runtime=arm/'runtime-v1'
    shutil.copytree(source/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for folder in ('inputs','upstream'):shutil.copytree(source/folder,arm/folder)
    for module in MODULES:shutil.copy2(REPO/'src/vllm_kvmem'/module,runtime/'src/vllm_kvmem'/module)
    # Keep every accepted GPU assembly kernel byte-for-byte. Local work also
    # contains a separate original-RoPE experiment, outside this optimization.
    marker='@triton.jit\ndef _assemble_packed'
    target=runtime/'src/vllm_kvmem/packed_assembly.py'
    old=target.read_text();new=(REPO/'src/vllm_kvmem/packed_assembly.py').read_text()
    if old.count(marker)!=1 or new.count(marker)!=1:raise ValueError('Packed kernel boundary changed')
    target.write_text(new.split(marker)[0]+marker+old.split(marker)[1])
    serve=runtime/'src/vllm_kvmem/serve.py'
    anchor="    parser.add_argument('--layerwise-batched-reads', action='store_true')\n"
    replace_once(serve,anchor,anchor+"    parser.add_argument('--layerwise-paired-reads', action='store_true')\n")
    anchor="    plan = build_launch(profile, base, run_dir)\n"
    replace_once(serve,anchor,anchor+"    if args.layerwise_paired_reads:\n"
        "        if not args.layerwise_query_optimized:\n"
        "            raise ValueError('Paired layer reads require optimized Query execution')\n"
        "        plan['environment']['KVMEM_LAYERWISE_PAIRED_READS'] = '1'\n")
    shutil.copy2(REPO/'scripts/run_concurrent_layerwise_ruler.py',runtime/'scripts/run_concurrent_layerwise_ruler.py')
    (runtime/'tests').mkdir(exist_ok=True)
    for name in (*CPU_TESTS,GPU_TEST,'test_authority.py'):
        shutil.copy2(REPO/'tests'/name,runtime/'tests'/name)
    fixture,profile=deepcopy(p.read(source/'fixture.json')),deepcopy(p.read(source/'profile.json'))
    fixture.update(layerwise_paired_reads=True,cohort='paired-archive-read-full-input8')
    profile.update(port=19106,served_model_name='layerwise-paired-reads')
    p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
    p.save(arm/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
        if f.is_file() and '__pycache__' not in f.parts})
    p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,
        profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
        runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
    p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
    shutil.copy2(__file__,root/'launch.py')
    shutil.copy2(REPO/'scripts/run_layerwise_pressure_efficiency.py',root/'comparison.py')
    files={str(source/n):p.sha(source/n) for n in
        ('prepared.json','fixture.json','profile.json','runtime-proof.json','equivalence.json')}
    for row in fixture['rows']:
        case=source/'formal/sequential/cases'/row['id']
        files.update({str(case/n):p.sha(case/n) for n in ('request.json','response.json','evidence.json','result.json')})
    p.save(root/'plan.json',dict(source=str(source),previous=str(previous),arm=str(arm),
        source_files=files,source_checks=checks,expected_cases=8,
        prepared_sha256=p.sha(arm/'prepared.json'),launch_sha256=p.sha(root/'launch.py'),
        comparator_sha256=p.sha(root/'comparison.py'),cpu_component=str(cpu_evidence),
        cpu_component_sha256=p.sha(cpu_evidence),scheduled_monitor=False,no_automatic_retry=True,
        component_expected={'cpu':83,'gpu':6},
        acceptance='Actual paired CPU reads, native reuse, complete pressure/Query scope and selected IDs, all output IDs/text exactly equal. Performance remains a separate eight-owner gate.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))
    verify(arm)


def frozen(root):
    plan=p.read(root/'plan.json');arm=Path(plan['arm'])
    for file,key in [('launch.py','launch_sha256'),('comparison.py','comparator_sha256')]:
        if p.sha(root/file)!=plan[key]:raise ValueError('Frozen qualification driver changed')
    if p.sha(arm/'prepared.json')!=plan['prepared_sha256']:raise ValueError('Prepared runtime changed')
    for name,digest in plan['source_files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Accepted reference changed: '+name)
    if p.sha(Path(plan['cpu_component']))!=plan['cpu_component_sha256']:
        raise ValueError('CPU component evidence changed')
    verify(arm)
    return plan


def counts(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    return dict(tests=len(cases),failures=sum(c.find('failure') is not None for c in cases),
        errors=sum(c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases))


def run(root):
    from run_recall_comparison import gpu_lock,termination_guard,stop_owned
    from run_layerwise_pressure_efficiency import alive
    import benchmark_block_graph as lifecycle
    plan=frozen(root);arm=Path(plan['arm']);runtime=arm/'runtime-v1';previous=Path(plan['previous'])
    pointer=previous.parent/'value-recall-active.json'
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
    env.update(PYTHONPATH=os.pathsep.join((str(runtime/'src'),str(runtime/'scripts'))),
        PYTHONDONTWRITEBYTECODE='1',CUDA_HOME='/usr/local/cuda',
        PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'],
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if (root/'process.json').exists():raise ValueError('Preserve first attempts; no automatic retry')
    with termination_guard():
        with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
            if (p.read(pointer)['queue_root']!=str(previous) or
                    p.read(previous/'status.json')['status']!='completed' or
                    alive(p.read(previous/'process.json')['pid']) or lifecycle.gpu_processes()):
                raise RuntimeError('Previous GPU owner has not released the device')
            p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
            p.save(pointer,dict(queue_root=str(root),root=str(arm),status=str(root/'status.json'),
                process=str(root/'process.json'),experiment_type='paired-archive-read-qualification',at=time.time()))
            for kind in ('cpu','gpu'):
                p.save(root/'status.json',dict(status='running',stage=kind+'_components',at=time.time()))
                tests=['tests/'+f for f in CPU_TESTS] if kind=='cpu' else ['tests/'+GPU_TEST]
                command=[sys.executable,'-B','-m','pytest','-q',*tests,'--junitxml='+str(root/(kind+'-tests.xml'))]
                if kind=='cpu':command+=['-k','not gpu_assembly']
                test_env=dict(env)
                if kind=='gpu':test_env['KVMEM_RUN_GPU_TESTS']='1'
                with (root/(kind+'-tests.log')).open('x') as log:
                    code=subprocess.run(command,cwd=runtime,env=test_env,stdout=log,stderr=subprocess.STDOUT).returncode
                observed=counts(root/(kind+'-tests.xml')) if (root/(kind+'-tests.xml')).exists() else {}
                passed=code==0 and observed and all(observed[k]==0 for k in ('failures','errors','skipped'))
                passed=bool(passed and observed['tests']==plan['component_expected'][kind])
                p.save(root/(kind+'-audit.json'),dict(passed=passed,returncode=code,counts=observed,
                    test_sha256=p.sha(root/(kind+'-tests.xml')) if observed else None,
                    runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
                if not passed:raise RuntimeError(kind+' components failed; preserve evidence')
            frozen(root)
        p.save(root/'status.json',dict(status='running',stage='full_input8',active_root=str(arm),at=time.time()))
        command=[sys.executable,'-B',str(runtime/'scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
        with (arm/'driver.log').open('x') as log:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        p.save(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
        try:code=child.wait()
        except BaseException:
            stop_owned(child);raise
        if code:raise RuntimeError('Full-input qualification failed; keep every first answer')
        frozen(root)
        spec=importlib.util.spec_from_file_location('paired_read_comparison',root/'comparison.py')
        comparator=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparator)
        equality=comparator.compare(arm,Path(plan['source']))
        stats=p.read(arm/'formal/sequential/server-attempts/0001/server/capture/native_attention_stats.json')
        phase=stats['layer_phase_counts']
        native=(len(stats['producer_layers'])==16 and not stats['disabled_reason'] and
            not stats['validation_failures'] and phase.get('reuse:prefill',0)>0 and
            not any(v for k,v in phase.items() if k.startswith('fallback:prefill')))
        p.save(arm/'native-reuse-audit.json',dict(passed=bool(native),statistics=stats))
        exercised=[]
        for row in p.read(arm/'fixture.json')['rows']:
            evidence=p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')
            paired_query=[l['assembly_backend'].get('paired_read_chunks',0) for l in evidence['layers']]
            paired_pressure=[e['assembly_read_stats'].get('layerwise_paired_read_chunks',0)
                             for e in evidence['refresh'] if e['selection']=='pressure']
            exercised.append(dict(id=row['id'],query=paired_query,pressure=paired_pressure,
                passed=len(paired_query)==16 and all(x>0 for x in paired_query) and
                       bool(paired_pressure) and all(x>0 for x in paired_pressure)))
        good=bool(equality['passed'] and native and len(exercised)==8 and all(x['passed'] for x in exercised))
        p.save(root/'acceptance.json',dict(passed=good,cases=8,paired_reads=exercised,
            performance=p.read(arm/'concurrency-performance.json'),performance_target_met=False,
            scope='Correctness qualification only. Matched eight-owner throughput is still required.'))
        if not good:raise RuntimeError('Actual backend or complete-input equality failed')
        p.save(root/'status.json',dict(status='completed',cases=8,correctness_passed=True,
                                     efficiency_accepted=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    for name in ('root','source','previous','cpu-evidence'):
        parser.add_argument('--'+name,type=Path,required=name=='root')
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source.resolve(),args.previous.resolve(),args.cpu_evidence.resolve())
        else:run(root)
    except BaseException as error:
        if root.exists():p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
