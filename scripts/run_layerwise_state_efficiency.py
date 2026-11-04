"""Frozen native history qualification after the current matched measurement.

No first-answer retry, no scheduled monitor, and no Memory inference. The
candidate runs only if the preceding matched experiment misses its efficiency
target; its full inputs must match the already qualified paired-read version.
"""
import argparse
from copy import deepcopy
import fcntl
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


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def component(root):
    audit=p.read(root/'component-audit.json')
    for name,digest in p.read(root/'source-proof.json').items():
        if p.sha(root/name)!=digest:raise ValueError('Native history component changed: '+name)
    tests=list(ET.parse(root/'tests.xml').getroot().iter('testcase'))
    if (audit['status']!='passed' or len(tests)!=28 or
            any(t.find(k) is not None for t in tests for k in ('failure','error','skipped')) or
            audit['tests_sha256']!=p.sha(root/'tests.xml') or
            audit['source_proof_sha256']!=p.sha(root/'source-proof.json')):
        raise ValueError('Native history CPU qualification is incomplete')


def replace(path,before,after):
    text=path.read_text()
    if text.count(before)!=1:raise ValueError('Frozen patch anchor changed: '+str(path))
    path.write_text(text.replace(before,after))


def prepare(root,source,components,previous):
    from run_layerwise_query_efficiency import source_audit
    from run_layerwise_matched_followup import identity
    source_audit(source);component(components)
    if root.exists():raise ValueError('Use a fresh qualification root')
    owner=identity(p.read(previous/'process.json')['pid'])
    if owner is None:raise ValueError('Expected the current matched measurement to be live')
    root.mkdir(parents=True);arm=root/'candidate';arm.mkdir();runtime=arm/'runtime-v1'
    shutil.copytree(source/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('inputs','upstream'):shutil.copytree(source/name,arm/name)
    for name in ('buffered_prefill_attention.py','layerwise_pressure.py'):
        shutil.copy2(components/'runtime/vllm_kvmem'/name,runtime/'src/vllm_kvmem'/name)
    binary,=list((components/'runtime/vllm_kvmem').glob('_prefill_state_native*.so'))
    shutil.copy2(binary,runtime/'src/vllm_kvmem'/binary.name)
    for name in ('buffered_prefill_state.cpp','prefill_attention_commit.hpp'):
        (runtime/'source-proof').mkdir(exist_ok=True)
        shutil.copy2(components/name,runtime/'source-proof'/name)
    serve=runtime/'src/vllm_kvmem/serve.py'
    anchor="    parser.add_argument('--layerwise-paired-reads', action='store_true')\n"
    replace(serve,anchor,anchor+"    parser.add_argument('--layerwise-buffered-history', action='store_true')\n")
    anchor='    plan = build_launch(profile, base, run_dir)\n'
    replace(serve,anchor,anchor+"    if args.layerwise_buffered_history:\n"
        "        if not args.layerwise_prefill_pressure:\n"
        "            raise ValueError('Buffered layer histories require independent prefill pressure')\n"
        "        plan['environment']['KVMEM_LAYERWISE_BUFFERED_HISTORY'] = '1'\n")
    driver=runtime/'scripts/run_concurrent_layerwise_ruler.py'
    anchor="    if fixture.get('layerwise_paired_reads'):\n        command += ['--layerwise-paired-reads']\n"
    replace(driver,anchor,anchor+"    if fixture.get('layerwise_buffered_history'):\n"
        "        if not fixture['layerwise_prefill_pressure']:\n"
        "            raise ValueError('Buffered histories require independent prefill pressure')\n"
        "        command += ['--layerwise-buffered-history']\n")
    fixture,profile=deepcopy(p.read(source/'fixture.json')),deepcopy(p.read(source/'profile.json'))
    fixture.update(layerwise_buffered_history=True,cohort='buffered-native-history-full-input8')
    profile.update(port=19107,served_model_name='layerwise-buffered-history')
    p.save(arm/'fixture.json',fixture);p.save(arm/'profile.json',profile)
    p.save(arm/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
        if f.is_file() and '__pycache__' not in f.parts})
    p.save(arm/'prepared.json',dict(at=time.time(),expected_runs=8,
        profile_sha256=p.sha(arm/'profile.json'),fixture_sha256=p.sha(arm/'fixture.json'),
        runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
    p.save(arm/'status.json',dict(status='prepared',completed=0,expected=8))
    for src,dst in [(Path(__file__),'launch.py'),(REPO/'scripts/run_layerwise_pressure_efficiency.py','comparison.py'),
                    (REPO/'scripts/run_layerwise_matched_followup.py','owner.py')]:
        shutil.copy2(src,root/dst)
    files={str(source/n):p.sha(source/n) for n in ('prepared.json','fixture.json','profile.json','runtime-proof.json','equivalence.json')}
    for row in fixture['rows']:
        case=source/'formal/sequential/cases'/row['id']
        files.update({str(case/n):p.sha(case/n) for n in ('request.json','response.json','evidence.json','result.json')})
    p.save(root/'plan.json',dict(source=str(source),components=str(components),previous=str(previous),
        source_owner=owner,previous_plan_sha256=p.sha(previous/'plan.json'),source_files=files,
        prepared_sha256=p.sha(arm/'prepared.json'),component_audit_sha256=p.sha(components/'component-audit.json'),
        files={n:p.sha(root/n) for n in ('launch.py','comparison.py','owner.py')},
        no_automatic_retry=True,scheduled_monitor=False,
        scope='Same full inputs, budgets, source arithmetic, candidates and generation. Only sixteen per-layer CPU score states change representation. Global attention, GPU math, archive reads and KV assembly stay qualified.'))
    verify(arm);p.save(root/'status.json',dict(status='prepared',at=time.time()))


def frozen(root):
    plan=p.read(root/'plan.json');component(Path(plan['components']));verify(root/'candidate')
    if (p.sha(root/'candidate/prepared.json')!=plan['prepared_sha256'] or
            p.sha(Path(plan['components'])/'component-audit.json')!=plan['component_audit_sha256'] or
            p.sha(Path(plan['previous'])/'plan.json')!=plan['previous_plan_sha256']):
        raise ValueError('Qualification dependency changed')
    for name,digest in plan['files'].items():
        if p.sha(root/name)!=digest:raise ValueError('Frozen qualification driver changed')
    for name,digest in plan['source_files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Accepted first-answer reference changed')
    return plan


def run(root):
    from run_recall_comparison import gpu_lock,termination_guard,stop_owned
    import benchmark_block_graph as lifecycle
    plan=frozen(root);previous=Path(plan['previous']);arm=root/'candidate';runtime=arm/'runtime-v1'
    owner=module(root/'owner.py','buffered_owner');pointer=previous.parent/'value-recall-active.json'
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'process.json').exists():raise ValueError('Preserve the first attempted qualification')
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        p.save(root/'status.json',dict(status='waiting_for_matched_performance',source_owner=plan['source_owner'],at=time.time()))
        deadline=time.monotonic()+7200
        while True:
            if p.read(pointer)['queue_root']!=str(previous):raise RuntimeError('Another experiment acquired GPU scheduling')
            if owner.source_ready(p.read(previous/'status.json')['status'],owner.identity(plan['source_owner']['pid']),plan['source_owner']):break
            if time.monotonic()>=deadline:raise TimeoutError('Measurement still active; leave it untouched')
            time.sleep(15)
        frozen(root)
        performance=p.read(previous/'performance-comparison.json')
        p.save(root/'preceding-performance.json',dict(source=str(previous),sha256=p.sha(previous/'performance-comparison.json'),goal_met=performance['goal_met']))
        if performance['goal_met']:
            p.save(root/'status.json',dict(status='not_run',reason='Prior qualified version reached its matched target; review for Memory before adding another model experiment.',at=time.time()))
            return
        with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
            if lifecycle.gpu_processes():raise RuntimeError('GPU has another owner')
            if owner.identity(p.read(previous/'child.json')['pid']) is not None:
                raise RuntimeError('Previous model driver has not exited')
            p.save(pointer,dict(queue_root=str(root),root=str(arm),status=str(root/'status.json'),
                process=str(root/'process.json'),experiment_type='native-history-full-input-qualification',at=time.time()))
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        env.update(PYTHONPATH=os.pathsep.join((str(runtime/'src'),str(runtime/'scripts'))),
            PYTHONDONTWRITEBYTECODE='1',CUDA_HOME='/usr/local/cuda',
            PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'],
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        p.save(root/'status.json',dict(status='running',stage='full_input8',active_root=str(arm),at=time.time()))
        command=[sys.executable,'-B',str(runtime/'scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
        with (arm/'driver.log').open('x') as log:
            child=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        p.save(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
        try:code=child.wait()
        except BaseException:
            stop_owned(child);raise
        if code:raise RuntimeError('Full-input history qualification failed; preserve all first answers')
        frozen(root);equality=module(root/'comparison.py','buffered_comparison').compare(arm,Path(plan['source']))
        native=p.read(arm/'formal/sequential/server-attempts/0001/server/capture/native_attention_stats.json')
        phases=native['layer_phase_counts']
        good=(equality['passed'] and len(native['producer_layers'])==16 and not native['disabled_reason'] and
            not native['validation_failures'] and phases.get('reuse:prefill',0)>0 and
            not any(v for k,v in phases.items() if k.startswith('fallback:prefill')))
        p.save(arm/'native-reuse-audit.json',dict(passed=bool(good),statistics=native))
        backend=[]
        for row in p.read(arm/'fixture.json')['rows']:
            evidence=p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')
            pressure=[e for e in evidence['events'] if e['event']=='pressure_selected']
            passed=bool(pressure) and all(len(e['layers'])==16 and all(
                l.get('attention_state_backend')=='native_buffered' and l.get('buffered_commits',0)>0
                for l in e['layers']) for e in pressure)
            backend.append(dict(id=row['id'],pressure_events=len(pressure),passed=passed))
        good=bool(good and len(backend)==8 and all(r['passed'] for r in backend))
        p.save(root/'acceptance.json',dict(passed=good,cases=8,backend=backend,efficiency_accepted=False,
            scope='All pressure/Query scopes, blocks and full output IDs/text must match; matched eight-owner timing is separate.'))
        if not good:raise RuntimeError('Native history full-input/backend equality failed')
        p.save(root/'status.json',dict(status='completed',cases=8,correctness_passed=True,efficiency_accepted=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    for name in ('source','components','previous'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source.resolve(),args.components.resolve(),args.previous.resolve())
        else:run(root)
    except BaseException as error:
        if root.exists() and args.action=='run':p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
