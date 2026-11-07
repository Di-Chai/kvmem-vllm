"""Fixed-cohort full-input correctness diagnostic, not a throughput candidate."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_layerwise_gilopt import compare, read, save_new, sha, tree

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
SOURCE = REPO/'artifacts/prefill-batch-overlap-20261105-r4/c1/candidate'
CANDIDATE = REPO/'artifacts/multiowner-6000-goal-20261105-r1/candidate-v2'
FORMAL = REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal'


def prepare(root, owners, repeats):
    if root.exists(): raise ValueError('Preserve previous evidence')
    assert tree(SOURCE/'runtime-v1') == read(SOURCE/'runtime-proof.json')
    root.mkdir(parents=True)
    assert owners == 2 and repeats == 2
    assert tree(CANDIDATE/'runtime-v1') == read(CANDIDATE/'runtime-proof.json')
    assert '6 passed' in (CANDIDATE/'gpu-tests.log').read_text()
    assert '9 passed' in (CANDIDATE/'cohort-tests.log').read_text()
    fixture = read(FORMAL/'fixture.json')
    rows = [next(r for r in fixture['rows'] if r['id'].startswith(group+'_')
                 and r['id'].endswith('_q0002')) for group in ('005','006','007','008','009','010')[:owners]]
    ids = [r['id'] for r in rows]
    fixture.update(rows=rows,case_ids=ids,cases=owners,qualification=False,
                   restore_digest_qualification=False,diagnostic_only=True,
                   performance_observation=True)
    fixture['bindings'] = [b for b in fixture['bindings'] if b['id'] in ids]
    fixture['execution_kvmem_overrides']=read(SOURCE/'fixture.json')['execution_kvmem_overrides']
    profile = read(SOURCE/'profile.json')
    kv = profile['kvmem']
    kv['cpu_read_cache_bytes'] = 112000000000
    kv['prefill_physical_multi_owner'] = True
    assert [kv[k] for k in ['select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks']] == [36864,16384,32,41,0]
    assert profile['max_num_seqs'] == 8 and profile['max_num_batched_tokens'] == 16424
    reservations = [((r['input_tokens']+2*kv['gen_budget']+31)//32)*32*17*4*256*2 for r in rows]
    assert sum(reservations) < kv['cpu_read_cache_bytes'] - 16*1024**3, 'All phase owners must fit without waiting for a gated release'
    files = {str(FORMAL/'fixture.json'):sha(FORMAL/'fixture.json'),
             str(SOURCE/'runtime-proof.json'):sha(SOURCE/'runtime-proof.json')}
    order=['candidate'+str(i+1) for i in range(repeats)]
    for arm in order:
        dest = root/arm; dest.mkdir()
        runtime = dest/'runtime-v1'
        shutil.copytree(CANDIDATE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(REPO/'src/vllm_kvmem/prefill_phase_gate.py',runtime/'src/vllm_kvmem/prefill_phase_gate.py')
        with (runtime/'src/vllm_kvmem/layerwise_query.py').open('a') as stream:
            stream.write("\n# Measurement only: fixed cohort gate; no Query/decode in measured prefill.\n"
                         f"os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '{owners}'\n"
                         "from .prefill_phase_gate import install as _install_measurement_gate\n"
                         "_install_measurement_gate()\n")
        shutil.copy2(REPO/'src/vllm_kvmem/prefill_cohort_gate.py',runtime/'src/vllm_kvmem/prefill_cohort_gate.py')
        with (runtime/'src/vllm_kvmem/scheduler.py').open('a') as stream:
            stream.write("\n# Numerical diagnostic only: fixed cohort waves and serial final decoding.\n"
                + "os.environ['KVMEM_DIAGNOSTIC_PREFILL_COHORT_LENGTHS'] = "
                + repr(json.dumps([r['input_tokens'] for r in rows])) + "\n"
                + "from .prefill_cohort_gate import install as _install_cohort_gate\n_install_cohort_gate()\n")
        arm_profile = json.loads(json.dumps(profile))
        assert arm_profile['kvmem']['prefill_publication_pipeline']
        save_new(dest/'fixture.json',fixture)
        save_new(dest/'profile.json',arm_profile)
        save_new(dest/'runtime-proof.json',tree(runtime))
        save_new(dest/'status.json',dict(status='prepared'))
        files.update({str(dest/n):sha(dest/n) for n in ['fixture.json','profile.json','runtime-proof.json']})
    assert all(read(root/arm/'runtime-proof.json') == read(root/order[0]/'runtime-proof.json') for arm in order)
    files.update({b['request']:b['sha256'] for b in fixture['bindings']})
    shutil.copy2(__file__,root/'launch.py')
    files[str(root/'launch.py')] = sha(root/'launch.py')
    save_new(root/'plan.json',dict(order=order,cases=ids,files=files,cpu_cache_bytes=112000000000,
        cpu_reservation_bytes=sum(reservations),client_credit_budget_bytes=kv['cpu_read_cache_bytes']-16*1024**3,
        scope='Original complete cold MAB prefixes; candidate v2 with fixed-cohort barriers and serialized final decoding; correctness diagnostic only. Native batch 16424, Active 36864 / Gen 16384, logical 1K, sampling and recall unchanged. Final Query gated for finite pure prefill measurement; not continuous service TPS.',
        formal_resume=False,external_grading=False,automatic_retry=False))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json') == read(root/'prepared-proof.json')['plan_sha256']
    plan = read(root/'plan.json')
    for name,digest in plan['files'].items(): assert sha(Path(name)) == digest,name
    for arm in plan['order']:
        assert tree(root/arm/'runtime-v1') == read(root/arm/'runtime-proof.json')
    return plan


def run_arm(root,arm):
    plan = verify(root)
    assert arm in plan['order']
    dest = root/arm
    runtime = dest/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('frozen_memory',runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec); spec.loader.exec_module(memory)
    def validate(path):
        assert path == dest
        verify(root)
        return read(dest/'fixture.json'),read(dest/'profile.json')
    memory.verify = validate
    memory.run(dest)


def run(root):
    plan = verify(root)
    if (root/'process.json').exists(): raise ValueError('Do not repeat attempts')
    save_new(root/'process.json',dict(pid=os.getpid(),at=time.time()))
    from run_recall_comparison import stop_owned,termination_guard
    child = None
    with termination_guard():
        try:
            for arm in plan['order']:
                (root/'status.json').write_text(json.dumps(dict(status='running',arm=arm)))
                with (root/(arm+'.log')).open('x') as log:
                    child = subprocess.Popen([sys.executable,'-B',str(root/'launch.py'),'arm',
                        '--root',str(root),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait()
                if child.returncode: raise RuntimeError(arm+' failed; preserve first responses, no retry')
            evidence = compare(root/plan['order'][0],root/plan['order'][1]) if len(plan['order'])==2 else dict(passed=False,scope='Single first candidate measurement; exact repeatability not yet qualified')
            save_new(root/'equivalence.json',evidence)
            (root/'status.json').write_text(json.dumps(dict(status='completed',exact_equivalence=evidence['passed'])))
        except BaseException as error:
            (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
            raise
        finally:
            stop_owned(child)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','arm'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=['candidate1','candidate2'])
    parser.add_argument('--owners',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=2)
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'prepare': prepare(root,args.owners,args.repeats)
    elif args.action == 'run': run(root)
    else: run_arm(root,args.arm)
