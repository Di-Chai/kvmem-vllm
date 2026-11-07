"""Finite four-prefix MAB prefill comparison, preserving original controls."""
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
SOURCE = REPO/'artifacts/mab-efficiency-20261105-r1/query-handoff-qualification/baseline'
FORMAL = REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal'


def prepare(root):
    if root.exists(): raise ValueError('Preserve previous evidence')
    assert tree(SOURCE/'runtime-v1') == read(SOURCE/'runtime-proof.json')
    root.mkdir(parents=True)
    fixture = read(FORMAL/'fixture.json')
    rows = [next(r for r in fixture['rows'] if r['id'].startswith(group+'_')
                 and r['id'].endswith('_q0002')) for group in ('005','006','007','008')]
    ids = [r['id'] for r in rows]
    fixture.update(rows=rows,case_ids=ids,cases=4,qualification=False,
                   restore_digest_qualification=False,diagnostic_only=True,
                   performance_observation=True)
    fixture['bindings'] = [b for b in fixture['bindings'] if b['id'] in ids]
    profile = read(SOURCE/'profile.json')
    kv = profile['kvmem']
    assert [kv[k] for k in ['select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks']] == [36864,16384,32,41,0]
    assert profile['max_num_seqs'] == 8 and profile['max_num_batched_tokens'] == 16424
    reservations = [((r['input_tokens']+2*kv['gen_budget']+31)//32)*32*17*4*256*2 for r in rows]
    assert sum(reservations) < kv['cpu_read_cache_bytes'] - 16*1024**3, 'All phase owners must fit without waiting for a gated release'
    files = {str(FORMAL/'fixture.json'):sha(FORMAL/'fixture.json'),
             str(SOURCE/'runtime-proof.json'):sha(SOURCE/'runtime-proof.json')}
    for arm, enabled in [('baseline',False),('pipeline',True)]:
        dest = root/arm; dest.mkdir()
        runtime = dest/'runtime-v1'
        shutil.copytree(SOURCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(REPO/'src/vllm_kvmem/prefill_phase_gate.py',runtime/'src/vllm_kvmem/prefill_phase_gate.py')
        with (runtime/'src/vllm_kvmem/layerwise_query.py').open('a') as stream:
            stream.write("\n# Measurement only: identical fixed four-owner gate in both profiles.\n"
                         "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '4'\n"
                         "from .prefill_phase_gate import install as _install_measurement_gate\n"
                         "_install_measurement_gate()\n")
        arm_profile = json.loads(json.dumps(profile))
        if enabled: arm_profile['kvmem']['prefill_publication_pipeline'] = True
        save_new(dest/'fixture.json',fixture)
        save_new(dest/'profile.json',arm_profile)
        save_new(dest/'runtime-proof.json',tree(runtime))
        save_new(dest/'status.json',dict(status='prepared'))
        files.update({str(dest/n):sha(dest/n) for n in ['fixture.json','profile.json','runtime-proof.json']})
    assert read(root/'baseline/runtime-proof.json') == read(root/'pipeline/runtime-proof.json')
    changed = {n for n,d in read(root/'baseline/runtime-proof.json').items()
               if read(SOURCE/'runtime-proof.json').get(n) != d}
    assert changed == {'src/vllm_kvmem/prefill_phase_gate.py','src/vllm_kvmem/layerwise_query.py'}
    files.update({b['request']:b['sha256'] for b in fixture['bindings']})
    shutil.copy2(__file__,root/'launch.py')
    files[str(root/'launch.py')] = sha(root/'launch.py')
    save_new(root/'plan.json',dict(order=['baseline','pipeline'],cases=ids,files=files,
        cpu_reservation_bytes=sum(reservations),client_credit_budget_bytes=kv['cpu_read_cache_bytes']-16*1024**3,
        scope='Four original full cold prefixes, same budgets/sampling/runtime and identical finite history gate. Only publication pipeline differs. Gate excludes all Query/decode but changes service scheduling; diagnostic rate, not continuous MAB throughput.',
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
            evidence = compare(root/'baseline',root/'pipeline')
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
    parser.add_argument('--arm',choices=['baseline','pipeline'])
    args = parser.parse_args(); root = args.root.resolve()
    if args.action == 'prepare': prepare(root)
    elif args.action == 'run': run(root)
    else: run_arm(root,args.arm)
