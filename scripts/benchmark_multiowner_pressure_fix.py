"""Predeclared full-input pressure handoff/view-reuse comparison.

Only execution deltas are ported onto the measured six-owner runtime. Original
RoPE, accepted scores, selection, budgets, inputs and sampling are untouched.
The cohort gate isolates historical prefill; these are not service TPS results.
"""
import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_layerwise_gilopt import compare, read, save_new, sha, tree
from benchmark_mab_query_handoff import insert_once, method_source
from freeze_pressure_layer_views import view_sources

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
ORDER = ('submit', 'combined1', 'combined2', 'control')


def pressure_sources(runtime, kind):
    changes = {}
    if kind in ('submit', 'combined1', 'combined2'):
        name = 'src/vllm_kvmem/async_refresh.py'
        code = (runtime/name).read_text()
        old = method_source(code, '_submit')
        new = method_source((REPO/name).read_text(), '_submit')
        marker = "        self.emit(job,'assembly_queued')\n"
        assert new[:new.index(marker)+len(marker)].strip() == old.strip()
        changes[name] = insert_once(code, old, new)
        name = 'src/vllm_kvmem/layerwise_pressure.py'
        code = (runtime/name).read_text()
        anchor = next(line for line in code.splitlines(True) if line.startswith('        self.single_assembly = '))
        changes[name] = insert_once(code, anchor, anchor+'        self.submit_on_prepare = True  # Frozen execution candidate.\n')
    if kind.startswith('combined'):
        # Reuse only the already tested transaction-local borrowed-view delta.
        # This helper preserves frozen original-RoPE assembly and selection.
        for name, code in view_sources(runtime).items():
            if name in changes:
                anchor = next(line for line in code.splitlines(True) if line.startswith('        self.single_assembly = '))
                code = insert_once(code, anchor, anchor+'        self.submit_on_prepare = True  # Frozen execution candidate.\n')
            changes[name] = code
    return changes


def prepare(root, source):
    assert not root.exists(), 'Preserve every prior attempt'
    assert read(source/'status.json')['status'] == 'completed'
    original = read(source/'runtime-proof.json')
    assert tree(source/'runtime-v1') == original
    profile = read(source/'profile.json'); kv = profile['kvmem']
    assert [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] == [36864,16384,32,41,0]
    assert kv['cpu_read_cache_bytes'] == 112000000000 and kv['prefill_physical_multi_owner']
    assert profile['max_num_batched_tokens'] == 16424
    assert read(source/'fixture.json')['cases'] == 6
    root.mkdir(parents=True)
    files = {}
    for kind in ORDER:
        dest = root/kind; dest.mkdir(); runtime = dest/'runtime-v1'
        shutil.copytree(source/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        patch = []
        for name, code in pressure_sources(runtime,kind).items():
            p = runtime/name; before = p.read_text(); p.write_text(code)
            patch.extend(difflib.unified_diff(before.splitlines(True),code.splitlines(True),
                fromfile='a/'+name,tofile='b/'+name))
        (dest/'execution.patch').write_text(''.join(patch))
        for name in ('fixture.json','profile.json'):
            shutil.copy2(source/name,dest/name)
        current = tree(runtime)
        save_new(dest/'runtime-proof.json',current)
        save_new(dest/'status.json',dict(status='prepared'))
        save_new(dest/'changes.json',dict(changed=sorted(n for n in current if current[n]!=original.get(n)),
            source=str(source),strategy=kind,recall_changed=False))
        files.update({str(dest/n):sha(dest/n) for n in ('fixture.json','profile.json','runtime-proof.json','execution.patch','changes.json')})
    assert tree(root/'combined1/runtime-v1') == tree(root/'combined2/runtime-v1')
    assert tree(root/'control/runtime-v1') == original
    for binding in read(source/'fixture.json')['bindings']:
        files[binding['request']] = binding['sha256']
    shutil.copy2(__file__,root/'launch.py'); files[str(root/'launch.py')] = sha(root/'launch.py')
    save_new(root/'plan.json',dict(order=ORDER,source=str(source),files=files,
        scope='Six unchanged complete cold MAB inputs. Submit-only attribution, two combined runs, and a following unchanged control; previous two controls preserved. No profiler. Original phase gate excludes Query/decode.',
        automatic_retry=False,formal_resume=False,external_grading=False,
        exact_dynamic_batch_equivalence_required=False,
        correctness='Existing multiowner A/A changes batch shapes. Preserve every answer and report differences; component byte checks and separate fixed-shape full-input qualification are required.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json') == read(root/'prepared-proof.json')['plan_sha256']
    plan = read(root/'plan.json')
    for name, digest in plan['files'].items(): assert sha(Path(name)) == digest, name
    for kind in plan['order']:
        assert tree(root/kind/'runtime-v1') == read(root/kind/'runtime-proof.json')
    return plan


def arm(root, kind):
    plan = verify(root); assert kind in plan['order']
    dest = root/kind; runtime = dest/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('frozen_memory',runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec); spec.loader.exec_module(memory)
    def frozen(path):
        assert path == dest; verify(root)
        return read(dest/'fixture.json'),read(dest/'profile.json')
    memory.verify = frozen
    memory.run(dest)


def run(root):
    plan = verify(root)
    assert read(root/'qualification.json')['passed'], 'CPU and GPU checks must precede full inputs'
    save_new(root/'process.json',dict(pid=os.getpid(),at=time.time()))
    from run_recall_comparison import termination_guard, stop_owned
    from analyze_mab_prefill_concurrency import measure
    child = None
    with termination_guard():
        try:
            for kind in plan['order']:
                (root/'status.json').write_text(json.dumps(dict(status='running',arm=kind)))
                with (root/(kind+'.log')).open('x') as log:
                    child = subprocess.Popen([sys.executable,'-B',str(root/'launch.py'),'arm',
                        '--root',str(root),'--kind',kind],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait()
                if child.returncode: raise RuntimeError('Preserved first technical failure: '+kind)
                save_new(root/(kind+'-phase.json'),measure(root/kind))
                save_new(root/(kind+'-comparison.json'),compare(Path(plan['source']),root/kind))
            (root/'status.json').write_text(json.dumps(dict(status='completed')))
        except BaseException as error:
            (root/'status.json').write_text(json.dumps(dict(status='failed',arm=kind,error=repr(error))))
            raise
        finally: stop_owned(child)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','arm'))
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--kind',choices=ORDER)
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare': prepare(root,a.source.resolve())
    elif a.action == 'arm': arm(root,a.kind)
    else: run(root)
