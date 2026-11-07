"""One complete cold MAB request for immediate pressure assembly submission."""
import argparse
import json
from pathlib import Path
import shutil

from audit_layerwise_gilopt import read, save_new, sha, tree
from benchmark_mab_query_handoff import method_source, insert_once
from benchmark_mab_direct_restore import run

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())


def prepare(root, baseline):
    if root.exists():raise ValueError('Preserve previous attempt')
    assert tree(baseline/'runtime-v1') == read(baseline/'runtime-proof.json')
    assert read(baseline/'status.json')['status'] == 'completed'
    root.mkdir(parents=True)
    candidate = root/'candidate';candidate.mkdir()
    runtime = candidate/'runtime-v1'
    shutil.copytree(baseline/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    original = tree(runtime)
    path = runtime/'src/vllm_kvmem/async_refresh.py'
    text = path.read_text()
    old = method_source(text,'_submit')
    current = method_source((REPO/'src/vllm_kvmem/async_refresh.py').read_text(),'_submit')
    marker = "        self.emit(job,'assembly_queued')\n"
    assert current[:current.index(marker)+len(marker)].strip() == old.strip()
    path.write_text(insert_once(text,old,current))
    path = runtime/'src/vllm_kvmem/layerwise_pressure.py'
    text = path.read_text()
    anchor = next(l for l in text.splitlines(keepends=True) if l.startswith('        self.single_assembly = '))
    path.write_text(insert_once(text,anchor,anchor+'        self.submit_on_prepare = True\n'))
    path = runtime/'src/vllm_kvmem/layerwise_query.py'
    text = path.read_text()
    text = insert_once(text,'                    diagnostic.prepare(self, job, queries)\n',
        "                    self.emit(job, 'assembly_worker_started')\n                    diagnostic.prepare(self, job, queries)\n")
    path.write_text(text)
    frozen = tree(runtime)
    changed = sorted(k for k in frozen if frozen[k] != original.get(k))
    assert changed == ['src/vllm_kvmem/async_refresh.py','src/vllm_kvmem/layerwise_pressure.py','src/vllm_kvmem/layerwise_query.py']
    fixture = read(baseline/'fixture.json')
    row = fixture['rows'][0]
    fixture.update(rows=[row],case_ids=[row['id']],cases=1,
        bindings=[b for b in fixture['bindings'] if b['id']==row['id']],diagnostic_only=True)
    save_new(candidate/'fixture.json',fixture)
    for name in ('profile.json','verification-profile.json'):shutil.copy2(baseline/name,candidate/name)
    save_new(candidate/'runtime-proof.json',frozen)
    save_new(candidate/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py')
    files={str(candidate/n):sha(candidate/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')}
    files.update({b['request']:b['sha256'] for b in fixture['bindings']})
    files.update({str(p):sha(p) for p in (baseline/'run/answers/cases'/row['id']).iterdir() if p.is_file()})
    for p in [root/'launch.py',baseline/'runtime-proof.json',REPO/'scripts/benchmark_mab_direct_restore.py',
              REPO/'scripts/benchmark_mab_query_handoff.py']:
        files[str(p)]=sha(p)
    save_new(root/'plan.json',dict(baseline=str(baseline),files=files,changed=changed,
        formal_resume=False,external_grading=False,scope='Original full cold input; pressure submission scheduling only. Native lane/byte admission, selection and KV contents unchanged. Pure prefill ends before final Query.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve(),args.baseline.resolve())
    else:run(args.root.resolve())
