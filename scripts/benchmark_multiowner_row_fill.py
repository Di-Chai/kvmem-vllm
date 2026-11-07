"""Reuse the existing native row capacity, then measure complete six-owner inputs."""
import argparse
import ast
import difflib
from pathlib import Path
import shutil

from audit_layerwise_gilopt import read,save_new,sha,tree
from benchmark_mab_query_handoff import insert_once,method_source
from benchmark_multiowner_pressure_fix import run,arm

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'artifacts/multiowner-pressure-fix-bench-20261105-r2/combined1'
ORDER=('b1','b2','control')


def sources(runtime):
    result={}
    name='src/vllm_kvmem/options.py';code=(runtime/name).read_text()
    for old,new in (
        ('    prefill_physical_multi_owner: bool = False\n',
         '    prefill_physical_multi_owner: bool = False\n    prefill_physical_fill_rows: bool = False\n'),
        ("            raise ValueError('Multi-owner prefill requires physical coalescing')\n",
         "            raise ValueError('Multi-owner prefill requires physical coalescing')\n"
         "        if type(self.prefill_physical_fill_rows) is not bool:\n"
         "            raise ValueError('prefill_physical_fill_rows must be boolean')\n"
         "        if self.prefill_physical_fill_rows and not self.prefill_physical_multi_owner:\n"
         "            raise ValueError('Filling physical row reservations requires multi-owner prefill')\n"),
        ("            ('prefill_physical_multi_owner', False),\n",
         "            ('prefill_physical_multi_owner', False),\n            ('prefill_physical_fill_rows', False),\n"),
    ):code=insert_once(code,old,new)
    result[name]=code
    name='src/vllm_kvmem/scheduler.py';code=(runtime/name).read_text()
    for old,new in (
        ("        self.prefill_physical_multi_owner = getattr(options, 'prefill_physical_multi_owner', False)\n",
         "        self.prefill_physical_multi_owner = getattr(options, 'prefill_physical_multi_owner', False)\n"
         "        self.prefill_physical_fill_rows = getattr(options, 'prefill_physical_fill_rows', False)\n"),
        ('        self._logical_prefill_owners = {}\n',
         '        self._logical_prefill_owners = {}\n        self._logical_owner_chunks = {}\n'),
        ('                chunk, self.prefill_physical_max_chunks)\n            if limit >= 2:\n',
         "                chunk, self.prefill_physical_max_chunks)\n"
         "            if getattr(self, 'prefill_physical_fill_rows', False):\n"
         "                from .multiowner_prefill import owner_chunk_allocations\n"
         "                limits = owner_chunk_allocations(len(requests), self._physical_native_rows,\n"
         "                    chunk, self.prefill_physical_max_chunks, rotation=step)\n"
         "                limit = max(limits)\n"
         "                self._logical_owner_chunks = {r.request_id:n for r,n in zip(requests,limits)}\n"
         "            if limit >= 2:\n"),
        ("        maximum = min(getattr(self, '_logical_owner_max_chunks', getattr(self, 'prefill_physical_max_chunks', 2)), budget//chunk,\n",
         "        owner_max = getattr(self, '_logical_owner_chunks', {}).get(request.request_id,\n"
         "            getattr(self, '_logical_owner_max_chunks', getattr(self, 'prefill_physical_max_chunks', 2)))\n"
         "        maximum = min(owner_max, budget//chunk,\n"),
    ):code=insert_once(code,old,new)
    result[name]=code
    name='src/vllm_kvmem/multiowner_prefill.py';code=(runtime/name).read_text()
    helper=method_source((REPO/name).read_text(),'owner_chunk_allocations')
    result[name]=code+'\n\n'+helper+'\n'
    for code in result.values():ast.parse(code)
    return result


def prepare(root):
    assert not root.exists()
    assert read(SOURCE/'status.json')['status']=='completed'
    original=read(SOURCE/'runtime-proof.json');assert tree(SOURCE/'runtime-v1')==original
    root.mkdir(parents=True)
    files={}
    for kind in ORDER:
        dest=root/kind;dest.mkdir();runtime=dest/'runtime-v1'
        shutil.copytree(SOURCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        patches=[]
        if kind!='control':
            for name,after in sources(runtime).items():
                p=runtime/name;before=p.read_text();p.write_text(after)
                patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        (dest/'execution.patch').write_text(''.join(patches))
        shutil.copy2(SOURCE/'fixture.json',dest/'fixture.json')
        profile=read(SOURCE/'profile.json')
        if kind!='control':profile['kvmem']['prefill_physical_fill_rows']=True
        save_new(dest/'profile.json',profile)
        save_new(dest/'runtime-proof.json',tree(runtime));save_new(dest/'status.json',dict(status='prepared'))
        files.update({str(dest/n):sha(dest/n) for n in ('fixture.json','profile.json','runtime-proof.json','execution.patch')})
    shutil.copy2(__file__,root/'launch.py');files[str(root/'launch.py')]=sha(root/'launch.py')
    for binding in read(SOURCE/'fixture.json')['bindings']:files[binding['request']]=binding['sha256']
    save_new(root/'plan.json',dict(order=ORDER,source=str(SOURCE),files=files,
        scope='Same six complete cold inputs and phase gate. Distribute only unused integral logical chunks within existing448 native rows, max8 chunks per owner, native batch16424. Active/Gen, sampling and recall unchanged. Physical numerical batch shapes change under prior authorization.',
        automatic_retry=False,formal_resume=False,external_grading=False))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','arm'))
    p.add_argument('--root',required=True,type=Path);p.add_argument('--kind',choices=ORDER)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='arm':arm(root,a.kind)
    else:run(root)
