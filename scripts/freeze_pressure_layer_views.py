"""Freeze only the validated-view delta on the previous P4 runtime."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'artifacts/g4k-gilopt-prefill-diagnostic-20261104-r1/cohort-1/runtime-v1'
P4=REPO/'artifacts/g4k-pressure-implementation-20261104-r1/post-forward-reap-serial/runtime-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def replace(text,old,new):
    if text.count(old)!=1:raise ValueError('Patch anchor is not unique: '+old[:100])
    return text.replace(old,new)
def method(text,name):
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='NativeWindowAssembler')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(text.splitlines(keepends=True)[start:node.end_lineno])

def view_sources(runtime, implementation=REPO):
    names=('window_assembly.py','layerwise_query.py','layerwise_pressure.py','layerwise_pressure_efficiency.py')
    patches={}
    for name in names:
        relative='src/vllm_kvmem/'+name
        before=(runtime/relative).read_text();current=(implementation/relative).read_text();after=before
        if name=='window_assembly.py':
            after=replace(after,method(before,'__init__'),method(current,'__init__'))
            methods='\n\n'.join(method(current,n).rstrip() for n in
                ('_layer_metadata','_validate_borrowed_layers','from_pressure_parent'))
            anchor='    @classmethod\n    def from_runner('
            after=replace(after,anchor,methods+'\n\n'+anchor)
            after=replace(after,'            assembler._shared_staging = True\n        return assembler',
                          '            assembler._shared_staging = True\n        assembler._runner_binding = (runner, request_id)\n        return assembler')
        elif name=='layerwise_query.py':
            lo=before.index('        assembler = NativeWindowAssembler.from_runner(',before.index('    def assemble_layer_pinned('))
            hi=before.index('        layer, staging = assembler.layers[0]',lo)
            start=current.index("        parent = state.get('pressure_parent_assembler')")
            end=current.index('        layer, staging = assembler.layers[0]',start)
            after=before[:lo]+current[start:end]+before[hi:]
        elif name=='layerwise_pressure.py':
            anchor="        self.single_assembly = os.getenv('KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY') == '1'\n"
            after=replace(after,anchor,anchor+'        self.reuse_layer_views = True  # Frozen candidate only.\n')
        else:
            anchor='                     pressure_stage=job.command.pressure_stage)\n'
            after=replace(after,anchor,anchor+"        if job.command.pressure_stage == 'prefill' and getattr(controller, 'reuse_layer_views', False):\n            state['pressure_parent_assembler'] = self\n")
            anchor='            dict(layerwise_single_assembly=True, layerwise_assembled_layers=len(names),\n'
            after=replace(after,anchor,anchor+"                 layerwise_reused_layer_views='pressure_parent_assembler' in state,\n")
        ast.parse(after);patches[relative]=after
    return patches


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,root/'freeze.py')
    previous=json.loads((P4.parent/'runtime-proof.json').read_text())
    assert all(sha(P4/n)==digest for n,digest in previous.items())
    patches=view_sources(P4)
    patch=[];expected={};source={}
    for relative in previous:
        before=(BASE/relative).read_bytes();after=patches.get(relative)
        data=after.encode() if after is not None else (P4/relative).read_bytes()
        if before==data:continue
        expected[relative]=hashlib.sha256(data).hexdigest()
        patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),data.decode().splitlines(keepends=True),
                                         fromfile='a/'+relative,tofile='b/'+relative))
    path=root/'candidate.patch';path.write_text(''.join(patch))
    save(root/'candidate.json',dict(patch=str(path),patch_sha256=sha(path),expected=expected,
        source_files={str(REPO/n):sha(REPO/n) for n in patches},
        scope='P4 plus transaction-local validated pressure views; no new prepared source/native score changes.'))
    save(root/'status.json',dict(status='frozen_candidate',model_qualified=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    freeze(p.parse_args().root.resolve())
