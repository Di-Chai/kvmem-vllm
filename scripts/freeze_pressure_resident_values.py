"""Apply only resident-Value reuse to the model-qualified P4 runtime."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

REPO=Path(__file__).resolve().parents[1]
P4=REPO/'artifacts/g4k-pressure-implementation-20261104-r1/post-forward-reap-serial'
BASE=REPO/'artifacts/g4k-gilopt-prefill-diagnostic-20261104-r1/cohort-1/runtime-v1'


def replace(text,old,new):
    if text.count(old)!=1:raise ValueError('Non-unique frozen patch anchor: '+old[:100])
    return text.replace(old,new)


def resident_sources(runtime, implementation=REPO):
    """Return only the Value-reuse delta against a qualified runtime.

    The older MAB runtime has no rolling-decode variable. Gate on the original
    command stage instead, so its decode policy remains completely unchanged.
    Exact anchors reject other revisions rather than replacing whole modules.
    """
    changes={}
    prefix='src/vllm_kvmem/'
    name=prefix+'layerwise_pressure.py';s=(runtime/name).read_text()
    anchor="        self.single_assembly = os.getenv('KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY') == '1'\n"
    s=replace(s,anchor,anchor+'        self.resident_values = True  # Frozen explicit candidate.\n')
    s=replace(s,'        chosen, evidence = [], []\n',
        "        chosen, evidence, source_maps = [], [], []\n        capture_values = job.command.pressure_stage == 'prefill' and self.resident_values\n")
    anchor='            mapping = self.records(rid, authority, job.history, local, layer)\n'
    s=replace(s,anchor,anchor+'            if capture_values:\n                source_maps.append(tuple(mapping))\n')
    anchor='        job.layerwise_pressure_plan = chosen, evidence\n'
    s=replace(s,anchor,anchor+'        if capture_values:\n            job.layerwise_pressure_value_sources = (rid, job.history, local, tuple(source_maps))\n')
    changes[name]=s
    name=prefix+'layerwise_pressure_efficiency.py';s=(runtime/name).read_text()
    current=(implementation/name).read_text();lo=current.index("        if job.command.pressure_stage == 'prefill' and getattr(controller, 'resident_values'")
    hi=current.index('        started = time.perf_counter()',lo)
    anchor='                     pressure_stage=job.command.pressure_stage)\n'
    s=replace(s,anchor,anchor+current[lo:hi])
    s=replace(s,'        inline_chunks = 0\n','        inline_chunks = resident_v_tokens = 0\n')
    anchor="            inline_chunks += stats.get('inline_read_chunks',0)\n"
    s=replace(s,anchor,anchor+"            resident_v_tokens += stats.get('resident_v_reused_tokens',0)\n")
    s=replace(s,'                 layerwise_inline_read_chunks=inline_chunks),',
        '                 layerwise_inline_read_chunks=inline_chunks,\n                 layerwise_resident_v_reused_tokens=resident_v_tokens),')
    changes[name]=s
    name=prefix+'layerwise_query.py';s=(runtime/name).read_text();current=(implementation/name).read_text()
    lo=current.index("            if 'pressure_value_sources' in state:")
    hi=current.index('            for cursor, group in groups:',lo)
    anchor='        staging.claim(assembler)\n        try:\n'
    changes[name]=replace(s,anchor,anchor+current[lo:hi])
    name=prefix+'pressure_resident_values.py';changes[name]=(implementation/name).read_text()
    name=prefix+'packed_assembly.py';s=(runtime/name).read_text()
    s=replace(s,'not wait or regions is None or destination is None',
              'not wait or (regions is None and region is None) or destination is None')
    changes[name]=replace(s,'synchronous paired CPU regions','synchronous CPU regions')
    name=prefix+'storage.py';s=(runtime/name).read_text()
    s=replace(s,'def read_region(self, namespace, leases, offset, length, *, destination=None, _sources=None):',
        'def read_region(self, namespace, leases, offset, length, *, destination=None, _sources=None, inline=False):')
    changes[name]=replace(s,'return self._read_regions(namespace, leases, ((offset, length),),\n                                  destination=destination, _sources=_sources)',
        'return self._read_regions(namespace, leases, ((offset, length),),\n                                  destination=destination, _sources=_sources, inline=inline)')
    name=prefix+'split_assembly.py';s=(runtime/name).read_text();current=(implementation/name).read_text()
    if 'def _gather_v_tiled(' not in current:
        return changes
    lo=current.index('@triton.jit\ndef _gather_v_tiled(')
    hi=current.index('\ndef gather_resident_values(',lo)
    anchor='def gather_resident_values(layers, from_bases, block_tokens,destination=None):'
    s=replace(s,anchor,current[lo:hi]+'\n'+anchor.replace('destination=None):','destination=None, *, tiled=False):'))
    lo=current.index('        if tiled:',hi);hi=current.index('        _gather_v[',lo)
    anchor='        _gather_v[(from_bases.numel()*block_tokens,heads)]('
    changes[name]=replace(s,anchor,current[lo:hi]+anchor)
    return changes


def freeze(root):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'freeze.py')
    proof=json.loads((P4/'runtime-proof.json').read_text())
    changes=resident_sources(P4/'runtime-v1');prefix='src/vllm_kvmem/'
    expected={};patch=[]
    for name in [*proof,prefix+'pressure_resident_values.py']:
        before=(BASE/name).read_bytes() if (BASE/name).exists() else b''
        after=changes[name].encode() if name in changes else (P4/'runtime-v1'/name).read_bytes()
        if before==after:continue
        expected[name]=hashlib.sha256(after).hexdigest()
        patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),
            fromfile='a/'+name if before else '/dev/null',tofile='b/'+name))
    path=root/'candidate.patch';path.write_text(''.join(patch))
    with (root/'candidate.json').open('x') as f:json.dump(dict(patch=str(path),
        patch_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),expected=expected,
        scope='P4 plus resident V reuse only; excludes P2, borrowed layer views and native score export. '
              'Existing bounded V staging, same raw K read and split assembly kernel.'),f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    freeze(p.parse_args().root.resolve())
