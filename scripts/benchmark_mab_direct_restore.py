"""Qualify direct layerwise P restoration against an immutable MAB control."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from audit_layerwise_gilopt import compare, read, save_new, sha, tree
from benchmark_mab_query_handoff import method_source, insert_once

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())


def prepare(root, baseline, qualification=None):
    if root.exists(): raise ValueError('Preserve previous attempt')
    assert read(baseline/'status.json')['status'] == 'completed'
    assert not read(baseline/'run/cleanup.json')['gpu_processes']
    assert tree(baseline/'runtime-v1') == read(baseline/'runtime-proof.json')
    root.mkdir(parents=True)
    candidate = root/'candidate'
    candidate.mkdir()
    runtime = candidate/'runtime-v1'
    shutil.copytree(baseline/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('fixture.json','profile.json','verification-profile.json'):
        shutil.copy2(baseline/name,candidate/name)
    original = tree(runtime)
    for name, methods in [('layerwise_checkpoints.py',('_resolve','prepare_restore')),
                          ('session_worker.py',('_restore',))]:
        path = runtime/'src/vllm_kvmem'/name
        text = path.read_text()
        source = (REPO/'src/vllm_kvmem'/name).read_text()
        for method in methods:
            text = insert_once(text,method_source(text,method),method_source(source,method))
        path.write_text(text)
    path = runtime/'src/vllm_kvmem/layerwise_query.py'
    text = path.read_text()
    anchor = "        self.prepared_index = os.getenv('KVMEM_LAYERWISE_PREPARED_INDEX') == '1'\n"
    text = insert_once(text,anchor,anchor+'        self.direct_prefix_restore = True\n')
    path.write_text(text)
    shutil.copy2(REPO/'src/vllm_kvmem/layerwise_restore_efficiency.py',runtime/'src/vllm_kvmem/layerwise_restore_efficiency.py')
    frozen = tree(runtime)
    changed = sorted(k for k in frozen if original.get(k) != frozen[k])
    assert changed == ['src/vllm_kvmem/layerwise_checkpoints.py','src/vllm_kvmem/layerwise_query.py',
                       'src/vllm_kvmem/layerwise_restore_efficiency.py','src/vllm_kvmem/session_worker.py']
    save_new(candidate/'runtime-proof.json',frozen)
    save_new(candidate/'status.json',dict(status='prepared'))
    files = {str(candidate/n):sha(candidate/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')}
    if not read(candidate/'fixture.json')['restore_digest_qualification']:
        if qualification is None:
            raise ValueError('Performance requires a passed byte/output qualification')
        gate = read(qualification/'equivalence.json')
        assert gate['passed'] and gate['cases'] == 4
        assert read(qualification/'status.json')['status'] == 'completed'
        assert tree(qualification/'candidate/runtime-v1') == read(qualification/'candidate/runtime-proof.json')
        for name in ('equivalence.json','candidate/runtime-proof.json','candidate/execution-audit.json'):
            files[str(qualification/name)] = sha(qualification/name)
    for binding in read(candidate/'fixture.json')['bindings']:files[binding['request']] = binding['sha256']
    for path in (baseline/'run/answers/cases').rglob('*'):
        if path.is_file():files[str(path)] = sha(path)
    files[str(baseline/'runtime-proof.json')] = sha(baseline/'runtime-proof.json')
    shutil.copy2(__file__,root/'launch.py')
    files[str(root/'launch.py')] = sha(root/'launch.py')
    save_new(root/'plan.json',dict(baseline=str(baseline),files=files,changed=changed,
        formal_resume=False,external_grading=False,scope='Direct final P views; same archive, target/draft bytes, model state, budgets, score policy and full inputs.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def run(root):
    assert sha(root/'plan.json') == read(root/'prepared-proof.json')['plan_sha256']
    plan = read(root/'plan.json')
    candidate = root/'candidate'
    sys.path[:0] = [str(candidate/'runtime-v1/src'),str(candidate/'runtime-v1/scripts')]
    spec = importlib.util.spec_from_file_location('direct_restore_memory',candidate/'runtime-v1/scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    def verify(path):
        assert path == candidate
        for name,digest in plan['files'].items():assert sha(Path(name)) == digest, name
        assert tree(candidate/'runtime-v1') == read(candidate/'runtime-proof.json')
        assert tree(Path(plan['baseline'])/'runtime-v1') == read(Path(plan['baseline'])/'runtime-proof.json')
        return read(candidate/'fixture.json'),read(candidate/'profile.json')
    memory.verify = verify
    try:
        (root/'status.json').write_text(json.dumps(dict(status='running')))
        memory.run(candidate)
        evidence = compare(Path(plan['baseline']),candidate)
        save_new(root/'equivalence.json',evidence)
        (root/'status.json').write_text(json.dumps(dict(status='completed',equivalent=evidence['passed'])))
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--qualification',type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':prepare(args.root.resolve(),args.baseline.resolve(),args.qualification.resolve() if args.qualification else None)
    else:run(args.root.resolve())
