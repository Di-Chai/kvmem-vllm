"""Reuse qualified static FA2 statistics in the unchanged six-prefix cohort."""
import argparse
from pathlib import Path
import shutil

from audit_layerwise_gilopt import read, save_new, sha, tree
from benchmark_multiowner_pressure_fix import arm, run

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
SOURCE = REPO/'artifacts/multiowner-pressure-fix-bench-20261105-r2/combined1'
ORDER = ('b1', 'b2', 'control')


def prepare(root):
    assert not root.exists()
    assert read(SOURCE/'status.json')['status'] == 'completed'
    assert tree(SOURCE/'runtime-v1') == read(SOURCE/'runtime-proof.json')
    root.mkdir(parents=True)
    files = {}
    for kind in ORDER:
        dest = root/kind; dest.mkdir()
        shutil.copytree(SOURCE/'runtime-v1', dest/'runtime-v1',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(SOURCE/'fixture.json', dest/'fixture.json')
        profile = read(SOURCE/'profile.json')
        if kind != 'control': profile['kvmem']['action_native_prefill_static_aligned'] = True
        save_new(dest/'profile.json', profile)
        save_new(dest/'runtime-proof.json', tree(dest/'runtime-v1'))
        save_new(dest/'status.json', dict(status='prepared'))
        files.update({str(dest/n):sha(dest/n) for n in ('fixture.json', 'profile.json', 'runtime-proof.json')})
    components = root/'components'; components.mkdir()
    for name in ('test_native_prefill_static_aligned.py', 'test_native_prefill_static_aligned_gpu.py',
                 'test_multiowner_prefill_gpu.py', 'test_physical_prefill_stats_gpu.py',
                 'test_options.py', 'test_native_prefill_aligned_pieces.py'):
        shutil.copy2(REPO/'tests'/name, components/name)
    save_new(components/'proof.json', {p.name:sha(p) for p in components.glob('*.py')})
    files[str(components/'proof.json')] = sha(components/'proof.json')
    shutil.copy2(__file__, root/'launch.py'); files[str(root/'launch.py')] = sha(root/'launch.py')
    for binding in read(SOURCE/'fixture.json')['bindings']: files[binding['request']] = binding['sha256']
    save_new(root/'plan.json', dict(order=ORDER, source=str(SOURCE), files=files,
        scope='Only existing static native statistics dispatch enabled; original rows448/6GiB, physical coalescing, Active/Gen, complete inputs, sampling, recall and measurement gate unchanged.',
        expected_component_tests=71, automatic_retry=False, formal_resume=False, external_grading=False))
    save_new(root/'prepared-proof.json', dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json', dict(status='prepared'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'arm'))
    p.add_argument('--root', required=True, type=Path); p.add_argument('--kind', choices=ORDER)
    args = p.parse_args(); root = args.root.resolve()
    if args.action == 'prepare': prepare(root)
    elif args.action == 'arm': arm(root, args.kind)
    else: run(root)
