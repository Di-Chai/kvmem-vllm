"""The layer import rule (docs/layering.md, step 7).

vllm_kvmem is split into core, adapters, serving and apps; each imports only
itself and the layers below it. The strategy package imports nothing outside
itself and every layer may import it. The modules at the old flat paths are
aliases of the moved ones, kept for one release.
"""
import ast
import importlib
from pathlib import Path

import pytest

import vllm_kvmem

PACKAGE = Path(vllm_kvmem.__file__).parent
LAYERS = ('core', 'adapters', 'serving', 'apps')
BELOW = {layer: set(LAYERS[:i + 1]) for i, layer in enumerate(LAYERS)}
# Native extensions stay at the package root; they are core.
ROOT_CORE = {'_native', '_prefill_state_native'}
# Imports that point up today. Each needs its dependency inverted; the list may
# only shrink (a fixed entry fails test_exceptions_are_still_needed).
EXCEPTIONS = frozenset({
    ('adapters.q8_adapter', 'serving.integration'),
    ('core.checkpoint_storage', 'serving.request_checkpoint'),
    ('core.memory', 'serving.pressure_pipeline'),
    ('core.memory', 'serving.pressure_reader'),
    ('core.memory', 'serving.tensor_parallel'),
    ('core.memory_storage', 'serving.pressure_reader'),
    ('core.prepared_regions', 'serving.pressure_reader'),
})
SHIM_MARK = '(layering step 7). This alias is removed after the next release.'


def _exists(dotted):
    path = PACKAGE.joinpath(*dotted.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file() or \
        any(PACKAGE.glob(dotted + '.*.so'))


def imports(path):
    """(source module, target module) for every package-internal import in a file."""
    relative = path.relative_to(PACKAGE).with_suffix('')
    module = '.'.join(relative.parts)
    package = relative.parts[:-1] if path.name != '__init__.py' else relative.parts[:-1]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package)[:len(package) - (node.level - 1)]
            elif node.module and node.module.split('.')[0] == 'vllm_kvmem':
                base = []
                node = ast.ImportFrom(module='.'.join(node.module.split('.')[1:]) or None, names=node.names, level=0)
            else:
                continue
            if node.module:
                yield module, '.'.join(base + node.module.split('.'))
            else:
                for alias in node.names:
                    target = '.'.join(base + [alias.name])
                    yield module, target if _exists(target) else '.'.join(base)


def layer_of(dotted):
    top = dotted.split('.')[0]
    if top in LAYERS:
        return top
    if top in ROOT_CORE:
        return 'core'
    if top == 'strategy':
        return 'strategy'
    return 'root'


def unit(dotted):
    """Layer plus top-level module: the granularity of EXCEPTIONS."""
    parts = dotted.split('.')
    return '.'.join(parts[:2]) if parts[0] in LAYERS else parts[0]


def edges():
    for layer in LAYERS:
        for path in sorted((PACKAGE / layer).rglob('*.py')):
            yield from imports(path)


def test_each_layer_imports_only_itself_and_the_layers_below():
    found = set()
    for source, target in edges():
        layer, kind = layer_of(source), layer_of(target)
        if kind == 'strategy' or (kind == 'root' and target == ''):
            continue
        assert kind != 'root', '%s imports %s through its old path; use the new one' % (source, target)
        if kind not in BELOW[layer]:
            found.add((unit(source), unit(target)))
    assert found <= EXCEPTIONS, sorted(found - EXCEPTIONS)


def test_exceptions_are_still_needed():
    found = {(unit(s), unit(t)) for s, t in edges()}
    assert EXCEPTIONS <= found, 'fixed, remove from EXCEPTIONS: %s' % sorted(EXCEPTIONS - found)


def test_the_package_root_holds_only_glue_and_aliases():
    for path in sorted(PACKAGE.glob('*.py')):
        if path.name in ('__init__.py', '__main__.py'):
            continue
        assert SHIM_MARK in path.read_text(), path.name
    for directory in sorted(p for p in PACKAGE.iterdir() if p.is_dir() and (p / '__init__.py').is_file()):
        if directory.name in LAYERS or directory.name == 'strategy':
            continue
        for path in directory.rglob('*.py'):     # old subpackage paths: aliases only
            assert 'Moved to vllm_kvmem.' in path.read_text(), path


@pytest.mark.parametrize('old,new', [('vllm_kvmem.serve', 'vllm_kvmem.serving.serve'),
                                     ('vllm_kvmem.memory', 'vllm_kvmem.core.memory'),
                                     ('vllm_kvmem.capture', 'vllm_kvmem.adapters.capture'),
                                     ('vllm_kvmem.planner.costs', 'vllm_kvmem.apps.planner.costs'),
                                     ('vllm_kvmem.settings', 'vllm_kvmem.core.settings')])
def test_an_old_path_is_the_moved_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_every_old_path_aliases_a_moved_module():
    # In a subprocess: several modules patch vLLM or process state when imported.
    import subprocess
    import sys
    code = """
import importlib, pathlib, sys, vllm_kvmem
root = pathlib.Path(vllm_kvmem.__file__).parent
names = [p.stem for p in root.glob('*.py') if p.stem not in ('__init__', '__main__')]
for d in [p for p in root.iterdir() if (p / '__init__.py').is_file() and p.name not in %r]:
    names += [d.name + '.' + p.stem for p in d.glob('*.py') if p.stem != '__init__']
bad = [n for n in names if importlib.import_module('vllm_kvmem.' + n).__name__.split('.')[1] not in %r]
print(len(names), bad)
sys.exit(1 if bad or not names else 0)
""" % (LAYERS + ('strategy',), LAYERS)
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env=dict(__import__('os').environ, PYTHONPATH=str(PACKAGE.parent)))
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
