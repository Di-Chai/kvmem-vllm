"""The layer import rule (docs/layering.md, step 7).

vllm_kvmem is split into core, adapters, serving and apps; each imports only
itself and the layers below it. The strategy package imports nothing outside
itself and every layer may import it. The old flat module paths were removed
after the move; every reference uses the layer path.
"""
import ast
from pathlib import Path

import pytest

import vllm_kvmem

PACKAGE = Path(vllm_kvmem.__file__).parent
LAYERS = ('core', 'adapters', 'serving', 'apps')
BELOW = {layer: set(LAYERS[:i + 1]) for i, layer in enumerate(LAYERS)}
# Native extensions stay at the package root; they are core.
ROOT_CORE = {'_native', '_prefill_state_native'}
# Imports allowed to point up. Empty since the seven upward imports left after the
# move were inverted (serving registers hooks on core instead of core importing it).
EXCEPTIONS = frozenset()


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


def test_the_package_root_holds_only_glue():
    assert sorted(p.name for p in PACKAGE.glob('*.py')) == ['__init__.py', '__main__.py']
    packages = sorted(p.name for p in PACKAGE.iterdir() if (p / '__init__.py').is_file())
    assert packages == sorted(LAYERS + ('strategy',))


def old_names():
    """Every module name that once sat at the flat package root."""
    names = set()
    for layer in LAYERS:
        names.update(p.stem for p in (PACKAGE / layer).glob('*.py') if p.stem != '__init__')
        names.update(p.name for p in (PACKAGE / layer).iterdir() if (p / '__init__.py').is_file())
    return names


def test_nothing_in_the_repository_uses_an_old_flat_path():
    """The aliases at the old paths were removed; every reference uses the layer path."""
    import re
    repo = PACKAGE.parents[1]
    alternatives = '|'.join(sorted(old_names(), key=len, reverse=True))
    module = re.compile(r'\bvllm_kvmem\.(%s)\b' % alternatives)
    either = re.compile(r'\bvllm_kvmem(?:\.|/)(%s)\b' % alternatives)
    found = []
    for top in ('src', 'tests', 'scripts', 'benchmarks', 'docs', 'examples', 'experiments', 'diagnostics', 'ui'):
        # Experiment harnesses and their tests also handle frozen runtimes, whose
        # files keep the flat layout: only module names are checked there.
        pattern = module if top in ('scripts', 'tests') else either
        for path in (repo / top).rglob('*') if (repo / top).is_dir() else ():
            if (not path.is_file() or 'node_modules' in path.parts or '__pycache__' in path.parts or
                    any(part.endswith('.egg-info') for part in path.parts) or
                    path.suffix not in ('.py', '.md', '.json', '.toml', '.sh', '.cjs', '.js', '.txt', '.yml')):
                continue
            text = path.read_text(errors='ignore')
            for match in pattern.finditer(text):
                found.append('%s: %s' % (path.relative_to(repo), match.group(0)))
    for name in ('pyproject.toml', 'setup.py', 'MANIFEST.in', 'README.md'):
        if (repo / name).is_file():
            found += ['%s: %s' % (name, m.group(0)) for m in either.finditer((repo / name).read_text())]
    assert found == [], found[:20]
