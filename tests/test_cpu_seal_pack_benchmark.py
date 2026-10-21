"""Small parameter/freeze checks; no archive work, numeric imports or performance."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/benchmark_cpu_seal_pack.py'
spec = importlib.util.spec_from_file_location('cpu_seal_pack_benchmark', PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fake_source(root):
    package = root / 'src/vllm_kvmem'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'storage.py').write_text('original source')
    (package / '_native.test.so').write_bytes(b'compiled-byte-proof')
    for name in ('csrc/bindings.cpp', 'tests/test_cpu_seal_pack.py',
                 'tests/test_cpu_seal_hash.py', 'tests/test_cpu_direct_record_seal.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return package


def test_geometry_and_counterbalanced_pairs_are_exact():
    assert bench.RECORD_BYTES == 1_114_112
    assert bench.validate_counts((32, 64)) == (32, 64)
    assert tuple(mode for pair in bench.abba_pairs(12) for mode in pair) == (1, 8, 8, 1) * 6


@pytest.mark.parametrize('pairs', [False, 0, 10, 13])
def test_reject_insufficient_or_unbalanced_samples(pairs):
    with pytest.raises(ValueError, match='12 matched'):
        bench.abba_pairs(pairs)


def test_freeze_copies_exact_native_library_and_rejects_tamper(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    package = fake_source(source)
    root = tmp_path / 'frozen'
    manifest = bench.prepare(root, source)
    assert manifest['native_binary'] == 'runtime/vllm_kvmem/_native.test.so'
    assert manifest['fixed_options']['cpu_seal_hash_workers'] == 8
    assert manifest['fixed_options']['gather_workers'] == 8
    assert manifest['geometry']['records'] == [32, 64]
    monkeypatch.setattr(bench, '__file__', str(root / 'benchmark_cpu_seal_pack.py'))
    (package / '_native.test.so').write_bytes(b'new-live-library')
    assert bench.verify_sources(root) == manifest
    (root / manifest['native_binary']).write_bytes(b'changed-frozen-library')
    with pytest.raises(ValueError, match='changed or missing'):
        bench.verify_sources(root)


def test_missing_binary_is_rejected_before_creating_experiment(tmp_path):
    source = tmp_path / 'source'
    package = fake_source(source)
    (package / '_native.test.so').unlink()
    with pytest.raises(ValueError, match='exactly one compiled'):
        bench.prepare(tmp_path / 'frozen', source)
    assert not (tmp_path / 'frozen').exists()


def test_prepare_has_no_torch_numpy_or_native_import(tmp_path):
    source = tmp_path / 'source'
    fake_source(source)
    code = '''import runpy,sys
from pathlib import Path
module=runpy.run_path(sys.argv[1],run_name='prepare_only')
module['prepare'](Path(sys.argv[2]),Path(sys.argv[3]))
assert not any(name in sys.modules for name in ('torch','numpy','vllm_kvmem._native'))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(PATH),
        str(tmp_path / 'frozen'), str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpu_work_requires_explicit_run_coordination(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(PATH), 'run', '--directory', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'external performance-job coordination' in result.stderr
