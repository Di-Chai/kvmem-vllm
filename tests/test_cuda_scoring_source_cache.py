"""CPU-only checks of immutable MeanK build inputs; load is always mocked."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from vllm_kvmem import cuda_scoring as scoring


CONTENTS = {
    'qw3_mean_k.cu': b'#include "qw3_mean_k_kernel.cuh"\n',
    'qw3_mean_k_kernel.cuh': b'#include "qw3_mean_k_tiled.cuh"\n',
    'qw3_mean_k_tiled.cuh': b'// original tiled implementation\n',
}


def source(directory, contents=CONTENTS):
    directory.mkdir(parents=True)
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    return directory


def file_state(directory):
    return {p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
            for p in directory.iterdir()}


def test_identical_frozen_paths_use_same_inputs_without_touching_mtime(tmp_path):
    first = source(tmp_path / 'frozen-a' / 'csrc')
    second = source(tmp_path / 'frozen-b' / 'csrc')
    build = tmp_path / 'build'
    result = scoring._prepare_build_sources(first, build)
    directory, cu, _ = result
    assert cu.is_relative_to(directory)
    assert {p.name: p.read_bytes() for p in cu.parent.iterdir()} == CONTENTS
    before = file_state(cu.parent)
    assert scoring._prepare_build_sources(second, build) == result
    assert file_state(cu.parent) == before


@pytest.mark.parametrize('changed_name', tuple(CONTENTS))
def test_changing_any_source_or_header_gets_an_isolated_build(tmp_path, changed_name):
    original = source(tmp_path / 'original')
    changed = source(tmp_path / 'changed', {**CONTENTS, changed_name: CONTENTS[changed_name] + b'// change\n'})
    old = scoring._prepare_build_sources(original, tmp_path / 'build')
    state = file_state(old[1].parent)
    new = scoring._prepare_build_sources(changed, tmp_path / 'build')
    assert all(a != b for a, b in zip(old, new))
    assert file_state(old[1].parent) == state
    assert (new[1].parent / changed_name).read_bytes().endswith(b'// change\n')


def test_hash_preserves_file_boundaries(tmp_path):
    names = tuple(CONTENTS)
    a = source(tmp_path / 'a', dict(zip(names, (b'a', b'bc', b'd'))))
    b = source(tmp_path / 'b', dict(zip(names, (b'ab', b'c', b'd'))))
    assert scoring._prepare_build_sources(a, tmp_path / 'build') != scoring._prepare_build_sources(b, tmp_path / 'build')


def test_original_bytes_are_read_once_for_hash_and_publication(tmp_path, monkeypatch):
    original = source(tmp_path / 'source')
    real_read = Path.read_bytes
    reads = []

    def read_once(path):
        data = real_read(path)
        if path.parent == original:
            reads.append(path.name)
            path.write_bytes(b'changed after snapshot\n')
        return data

    monkeypatch.setattr(Path, 'read_bytes', read_once)
    result = scoring._prepare_build_sources(original, tmp_path / 'build')
    assert reads == list(CONTENTS)
    assert {p.name: p.read_bytes() for p in result[1].parent.iterdir()} == CONTENTS
    clean = source(tmp_path / 'clean')
    assert scoring._prepare_build_sources(clean, tmp_path / 'build') == result


def test_concurrent_publish_keeps_complete_files_and_first_publication_mtime(tmp_path, monkeypatch):
    original = source(tmp_path / 'source')
    barrier = threading.Barrier(8)
    real_link = scoring.os.link
    published = []

    def racing_link(temporary, final):
        if final.name == 'qw3_mean_k.cu':
            barrier.wait(timeout=10)
        real_link(temporary, final)
        published.append((final, final.stat().st_ino, final.stat().st_mtime_ns))

    monkeypatch.setattr(scoring.os, 'link', racing_link)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: scoring._prepare_build_sources(original, tmp_path / 'build'), range(8)))
    assert all(result == results[0] for result in results)
    assert len(published) == 3
    for path, inode, mtime in published:
        assert path.read_bytes() == CONTENTS[path.name]
        assert (path.stat().st_ino, path.stat().st_mtime_ns) == (inode, mtime)
    assert {p.name for p in results[0][1].parent.iterdir()} == set(CONTENTS)


@pytest.mark.parametrize('failure_stage', ['fsync', 'link'])
def test_failed_publication_never_exposes_partial_source_and_can_retry(tmp_path, monkeypatch, failure_stage):
    original = source(tmp_path / 'source')
    real_operation = getattr(scoring.os, failure_stage)

    def fail(*args, **kwargs):
        raise OSError('injected publication failure')

    monkeypatch.setattr(scoring.os, failure_stage, fail)
    with pytest.raises(OSError, match='injected publication failure'):
        scoring._prepare_build_sources(original, tmp_path / 'build')
    assert not [p for p in (tmp_path / 'build').rglob('*') if p.is_file()]
    monkeypatch.setattr(scoring.os, failure_stage, real_operation)
    _, cu, _ = scoring._prepare_build_sources(original, tmp_path / 'build')
    assert {p.name: p.read_bytes() for p in cu.parent.iterdir()} == CONTENTS


def test_missing_header_fails_before_publishing_snapshot(tmp_path):
    original = source(tmp_path / 'source')
    (original / 'qw3_mean_k_tiled.cuh').unlink()
    with pytest.raises(FileNotFoundError):
        scoring._prepare_build_sources(original, tmp_path / 'build')
    assert not (tmp_path / 'build').exists()


def test_corrupted_existing_source_fails_closed_without_rewriting(tmp_path):
    original = source(tmp_path / 'source')
    _, cu, _ = scoring._prepare_build_sources(original, tmp_path / 'build')
    header = cu.parent / 'qw3_mean_k_kernel.cuh'
    header.write_bytes(b'corrupted')
    before = file_state(cu.parent)
    with pytest.raises(RuntimeError, match='cached source content differs'):
        scoring._prepare_build_sources(original, tmp_path / 'build')
    assert file_state(cu.parent) == before


@pytest.mark.parametrize('packaged_sources', [True, False])
def test_kernel_uses_stable_sources_existing_flags_build_root_and_lru(tmp_path, monkeypatch, packaged_sources):
    loads = []
    module = object()

    def load(**kwargs):
        loads.append(kwargs)
        return module

    monkeypatch.setitem(sys.modules, 'torch.utils.cpp_extension', SimpleNamespace(load=load))
    monkeypatch.setattr(scoring.shutil, 'which', lambda name: '/fake/ninja')
    monkeypatch.setenv('KVMEM_CUDA_BUILD_DIR', str(tmp_path / 'selected-build-root'))
    monkeypatch.setenv('MAX_JOBS', '3')
    scoring.kernel.cache_clear()
    try:
        for name in ('frozen-a', 'frozen-b'):
            module_path = tmp_path / name / 'src/vllm_kvmem/cuda_scoring.py'
            directory = module_path.parent / 'cuda_sources' if packaged_sources else module_path.parents[2] / 'csrc'
            source(directory)
            monkeypatch.setattr(scoring, '__file__', str(module_path))
            scoring.kernel.cache_clear()
            assert scoring.kernel() is module
            assert scoring.kernel() is module
        assert len(loads) == 2
        assert loads[0] == loads[1]
        call = loads[0]
        assert call['extra_cuda_cflags'] == ['-O3'] and call['verbose'] is False
        assert call['name'] == Path(call['build_directory']).name
        assert Path(call['build_directory']).parent == tmp_path / 'selected-build-root'
        assert len(call['sources']) == 1
        cu = Path(call['sources'][0])
        assert cu.is_absolute() and cu.is_relative_to(Path(call['build_directory']))
        assert {p.name: p.read_bytes() for p in cu.parent.iterdir()} == CONTENTS
        assert scoring.os.environ['MAX_JOBS'] == '3'
    finally:
        scoring.kernel.cache_clear()
