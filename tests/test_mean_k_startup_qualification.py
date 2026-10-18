"""Qualification preparation and proof validation only; no Torch/GPU/compiler."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('mean_k_qualification',
    Path(__file__).resolve().parents[1] / 'scripts/qualify_mean_k_startup.py')
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    for name in qualification.PATCH_FILES:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# candidate ' + name)
    for name in ('__init__.py', 'retrieval.py'):
        (workspace / 'src/vllm_kvmem' / name).write_text('# original ' + name)
    for name in qualification.CUDA_FILES:
        path = workspace / 'csrc' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('// original ' + name)
    (workspace / 'tests' / qualification.ORACLE).write_text('# original CUDA oracle')
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'candidate.patch').write_text('candidate patch')
    qualification.save(candidate / 'manifest.json', dict(status='cpu_verified_gpu_pending',
        patch_sha256=qualification.sha(candidate / 'candidate.patch'), cpu_validation={'passed': 25},
        files={name: {'after_sha256': qualification.sha(workspace / name)} for name in qualification.PATCH_FILES}))

    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation must not spawn processes or inspect GPU')

    monkeypatch.setattr(qualification.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(qualification, 'gpu_snapshot', forbidden)
    output = tmp_path / 'qualification'
    manifest = qualification.prepare(output, candidate, workspace)
    return output, candidate, workspace, manifest


def test_prepare_freezes_two_distinct_identical_sources_without_building(prepared):
    output, candidate, workspace, manifest = prepared
    assert qualification.verify(output, frozen_entry=False) == manifest
    assert qualification.read(output / 'status.json')['status'] == 'prepared'
    assert not list(output.glob('build-*'))
    a, b = (output / 'snapshots' / label / 'vllm_kvmem' for label in ('a', 'b'))
    assert a != b and qualification.inventory(a) == qualification.inventory(b)
    assert set(qualification.inventory(a)) == set(qualification.PYTHON_FILES) | {
        'cuda_sources/' + name for name in qualification.CUDA_FILES}
    assert qualification.sha(output / 'oracle' / qualification.ORACLE) == qualification.sha(workspace / 'tests' / qualification.ORACLE)
    with pytest.raises(ValueError, match='frozen entry point'):
        qualification.verify(output)


@pytest.mark.parametrize('relative', [
    'snapshots/a/vllm_kvmem/cuda_scoring.py',
    'snapshots/b/vllm_kvmem/cuda_sources/qw3_mean_k_tiled.cuh',
    'oracle/test_original_cuda_scoring.py', 'provenance/candidate.patch',
])
def test_changed_frozen_provenance_is_rejected(prepared, relative):
    output, _, _, _ = prepared
    (output / relative).write_text('changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        qualification.verify(output, frozen_entry=False)


def test_missing_header_is_rejected(prepared):
    output, _, _, _ = prepared
    (output / 'snapshots/a/vllm_kvmem/cuda_sources/qw3_mean_k_kernel.cuh').unlink()
    with pytest.raises(ValueError, match='Frozen input changed'):
        qualification.verify(output, frozen_entry=False)


@pytest.mark.parametrize('fault', ['patch', 'source', 'missing_cpu', 'scope'])
def test_prepare_rejects_unverified_candidate_before_creating_output(prepared, fault):
    output, candidate, workspace, _ = prepared
    if fault == 'patch':
        (candidate / 'candidate.patch').write_text('different patch')
    elif fault == 'source':
        (workspace / 'src/vllm_kvmem/cuda_scoring.py').write_text('different loader')
    else:
        manifest = qualification.read(candidate / 'manifest.json')
        if fault == 'missing_cpu':
            manifest.pop('cpu_validation')
        else:
            manifest['files'].pop('src/vllm_kvmem/runner.py')
        qualification.save(candidate / 'manifest.json', manifest)
    next_output = output.parent / 'next'
    with pytest.raises(ValueError):
        qualification.prepare(next_output, candidate, workspace)
    assert not next_output.exists()


def fake_build(tmp_path):
    build = tmp_path / 'build'
    directory = build / 'kvmem_meank_key'
    inputs = directory / 'sources/key'
    inputs.mkdir(parents=True)
    for name in qualification.CUDA_FILES:
        (inputs / name).write_text('// ' + name)
    (directory / 'build.ninja').write_text('build kernel.o: cuda_compile ' + str(inputs / 'qw3_mean_k.cu') + '\n')
    (directory / 'kernel.o').write_bytes(b'complete object')
    (directory / 'kernel.so').write_bytes(b'complete library')
    return build, directory


def test_cache_proof_requires_exact_object_library_and_source_identity(tmp_path):
    build, directory = fake_build(tmp_path)
    first = qualification.build_evidence(build)
    qualification.unchanged_build(first, qualification.build_evidence(build))
    for group in ('artifacts', 'inputs'):
        for field in ('sha256', 'inode', 'mtime_ns'):
            changed = copy.deepcopy(first)
            item = next(iter(changed[group].values()))
            item[field] = 'changed'
            with pytest.raises(ValueError, match='Second process changed'):
                qualification.unchanged_build(first, changed)
    changed = copy.deepcopy(first)
    changed['ninja']['sha256'] = 'other compile input or flags'
    with pytest.raises(ValueError, match='Second process changed'):
        qualification.unchanged_build(first, changed)


@pytest.mark.parametrize('removed', ['kernel.o', 'kernel.so', 'build.ninja', 'sources/key/qw3_mean_k_tiled.cuh'])
def test_missing_build_evidence_cannot_pass(tmp_path, removed):
    build, directory = fake_build(tmp_path)
    (directory / removed).unlink()
    with pytest.raises(ValueError):
        qualification.build_evidence(build)


@pytest.mark.parametrize('tests,failures,errors,skipped,passes', [
    (9, 0, 0, 0, True), (9, 0, 0, 9, False), (9, 1, 0, 0, False),
    (9, 0, 1, 0, False), (8, 0, 0, 0, False), (0, 0, 0, 0, False),
])
def test_numerical_oracle_requires_all_nine_real_tests_without_skips(tmp_path, tests, failures, errors, skipped, passes):
    path = tmp_path / 'oracle.xml'
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}">' +
                    ''.join(f'<testcase name="case{i}" />' for i in range(tests)) + '</testsuite></testsuites>')
    if passes:
        assert qualification.numerical_evidence(path)['tests'] == 9
    else:
        with pytest.raises(ValueError):
            qualification.numerical_evidence(path)
