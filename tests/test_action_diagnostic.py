"""Diagnostic isolation gates; no model or CUDA execution."""
import asyncio
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1]/'scripts/diagnose_action_capture.py'
spec = importlib.util.spec_from_file_location('action_diagnostic_test', path)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


def test_observation_profile_preserves_model_sampling_and_memory_controls():
    reference = dict(port=8000, verify_session_restore=False, model='unchanged',
        max_num_seqs=8, max_model_len=262144, gpu_memory_utilization=.81,
        kvmem=dict(action_attention=True, rolling_active_window=True,
                   select_budget=65536, gen_budget=32768, cpu_read_cache_bytes=100_000_000_000))
    before = copy.deepcopy(reference)
    actual = diag.diagnostic_profile(reference, 18540)
    assert reference == before
    actual['port'] = 8000
    assert actual['kvmem'].pop('action_profiling') is True
    assert actual['kvmem'].pop('action_profile_interval') == 8
    assert actual['kvmem'].pop('action_profile_max_events') == 4096
    assert actual == reference
    for overrides in ({'verify_session_restore':True}, {'port':True}):
        bad = dict(reference, **overrides)
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            diag.diagnostic_profile(bad, bad['port'])


def overlay_fixture(tmp_path):
    base, overlay = tmp_path/'base', tmp_path/'overlay'
    rows = []
    for name in sorted(diag.FILES):
        (base/name).parent.mkdir(parents=True, exist_ok=True)
        (overlay/name).parent.mkdir(parents=True, exist_ok=True)
        if not name.endswith('/action_profile.py'):
            (base/name).write_text('original\n')
        (overlay/name).write_text('diagnostic only\n')
        rows.append(dict(path=name, base_sha256=diag.cap.sha(base/name) if (base/name).exists() else None,
                         overlay_sha256=diag.cap.sha(overlay/name)))
    return base, overlay, dict(files=rows)


def test_overlay_cannot_import_unqualified_extra_files_or_wrong_parent(tmp_path):
    base, overlay, manifest = overlay_fixture(tmp_path)
    expected = diag.overlay_contract(base, overlay, manifest)
    assert set(expected) == diag.FILES
    extra = copy.deepcopy(manifest)
    extra['files'].append(dict(path='src/vllm_kvmem/scheduler.py', base_sha256=None, overlay_sha256='bad'))
    with pytest.raises((ValueError, RuntimeError, AssertionError), match='five reviewed'):
        diag.overlay_contract(base, overlay, extra)
    (base/'src/vllm_kvmem/options.py').write_text('concurrent unrelated mutation\n')
    with pytest.raises((ValueError, RuntimeError, AssertionError), match='base runtime differs'):
        diag.overlay_contract(base, overlay, manifest)


def test_overlay_hash_change_is_rejected(tmp_path):
    base, overlay, manifest = overlay_fixture(tmp_path)
    (overlay/'src/vllm_kvmem/action_profile.py').write_text('mutated after review\n')
    with pytest.raises((ValueError, RuntimeError, AssertionError), match='overlay changed'):
        diag.overlay_contract(base, overlay, manifest)


def test_failed_validation_never_reaches_gpu_or_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, 'ROOT', tmp_path/'frozen')
    def invalid(_):
        raise ValueError('qualification or runtime changed')
    monkeypatch.setattr(diag, 'validate', invalid)
    monkeypatch.setattr(diag.ready.bench.q, 'gpu_state', lambda:pytest.fail('Touched GPU before qualification'))
    monkeypatch.setattr(diag.fcntl, 'flock', lambda *_:pytest.fail('Acquired run lock before validation'))
    with pytest.raises(ValueError, match='qualification or runtime changed'):
        asyncio.run(diag.run(SimpleNamespace(output=tmp_path)))
