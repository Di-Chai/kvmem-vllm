"""KVMEM_RUNNER_ARMED_LATCH: the probe 'armed' marker is only ever created, so a
latched runner skips later stat() calls; without the flag every check stats."""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from vllm_kvmem.runner import _probe_armed


def runner(tmp_path):
    return SimpleNamespace(_kvmem_probe_output=tmp_path)


def test_default_checks_the_marker_every_time(tmp_path, monkeypatch):
    monkeypatch.delenv('KVMEM_RUNNER_ARMED_LATCH', raising=False)
    r = runner(tmp_path)
    assert not _probe_armed(r)
    (tmp_path / 'armed').touch()
    assert _probe_armed(r)
    (tmp_path / 'armed').unlink()                 # unchanged semantics: re-stat each call
    assert not _probe_armed(r)


def test_latch_keeps_armed_once_seen(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_RUNNER_ARMED_LATCH', '1')
    r = runner(tmp_path)
    assert not _probe_armed(r) and not getattr(r, '_kvmem_armed_latched', False)
    (tmp_path / 'armed').touch()
    assert _probe_armed(r) and r._kvmem_armed_latched
    (tmp_path / 'armed').unlink()                 # never happens in practice; latched anyway
    assert _probe_armed(r)


def test_latch_value_is_validated(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_RUNNER_ARMED_LATCH', 'yes')
    (tmp_path / 'armed').touch()
    with pytest.raises(ValueError, match='KVMEM_RUNNER_ARMED_LATCH'):
        _probe_armed(runner(tmp_path))
