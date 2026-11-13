"""KVMEM_GIL_SWITCH_INTERVAL_US: opt-in, validated, applied to the interpreter."""
import sys

import pytest

from vllm_kvmem.serving import gil_switch


@pytest.fixture
def restore_interval():
    before = sys.getswitchinterval()
    yield
    sys.setswitchinterval(before)


def test_unset_keeps_the_interpreter_default(monkeypatch, restore_interval):
    monkeypatch.delenv('KVMEM_GIL_SWITCH_INTERVAL_US', raising=False)
    before = sys.getswitchinterval()
    assert gil_switch.apply() is None and sys.getswitchinterval() == before


def test_configured_interval_is_applied(monkeypatch, restore_interval):
    monkeypatch.setenv('KVMEM_GIL_SWITCH_INTERVAL_US', '1000')
    assert gil_switch.apply() == pytest.approx(.001)
    assert sys.getswitchinterval() == pytest.approx(.001)


@pytest.mark.parametrize('bad', ['0', '49', '100001', '1e3', '-5', ' 1000'])
def test_invalid_values_are_rejected(monkeypatch, bad):
    monkeypatch.setenv('KVMEM_GIL_SWITCH_INTERVAL_US', bad)
    with pytest.raises(ValueError):
        gil_switch.configured()


def test_diagnostic_gil_load_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_DIAG_GIL_LOAD', raising=False)
    assert gil_switch.load_config() is None
    monkeypatch.setenv('KVMEM_DIAG_GIL_LOAD', '0.1,1')
    assert gil_switch.load_config() == (0.1, 0.001)
    for bad in ('0,1', '1,1', '0.1,0', '0.1', 'x,y', '0.1,500'):
        monkeypatch.setenv('KVMEM_DIAG_GIL_LOAD', bad)
        with pytest.raises(ValueError):
            gil_switch.load_config()


def test_diagnostic_thread_names_reach_the_os(monkeypatch):
    import threading
    monkeypatch.setenv('KVMEM_DIAG_THREAD_NAMES', '1')
    original = threading.Thread._bootstrap_inner
    try:
        assert gil_switch.name_threads() is True
        seen = []
        def read():
            with open(f'/proc/self/task/{threading.get_native_id()}/comm') as f:
                seen.append(f.read().strip())
        t = threading.Thread(target=read, name='kvmem-refresh_0')
        t.start(); t.join()
        assert seen == ['kvmem-refresh_0']
    finally:
        threading.Thread._bootstrap_inner = original
        threading.Thread._kvmem_named = False
