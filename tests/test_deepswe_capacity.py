"""Retain original conservative slot provisioning while cleanup releases it."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'deepswe_capacity_runner', Path(__file__).resolve().parents[1] / 'scripts/benchmark_deepswe_first16.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_slot_budget_is_preserved_with_explicit_namespace_release():
    profile = {'kvmem': {'cpu_history_coordination': True, 'session_capacity': 17}}
    with pytest.raises(ValueError, match='16 tasks \\+ 10 preflight'):
        runner.validate_session_capacity(profile, 16)
    profile['kvmem']['session_capacity'] = 25
    with pytest.raises(ValueError, match='at least 26'):
        runner.validate_session_capacity(profile, 16)
    profile['kvmem']['session_capacity'] = 26
    assert runner.validate_session_capacity(profile, 16)['required'] == 26
    assert runner.validate_session_capacity(profile, 16)['cleanup_retains_namespace'] is False


def test_legacy_lru_mode_does_not_require_all_finished_namespaces():
    profile = {'kvmem': {'cpu_history_coordination': False, 'session_capacity': 17}}
    assert runner.validate_session_capacity(profile, 16)['configured'] == 17


def test_128k_profile_qualifies_four_full_sized_contexts():
    profile = {'max_num_seqs': 4, 'kvmem': {'cpu_history_coordination': True,
               'session_capacity': 22, 'select_budget': 131072}}
    assert runner.preflight_shape(profile) == (4, 141870, 145842)
    assert runner.validate_session_capacity(profile, 16)['required'] == 22
