import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('phased_benchmark',
    Path(__file__).parents[1]/'scripts/benchmark_phased_agents.py')
phased = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phased)


def profile():
    return json.loads((Path(__file__).parents[1]/
        'examples/qwen-nvfp4-offline-agent-k128-g128.json').read_text())


def test_phase_comparison_changes_only_the_gate_and_preserves_attention_and_rpc_contract():
    original = profile()
    before = copy.deepcopy(original)
    profiles = phased.phase_profiles(original, 8192, 1000.)
    assert original == before
    baseline, candidate = profiles['A0'], profiles['P1']
    assert baseline['kvmem']['offline_agent_policy'] and candidate['kvmem']['offline_agent_policy']
    assert baseline['kvmem'].pop('offline_phase_policy') is False
    assert candidate['kvmem'].pop('offline_phase_policy') is True
    assert baseline == candidate
    assert candidate['kvmem']['action_attention'] and candidate['kvmem']['prefill_attention']
    assert candidate['draft_tokens'] == 4 and candidate['max_num_seqs'] == 4


def test_phase_frozen_validation_rejects_hidden_policy_or_model_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(phased, 'ROOT', tmp_path/'frozen')
    (tmp_path/'manifest.json').write_text(json.dumps({'comparison_kind':'public_phases'}))
    profiles = phased.phase_profiles(profile(), 8192, 1000.)
    for arm, value in profiles.items():
        (tmp_path/(arm+'.json')).write_text(json.dumps(value))
    phased.validate_prepared(tmp_path)
    profiles['P1']['draft_tokens'] = 3
    (tmp_path/'P1.json').write_text(json.dumps(profiles['P1']))
    with pytest.raises(ValueError, match='only'):
        phased.validate_prepared(tmp_path)
