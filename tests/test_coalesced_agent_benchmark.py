import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('coalesced_bench',Path(__file__).parents[1]/'scripts/benchmark_coalesced_agents.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def profile():
    return json.loads((Path(__file__).parents[1]/'examples/qwen-nvfp4-offline-agent-k128-g128.json').read_text())


def test_comparison_changes_only_coalescing_and_keeps_attention_native_mtp_and_rpc_settings():
    original=profile();before=copy.deepcopy(original)
    p=bench.coalesced_profiles(original,4096);a,b=p['A0'],p['P1']
    assert original==before
    assert a['kvmem'].pop('offline_prefill_coalescing') is False
    assert b['kvmem'].pop('offline_prefill_coalescing') is True
    assert a==b and a['kvmem']['offline_agent_policy'] and not a['kvmem']['offline_phase_policy']
    assert a['kvmem']['step_latency_interval']==1
    assert a['kvmem']['action_attention'] and a['kvmem']['prefill_attention']
    assert a['draft_tokens']==4 and a['max_num_seqs']==4


def test_frozen_validation_rejects_hidden_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,'ROOT',tmp_path/'frozen')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(comparison_kind='offline_prefill_coalescing')))
    p=bench.coalesced_profiles(profile(),4096)
    for a,v in p.items():(tmp_path/(a+'.json')).write_text(json.dumps(v))
    bench.validate_prepared(tmp_path)
    p['P1']['draft_tokens']=3;(tmp_path/'P1.json').write_text(json.dumps(p['P1']))
    with pytest.raises(ValueError,match='Only'):bench.validate_prepared(tmp_path)
