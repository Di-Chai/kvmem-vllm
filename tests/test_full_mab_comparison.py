"""Guard the full-input comparison against budget/method and provenance drift."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from vllm_kvmem.serving.full_history_benchmark import validate_full_history
from vllm_kvmem.serving.serve import build_launch
from run_full_mab_comparison import method_profile, audit_global, load_source, BASE, read, explicit_execution


def setup_capacity(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(model_type='qwen3_5_text',
        layer_types=['full_attention']*16,num_key_value_heads=4,head_dim=256)))
    cfg=dict(model_format='safetensors',draft_method='mtp',max_model_len=2097152,hf_config_path=str(tmp_path))
    options=NS(memory=NS(select_budget=36864,gen_budget=16384,repository_blocks=67584,block_tokens=32),
        request_planning=True,archive_backend='cpu',cpu_history_coordination=True,cpu_progress_reserve=True,
        session_freeze_prompt=True,session_capacity=8,historical_gc=None,repository_gc=False,
        rolling_active_window=False,cpu_read_cache_bytes=112000000000)
    return cfg,options


def test_finite_input_capacity_and_gc_are_independent(tmp_path):
    cfg,options=setup_capacity(tmp_path)
    evidence=validate_full_history(cfg,options)
    assert evidence['full_history_bytes']==(2097152+32768)*17*4*256*2
    options.memory.repository_blocks=32768
    with pytest.raises(ValueError,match='complete input'):validate_full_history(cfg,options)


@pytest.mark.parametrize('key,value', [('historical_gc',object()),('rolling_active_window',True),
    ('session_freeze_prompt',False),('cpu_read_cache_bytes',20000000000)])
def test_full_input_mode_refuses_lossy_or_underprovisioned_history(tmp_path,key,value):
    cfg,options=setup_capacity(tmp_path);setattr(options,key,value)
    with pytest.raises(ValueError):validate_full_history(cfg,options)


def test_paired_launch_controls_and_opt_in_guard(tmp_path):
    source=read(BASE/'profile.json');before=deepcopy(source)
    profiles=[method_profile(source,arm) for arm in ('mean_k','layerwise')]
    for profile in profiles:
        launch=build_launch(profile,BASE,tmp_path/'run')
        assert launch['environment']['KVMEM_LOGICAL_KV_CAPACITY']=='1'
        assert [profile['kvmem'][k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')]==[36864,16384,32,41,0]
        profile['full_history_benchmark']=False
        with pytest.raises(ValueError,match='above 1M'):build_launch(profile,BASE,tmp_path/'invalid')
    assert source==before
    assert not profiles[0]['kvmem']['action_attention'] and profiles[1]['kvmem']['prefill_attention']


def test_mean_k_audit_rejects_accidental_attention_fallback():
    row=dict(query_scoring_method='mean_k',action_attention_enabled=False,attention_source='none',
        prefill_attention_samples=0,selected_tokens=36864,selection='pressure')
    assert audit_global(row,[row],[],dict(select_budget=36864))==[row]
    bad=dict(row,action_attention_enabled=True)
    with pytest.raises(ValueError,match='accumulated'):audit_global(bad,[bad],[],dict(select_budget=36864))
    bad=dict(row,selected_tokens=36896)
    with pytest.raises(ValueError,match='budget'):audit_global(bad,[bad],[],dict(select_budget=36864))


def test_optional_execution_preserves_legacy_fixture_and_rejects_budget_changes():
    assert explicit_execution(dict(arm='layerwise'))=={}
    controls=dict(KVMEM_LAYERWISE_PRESSURE_PREPARED_COMMIT='1',
                  KVMEM_LAYERWISE_PRESSURE_PIPELINE_PROGRESS='1')
    fixture=dict(arm='layerwise',execution_environment=controls)
    assert explicit_execution(fixture)==controls
    with pytest.raises(ValueError,match='Undeclared'):
        explicit_execution(dict(arm='layerwise',execution_environment={'KVMEM_SELECT_BUDGET':'16384'}))
    with pytest.raises(ValueError,match='Pure Mean-K'):
        explicit_execution(dict(arm='mean_k',execution_environment=controls))


def test_full_history_native_planner_uses_bounded_window_without_changing_logical_limit(tmp_path,monkeypatch):
    from test_logical_capacity import config,specs
    from vllm_kvmem.serving.logical_capacity import physical_config,wrap_planner
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    cfg=config(tmp_path)
    cfg.model_config.max_model_len=cfg.model_config.original_max_model_len=2097152
    cfg.model_config.hf_text_config=NS(model_type='qwen3_5_text',layer_types=['full_attention']*16,
        num_key_value_heads=4,head_dim=256)
    cfg.kv_transfer_config.kv_connector_extra_config['kvmem']=method_profile(read(BASE/'profile.json'),'mean_k')['kvmem']
    monkeypatch.delenv('KVMEM_RECALL_METHOD',raising=False)
    monkeypatch.delenv('KVMEM_FULL_HISTORY_BENCHMARK',raising=False)
    with pytest.raises(ValueError,match='Unqualified bounded'):physical_config(cfg)
    monkeypatch.setenv('KVMEM_FULL_HISTORY_BENCHMARK','1')
    bounded=physical_config(cfg)
    assert bounded.model_config.max_model_len==36864+16384
    assert cfg.model_config.max_model_len==2097152
    planner=wrap_planner(get_kv_cache_configs)
    result=planner(cfg,specs(),[40*2**30])
    assert result
    with pytest.raises(ValueError,match='cannot fit 8'):
        planner(cfg,specs(),[int(2.97*2**30)])
    cfg.kv_transfer_config.kv_connector_extra_config['kvmem']['repository_blocks']=32768
    with pytest.raises(ValueError,match='complete input'):physical_config(cfg)
