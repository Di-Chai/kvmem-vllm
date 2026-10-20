"""Experimental DFlash planning never silently changes supported deployment modes."""
import json
import pytest
from test_serving_launcher import nvfp4_profile
from vllm_kvmem.serve import build_launch


def configured(tmp_path):
    cfg=nvfp4_profile(tmp_path)
    cfg.update(draft_method='dflash',draft_tokens=7,execution_mode='full_graph',
        experimental_dflash_query_planning=True,prefill_chunk_tokens=1024)
    cfg['kvmem'].update(block_tokens=32,select_budget=4096,gen_budget=4096,
        repository_blocks=2048,recent_blocks=64,request_planning=True,
        session_checkpoint_policy='pm',session_capture_terminal=True,
        action_attention=True,prefill_attention=True,action_fraction=0.,
        action_scope='non-thinking',graph_mode='multi',graph_scoring_version='v2',graph_alpha=.2,
        pressure_policy='graph')
    return cfg


def test_experimental_query_plan_retains_native_draft_cache_and_full_graphs(tmp_path):
    cfg=configured(tmp_path);plan=build_launch(cfg,tmp_path,tmp_path/'run')
    argv=plan['vllm_argv'];spec=json.loads(argv[argv.index('--speculative-config')+1])
    assert spec['method']=='dflash' and spec['num_speculative_tokens']==7
    assert spec['kv_cache_dtype']=='bfloat16'
    assert json.loads(argv[argv.index('--compilation-config')+1])=={'mode':0,'cudagraph_mode':'FULL'}
    assert plan['profile']['kvmem']['pressure_policy']=='graph'
    assert 'KVMEM_REQUEST_PLAN_CONFIG' in plan['environment']
    assert build_launch(plan['profile'],tmp_path,tmp_path/'replay')['vllm_argv']==argv


@pytest.mark.parametrize('change,match',[
    ({'experimental_dflash_query_planning':False},'explicit experimental'),
    ({'experimental_dflash_query_planning':1},'must be boolean'),
    ({'execution_mode':'compiled_graph'},'native safetensors Qwen MTP'),
    ({'execution_mode':'eager'},'FULL graphs'),
    ({'draft_tokens':3},'k=7'),
    ({'async_refresh':True},'synchronous'),
    ({'guided_query_tokens':32},'no guided Query'),
    ({'recent_blocks':63},'recent band'),
])
def test_unqualified_modes_remain_rejected(tmp_path,change,match):
    cfg=configured(tmp_path)
    for key,value in change.items():
        (cfg['kvmem'] if key in cfg['kvmem'] or key in ('async_refresh','guided_query_tokens') else cfg)[key]=value
    with pytest.raises(ValueError,match=match):build_launch(cfg,tmp_path,tmp_path/'bad')
