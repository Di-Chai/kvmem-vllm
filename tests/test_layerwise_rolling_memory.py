from copy import deepcopy
from pathlib import Path

import pytest

from run_layerwise_memory import memory_generation_capacity_audit, memory_pressure_audit, server_command
from run_layerwise_rolling_memory import rolling_profile, rolling_fixture


def test_actual_native_batch_change_is_rejected_before_the_first_answer():
    from run_layerwise_rolling_memory import matched_native_arguments
    before=['--max-num-batched-tokens','16424','--kv-transfer-config','old-budgets']
    after=['--max-num-batched-tokens','16424','--kv-transfer-config','authorized-new-budgets']
    matched_native_arguments(before,after)
    after[1]='4136'
    with pytest.raises(ValueError,match='batching'):matched_native_arguments(before,after)


def test_rolling_keeps_original_sampling_concurrency_active_and_output_caps():
    profile=dict(model='frozen-nvfp4',max_num_seqs=8,draft_tokens=4,query_replay_budget=16384,
        verify_session_restore=False,kvmem=dict(select_budget=16384,gen_budget=16384,
        rolling_active_window=False,sink_blocks=41,block_tokens=32))
    original=deepcopy(profile)
    changed=rolling_profile(profile)
    assert profile==original
    changed['kvmem']['gen_budget']=16384
    changed['kvmem']['rolling_active_window']=False
    changed['query_replay_budget']=16384
    assert changed.pop('request_planning_batch_tokens')==16424
    assert changed==original
    fixture=dict(rows=[dict(id='first',input_tokens=108368)],
        bindings=[dict(max_tokens=16384,thinking_token_budget=4096,temperature=.6)],
        execution_flags=dict(layerwise_prefill_pressure=True,layerwise_buffered_history=True))
    new=rolling_fixture(fixture)
    assert new['rows']==fixture['rows'] and new['bindings']==fixture['bindings']
    assert '--layerwise-rolling-decode' in server_command(Path('/new'),new)
    assert '--layerwise-rolling-decode' not in server_command(Path('/old'),fixture)


def test_rolling_audit_allows_long_output_without_changing_the_limit():
    kv=dict(select_budget=16384,gen_budget=4096,block_tokens=32,rolling_active_window=True)
    result=memory_generation_capacity_audit(dict(history_begin=108320),
        [dict(charged_tokens=16384)]*16,108368,16384,kv)
    assert result['max_output_tokens']==16384 and result['physical_capacity_tokens']==20480
    assert result['initial_generation_headroom_tokens']==4048
    assert not result['no_post_query_pressure_needed']
    with pytest.raises(ValueError):
        memory_generation_capacity_audit(dict(history_begin=108320),
            [dict(charged_tokens=16384)]*16,112400,16384,kv)


def test_formal_pressure_audit_requires_explicit_rolling_and_accepted_output_scores():
    starts=list(range(0,1312,32))
    event=dict(event='pressure_selected',event_type='decode_pressure',fallback_reason=None,
        actual_method='accumulated_attention_per_layer',accepted_history=112384,selected_tokens=16384,
        layers=[dict(layer=i,charged_tokens=16384,selected_tokens=16384,accepted_score_end=112384,
            candidate_starts=starts,selected_starts=starts,attention_state_backend='native_decode',
            score_scope='all_accepted_including_thinking',query_samples=250) for i in range(16)])
    refresh=[dict(selection='pressure')]
    assert memory_pressure_audit([event],refresh,True,True,16384,rolling=True)==refresh
    with pytest.raises(ValueError):memory_pressure_audit([event],refresh,True,True,16384)
    for key,value in [('query_samples',0),('score_scope','non-thinking'),
                      ('accepted_score_end',112385),('charged_tokens',16416)]:
        bad=deepcopy(event);bad['layers'][15][key]=value
        with pytest.raises(ValueError):memory_pressure_audit([bad],refresh,True,True,16384,rolling=True)
