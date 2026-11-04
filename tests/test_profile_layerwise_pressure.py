from copy import deepcopy

import pytest

from profile_layerwise_pressure import controls, exact_comparison, handoff_ready, save
from test_layerwise_efficiency_equivalence import evidence


def profile():
    return dict(port=19092, served_model_name='source', max_num_seqs=1,
        execution_mode='eager', prefill_chunk_tokens=512,
        kvmem=dict(select_budget=36864, gen_budget=16384, block_tokens=32,
                   sink_blocks=41, recent_blocks=0, pressure_policy='attention'))


def test_only_endpoint_changes_are_allowed():
    original = profile()
    changed = deepcopy(original)
    changed.update(port=19104, served_model_name='diagnostic')
    controls(original, changed)
    for key, value in [('max_num_seqs', 8), ('prefill_chunk_tokens',2048),
                       ('execution_mode','compiled_graph')]:
        different = deepcopy(changed)
        different[key] = value
        with pytest.raises(ValueError):
            controls(original,different)
    for key, value in [('select_budget',32768), ('gen_budget',8192),
                       ('pressure_policy','mean_k')]:
        different = deepcopy(changed)
        different['kvmem'][key] = value
        with pytest.raises(ValueError):
            controls(original,different)
    assert original == profile()


def test_full_output_is_checked_even_when_choices_match():
    before = evidence()
    cfg = dict(kvmem=dict(select_budget=64,gen_budget=32,block_tokens=32))
    response = dict(output_ids=[2,3],text='answer')
    check = exact_comparison(before,deepcopy(before),response,dict(response),cfg)
    assert all(check[k] for k in ('same_pressure','same_query','same_output_tokens','same_text'))
    check = exact_comparison(before,deepcopy(before),response,
        dict(output_ids=[2,4],text='other'),cfg)
    assert check['same_pressure'] and check['same_query']
    assert not check['same_output_tokens'] and not check['same_text']


def test_candidate_scope_change_is_not_hidden_by_same_output():
    before = evidence()
    after = deepcopy(before)
    after['refresh'][0]['query_score_tokens'] = 1
    cfg = dict(kvmem=dict(select_budget=64,gen_budget=32,block_tokens=32))
    response = dict(output_ids=[2,3],text='answer')
    check = exact_comparison(before,after,response,response,cfg)
    assert not check['same_query']


def test_failed_diagnostic_requires_terminal_controller_and_clean_owned_processes(tmp_path):
    save(tmp_path/'status.json',dict(status='failed'))
    save(tmp_path/'process.json',dict(pid=123))
    save(tmp_path/'plan.json',dict(diagnostic_only=True))
    save(tmp_path/'cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))
    with pytest.raises(ValueError,match='still live'):
        handoff_ready(tmp_path,is_alive=lambda pid:True)
    handoff_ready(tmp_path,is_alive=lambda pid:False)
    save(tmp_path/'cleanup.json',dict(remaining_owned_groups={123:[124]},gpu_processes=[]))
    with pytest.raises(ValueError,match='ownership'):
        handoff_ready(tmp_path,is_alive=lambda pid:False)
