from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from ruler_query_only_protocol import (PROFILE, audit_native_stats, audit_query_report,
                                       expected_profile, validate_execution)


def fixture():
    reference = dict(model='nvfp4', tokenizer='original', hf_config_path='nvfp4', host='127.0.0.1',
        port=18850, execution_mode='compiled_graph', draft_method='mtp', draft_tokens=4,
        max_num_seqs=8, prefill_chunk_tokens=2048, max_num_batched_tokens=16424,
        linear_backend_per_quant={'nvfp4_w4a4':'auto'}, kvmem=dict(select_budget=36864,
        gen_budget=16384, sink_blocks=41, recent_blocks=0, block_tokens=32,
        graph_mode='multi', graph_alpha=.2, graph_scoring_version='v2', pressure_policy='attention',
        action_fraction=0., query_history=False, guided_query_tokens=0,
        prefill_physical_coalescing=True, prefill_attention_stride=32))
    overrides=dict(model='awq', tokenizer='frozen', hf_config_path='awq', host='127.0.0.1', port=18856)
    contract=dict(execution_profile=PROFILE, reference_profile=reference,
        platform_path_overrides=overrides, cases=260, samples_per_task=20, target_length=262144,
        workers=8, cpu_payload_credits_bytes=82820130816, request_timeout_seconds=3600,
        starts_in_thinking=True, sampling=dict(max_tokens=8192, temperature=1., top_p=.95,
        top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096))
    return expected_profile(reference, overrides), contract


def test_explicit_ablation_preserves_every_other_reference_setting():
    profile, contract = fixture()
    before=deepcopy(contract)
    validate_execution(profile,contract)
    assert contract==before
    assert profile['kvmem']['graph_alpha']==0
    assert profile['kvmem']['pressure_policy']=='attention'
    assert profile['prefill_chunk_tokens']==2048


@pytest.mark.parametrize('key,value', [('gen_budget',65536),('select_budget',32768),
    ('pressure_policy','graph'),('action_fraction',.5),('prefill_attention_stride',16),
    ('graph_alpha',.2),('graph_mode','off')])
def test_unrelated_policy_changes_are_rejected(key,value):
    profile, contract=fixture()
    profile['kvmem'][key]=value
    with pytest.raises(ValueError): validate_execution(profile,contract)


@pytest.mark.parametrize('path,value',[('workers',4),('cpu_payload_credits_bytes',10**12),
    ('request_timeout_seconds',7200),('samples_per_task',500)])
def test_cohort_and_admission_cannot_silently_drift(path,value):
    profile,contract=fixture()
    contract[path]=value
    with pytest.raises(ValueError): validate_execution(profile,contract)


def report():
    return dict(mode='multi', alpha=0., seed_source='mean_k', accepted_through=262000,
        identity=True, locality_weight=0., attention_weight=0., raw_l1_change=0.,
        normalized_l1_change=0., added_blocks=0, displaced_blocks=0)


@pytest.mark.parametrize('key,value',[('alpha',.2),('seed_source','prefill'),('identity',False),
    ('locality_weight',.1),('attention_weight',.1),('raw_l1_change',1e-8),('added_blocks',1)])
def test_actual_query_must_retain_raw_scores_and_picked_pages(key,value):
    good=report();audit_query_report(good,261999)
    good[key]=value
    with pytest.raises(ValueError): audit_query_report(good,261999)


def test_alpha_zero_returns_exact_query_vector_despite_a_rank_changing_edge():
    from vllm_kvmem.block_graph import GraphRow
    from vllm_kvmem.graph_scoring import score_graph
    raw=[.49,.51]
    snapshot=NS(block_tokens=4, rows=(GraphRow(4,(0,),(1.,),1),), partial=None,
        message_spans=(), closed=False, revision=1, epoch=1, accepted_through=8, completed_through=8,
        bytes=0)
    with score_graph(raw,[0,4],[True,True],snapshot,mode='multi',scoring_version='v2',
                     alpha=0.,native=True) as result:
        assert result.scores is raw and result.diagnostics['identity']
        assert result.diagnostics['raw_l1_change']==0
        assert max(range(2),key=lambda i:result.scores[i])==1


def test_only_the_documented_a800_native_score_fallback_is_accepted():
    native=dict(requested_mode='reuse',disabled_reason='unsupported_architecture_requires_sm12x',
                allocated_bytes=0,validation_failures=0,layer_phase_counts={'fallback:prefill:sm80':3})
    audit_native_stats(native)
    native['disabled_reason']='invalid_mass'
    with pytest.raises(ValueError): audit_native_stats(native)
