"""Explicit stage policies reuse accepted attention, without hidden fallback."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving import recall_experiment as recall
from vllm_kvmem.serving.serve import build_launch
from test_recall_experiment import native_authority, finish
from test_recall_launch import profile as launch_profile
from test_recall_refresh_dispatch import setup_dispatch


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'block_value_norm')
    monkeypatch.setenv('KVMEM_VALUE_RECALL_REFERENCE', '1')
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND', 'streaming')
    monkeypatch.setenv('KVMEM_RECALL_DEVICE', 'cpu')
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW', '128')
    for stage in ('PREFILL', 'DECODE'):
        monkeypatch.setenv('KVMEM_' + stage + '_PRESSURE_METHOD', 'accumulated_attention')
    monkeypatch.delenv('KVMEM_VALUE_RECALL_DIAGNOSTICS', raising=False)


def history(authority):
    rows = list(authority.records.values())
    prefill = NS(samples=8, chunks=4, end=authority.history_end,
                 scores=lambda: {r.history_begin: (10. if i == 4 else .01) for i, r in enumerate(rows)})
    return NS(prefill=prefill, scores={r.history_begin: (100. if i == 6 else 0.) for i, r in enumerate(rows)},
              mass={rows[5].history_begin: 1000.}, samples=2, rounds=1,
              options=NS(action_decay=.8, action_scope='non-thinking'))


@pytest.mark.parametrize('stage,optional', [('prefill', (4, 7)), ('decode', (5, 6))])
def test_native_pressure_uses_cumulative_scores_with_fixed_keeps_and_no_q(hybrid, tmp_path, stage, optional):
    a = native_authority(count=256, budget=160, sink=1, recent=1)
    rows = list(a.records.values())
    a.action_history = history(a)
    # Any archive/Value access would be an incorrect use of the Query method.
    a.read_records = lambda *_: pytest.fail('Cumulative pressure read K/V')
    try:
        plan = recall.select_for_event(a, None, [rows[2].handle], max_tokens=128,
            event_type=stage + '_pressure', query_metadata={}, request_id='hybrid', output=tmp_path)
        event = a.recall_selection_report
        assert event['configured_method'] == 'block_value_norm'
        assert event['actual_method'] == event['configured_event_method'] == 'accumulated_attention'
        assert event['configuration_kind'] == 'pressure_attention_query_recall'
        assert event['fallback_reason'] is None
        # The existing pressure picker keeps sink/explicit pins, then scores;
        # semantic Query recent_blocks do not override pressure scores.
        assert set(event['selected_block_ids']) == {rows[i].handle.slot for i in (0, 2, *optional)}
        assert event['charged_tokens'] == event['selected_tokens'] == 128
        assert event['attention_source'] == ('prefill' if stage == 'prefill' else 'output')
        assert event['query_begin'] == 0 and event['query_end'] <= event['accepted_through']
        assert event['attention_normalization_domain'] == 'sampled_causal_active_windows'
        finish(a.memory)
        del plan
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('kind', ['no_state', 'zero_prefill', 'future_prefill', 'no_decode_mass'])
def test_missing_cumulative_state_fails_instead_of_using_recency(hybrid, tmp_path, kind):
    a = native_authority(count=256, budget=160)
    a.action_history = history(a)
    stage = 'prefill'
    if kind == 'no_state':
        a.action_history = None
    elif kind == 'zero_prefill':
        a.action_history.prefill.scores = lambda: {}
    elif kind == 'future_prefill':
        a.action_history.prefill.end += 1
    else:
        stage = 'decode'
        a.action_history.scores = {}
        a.action_history.mass = {}
        a.action_history.prefill.scores = lambda: {}
    try:
        with pytest.raises(ValueError, match='prefill|attention|fallback'):
            recall.select_for_event(a, None, [], max_tokens=128, event_type=stage + '_pressure',
                query_metadata={}, request_id='invalid-state', output=tmp_path)
    finally:
        assert a.memory.close()


def test_decode_native_prefill_seed_is_declared_when_no_output_attention_exists(hybrid, tmp_path):
    a = native_authority(count=256, budget=160)
    a.action_history = history(a)
    a.action_history.scores = {}
    a.action_history.mass = {}
    try:
        plan = recall.select_for_event(a, None, [], max_tokens=128, event_type='decode_pressure',
            query_metadata={}, request_id='prefill-seed', output=tmp_path)
        event = a.recall_selection_report
        assert event['decode_seed_source'] == event['attention_source'] == 'prefill'
        assert event['actual_method'] == 'accumulated_attention' and event['fallback_reason'] is None
        finish(a.memory)
        del plan
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('stage', ['prefill', 'decode'])
def test_live_dispatch_does_not_require_tail_q_or_value_scale_for_pressure(monkeypatch, tmp_path, stage):
    s = setup_dispatch(monkeypatch, tmp_path, method='block_value_norm', stage=stage)
    monkeypatch.setenv('KVMEM_' + stage.upper() + '_PRESSURE_METHOD', 'accumulated_attention')
    s.authority.pressure_query_window = None
    del s.manager.runner._kvmem_capture_runtime.scales
    s.manager._assemble(s.job, None, s.ready)
    selected, = s.calls.selections
    assert selected['queries'] is None and selected['query_metadata'] == {}
    assert selected['event_type'] == stage + '_pressure'
    assert not s.calls.reads and not s.calls.legacy


def test_public_query_still_uses_value_method_and_original_query(hybrid, monkeypatch, tmp_path):
    s = setup_dispatch(monkeypatch, tmp_path, method='block_deletion_error', selection='query')
    s.manager._assemble(s.job, s.public_q, s.ready)
    selected, = s.calls.selections
    assert selected['queries'] is s.public_q and selected['event_type'] == 'query'
    assert recall.configured_event_method('query') == 'block_deletion_error'
    assert s.authority.recall_value_scales.tolist() == [2., .5]


def test_launch_preserves_both_stage_policies_and_requires_attention_capture(tmp_path):
    config = launch_profile()
    config['recall_experiment'].update(method='block_value_norm', include_mandatory_in_softmax=True,
        value_backend='streaming', prefill_pressure_method='accumulated_attention',
        decode_pressure_method='accumulated_attention')
    config['kvmem'].update(action_attention=True, prefill_attention=True,
                          pressure_policy='attention', session_capacity=1, session_checkpoint_bytes=1,
                          async_archive_retirement=True, session_checkpoint_policy='pm',
                          session_capture_terminal=True)
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    for stage in ('PREFILL', 'DECODE'):
        assert plan['environment']['KVMEM_' + stage + '_PRESSURE_METHOD'] == 'accumulated_attention'
    bad = deepcopy(config)
    bad['kvmem']['prefill_attention'] = False
    with pytest.raises(ValueError, match='prefill attention'):
        build_launch(bad, tmp_path, tmp_path/'bad', check_paths=False)


def test_event_audit_accepts_only_declared_stage_mixing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    p = importlib.import_module('recall_comparison_protocol')
    from test_recall_comparison import profile, event
    config = profile()
    config['recall_experiment'] = dict(prefill_pressure_method='accumulated_attention',
        decode_pressure_method='accumulated_attention', include_mandatory_in_softmax=True)
    rows = []
    for kind in p.EVENT_TYPES:
        e = event('block_value_norm', kind)
        e.update(configuration_kind='pressure_attention_query_recall',
                 prefill_pressure_method='accumulated_attention', decode_pressure_method='accumulated_attention')
        if kind == 'query':
            e.update(query_processed_through=256, public_input_end=260,
                attention_normalization_domain='candidate_and_mandatory', mandatory_in_normalization=True,
                score_mask_count=e['candidate_block_count'])
        else:
            e.update(actual_method='accumulated_attention', query_source='accepted_prefill_accumulation',
                query_begin=0, attention_source='prefill', prefill_attention_samples=8,
                prefill_attention_chunks=4, prefill_score_normalization='decayed_mass_over_exposure',
                attention_normalization_domain='sampled_causal_active_windows')
        e['configured_event_method'] = e['actual_method']
        rows.append(e)
    audit = p.audit_events(rows, 'block_value_norm', config, required_types=p.EVENT_TYPES)
    assert audit['actual_methods'] == ['accumulated_attention', 'block_value_norm']
    changed = deepcopy(rows)
    changed[0]['actual_method'] = 'block_value_norm'
    with pytest.raises(ValueError, match='Mixed-method'):
        p.audit_events(changed, 'block_value_norm', config)
    with pytest.raises(ValueError, match='Mixed-method'):
        p.audit_events(rows, 'block_value_norm', profile())


def test_hybrid_fixture_declares_two_query_methods_without_exact_mass(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    p = importlib.import_module('recall_comparison_protocol')
    f = dict(value_recall_reference=True, pressure_attention_hybrid=True,
             full_value_comparison=False, methods=['block_value_norm', 'block_deletion_error'])
    assert p.campaign_methods(f) == ('block_value_norm', 'block_deletion_error')
    f['methods'].insert(0, 'exact_mass')
    with pytest.raises(ValueError, match='two Query methods'):
        p.campaign_methods(f)
