"""Launch-time strategy capability checks (vllm_kvmem.capabilities)."""
import pytest

from vllm_kvmem import capabilities

BASE = dict(action_attention=True, prefill_attention=True, pressure_policy='attention')


def test_the_built_in_strategies_pass_on_both_paths():
    report = capabilities.check(dict(BASE), query_mode=None, pressure=False)
    assert (report['path'], report['query_scorer'], report['pressure_scorer'], report['picker']) == (
        'global', 'meank', 'attention', 'native')
    report = capabilities.check(dict(BASE), query_mode='sequential', pressure=True)
    assert report['path'] == 'layer' and report['shared_frame_only'] == []


def test_the_global_path_reads_the_older_options():
    report = capabilities.check(dict(BASE, query_score_mode='key_fps_4', pressure_policy='recency'),
                                query_mode=None, pressure=False)
    assert (report['query_scorer'], report['pressure_scorer']) == ('key_fps', 'recency')
    assert 'key_fps_index' in report['signals']
    report = capabilities.check(dict(BASE, action_pressure=False, pressure_policy=None), query_mode=None, pressure=False)
    assert report['pressure_scorer'] == 'recency'


def test_a_strategy_whose_signals_are_missing_is_refused():
    with pytest.raises(ValueError, match='needs block_graph'):
        capabilities.check(dict(BASE, pressure_policy='graph', graph_mode='off'), query_mode=None, pressure=False)
    with pytest.raises(ValueError, match='needs layer_attention'):
        capabilities.check(dict(BASE, action_attention=False), query_mode=None, pressure=False)


def test_per_layer_refusals_come_at_launch():
    layer = dict(query_mode='sequential', pressure=True)
    with pytest.raises(ValueError, match='does not run on the layer path'):
        capabilities.check(dict(BASE, strategy_query_scorer='key_fps'), **layer)
    with pytest.raises(ValueError, match='pressure_policy=attention'):
        capabilities.check(dict(BASE, pressure_policy='recency'), **layer)
    # A Query-only per-layer diagnostic runs no per-layer pressure: nothing to check there.
    capabilities.check(dict(BASE, pressure_policy='recency', action_attention=False),
                       query_mode='sequential', pressure=False)
    with pytest.raises(ValueError, match='no guidance or Query history'):
        capabilities.check(dict(BASE, guided_query_tokens=64), **layer)
    with pytest.raises(ValueError, match='no guidance or Query history'):
        capabilities.check(dict(BASE, query_history=True), **layer)
    with pytest.raises(ValueError, match='requires prefill_attention'):
        capabilities.check(dict(BASE, prefill_attention=False), **layer)
    with pytest.raises(ValueError, match='batch selection'):
        capabilities.check(dict(BASE, strategy_pressure_scorer='recency'), batch_selection=True, **layer)
    capabilities.check(dict(BASE, strategy_pressure_scorer='recency'), **layer)
    with pytest.raises(ValueError, match='--layerwise-prefill-pressure'):
        capabilities.check(dict(BASE, strategy_pressure_scorer='recency'), query_mode='sequential', pressure=False)
    with pytest.raises(ValueError, match='sequential per-layer Query'):
        capabilities.check(dict(BASE, strategy_query_scorer='recency'), query_mode='global', pressure=True)


def test_global_only_strategies_are_reported_on_the_per_layer_path():
    kvmem = dict(BASE, query_score_mode='key_fps_4', graph_mode='multi', query_mean_quota_fraction=.5)
    report = capabilities.check(kvmem, query_mode='sequential', pressure=True,
                                environment={'KVMEM_RECALL_METHOD': 'fps4'})
    assert report['shared_frame_only'] == ['query_score_mode=key_fps_4', 'query_mean_quota_fraction',
                                           'graph_mode=multi', 'KVMEM_RECALL_METHOD=fps4']
    # graph collection alone is not a strategy
    assert capabilities.check(dict(BASE, graph_mode='collect'), query_mode='sequential',
                              pressure=True)['shared_frame_only'] == []


def test_the_strategy_table_is_generated_from_the_registry():
    text = capabilities.table()
    assert '| query scorer | `key_fps` | runs (built in) | refused at launch |' in text
    assert '| pressure scorer | `recency` | runs (built in) | runs (strategy object) |' in text
    assert '`overlap` (legacy)' in text
    assert capabilities.main(['table']) == 0


def test_costly_signals_no_strategy_reads_are_reported():
    layer = dict(query_mode='sequential', pressure=True)
    assert capabilities.check(dict(BASE), **layer)['unused_signals'] == []
    recency = dict(BASE, strategy_query_scorer='recency', strategy_pressure_scorer='recency')
    assert capabilities.check(recency, **layer)['unused_signals'] == ['layer_attention']
    report = capabilities.check(dict(BASE, graph_mode='multi'), query_mode=None, pressure=False)
    assert report['unused_signals'] == []     # graph propagation reads the block graph
