"""The 16/8 scoring experiment has explicit, reproducible launch settings."""
from copy import deepcopy
import json

import pytest

from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from test_graph_options import model_profile, runtime, settings, worker_payload


def overlap_settings(**updates):
    values = settings(query_score_mode='overlap_16_8',
        graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=.2,
        cpu_read_cache_bytes=4*1024**3)
    values.update(updates)
    return values


@pytest.mark.parametrize('alpha', [0., .2])
def test_default_and_switch_preserve_configuration_and_physical_block_geometry(alpha):
    default = RuntimeOptions()
    assert default.query_score_mode == 'mean_k'
    assert default.query_window_cache_bytes == 2147483648
    values = overlap_settings(query_window_cache_bytes=1048576, graph_alpha=alpha)
    before = deepcopy(values)
    options = runtime(values)
    assert values == before
    assert options.query_score_mode == 'overlap_16_8'
    assert options.query_window_cache_bytes == 1048576
    assert options.memory.block_tokens == 32
    assert (options.memory.select_budget, options.memory.gen_budget) == (8192, 4096)
    assert (options.graph_scoring_version, options.graph_aggregation, options.graph_alpha) == ('v2', 'sum', alpha)


@pytest.mark.parametrize('updates,match', [
    ({'query_score_mode': 'overlap_32_16'}, 'query_score_mode'),
    ({'query_score_mode': None}, 'query_score_mode'),
    ({'query_window_cache_bytes': 0}, 'query_window_cache_bytes'),
    ({'query_window_cache_bytes': -1}, 'query_window_cache_bytes'),
    ({'query_window_cache_bytes': True}, 'query_window_cache_bytes'),
    ({'query_window_cache_bytes': '2GiB'}, 'query_window_cache_bytes'),
    ({'cpu_read_cache_bytes': 0}, 'CPU memory budget'),
    ({'cpu_read_cache_bytes': 2147483648}, 'CPU memory budget'),
    ({'block_tokens': 16}, 'block_tokens=32'),
    ({'graph_mode': 'off'}, 'graph_mode=multi'),
    ({'graph_mode': 'locality'}, 'graph_mode=multi'),
    ({'graph_scoring_version': 'v1'}, 'graph_scoring_version'),
    ({'graph_aggregation': 'max'}, 'graph_aggregation=sum'),
    ({'graph_alpha': .3}, 'graph_alpha=0 or 0.2'),
    ({'query_history': True}, 'query_history=false'),
    ({'guided_query_tokens': 512}, 'guided_query_tokens=0'),
])
def test_unqualified_window_or_graph_semantics_are_rejected(updates, match):
    with pytest.raises(ValueError, match=match):
        runtime(overlap_settings(**updates))


@pytest.mark.parametrize('alpha', [0., .2])
def test_direct_profile_replay_preserves_mode_capacity_and_native_budget(tmp_path, alpha):
    profile = model_profile(tmp_path)
    profile['kvmem'] = overlap_settings(query_window_cache_bytes=1048576, graph_alpha=alpha)
    plan = build_launch(profile, tmp_path, tmp_path / 'run')
    observed = worker_payload(plan)
    for key in ('query_score_mode', 'query_window_cache_bytes', 'block_tokens',
                'select_budget', 'gen_budget', 'graph_scoring_version',
                'graph_aggregation', 'graph_alpha'):
        assert observed[key] == profile['kvmem'][key]
    replay = build_launch(plan['profile'], tmp_path, tmp_path / 'replay')
    assert worker_payload(replay) == observed
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('alpha', [0., .2])
def test_elastic_profile_forwards_mode_and_capacity_and_rejects_full_context(tmp_path, alpha):
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    local = model_profile(tmp_path)
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'overlap-test', 'api_key': 'local-test-only'}]}))
    profile = dict(model=local['model'], tokenizer=local['tokenizer'], users_file=str(users),
        context_mode='meank', draft_method='mtp', draft_tokens=4, block_tokens=32,
        cpu_kv_bytes=4*1024**3,
        active_context_tokens=8192, gen_budget_tokens=4096, max_output_tokens=512,
        request_planning=True, action_attention=True, action_fraction=0.,
        action_scope='non-thinking', prefill_attention=True, prefill_chunk_tokens=1024,
        graph_mode='multi', graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=alpha,
        query_score_mode='overlap_16_8', query_window_cache_bytes=1048576)
    resolved, _ = build_elastic(profile, tmp_path, tmp_path / 'elastic')
    backend = resolved['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path / 'native'))
    for key in ('query_score_mode', 'query_window_cache_bytes', 'graph_alpha'):
        assert resolved[key] == backend['kvmem'][key] == native[key] == profile[key]
    replay, _ = build_elastic(resolved, tmp_path, tmp_path / 'elastic-replay')
    assert replay['mean_k_backend_profile']['kvmem'] == backend['kvmem']
    with pytest.raises(ValueError, match='MeanK context management'):
        build_elastic({**profile, 'context_mode': 'full'}, tmp_path, tmp_path / 'bad')
    with pytest.raises(ValueError, match='graph_alpha=0 or 0.2'):
        build_elastic({**profile, 'graph_alpha': .3}, tmp_path, tmp_path / 'bad-alpha')
    with pytest.raises(ValueError, match='CPU memory budget'):
        build_elastic({**profile, 'cpu_kv_bytes': profile['query_window_cache_bytes']},
            tmp_path, tmp_path / 'insufficient-memory')
    assert not (tmp_path / 'insufficient-memory').exists()


@pytest.mark.parametrize('cpu_bytes', [0, 2147483648])
def test_direct_launcher_rejects_exhausted_cpu_budget_before_creating_run(tmp_path, cpu_bytes):
    profile = model_profile(tmp_path)
    profile['kvmem'] = overlap_settings(cpu_read_cache_bytes=cpu_bytes)
    with pytest.raises(ValueError, match='CPU memory budget'):
        build_launch(profile, tmp_path, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_overlap_rejects_both_explicit_and_inherited_token_experiment(tmp_path, monkeypatch):
    from vllm_kvmem import serve
    profile = model_profile(tmp_path)
    profile['kvmem'] = overlap_settings()
    config = tmp_path / 'profile.json'
    config.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='cannot be combined'):
        serve.main(['--config', str(config), '--run-dir', str(tmp_path / 'run'),
                    '--dry-run', '--experimental-query-token-scores', 'sum'])
    monkeypatch.setenv('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES', 'sum')
    with pytest.raises(ValueError, match='cannot be combined'):
        build_launch(profile, tmp_path, tmp_path / 'inherited')
