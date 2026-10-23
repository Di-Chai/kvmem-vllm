"""Public switch, native score domain and transaction checks for four-Key FPS."""
from copy import deepcopy
import json

import pytest
import torch

from test_graph_options import model_profile, runtime, settings, worker_payload
from vllm_kvmem.serve import build_launch


def fps_settings(**updates):
    return settings(query_score_mode='key_fps_4', graph_mode='multi',
        graph_scoring_version='v2', graph_aggregation='sum', graph_alpha=0.,
        cpu_read_cache_bytes=4*1024**3, **updates)


def test_fps_switch_reserves_cache_and_roundtrips_launch_without_changing_budget(tmp_path):
    from vllm_kvmem.query_window_budget import split_cpu_history_budget
    profile = model_profile(tmp_path)
    profile['kvmem'] = fps_settings(query_window_cache_bytes=1048576)
    original = deepcopy(profile)
    options = runtime(profile['kvmem'])
    assert split_cpu_history_budget(options) == (4*1024**3-1048576, 1048576)
    launch = build_launch(profile, tmp_path, tmp_path/'run')
    observed = worker_payload(launch)
    assert profile == original and not (tmp_path/'run').exists()
    for key in ('query_score_mode', 'query_window_cache_bytes', 'select_budget',
                'gen_budget', 'block_tokens', 'graph_alpha'):
        assert observed[key] == profile['kvmem'][key]
    assert worker_payload(build_launch(launch['profile'], tmp_path, tmp_path/'replay')) == observed


@pytest.mark.parametrize('update,match', [
    ({'graph_alpha': .2}, 'graph_alpha=0'),
    ({'graph_alpha': True}, 'graph_alpha'),
    ({'graph_aggregation': 'max'}, 'graph_aggregation=sum'),
    ({'graph_mode': 'off'}, 'graph_mode=multi'),
    ({'block_tokens': 16}, 'block_tokens=32'),
    ({'query_history': True}, 'query_history=false'),
    ({'guided_query_tokens': 512}, 'guided_query_tokens=0'),
    ({'cpu_read_cache_bytes': 2147483648}, 'CPU memory budget'),
])
def test_unqualified_fps_policies_fail_before_launch(update, match):
    profile = fps_settings()
    profile.update(update)
    with pytest.raises(ValueError, match=match):
        runtime(profile)


def test_fps_rejects_competing_token_experiment(tmp_path, monkeypatch):
    from vllm_kvmem import serve
    profile = model_profile(tmp_path)
    profile['kvmem'] = fps_settings()
    config = tmp_path/'profile.json'
    config.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='cannot be combined'):
        serve.main(['--config', str(config), '--run-dir', str(tmp_path/'run'),
                    '--dry-run', '--experimental-query-token-scores', 'sum'])
    monkeypatch.setenv('KVMEM_EXPERIMENTAL_QUERY_TOKEN_SCORES', 'sum')
    with pytest.raises(ValueError, match='cannot be combined'):
        build_launch(profile, tmp_path, tmp_path/'inherited')


def test_elastic_launcher_preserves_fps_mode_and_rejects_full_context(tmp_path):
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    local = model_profile(tmp_path)
    users = tmp_path/'users.json'
    users.write_text(json.dumps({'users': [{'id': 'fps-test', 'api_key': 'local-test-only'}]}))
    profile = dict(model=local['model'], tokenizer=local['tokenizer'], users_file=str(users),
        context_mode='meank', draft_method='mtp', draft_tokens=4, block_tokens=32,
        cpu_kv_bytes=4*1024**3, active_context_tokens=8192, gen_budget_tokens=4096,
        max_output_tokens=512, request_planning=True, action_attention=True,
        action_fraction=0., action_scope='non-thinking', prefill_attention=True,
        prefill_chunk_tokens=1024, graph_mode='multi', graph_scoring_version='v2',
        graph_aggregation='sum', graph_alpha=0., query_score_mode='key_fps_4',
        query_window_cache_bytes=1048576)
    profile.update(query_fps_halo_blocks=1, query_fps_optimized=True)
    resolved, _ = build_elastic(profile, tmp_path, tmp_path/'elastic')
    backend = resolved['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path/'native'))
    for key in ('query_score_mode', 'query_window_cache_bytes', 'graph_alpha',
                'query_fps_halo_blocks', 'query_fps_optimized'):
        assert resolved[key] == backend['kvmem'][key] == native[key] == profile[key]
    with pytest.raises(ValueError, match='MeanK context management'):
        build_elastic({**profile, 'context_mode': 'full'}, tmp_path, tmp_path/'bad')


def actual(tmp_path, capacity=65536, *, select_budget=64):
    from types import SimpleNamespace as NS
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.block_graph import BlockGraph, GraphSample
    from test_authority import drain
    cfg = KVMemConfig(block_tokens=32, select_budget=select_budget, gen_budget=128,
        repository_blocks=16, sink_blocks=1, recent_blocks=0,
        action_attention=True, action_fraction=0.)
    archive = AsyncArchive(str(tmp_path), 1*32*1*2*3, 16)
    authority = RequestAuthority('fps', cfg, archive, 1, 1, 2, resident_records=16,
        query_score_mode='key_fps_4', query_window_cache_bytes=capacity)
    generator = torch.Generator().manual_seed(123)
    raw = torch.randn(1, 192, 1, 2, generator=generator).bfloat16()
    assert authority.append(raw, torch.zeros_like(raw, dtype=torch.uint8),
                            history_begin=0, local_rope_begin=0)
    drain(authority)
    graph = authority.block_graph = BlockGraph(('tenant', 'fps'), 32)
    graph.prepare(0, 192, [GraphSample(160, (32, 64, 96), (.3, .5, .2), 'prefill')]).commit()
    authority.graph_options = NS(graph_mode='multi', graph_scoring_version='v2',
        graph_aggregation='sum', graph_alpha=0., graph_locality_weight=.1,
        graph_attention_weight=.2, graph_cross_message_weight=.25,
        graph_trace=False, graph_native_propagation=False)
    return authority, archive, graph


def test_fps_seed_identity_native_picker_fixed_keep_budget_and_release(tmp_path):
    from vllm_kvmem.authority import query_score_frame, _Ready
    from vllm_kvmem.query_key_fps import score_authority_fps
    from test_graph_lifecycle import selected_starts
    a, archive, graph = actual(tmp_path)
    q = torch.tensor([[[[2., 1.], [1., -2.]]]])
    with a.index_snapshot() as (records, means):
        with query_score_frame(a.memory, records) as frame:
            seeds, _ = score_authority_fps(a, records, q, means, frame.score_mask)
            slots = set(a.memory.preview_selection({r.handle.slot: float(s)
                for r, s in zip(records, seeds)}, (), action=False, prefill_scores=None))
            expected = [r.history_begin for r in records if r.handle.slot in slots]
    plan = a.select_for_queries(q)
    assert selected_starts(a, plan) == expected and plan.total_window_tokens == 64
    assert 0 in expected and len(expected) == 2
    assert a.graph_selection_report['seed_source'] == 'key_fps_4'
    assert a.graph_selection_report['identity'] is True
    assert a.graph_selection_report['propagation_applied'] is False
    assert a.query_key_fps_stats['cache_hit_blocks'] == 6
    a.memory.attach_completion(_Ready())
    assert a.memory.commit_selection()
    del plan
    assert a.close() and archive._query_window_cache_bytes == 0 and archive.close()


@pytest.mark.parametrize('fault', ['read', 'capacity', 'stale_graph'])
def test_failed_fps_prepare_cannot_partially_commit(tmp_path, monkeypatch, fault):
    from vllm_kvmem.authority import _Ready
    a, archive, graph = actual(tmp_path, capacity=1024 if fault == 'capacity' else 65536)
    before = a.memory.phase, graph.binding, graph.budget.used
    if fault == 'read':
        def fail(*args):
            raise OSError('injected FPS archive failure')
        monkeypatch.setattr(a, 'read_records', fail)
    if fault == 'stale_graph':
        plan = a.select_for_queries(torch.ones(1, 1, 1, 2))
        graph.invalidate()
        a.memory.attach_completion(_Ready())
        with pytest.raises(RuntimeError, match='Stale'):
            a.memory.commit_selection()
        del plan
    else:
        with pytest.raises((OSError, MemoryError)):
            a.select_for_queries(torch.ones(1, 1, 1, 2))
        assert (a.memory.phase, graph.binding, graph.budget.used) == before
    assert a.close() and archive._query_window_cache_bytes == 0 and archive.close()


def test_fps_trace_has_public_query_quota_independent_of_pressure(tmp_path):
    from test_graph_v2_ablation import export_fixture, export
    authority, rows, snapshot = export_fixture(tmp_path, nodes=4)
    authority.query_score_mode = 'key_fps_4'
    authority._graph_input_count = 8
    export(authority, rows, snapshot)
    assert authority._window_query_input_count == 1 and authority._graph_input_count == 8
    assert json.loads((tmp_path/'block_graph_inputs.jsonl').read_text())['seed_source'] == 'key_fps_4'
