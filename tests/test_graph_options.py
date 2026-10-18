"""Public graph configuration rejects unsupported modes before model startup."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_kvmem.options import GRAPH_DEFAULTS, RuntimeOptions, validate_graph_options
from vllm_kvmem.serve import build_launch

V1_GRAPH_DEFAULTS = {key: value for key, value in GRAPH_DEFAULTS.items() if key != 'graph_alpha'}


def runtime(payload=None):
    return RuntimeOptions.from_vllm(SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={'kvmem': payload or {}}),
        cache_config=SimpleNamespace(cache_dtype='fp8_e4m3')))


def settings(mode='multi', **updates):
    values = dict(block_tokens=32, select_budget=8192, gen_budget=4096,
        repository_blocks=8192, session_capacity=2, session_checkpoint_bytes=1024**3,
        async_archive_retirement=True, request_planning=True,
        session_checkpoint_policy='pm', session_capture_terminal=True,
        action_attention=True, action_fraction=0.0, action_scope='non-thinking',
        prefill_attention=True, graph_mode=mode)
    values.update(updates)
    return values


def test_graph_defaults_are_off_and_from_vllm_preserves_input():
    default = runtime()
    for key, value in GRAPH_DEFAULTS.items():
        assert getattr(default, key) == value
    payload = settings(graph_attention_weight=0.3, graph_trace=True)
    before = deepcopy(payload)
    result = runtime(payload)
    assert result.graph_attention_weight == 0.3 and result.graph_trace
    assert payload == before


@pytest.mark.parametrize('mode', ['off', 'collect', 'locality', 'attention', 'multi'])
@pytest.mark.parametrize('backend', ['torch_reference', 'qw3_cuda'])
def test_modes_support_both_existing_mean_k_backends(mode, backend):
    assert runtime(settings(mode, mean_k_backend=backend)).graph_mode == mode


def test_locality_does_not_require_attention_or_reject_inactive_default_fraction():
    result = runtime(settings('locality', action_attention=False,
        prefill_attention=False, action_fraction=0.25, action_scope='tool-call'))
    assert result.graph_mode == 'locality'
    assert not result.memory.action_attention and result.memory.action_fraction == 0.25


def test_collect_preserves_existing_query_history_pressure_and_fusion():
    result = runtime(settings('collect', query_history=True, action_pressure=True,
        action_fraction=0.5, action_combine='score', request_planning=True))
    assert result.query_history and result.action_pressure
    assert result.memory.action_fraction == 0.5 and result.memory.action_combine == 'score'
    # Collection can also accompany legacy sessions without explicit Query plans.
    assert runtime(settings('collect', request_planning=False)).graph_mode == 'collect'


@pytest.mark.parametrize('updates,match', [
    ({'graph_mode': 'invalid'}, 'graph_mode'),
    ({'graph_mode': []}, 'graph_mode'),
    ({'graph_locality_weight': float('nan')}, 'graph_locality_weight'),
    ({'graph_attention_weight': float('inf')}, 'graph_attention_weight'),
    ({'graph_attention_weight': -0.1}, 'graph_attention_weight'),
    ({'graph_locality_weight': True}, 'graph_locality_weight'),
    ({'graph_locality_weight': 0.9, 'graph_attention_weight': 0.2}, 'sum'),
    ({'graph_cross_message_weight': 1.1}, 'graph_cross_message_weight'),
    ({'graph_attention_min_mass': -0.1}, 'graph_attention_min_mass'),
    ({'graph_attention_topk': True}, 'graph_attention_topk'),
    ({'graph_attention_topk': 0}, 'graph_attention_topk'),
    ({'graph_max_bytes_per_session': 0}, 'graph_max_bytes_per_session'),
    ({'graph_max_total_bytes': -1}, 'graph_max_total_bytes'),
    ({'graph_max_bytes_per_session': 1024, 'graph_max_total_bytes': 512}, 'capacity'),
    ({'graph_trace': 1}, 'graph_trace'),
    ({'repository_gc': True}, 'repository_gc'),
    ({'session_capacity': 0}, 'owned sessions'),
    ({'request_planning': False}, 'request planning'),
    ({'query_history': True}, 'query_history'),
    ({'action_pressure': True}, 'action_pressure'),
    ({'action_fraction': 0.1}, 'legacy attention fusion'),
    ({'action_attention': False}, 'Attention graph'),
    ({'prefill_attention': False}, 'Attention graph'),
    ({'action_scope': 'tool-call'}, 'non-thinking'),
])
def test_invalid_graph_configuration_is_not_silently_coerced(updates, match):
    with pytest.raises(ValueError, match=match):
        runtime(settings(**updates))


@pytest.mark.parametrize('mode', ['collect', 'locality', 'attention', 'multi'])
def test_enabled_graph_rejects_repository_gc_and_full_context(mode):
    with pytest.raises(ValueError, match='repository_gc'):
        runtime(settings(mode, repository_gc=True))
    with pytest.raises(ValueError, match='context_mode=meank'):
        validate_graph_options({'graph_mode': mode}, context_mode='full')


def model_profile(tmp_path):
    model, tokenizer = tmp_path / 'model', tmp_path / 'tokenizer'
    model.mkdir(); tokenizer.mkdir()
    (model / 'config.json').write_text('{}')
    return dict(model_format='safetensors', model=str(model), tokenizer=str(tokenizer),
        hf_config_path=str(model), draft_method='mtp', draft_tokens=4,
        execution_mode='compiled_graph', prefill_chunk_tokens=1024,
        kvmem=settings())


def worker_payload(plan):
    argv = plan['vllm_argv']
    transfer = json.loads(argv[argv.index('--kv-transfer-config') + 1])
    return transfer['kv_connector_extra_config']['kvmem']


@pytest.mark.parametrize('mode', ['off', 'collect', 'locality', 'attention', 'multi'])
def test_direct_launcher_forwards_exact_graph_fields_and_replays_profile(tmp_path, mode):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(V1_GRAPH_DEFAULTS, graph_mode=mode, graph_trace=True,
        graph_locality_weight=0.2, graph_attention_weight=0.3)
    plan = build_launch(profile, tmp_path, tmp_path / 'run')
    assert (plan['environment'].get('KVMEM_REQUIRE_OWNED_SESSION') == '1') is (mode != 'off')
    for key in V1_GRAPH_DEFAULTS:
        assert worker_payload(plan)[key] == profile['kvmem'][key]
    replay = build_launch(plan['profile'], tmp_path, tmp_path / 'replay')
    assert worker_payload(replay) == worker_payload(plan)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('mode', ['off', 'collect', 'locality', 'attention', 'multi'])
def test_elastic_launcher_validates_and_forwards_all_graph_fields(tmp_path, mode):
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    direct = model_profile(tmp_path)
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'graph-test', 'api_key': 'local-test-only'}]}))
    profile = dict(model=direct['model'], tokenizer=direct['tokenizer'], users_file=str(users),
        context_mode='meank', draft_method='mtp', draft_tokens=4,
        active_context_tokens=8192, gen_budget_tokens=4096, max_output_tokens=512,
        request_planning=True, action_attention=True, action_fraction=0.0,
        action_scope='non-thinking', prefill_attention=True, prefill_chunk_tokens=1024,
        **{**V1_GRAPH_DEFAULTS, 'graph_mode': mode, 'graph_trace': True})
    resolved, _ = build_elastic(profile, tmp_path, tmp_path / 'elastic')
    backend = resolved['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path / 'native'))
    for key in V1_GRAPH_DEFAULTS:
        assert resolved[key] == backend['kvmem'][key] == native[key] == profile[key]
    for change, match in (({'repository_gc': True}, 'repository_gc'),
                          ({'context_mode': 'full'}, 'context_mode')):
        with pytest.raises(ValueError, match=match):
            build_elastic({**profile, 'graph_mode': 'multi', **change}, tmp_path, tmp_path / 'bad')


def test_example_is_a_complete_profile_and_all_defaults_are_explicit(tmp_path):
    profile = json.loads((Path(__file__).parents[1] / 'examples/query-block-graph-elastic.json').read_text())
    assert profile['graph_mode'] == 'multi'
    assert (set(V1_GRAPH_DEFAULTS) - {'graph_scoring_version'}).issubset(profile)
    assert not profile['query_history'] and not profile['action_pressure']
    assert profile['guided_query_tokens'] == 0 and profile['action_fraction'] == 0
    assert profile['max_output_tokens'] <= profile['gen_budget_tokens']
    local = model_profile(tmp_path)
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'example-test', 'api_key': 'local-test-only'}]}))
    profile.update(model=local['model'], tokenizer=local['tokenizer'], users_file=str(users))
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    resolved, _ = build_elastic(profile, tmp_path, tmp_path / 'example')
    assert resolved['mean_k_backend_profile']['kvmem']['graph_mode'] == 'multi'


@pytest.mark.parametrize('mode', ['off', 'collect', 'locality', 'attention', 'multi'])
def test_graph_rejects_legacy_multi_user_middleware_before_launch(tmp_path, mode):
    profile = model_profile(tmp_path)
    profile['kvmem']['graph_mode'] = mode
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'test', 'api_key': 'local-test-only'}]}))
    profile['multi_user'] = dict(users_file=str(users), max_inflight=2)
    if mode == 'off':
        assert build_launch(profile, tmp_path, tmp_path/'off')['profile']['multi_user']['max_inflight'] == 2
    else:
        with pytest.raises(ValueError, match='legacy direct multi_user'):
            build_launch(profile, tmp_path, tmp_path/'blocked')
