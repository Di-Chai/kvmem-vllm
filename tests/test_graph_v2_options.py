"""Versioned graph policy parsing and canonical launcher/worker transport."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from test_graph_options import model_profile, runtime, settings, worker_payload
from vllm_kvmem.elastic_serve import build_launch as build_elastic
from vllm_kvmem.options import RuntimeOptions, canonical_policy_payload
from vllm_kvmem.serve import build_launch


def v2_settings(**updates):
    return settings(graph_scoring_version='v2', **updates)


@pytest.mark.parametrize('aggregation', ['sum', 'max'])
def test_pooling_roundtrips_direct_launch_and_runtime(tmp_path, aggregation):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(graph_scoring_version='v2', graph_aggregation=aggregation,
                            pressure_policy='attention')
    plan = build_launch(profile, tmp_path, tmp_path/'pool')
    worker = worker_payload(plan)
    assert runtime(worker).graph_aggregation == aggregation
    assert worker['pressure_policy'] == 'attention'
    reloaded = build_launch(json.loads(json.dumps(plan['profile'])), tmp_path, tmp_path/'reload-pool')
    assert worker_payload(reloaded) == worker


@pytest.mark.parametrize('value', [False, None, [], {}, 'mean'])
def test_invalid_pooling_is_rejected_before_launch(value):
    with pytest.raises(ValueError, match='graph_aggregation'):
        runtime(v2_settings(graph_aggregation=value))


def test_max_pooling_requires_v2_and_can_be_preserved_by_dataclass_replacement():
    with pytest.raises(ValueError, match='requires graph_scoring_version=v2'):
        runtime(settings(graph_aggregation='max'))
    result = runtime(v2_settings(graph_aggregation='max'))
    assert replace(result, graph_trace=True).graph_aggregation == 'max'


def test_legacy_pressure_alias_and_v2_defaults_are_resolved_without_input_mutation():
    default = runtime()
    assert (default.graph_scoring_version, default.graph_alpha, default.pressure_policy,
            default.action_pressure) == ('v1', .2, 'recency', False)
    assert runtime(settings('collect', action_pressure=True)).pressure_policy == 'attention'
    payload = v2_settings(pressure_policy='graph')
    before = deepcopy(payload)
    resolved = runtime(payload)
    assert resolved.pressure_policy == 'graph' and resolved.action_pressure is True
    assert resolved.graph_alpha == .2
    assert payload == before
    assert replace(resolved, graph_trace=True).pressure_policy == 'graph'


@pytest.mark.parametrize('policy,legacy', [('recency', False), ('attention', True)])
def test_explicit_pressure_enum_accepts_only_equivalent_legacy_alias(policy, legacy):
    resolved = runtime(v2_settings(pressure_policy=policy, action_pressure=legacy))
    assert resolved.pressure_policy == policy and resolved.action_pressure is legacy
    with pytest.raises(ValueError, match='conflicts'):
        runtime(v2_settings(pressure_policy=policy, action_pressure=not legacy))


@pytest.mark.parametrize('legacy', [False, True])
def test_graph_pressure_rejects_any_explicit_legacy_alias(legacy):
    with pytest.raises(ValueError, match='conflicts'):
        runtime(v2_settings(pressure_policy='graph', action_pressure=legacy))


@pytest.mark.parametrize('updates,match', [
    ({'graph_scoring_version': 'v3'}, 'graph_scoring_version'),
    ({'graph_scoring_version': []}, 'graph_scoring_version'),
    ({'graph_alpha': True}, 'graph_alpha'),
    ({'graph_alpha': float('nan')}, 'graph_alpha'),
    ({'graph_alpha': float('inf')}, 'graph_alpha'),
    ({'graph_alpha': -.1}, 'graph_alpha'),
    ({'graph_alpha': 1.1}, 'graph_alpha'),
    ({'graph_locality_weight': .1}, 'legacy graph weights'),
    ({'graph_attention_weight': .2}, 'legacy graph weights'),
    ({'graph_alpha': .2, 'graph_attention_weight': .2}, 'legacy graph weights'),
    ({'pressure_policy': 'invalid'}, 'pressure_policy'),
    ({'pressure_policy': []}, 'pressure_policy'),
    ({'pressure_policy': None}, 'pressure_policy'),
    ({'action_pressure': 1}, 'action_pressure'),
])
def test_v2_rejects_explicit_ambiguous_or_invalid_configuration(updates, match):
    payload = v2_settings()
    payload.update(updates)
    with pytest.raises(ValueError, match=match):
        runtime(payload)


def test_alpha_never_silently_changes_or_gets_ignored_by_v1():
    with pytest.raises(ValueError, match='requires graph_scoring_version=v2'):
        runtime(settings(graph_alpha=.2))
    with pytest.raises(ValueError, match='legacy graph weights'):
        runtime(settings(graph_alpha=.2, graph_locality_weight=.1))
    with pytest.raises(ValueError, match='legacy graph weights'):
        RuntimeOptions(graph_scoring_version='v2', graph_locality_weight=.1)


def test_dataclass_replacement_keeps_default_provenance_and_validates_changes():
    resolved = runtime(v2_settings())
    assert replace(resolved, pressure_policy='attention').pressure_policy == 'attention'
    assert replace(resolved, graph_alpha=.6).graph_alpha == .6
    with pytest.raises(ValueError, match='legacy graph weights'):
        replace(resolved, graph_attention_weight=.4)
    with pytest.raises(ValueError, match='conflicts'):
        replace(runtime(v2_settings(pressure_policy='graph')), action_pressure=False)


@pytest.mark.parametrize('mode', ['off', 'collect', 'locality', 'attention', 'multi'])
@pytest.mark.parametrize('policy', ['recency', 'attention', 'graph'])
def test_policy_and_graph_mode_compatibility(mode, policy):
    payload = v2_settings(pressure_policy=policy)
    payload['graph_mode'] = mode
    if policy == 'graph' and mode in ('off', 'collect'):
        with pytest.raises(ValueError, match='pressure_policy=graph requires'):
            runtime(payload)
    else:
        assert runtime(payload).pressure_policy == policy


@pytest.mark.parametrize('updates', [
    {'graph_scoring_version': 'v1'},
    {'graph_mode': 'locality', 'action_attention': False, 'prefill_attention': False},
    {'graph_mode': 'locality', 'prefill_attention': False},
    {'graph_mode': 'locality', 'action_scope': 'tool-call'},
])
def test_graph_pressure_requires_v2_and_real_attention_seeds(updates):
    payload = v2_settings(pressure_policy='graph')
    payload.update(updates)
    with pytest.raises(ValueError, match='[Gg]raph pressure|pressure_policy=graph requires'):
        runtime(payload)


@pytest.mark.parametrize('policy', ['recency', 'attention', 'graph'])
def test_direct_resolve_serialize_reload_preserves_v2_worker_policy(tmp_path, policy):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(graph_scoring_version='v2', graph_alpha=.35,
                            pressure_policy=policy)
    before = deepcopy(profile)
    plan = build_launch(profile, tmp_path, tmp_path / 'run')
    worker = worker_payload(plan)
    assert worker['graph_scoring_version'] == 'v2'
    assert worker['graph_alpha'] == .35 and worker['pressure_policy'] == policy
    assert not {'graph_locality_weight', 'graph_attention_weight', 'action_pressure'} & worker.keys()
    reloaded = build_launch(json.loads(json.dumps(plan['profile'])), tmp_path, tmp_path / 'reload')
    assert worker_payload(reloaded) == worker
    assert runtime(worker).pressure_policy == policy
    assert profile == before


def elastic_profile(tmp_path):
    direct = model_profile(tmp_path)
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'v2-test', 'api_key': 'local-test-only'}]}))
    return dict(model=direct['model'], tokenizer=direct['tokenizer'], users_file=str(users),
        context_mode='meank', draft_method='mtp', draft_tokens=4,
        active_context_tokens=8192, gen_budget_tokens=4096, max_output_tokens=512,
        request_planning=True, action_attention=True, action_fraction=0.0,
        action_scope='non-thinking', prefill_attention=True, prefill_chunk_tokens=1024,
        graph_mode='multi', graph_scoring_version='v2', pressure_policy='graph')


@pytest.mark.parametrize('aggregation', ['sum', 'max'])
def test_elastic_default_injection_and_serialized_profile_roundtrip(tmp_path, aggregation):
    profile = elastic_profile(tmp_path)
    profile['graph_aggregation'] = aggregation
    before = deepcopy(profile)
    resolved, _ = build_elastic(profile, tmp_path, tmp_path / 'elastic')
    backend = resolved['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path / 'native'))
    for payload in (resolved, backend['kvmem'], native):
        assert payload['graph_scoring_version'] == 'v2'
        assert payload['graph_alpha'] == .2 and payload['pressure_policy'] == 'graph'
        assert payload['graph_aggregation'] == aggregation
        assert not {'graph_locality_weight', 'graph_attention_weight', 'action_pressure'} & payload.keys()
    restored, _ = build_elastic(json.loads(json.dumps(resolved)), tmp_path, tmp_path / 'elastic2')
    assert restored['mean_k_backend_profile'] == backend
    assert runtime(native).pressure_policy == 'graph'
    assert profile == before


@pytest.mark.parametrize('updates,match', [
    ({'graph_locality_weight': .1}, 'legacy graph weights'),
    ({'graph_attention_weight': .2}, 'legacy graph weights'),
    ({'action_pressure': False}, 'conflicts'),
    ({'action_pressure': True}, 'conflicts'),
])
def test_elastic_rejects_explicit_fields_even_when_equal_to_old_defaults(tmp_path, updates, match):
    profile = elastic_profile(tmp_path)
    profile.update(updates)
    with pytest.raises(ValueError, match=match):
        build_elastic(profile, tmp_path, tmp_path / 'invalid')


def test_v1_serialization_omits_inactive_alpha_and_preserves_legacy_weights():
    payload = settings('collect', action_pressure=True, graph_attention_weight=.4)
    resolved = runtime(payload)
    serialized = canonical_policy_payload(payload, resolved)
    assert 'graph_alpha' not in serialized
    assert serialized['graph_attention_weight'] == .4
    assert serialized['action_pressure'] is True
    assert serialized['pressure_policy'] == 'attention'
    assert runtime(json.loads(json.dumps(serialized))) == resolved


@pytest.mark.parametrize('kind', ['direct', 'elastic'])
def test_complete_v2_examples_validate_and_leave_legacy_example_unchanged(tmp_path, kind):
    examples = Path(__file__).parents[1] / 'examples'
    original = json.loads((examples / 'query-block-graph-elastic.json').read_text())
    profile = json.loads((examples / f'graph-scoring-v2-{kind}.json').read_text())
    local = model_profile(tmp_path)
    profile.update(model=local['model'], tokenizer=local['tokenizer'])
    if kind == 'direct':
        profile['hf_config_path'] = local['model']
        worker = worker_payload(build_launch(profile, tmp_path, tmp_path / 'direct'))
        assert worker['select_budget'] == original['active_context_tokens']
        assert worker['gen_budget'] == original['gen_budget_tokens']
    else:
        users = tmp_path / 'users.json'
        users.write_text(json.dumps({'users': [{'id': 'example', 'api_key': 'local-only'}]}))
        profile['users_file'] = str(users)
        resolved, _ = build_elastic(profile, tmp_path, tmp_path / 'elastic')
        worker = resolved['mean_k_backend_profile']['kvmem']
        for name in ('active_context_tokens', 'gen_budget_tokens', 'max_num_batched_tokens'):
            assert profile[name] == original[name]
    assert worker['graph_scoring_version'] == 'v2'
    assert worker['graph_alpha'] == .2 and worker['pressure_policy'] == 'graph'
    assert runtime(worker).pressure_policy == 'graph'
    assert original['action_pressure'] is False and 'graph_scoring_version' not in original
