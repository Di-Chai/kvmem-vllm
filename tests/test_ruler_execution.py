"""An execution optimization must not silently change RULER's quality policy."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ruler_execution import ENABLED, OPTIMIZED, optimized, validate_execution, audit_logical_publication
from ruler_execution import audit_query_replay


@pytest.fixture
def profile():
    return dict(model_format='safetensors', execution_mode='compiled_graph',
        draft_method='mtp', draft_tokens=4, verify_session_restore=False,
        max_num_seqs=8, max_num_batched_tokens=16424, prefill_chunk_tokens=2048,
        kvmem=dict(**{name: True for name in ENABLED},
            select_budget=36864, gen_budget=16384, block_tokens=32, sink_blocks=41, recent_blocks=0,
            graph_mode='multi', graph_scoring_version='v2', graph_alpha=.2, pressure_policy='attention',
            action_fraction=0., query_history=False, archive_backend='cpu',
            cpu_read_cache_bytes=100000000000, repository_gc=False,
            guided_reselect='boundary', guided_query_backend='eager', guided_query_tokens=0,
            session_freeze_prompt=False, action_native_scores='reuse', assembly_chunk_blocks=128,
            prefill_physical_max_chunks=7, action_native_max_rows=512, cpu_seal_hash_workers=8,
            cpu_seal_pack_workers=1, prefill_attention_interval=1024, prefill_attention_stride=32,
            prefill_attention_half_life=8192, action_layers='spread3', action_scope='non-thinking'))


def test_explicit_optimized_profile_preserves_selection_and_sampling(profile):
    before = deepcopy(profile)
    validate_execution(profile, dict(execution_profile=OPTIMIZED))
    assert profile == before
    with pytest.raises(ValueError, match='prefill chunk'):
        validate_execution(profile, {})


@pytest.mark.parametrize('key,value', [
    ('select_budget', 32768), ('gen_budget', 65536), ('guided_query_tokens', 512),
    ('pressure_policy', 'query'), ('graph_alpha', .5), ('query_history', True),
    ('prefill_attention_stride', 64), ('action_native_scores', 'off'),
    ('cpu_seal_serial_allocate', False), ('prefill_physical_coalescing', False),
    ('action_native_prefill_static_aligned', True), ('cpu_mean_index_simd', True),
])
def test_optimized_profile_rejects_policy_drift_or_unqualified_mode(profile, key, value):
    profile['kvmem'][key] = value
    with pytest.raises(ValueError):
        validate_execution(profile, dict(execution_profile=OPTIMIZED))


def test_historical_profile_and_diagnostics_stay_explicit(profile):
    profile['prefill_chunk_tokens'] = 1024
    validate_execution(profile, {})
    profile['prefill_chunk_tokens'] = 2048
    with pytest.raises(ValueError, match='diagnostic cohort'):
        validate_execution(profile, {}, diagnostic_chunk=2048)
    validate_execution(profile, {}, diagnostic_cases=8, diagnostic_chunk=2048)
    with pytest.raises(ValueError, match='frozen G16K'):
        validate_execution(profile, dict(execution_profile=OPTIMIZED),
                           diagnostic_cases=8, diagnostic_chunk=2048)
    with pytest.raises(ValueError, match='Unknown'):
        optimized(dict(execution_profile='typo'))


def test_concurrent_run_may_have_no_eligible_physical_coalescing():
    assert audit_logical_publication(dict(enabled=True))['effective'] is False
    counts = dict(enabled=True, planned_physical_groups=2, projected_physical_groups=2,
                  planned_logical_chunks=14, projected_logical_chunks=14)
    assert audit_logical_publication(counts)['effective'] is True
    for change in (dict(projected_logical_chunks=13), dict(planned_physical_groups=3),
                   dict(planned_logical_chunks=15, projected_logical_chunks=15)):
        with pytest.raises(ValueError, match='incomplete'):
            audit_logical_publication({**counts, **change})


def test_query_replay_configuration_is_bound_to_frozen_fixture(profile):
    fixture = dict(execution_profile=OPTIMIZED,
                   query_replay_execution=dict(backend='compiled', cobatch=True))
    with pytest.raises(ValueError, match='Frozen Query'):
        validate_execution(profile, fixture)
    profile['kvmem'].update(query_replay_backend='compiled', query_replay_cobatch=True)
    validate_execution(profile, fixture)


def test_query_replay_audit_requires_actual_execution_and_clean_lifecycle(profile):
    profile['kvmem'].update(query_replay_backend='compiled', query_replay_cobatch=True,
                           query_replay_max_bytes=1024)
    snapshot = dict(enabled=True, active=False, poisoned=False, peak_bytes=1000)
    rows = [dict(selection='query', query_execution_counts=dict(compiled=1),
                 query_replay_cobatch=snapshot),
            dict(selection='query', query_execution_counts=dict(cobatch=1),
                 query_replay_cobatch=snapshot)]
    assert audit_query_replay(rows, profile)['counts'] == dict(eager=0, compiled=1, cobatch=1)
    for damage in ('no_execution', 'wrong_mode', 'poisoned', 'over_budget', 'active'):
        changed = deepcopy(rows)
        if damage == 'no_execution':
            changed[1]['query_execution_counts'] = dict(compiled=1)
        elif damage == 'wrong_mode':
            changed[1]['query_execution_counts'] = dict(eager=1)
        else:
            changed[1]['query_replay_cobatch'].update(
                {dict(poisoned='poisoned', over_budget='peak_bytes', active='active')[damage]:
                 1025 if damage == 'over_budget' else True})
        with pytest.raises(ValueError):
            audit_query_replay(changed, profile)
