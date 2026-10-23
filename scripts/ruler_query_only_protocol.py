"""Explicit A800 Query-only ablation of the frozen optimized RULER profile.

The frozen campaign adapters reuse the existing client, queue and scorer. Only
their policy validation, hardware audit and report labels use this module.
Alpha zero retains the same Query picker with action=False and no prefill-score
fallback, and leaves the cumulative-attention pressure policy intact.
"""
from copy import deepcopy
import json
from pathlib import Path

from graph_v2_protocol import SCORE_DOMAIN, require


PROFILE = 'query-only-a800-20261022'
PATH_OVERRIDES = ('model', 'tokenizer', 'hf_config_path', 'host', 'port')


def optimized(fixture):
    require(fixture.get('execution_profile') == PROFILE, 'Wrong Query-only fixture profile')
    return True


def expected_profile(reference, overrides):
    require(set(overrides) == set(PATH_OVERRIDES), 'Unexpected platform override')
    require(reference['kvmem']['graph_mode'] == 'multi' and
            reference['kvmem']['graph_alpha'] == .2 and
            reference['kvmem']['pressure_policy'] == 'attention', 'Wrong reference policy')
    result = deepcopy(reference)
    result.update(overrides)
    result.pop('linear_backend_per_quant', None)
    result['kvmem'].update(graph_alpha=0., prefill_physical_coalescing=False)
    return result


def validate_profile(profile, *args, **kwargs):
    kv = profile['kvmem']
    require(profile['execution_mode'] == 'compiled_graph' and
            profile['draft_method'] == 'mtp' and profile['draft_tokens'] == 4,
            'Keep compiled native MTP4')
    require((kv['select_budget'], kv['gen_budget'], kv['sink_blocks'],
             kv['recent_blocks'], kv['block_tokens']) == (36864, 16384, 41, 0, 32),
            'Query-only budget differs from local reference')
    require(kv['graph_mode'] == 'multi' and kv['graph_alpha'] == 0. and
            kv['graph_scoring_version'] == 'v2' and kv['pressure_policy'] == 'attention' and
            kv['action_fraction'] == 0. and not kv['query_history'] and kv['guided_query_tokens'] == 0,
            'Query-only ablation changed another selection policy')


def validate_execution(profile, fixture, *, diagnostic_cases=None,
                       diagnostic_gen_budget=16384, diagnostic_chunk=1024):
    optimized(fixture)
    require(diagnostic_cases is None and diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024,
            'This campaign requires the complete frozen cohort')
    require(profile == expected_profile(fixture['reference_profile'], fixture['platform_path_overrides']),
            'A setting outside the explicit ablation/platform allowlist changed')
    validate_profile(profile)
    require(fixture['cases'] == 260 and fixture['samples_per_task'] == 20 and
            fixture['target_length'] == 262144 and fixture['workers'] == 8 and
            fixture['cpu_payload_credits_bytes'] == 82820130816 and
            fixture['request_timeout_seconds'] == 3600 and fixture['starts_in_thinking'] is True,
            'Cohort, admission or timeout differs from the local reference')
    require(fixture['sampling'] == dict(max_tokens=8192, temperature=1., top_p=.95,
            top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
            'Sampling or output budget differs from the local reference')


def audit_case(capture, response_id, profile, query, *, require_pressure=False, event_reader=None):
    import qualify_block_graph as q
    reader = event_reader or q.records
    matches = lambda event: q.native_matches({'response_id': response_id}, event)
    refresh = [r for r in reader(Path(capture)/'bounded_refresh.jsonl') if matches(r)]
    graphs = [r for r in reader(Path(capture)/'block_graph.jsonl') if matches(r)]
    queries = [r for r in refresh if r['selection'] == 'query']
    pressure = [r for r in refresh if r['selection'] == 'pressure']
    require(len(queries) == 1, 'Independent case must execute exactly one semantic Query')
    selected = queries[0]
    require(selected['query_span'] == list(query) and selected['mean_k_backend'] == 'qw3_cuda' and
            not selected['query_history_enabled'] and selected['action_history_pages'] == 0 and
            selected['guided_query_tokens'] == 0 and selected.get('prefill_attention_samples', 0) > 0,
            'Query inputs/backend, guidance or answer isolation changed')
    require(not require_pressure or pressure, 'Full input did not execute pressure selection')
    for row in refresh:
        require(0 < row['selected_tokens'] <= profile['kvmem']['select_budget'] and
                row['scoring_version'] == 'v2' and row['score_domain_version'] == SCORE_DOMAIN and
                row['pressure_policy'] == 'attention' and row['alpha'] == 0.,
                'Actual selection policy differs from the Query-only profile')
    pressure_graphs = [r for r in graphs if r['phase'] == 'pressure']
    require(len(pressure_graphs) == len(pressure), 'Missing pressure evidence')
    for event, picked in zip(pressure_graphs, pressure):
        require(picked['seed_source'] == 'prefill' and picked['guided_query_tokens'] == 0 and
                event['propagation_applied'] is False and event['transport_mask_count'] == 0 and
                all(event[key] == picked[key] for key in (
                    'seed_source', 'selected_tokens', 'score_mask_count', 'transport_mask_count')),
                'Pressure did not use unchanged cumulative input attention')
    semantic = [r for r in graphs if r['phase'] == 'query']
    require(len(semantic) == 1, 'Missing Query selection audit')
    audit_query_report(semantic[0], selected['history_begin'])
    return dict(query=selected, pressure=pressure, graph=graphs,
                phases=dict(query=1, pressure=len(pressure)),
                seed_sources=sorted({r['seed_source'] for r in refresh}),
                score_domain_version=SCORE_DOMAIN)


def audit_query_report(row, history_begin=0):
    require(row['mode'] == 'multi' and row['alpha'] == 0. and row['seed_source'] == 'mean_k' and
            row['accepted_through'] >= history_begin and row['identity'] is True and
            row['locality_weight'] == row['attention_weight'] == 0. and
            row['raw_l1_change'] == row['normalized_l1_change'] == 0. and
            row['added_blocks'] == row['displaced_blocks'] == 0,
            'Graph changed the raw Query score vector or selected pages')


def audit_native_stats(native):
    require(native['requested_mode'] == 'reuse' and
            native['disabled_reason'] == 'unsupported_architecture_requires_sm12x' and
            native['allocated_bytes'] == native['validation_failures'] == 0 and
            sum(value for key, value in native['layer_phase_counts'].items()
                if key.startswith('fallback:')) > 0,
            'Unexpected A800 native-score execution or fallback')


def audit_optimized(capture, profile):
    read = lambda name: json.loads((capture/name).read_text())
    native, progress, archive = (read(name) for name in
        ('native_attention_stats.json', 'progress.json', 'archive.json'))
    audit_native_stats(native)
    require(not profile['kvmem']['prefill_physical_coalescing'] and
            not progress.get('logical_prefill_stats', {}).get('enabled'),
            'Unqualified native-QK coalescing ran on A800')
    require(archive.get('disk_read_bytes', 0) == archive.get('disk_write_bytes', 0) == 0,
            'CPU history used disk')
    packing = progress.get('capture_pack_stats', {})
    require(packing.get('fallback:device_capability', 0) > 0 and
            not packing.get('fused_batches', 0), 'Unexpected A800 capture packing backend')
    return dict(passed=True, native_qk_fallback=native['disabled_reason'],
                native_score_counts=native['layer_phase_counts'], capture_packing=packing,
                physical_coalescing=False, query_alpha=0., pressure_policy='attention')
