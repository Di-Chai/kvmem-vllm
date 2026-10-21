"""Two-arm guided Query accuracy experiment; original inputs and scorers."""
from copy import deepcopy

from graph_v2_protocol import SCORE_DOMAIN, release_owner, require, sha

SCHEMA = 'kvmem-guided-query-benchmark-1'


def infer_arm(profile):
    return 'guided' if profile['kvmem']['guided_query_tokens'] else 'off'


def validate_profile(profile, arm, suite='niah', *, expected_gen_budget=16384, expected_chunk=1024):
    kv = profile['kvmem']
    require(arm in ('off', 'guided') and infer_arm(profile) == arm, 'Unknown guided arm')
    require(profile['model_format'] == 'safetensors' and profile['draft_method'] == 'mtp'
            and profile['draft_tokens'] == 4 and profile['execution_mode'] == 'compiled_graph',
            'Keep the previously qualified C arm: NVFP4/FP8/compiled MTP4')
    require(expected_gen_budget in (16384, 65536), 'Unsupported diagnostic generation budget')
    require((kv['select_budget'], kv['gen_budget'], kv['block_tokens'],
             kv['sink_blocks'], kv['recent_blocks']) == (36864, expected_gen_budget, 32, 41, 0),
            'Keep K36K/sink1312/recent0 and the explicitly expected generation budget')
    require(kv['graph_mode'] == 'multi' and kv['graph_scoring_version'] == 'v2'
            and kv['graph_alpha'] == .2 and kv['pressure_policy'] == 'attention'
            and kv['action_fraction'] == 0 and not kv['query_history'],
            'Keep current attention pressure and Query graph policy')
    require(kv['archive_backend'] == 'cpu' and kv['cpu_read_cache_bytes'] == 100000000000
            and not kv['repository_gc'] and not profile['verify_session_restore'],
            'Accuracy run requires CPU history, no GC and no duplicate verification forwards')
    require(kv['guided_reselect'] == 'boundary' and kv['guided_query_backend'] == 'eager'
            and kv['guided_query_tokens'] == (512 if arm == 'guided' else 0),
            'Independent explicit queries use qw3 raw boundary guidance')
    require(expected_chunk in (1024, 2048), 'Unsupported diagnostic prefill chunk')
    require(profile['max_num_seqs'] == 8 and profile['prefill_chunk_tokens'] == expected_chunk
            and profile['max_num_batched_tokens'] == 16424,
            'Keep eight clients, batch16424 and the explicitly expected prefill chunk')
    require(kv['session_freeze_prompt'] == (suite == 'memoryagentbench'),
            'MAB must restore frozen P; independent histories must not share answers')


def validate_pair(off, guided, suite):
    for name, profile in (('off', off), ('guided', guided)):
        validate_profile(profile, name, suite)
    copies = [deepcopy(p) for p in (off, guided)]
    for profile in copies:
        profile['kvmem'].pop('guided_query_tokens')
    require(copies[0] == copies[1], 'A non-guided setting differs between arms')


def audit_case(capture, response_id, profile, query, *, require_pressure=False, event_reader=None):
    from pathlib import Path
    import qualify_block_graph as q
    reader = event_reader or q.records
    match = lambda e: q.native_matches({'response_id': response_id}, e)
    refresh = [e for e in reader(Path(capture)/'bounded_refresh.jsonl') if match(e)]
    graph = [e for e in reader(Path(capture)/'block_graph.jsonl') if match(e)]
    queries = [e for e in refresh if e['selection'] == 'query']
    pressure = [e for e in refresh if e['selection'] == 'pressure']
    require(len(queries) == 1, 'Independent case must execute exactly one semantic Query')
    selected = queries[0]
    kv = profile['kvmem']
    require(selected['query_span'] == list(query) and selected['mean_k_backend'] == 'qw3_cuda'
            and not selected['query_history_enabled'] and selected['action_history_pages'] == 0,
            'Query coordinates/backend or independent-answer isolation changed')
    require(0 <= selected['guided_query_tokens'] <= kv['guided_query_tokens'], 'Private cap exceeded')
    require(selected.get('prefill_attention_samples', 0) > 0, 'Missing input attention')
    if selected.get('guided_published_tokens', 0):
        require(selected['guided_natural_termination'] and selected['guided_fallback'] is None
                and selected['query_score_tokens'] == selected['guided_published_tokens'],
                'Incomplete private query entered scoring')
    if require_pressure:
        require(pressure, 'Full long input did not execute pressure ingestion')
    for event in refresh:
        require(0 < event['selected_tokens'] <= kv['select_budget'], 'Selection exceeds K')
        require(event['scoring_version'] == 'v2' and event['score_domain_version'] == SCORE_DOMAIN
                and event['pressure_policy'] == 'attention' and event['alpha'] == .2,
                'Actual score policy differs from requested policy')
    for event in pressure:
        require(event['guided_query_tokens'] == 0 and event['seed_source'] == 'prefill',
                'Cold prefix pressure changed strategy')
    pressure_graph = [e for e in graph if e['phase'] == 'pressure']
    require(len(pressure_graph) == len(pressure), 'Missing actual pressure graph reports')
    for event, selected_pressure in zip(pressure_graph, pressure):
        require(event['propagation_applied'] is False and event['transport_mask_count'] == 0
                and all(event[key] == selected_pressure[key] for key in (
                    'seed_source', 'selected_tokens', 'score_mask_count', 'transport_mask_count')),
                'Graph propagation leaked into ordinary attention pressure')
    semantic = [e for e in graph if e['phase'] == 'query']
    require(len(semantic) == 1 and semantic[0]['mode'] == 'multi'
            and semantic[0]['alpha'] == .2 and semantic[0]['accepted_through'] >= selected['history_begin'],
            'Missing actual Query graph propagation')
    return dict(query=selected, pressure=pressure, graph=graph,
                phases=dict(query=len(queries), pressure=len(pressure)),
                seed_sources=sorted({e['seed_source'] for e in refresh}),
                score_domain_version=SCORE_DOMAIN)
