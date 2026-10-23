"""Frozen Pro6000 max / Query-only / max+guided RULER ablations."""
from copy import deepcopy
import json
from pathlib import Path

from graph_v2_protocol import SCORE_DOMAIN, require

PROFILE = 'pooling-pro6000-20261022'
ARMS = ('max', 'query_only', 'max_guided')
PORTS = dict(zip(ARMS, (18858, 18859, 18860)))


def expected_profile(reference, arm):
    require(arm in ARMS, 'Unknown pooling arm')
    p = deepcopy(reference)
    p['port'] = PORTS[arm]
    # Guidance cannot use physical coalescing; disable it in every arm so the
    # max/guided pair differs only in its private generation cap.
    p['kvmem'].update(prefill_physical_coalescing=False,
        graph_aggregation='sum' if arm == 'query_only' else 'max',
        graph_alpha=0. if arm == 'query_only' else .2,
        guided_query_tokens=512 if arm == 'max_guided' else 0)
    return p


def optimized(fixture):
    require(fixture.get('execution_profile') == PROFILE, 'Wrong pooling fixture')
    return True


def validate_execution(profile, fixture, *, diagnostic_cases=None,
                       diagnostic_gen_budget=16384, diagnostic_chunk=1024):
    optimized(fixture)
    require(diagnostic_cases is None and diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024,
            'Pooling campaign requires all 260 original cases')
    require(profile == expected_profile(fixture['reference_profile'], fixture['pooling_arm']),
            'A setting outside the explicit pooling allowlist changed')
    kv = profile['kvmem']
    require(profile['execution_mode'] == 'compiled_graph' and profile['draft_method'] == 'mtp'
            and profile['draft_tokens'] == 4 and profile['prefill_chunk_tokens'] == 2048
            and profile['max_num_seqs'] == 8 and profile['max_num_batched_tokens'] == 16424,
            'Keep local compiled MTP4 and eight-owner 2K execution')
    require((kv['select_budget'], kv['gen_budget'], kv['sink_blocks'], kv['recent_blocks'],
             kv['block_tokens']) == (36864, 16384, 41, 0, 32)
            and kv['pressure_policy'] == 'attention' and kv['graph_scoring_version'] == 'v2'
            and kv['graph_mode'] == 'multi' and not kv['query_history'] and kv['action_fraction'] == 0,
            'Budget, score domain or pressure policy changed')
    require(kv['guided_query_backend'] == 'eager' and kv['guided_reselect'] == 'boundary',
            'Use the previously qualified static explicit-Query guidance')
    require(fixture['cases'] == 260 and fixture['samples_per_task'] == 20
            and fixture['target_length'] == 262144 and fixture['workers'] == 8
            and fixture['cpu_payload_credits_bytes'] == 82820130816
            and fixture['request_timeout_seconds'] == 3600 and fixture['starts_in_thinking'] is True,
            'Cohort, admission or response extraction changed')
    require(fixture['sampling'] == dict(max_tokens=8192, temperature=1., top_p=.95,
            top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
            'Sampling differs from the frozen local cohort')


def audit_case(capture, response_id, profile, query, *, require_pressure=False, event_reader=None):
    import qualify_block_graph as q
    reader = event_reader or q.records
    match = lambda e: q.native_matches({'response_id': response_id}, e)
    refresh = [e for e in reader(Path(capture)/'bounded_refresh.jsonl') if match(e)]
    graphs = [e for e in reader(Path(capture)/'block_graph.jsonl') if match(e)]
    queries = [e for e in refresh if e['selection'] == 'query']
    pressure = [e for e in refresh if e['selection'] == 'pressure']
    require(len(queries) == 1, 'Independent case must execute exactly one semantic Query')
    selected, kv = queries[0], profile['kvmem']
    require(selected['query_span'] == list(query) and selected['mean_k_backend'] == 'qw3_cuda'
            and not selected['query_history_enabled'] and selected['action_history_pages'] == 0
            and selected.get('prefill_attention_samples', 0) > 0,
            'Query coordinates/backend or accepted-history isolation changed')
    require(0 <= selected['guided_query_tokens'] <= kv['guided_query_tokens'], 'Private generation exceeded cap')
    published = selected.get('guided_published_tokens', 0)
    if published:
        require(kv['guided_query_tokens'] and selected['guided_natural_termination']
                and selected['guided_fallback'] is None and selected['query_score_tokens'] == published,
                'Incomplete private generation entered Query scoring')
    elif kv['guided_query_tokens']:
        require(selected.get('guided_fallback') in ('capacity','empty-output','structured-output','unterminated-output'),
                'Requested private Query neither published nor reported a qualified fallback')
    require(not require_pressure or pressure, 'Full input lacks pressure selections')
    for e in refresh:
        require(0 < e['selected_tokens'] <= kv['select_budget']
                and e['scoring_version'] == 'v2' and e['score_domain_version'] == SCORE_DOMAIN
                and e['pressure_policy'] == 'attention' and e['alpha'] == kv['graph_alpha'],
                'Actual selection policy differs')
    pg = [e for e in graphs if e['phase'] == 'pressure']
    require(len(pg) == len(pressure), 'Missing pressure evidence')
    for e,p in zip(pg,pressure):
        require(p['seed_source'] == 'prefill' and p['guided_query_tokens'] == 0
                and e['propagation_applied'] is False and e['transport_mask_count'] == 0
                and all(e[k] == p[k] for k in ('seed_source','selected_tokens','score_mask_count','transport_mask_count')),
                'Ordinary prefill pressure changed from cumulative attention')
    semantic = [e for e in graphs if e['phase'] == 'query']
    require(len(semantic) == 1, 'Missing Query scoring report')
    e = semantic[0]
    require(e['mode'] == 'multi' and e['alpha'] == kv['graph_alpha']
            and e.get('aggregation','sum') == kv['graph_aggregation']
            and e['seed_source'] == 'mean_k' and e['accepted_through'] >= selected['history_begin'],
            'Wrong actual pooling operator')
    if kv['graph_alpha'] == 0:
        require(e['identity'] and e['raw_l1_change'] == 0
                and e['added_blocks'] == e['displaced_blocks'] == 0, 'Query-only arm changed raw ranking')
    else:
        require(e['pooling_source'] == 'raw_seed' and e['score_sum_conserved'] is False,
                'Expected raw-seed max pooling')
    return dict(query=selected, pressure=pressure, graph=graphs,
        phases=dict(query=1,pressure=len(pressure)), seed_sources=sorted({e['seed_source'] for e in refresh}),
        score_domain_version=SCORE_DOMAIN)


def audit_optimized(capture, profile):
    from benchmark_matched_prefill import audit_native_stats
    from benchmark_prefill_publication_pipeline import audit_pipeline
    from benchmark_prefill_variant import audit_optional_branches
    read = lambda name: json.loads((capture/name).read_text())
    native, progress, archive = (read(name) for name in
        ('native_attention_stats.json','progress.json','archive.json'))
    audit_native_stats(native)
    require(not progress.get('logical_prefill_stats',{}).get('enabled'), 'Disabled physical coalescing executed')
    require(archive.get('disk_read_bytes',0) == archive.get('disk_write_bytes',0) == 0, 'CPU history used SSD')
    rows = [json.loads(line) for line in (capture/'bounded_refresh.jsonl').open()]
    queries = [e for e in rows if e['selection'] == 'query']
    allowed = {'compiled','cobatch'} | ({'eager'} if profile['kvmem']['guided_query_tokens'] else set())
    require(queries and all(e.get('query_execution_counts') and
            set(e['query_execution_counts']) <= allowed for e in queries), 'Unexpected private-forward backend')
    guided = dict(requested_cap=profile['kvmem']['guided_query_tokens'],
        published=sum(e.get('guided_published_tokens',0) > 0 for e in queries),
        fallback=sum(e.get('guided_fallback') is not None for e in queries),
        consumed_tokens=sum(e['guided_query_tokens'] for e in queries))
    if guided['requested_cap']:
        require(guided['published'] > 0, 'Guided arm published no private Query')
    return dict(passed=True,native=native,publication=audit_pipeline(progress,True),
        branches=audit_optional_branches(profile,progress,archive,native),guided=guided,
        graph_aggregation=profile['kvmem']['graph_aggregation'],alpha=profile['kvmem']['graph_alpha'])
