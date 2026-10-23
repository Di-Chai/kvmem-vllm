"""Guided prefill pressure with unchanged original-Query max propagation."""
import json
from pathlib import Path

from graph_v2_protocol import SCORE_DOMAIN, require
from ruler_pooling_protocol import expected_profile as pooling_profile

PROFILE = 'guided-pressure-pro6000-20261023'


def expected_profile(reference):
    profile = pooling_profile(reference, 'max')
    profile['port'] = 18861
    profile['kvmem'].update(guided_reselect='prefill', guided_query_tokens=512)
    return profile


def validate_execution(profile, fixture, *, diagnostic_cases=None,
                       diagnostic_gen_budget=16384, diagnostic_chunk=1024):
    require(fixture['execution_profile'] == PROFILE, 'Wrong guided-pressure fixture')
    require(profile == expected_profile(fixture['reference_profile']), 'Guided-pressure profile changed')
    require(diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024,
            'Guided-pressure diagnostic may not change K/G or chunks')
    require(diagnostic_cases is None or type(diagnostic_cases) is int and 1 <= diagnostic_cases <= 8,
            'Use an explicitly labelled fixed-prefix qualification of at most eight cases')
    require(fixture['cases'] == 260 and fixture['samples_per_task'] == 20
            and fixture['target_length'] == 262144 and fixture['workers'] == 8
            and fixture['cpu_payload_credits_bytes'] == 82820130816,
            'Original RULER cohort/admission changed')
    require(fixture['sampling'] == dict(max_tokens=8192, temperature=1., top_p=.95,
        top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
        'Sampling changed')


def audit_case(capture, response_id, profile, query, *, require_pressure=False, event_reader=None):
    import qualify_block_graph as q
    reader = event_reader or q.records
    match = lambda e: q.native_matches({'response_id': response_id}, e)
    refresh = [e for e in reader(Path(capture)/'bounded_refresh.jsonl') if match(e)]
    graphs = [e for e in reader(Path(capture)/'block_graph.jsonl') if match(e)]
    queries = [e for e in refresh if e['selection'] == 'query']
    pressure = [e for e in refresh if e['selection'] == 'pressure']
    require(len(queries) == 1, 'Need one original public Query selection')
    raw = queries[0]
    require(raw['query_span'] == list(query) and raw['guided_query_tokens'] == 0
            and raw['query_score_tokens'] == query[1]-query[0]
            and raw['mean_k_backend'] == 'qw3_cuda' and not raw['query_history_enabled'],
            'Final original Query was replaced or contaminated')
    require(not require_pressure or pressure, 'Missing prefill pressure')
    pg = [e for e in graphs if e['phase'] == 'pressure']
    require(len(pg) == len(pressure), 'Missing guided-pressure graph evidence')
    for r,g in zip(pressure, pg):
        require(r.get('guided_scope') == 'prefill-pressure'
                and r['guided_context_end'] == r['history_begin']
                and r['public_input_replayed_for_guidance'] == 0
                and r['history_begin'] <= query[0]
                and not g['propagation_applied'] and g['transport_mask_count'] == 0,
                'Guidance used future input or changed pressure propagation')
        published = r.get('guided_published_tokens', 0)
        if published:
            require(r['seed_source'] == 'guided_query' and g['seed_source'] == 'guided_query'
                    and r['pressure_policy'] == 'guided_query'
                    and r['query_score_tokens'] == published
                    and r['guided_natural_termination'] and r['guided_fallback'] is None,
                    'Guided pressure did not actually consume the completed private Query')
        else:
            require(r['guided_fallback'] in ('capacity','empty-output','structured-output','unterminated-output')
                    and r['seed_source'] == 'prefill' and r['query_score_tokens'] == 0,
                    'Guided failure was not an explicit cumulative-attention fallback')
        require(0 < r['selected_tokens'] <= profile['kvmem']['select_budget']
                and r['score_domain_version'] == SCORE_DOMAIN, 'Wrong selection domain/budget')
    semantic = [e for e in graphs if e['phase'] == 'query']
    require(len(semantic) == 1 and semantic[0]['aggregation'] == 'max'
            and semantic[0]['alpha'] == .2 and semantic[0]['seed_source'] == 'mean_k',
            'Final original Query graph policy changed')
    return dict(query=raw, pressure=pressure, graph=graphs,
        phases=dict(query=1, pressure=len(pressure)),
        seed_sources=sorted({e['seed_source'] for e in refresh}), score_domain_version=SCORE_DOMAIN)


def audit_optimized(capture, profile):
    from ruler_pooling_protocol import audit_optimized as common
    # Common native-QK/publication/CPU-history checks expect private boundary
    # guidance for a nonzero cap. Suppress only that old location-specific test.
    from copy import deepcopy
    base = deepcopy(profile)
    base['kvmem']['guided_query_tokens'] = 0
    result = common(capture, base)
    rows = [json.loads(line) for line in (capture/'bounded_refresh.jsonl').open()]
    pressures = [r for r in rows if r['selection'] == 'pressure']
    queries = [r for r in rows if r['selection'] == 'query']
    require(pressures and all(r.get('guided_scope') == 'prefill-pressure' for r in pressures),
            'An ordinary pressure boundary bypassed requested guidance')
    require(all(r['guided_query_tokens'] == 0 for r in queries), 'Final Query generated private guidance')
    published = sum(r.get('guided_published_tokens', 0) > 0 for r in pressures)
    require(published > 0, 'No guided-pressure Query was published')
    result['guided'] = dict(scope='prefill-pressure', requested_cap=512,
        attempts=len(pressures), published=published,
        fallback=sum(bool(r.get('guided_fallback')) for r in pressures),
        consumed_tokens=sum(r['guided_query_tokens'] for r in pressures),
        rollback_verified=sum(r.get('guided_state_verified', False) for r in pressures),
        final_query_guided_attempts=0)
    return result
