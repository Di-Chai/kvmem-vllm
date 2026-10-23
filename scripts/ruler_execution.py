"""Explicit execution profiles for the frozen 13-task RULER cohort.

The optimized profile changes execution, retaining the established K/G,
sampling and selection policy. Historical fixtures still require 1K chunks.
"""
from guided_benchmark_protocol import require, validate_profile


OPTIMIZED = 'optimized-20261022'
ENABLED = (
    'action_fused_scoring', 'graph_native_propagation',
    'action_native_decode_commit', 'action_download_pool',
    'prefill_publication_pipeline', 'prefill_native_plan_arm',
    'prefill_capture_ahead', 'prefill_action_before_mtp',
    'capture_fused_pack', 'capture_opaque_projection',
    'cpu_direct_record_seal', 'cpu_seal_serial_allocate',
    'prefill_physical_coalescing',
)
DISABLED = (
    'prefill_deferred_validation', 'prefill_scatter_after_target',
    'cpu_mean_index_simd', 'action_native_prefill_aligned_pieces',
    'action_native_prefill_cta_gate', 'action_native_prefill_static_aligned',
)


def optimized(fixture):
    mode = fixture.get('execution_profile', 'historical')
    if mode in ('pooling-pro6000-20261022', 'guided-pressure-pro6000-20261023'):
        return True
    require(mode in ('historical', OPTIMIZED), 'Unknown RULER execution profile')
    return mode == OPTIMIZED


def validate_execution(profile, fixture, *, diagnostic_cases=None,
                       diagnostic_gen_budget=16384, diagnostic_chunk=1024):
    if fixture.get('execution_profile') == 'guided-pressure-pro6000-20261023':
        from ruler_guided_pressure_protocol import validate_execution as validate_guided_pressure
        return validate_guided_pressure(profile, fixture, diagnostic_cases=diagnostic_cases,
            diagnostic_gen_budget=diagnostic_gen_budget, diagnostic_chunk=diagnostic_chunk)
    if fixture.get('execution_profile') == 'pooling-pro6000-20261022':
        from ruler_pooling_protocol import validate_execution as validate_pooling
        return validate_pooling(profile, fixture, diagnostic_cases=diagnostic_cases,
            diagnostic_gen_budget=diagnostic_gen_budget, diagnostic_chunk=diagnostic_chunk)
    use_optimized = optimized(fixture)
    require((diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024) or
            diagnostic_cases is not None,
            'Changing budget or chunk requires an explicit diagnostic cohort')
    require(not use_optimized or (diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024),
            'The optimized suite has frozen G16K and logical 2K execution')
    validate_profile(profile, 'off', 'niah', expected_gen_budget=diagnostic_gen_budget,
                     expected_chunk=2048 if use_optimized else diagnostic_chunk)
    if not use_optimized:
        return
    kv = profile['kvmem']
    requested = fixture.get('query_replay_execution')
    if requested is not None:
        require(requested == dict(backend=kv.get('query_replay_backend', 'eager'),
                                  cobatch=kv.get('query_replay_cobatch', False)) and
                requested['backend'] in ('eager', 'compiled') and type(requested['cobatch']) is bool,
                'Frozen Query replay execution differs')
    require(all(kv.get(name) is True for name in ENABLED), 'Missing qualified RULER optimization')
    require(all(kv.get(name, False) is False for name in DISABLED),
            'Unqualified prototype entered the RULER profile')
    require(kv['action_native_scores'] == 'reuse' and kv['assembly_chunk_blocks'] == 128 and
            kv['prefill_physical_max_chunks'] == 7 and kv['action_native_max_rows'] == 512 and
            kv['cpu_seal_hash_workers'] == 8 and kv['cpu_seal_pack_workers'] == 1,
            'Optimized RULER execution capacity differs')
    require((kv['prefill_attention_interval'], kv['prefill_attention_stride'],
             kv['prefill_attention_half_life']) == (1024, 32, 8192) and
            kv['action_layers'] == 'spread3' and kv['action_scope'] == 'non-thinking',
            'Attention sampling policy changed')


def audit_optimized(capture, profile):
    import json
    from benchmark_matched_prefill import audit_native_stats
    from benchmark_prefill_publication_pipeline import audit_pipeline
    from benchmark_prefill_variant import audit_optional_branches

    read = lambda name: json.loads((capture / name).read_text())
    native, progress, archive = (read(name) for name in
        ('native_attention_stats.json', 'progress.json', 'archive.json'))
    audit_native_stats(native)
    pipeline = audit_pipeline(progress, True)
    branches = audit_optional_branches(profile, progress, archive, native)
    logical = audit_logical_publication(progress.get('logical_prefill_stats', {}))
    query_replay = None
    if (profile['kvmem'].get('query_replay_backend', 'eager') != 'eager' or
            profile['kvmem'].get('query_replay_cobatch', False)):
        rows = [json.loads(line) for line in (capture/'bounded_refresh.jsonl').open()]
        query_replay = audit_query_replay(rows, profile)
    return dict(native=native, publication=pipeline, branches=branches, logical=logical,
                query_replay=query_replay,
                fallback_counts={k: v for k, v in native['layer_phase_counts'].items()
                                 if k.startswith('fallback:')})


def audit_query_replay(rows, profile):
    kv = profile['kvmem']
    backend, cobatch = kv.get('query_replay_backend', 'eager'), kv.get('query_replay_cobatch', False)
    queries = [r for r in rows if r['selection'] == 'query']
    allowed = {backend, 'cobatch'} if cobatch else {backend}
    require(queries and all(r.get('query_execution_counts') and
            all(mode in allowed and type(count) is int and count > 0
                for mode, count in r['query_execution_counts'].items()) for r in queries),
            'Query replay executed an unexpected or unreported mode')
    counts = {mode: sum(r['query_execution_counts'].get(mode, 0) for r in queries)
              for mode in ('eager', 'compiled', 'cobatch')}
    if cobatch:
        require(counts['cobatch'] > 0, 'Requested Query co-batching never executed')
        snapshots = [r.get('query_replay_cobatch') for r in queries]
        require(all(r and r['enabled'] and not r['poisoned'] and not r['active'] and
                    not r.get('quarantined', 0) and r.get('peak_bytes', 0) <= kv['query_replay_max_bytes']
                    for r in snapshots), 'Query co-batch lifecycle or memory budget failed')
    return dict(counts=counts, queries=len(queries), passed=True)


def audit_logical_publication(logical):
    # The qualified scheduler deliberately coalesces only one live request.
    # An eight-client quality run can therefore correctly execute zero groups.
    keys = ('planned_physical_groups', 'projected_physical_groups',
            'planned_logical_chunks', 'projected_logical_chunks')
    counts = {key: logical.get(key, 0) for key in keys}
    require(logical.get('enabled') is True and
            all(type(value) is int and value >= 0 for value in counts.values()) and
            counts['planned_physical_groups'] == counts['projected_physical_groups'] and
            counts['planned_logical_chunks'] == counts['projected_logical_chunks'] and
            2 * counts['projected_physical_groups'] <= counts['projected_logical_chunks'] <=
                7 * counts['projected_physical_groups'],
            'Physical coalescing has incomplete logical publication')
    return dict(**logical, effective=counts['projected_physical_groups'] > 0,
                eligibility='Only one live request; concurrent owners retain ordinary logical prefill.')
