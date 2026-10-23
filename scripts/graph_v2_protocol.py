"""Shared, fail-closed A/C/D benchmark contracts; no model execution on import."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'kvmem-graph-v2-benchmark-1'
SCORE_DOMAIN = 'qw3-native-v2'
ARMS = ('A', 'C', 'D')
SUITES = ('niah', 'memoryagentbench', 'longmemeval_s')
ARM_POLICY = {'A': ('collect', 'attention'), 'C': ('multi', 'attention'),
              'D': ('multi', 'graph')}
HISTORY_BYTES_PER_TOKEN = 17 * 4 * 256 * 2
STREAM_SCHEMA = 'graph-v2-fixed256k-stream-1'
STREAM_WINDOW_SECONDS = 300.
STREAM_WARMUP_COMPLETIONS = 8
STREAM_SLOTS = 8


def frozen_prefix_graph_capacity(repository_blocks, owners):
    """Conservative graph cap for complete P histories, not allocated KV bytes.

    Budget 8 KiB per logical block for Python top-4 rows, live/P/scoring copies
    and bounded acceptance scratch. The 1M repository needs a larger graph
    allowance than the 256K cold cases; a live-row-only estimate misses retained
    immutable owners. All memory arms use the same non-scoring capacity.
    """
    per_owner = repository_blocks * 8192
    return per_owner, owners * per_owner


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profiles(base, port, suite='niah'):
    require(suite in SUITES, 'Unknown V2 suite')
    common = deepcopy(base)
    memory = suite == 'memoryagentbench'
    common.update(port=port, served_model_name='kvmem-graph-v2', verify_session_restore=False,
        max_model_len=1048576 if memory else 524288, max_num_seqs=8, max_num_batched_tokens=16424,
        query_replay_budget=16384, long_prefill_token_threshold=1024,
        prefill_chunk_tokens=1024, profiler_config=None)
    kv = common['kvmem']
    for name in ('graph_locality_weight', 'graph_attention_weight', 'action_pressure'):
        kv.pop(name, None)
    kv.update(select_budget=36864, gen_budget=16384, repository_blocks=32768 if memory else 16384,
        cpu_read_cache_bytes=100000000000, cpu_history_cache_bytes_per_request=20000000000,
        session_capacity=8, session_checkpoint_bytes=4*1024**3 if memory else 1,
        session_capture_terminal=not memory, session_capture_decode=False,
        session_checkpoint_policy='sealed' if memory else 'pm', session_freeze_prompt=memory,
        query_history=False, guided_query_tokens=0, action_fraction=0.,
        action_attention=True, prefill_attention=True, action_scope='non-thinking',
        action_interval=16, action_layers='spread3', prefill_attention_interval=1024,
        prefill_attention_stride=32, prefill_attention_half_life=8192,
        block_tokens=32, sink_blocks=41, recent_blocks=0,
        gpu_raw_k_cache_bytes=kv.get('gpu_raw_k_cache_bytes', 0),
        repository_gc=False, graph_trace=False, graph_scoring_version='v2', graph_alpha=.2,
        graph_attention_topk=4, graph_attention_min_mass=0., graph_cross_message_weight=.25,
        graph_max_bytes_per_session=33554432, graph_max_total_bytes=268435456,
        rolling_active_window=False, offline_agent_policy=False, offline_phase_policy=False,
        offline_prefill_coalescing=False, active_resident_pressure_only=False,
        active_resident_ttl_ms=0., adaptive_prefill=False, step_latency_interval=1)
    if memory:
        # Keep the formal accuracy profile on the original synchronous P
        # import. Async frozen-P reuse is a separate opt-in performance
        # candidate; do not silently change all three benchmark controls.
        kv['async_restore'] = False
        kv['graph_max_bytes_per_session'], kv['graph_max_total_bytes'] = frozen_prefix_graph_capacity(
            kv['repository_blocks'], kv['session_capacity'])
    result = {}
    for arm, (mode, policy) in ARM_POLICY.items():
        result[arm] = deepcopy(common)
        result[arm]['kvmem'].update(graph_mode=mode, pressure_policy=policy)
        validate_profile(result[arm], arm, suite)
    validate_profiles(result, suite)
    return result


def validate_profile(profile, arm, suite='niah'):
    require(arm in ARMS and suite in SUITES, 'Unknown V2 arm or suite')
    kv = profile['kvmem']
    require(profile.get('model_format') == 'safetensors' and
        profile.get('execution_mode') == 'compiled_graph' and
        profile.get('draft_method') == 'mtp' and profile.get('draft_tokens') == 4,
        'V2 requires native safetensors NVFP4 / compiled MTP4')
    require(profile.get('max_num_seqs') == 8 and profile.get('max_num_batched_tokens') == 16424 and
        profile.get('query_replay_budget') == 16384 and profile.get('prefill_chunk_tokens') == 1024,
        'V2 requires C8 / batch16424 / Query16384 / chunk1024')
    require(kv.get('mean_k_backend') == 'qw3_cuda' and kv.get('archive_backend') == 'cpu' and
        kv.get('request_planning') and kv.get('session_capacity') == 8,
        'V2 requires eight owned CPU-history MeanK sessions')
    require(kv.get('kv_dtype', 'fp8_e4m3') == 'fp8_e4m3', 'V2 keeps FP8 KV')
    require((kv.get('graph_mode'), kv.get('pressure_policy')) == ARM_POLICY[arm],
        'V2 graph mode / pressure policy does not match arm')
    require(kv.get('graph_scoring_version') == 'v2' and kv.get('graph_alpha') == .2 and
        kv.get('graph_aggregation', 'sum') == 'sum' and
        not {'graph_locality_weight', 'graph_attention_weight', 'action_pressure'} & kv.keys(),
        'V2 requires canonical version/alpha/policy fields')
    require(kv.get('action_attention') and kv.get('prefill_attention') and
        kv.get('action_scope') == 'non-thinking' and kv.get('action_fraction') == 0 and
        not kv.get('query_history') and kv.get('guided_query_tokens') == 0 and
        not kv.get('repository_gc'), 'All V2 arms share attention capture and aligned MeanK only')
    require((kv.get('block_tokens'), kv.get('select_budget'), kv.get('gen_budget'),
             kv.get('sink_blocks'), kv.get('recent_blocks')) == (32, 36864, 16384, 41, 0),
        'V2 frozen block/K/G/sink/recent changed')
    require(kv.get('step_latency_interval') == 1 and not kv.get('graph_trace'),
        'Require actual per-step batch evidence without detailed scoring trace')
    require(not kv.get('rolling_active_window') and not kv.get('offline_agent_policy') and
        not kv.get('offline_phase_policy') and not kv.get('offline_prefill_coalescing'),
        'Keep independent temporal scheduling policies disabled')
    if suite == 'memoryagentbench':
        require(profile.get('max_model_len') == 1048576 and kv.get('repository_blocks') == 32768,
            'MAB requires the original 1M context and 32768 CPU history blocks')
        per_owner, total = frozen_prefix_graph_capacity(kv['repository_blocks'], kv['session_capacity'])
        require(kv.get('graph_max_bytes_per_session', 0) >= per_owner and
            kv.get('graph_max_total_bytes', 0) >= total,
            'Frozen-P graph capacity must cover live/checkpoint/scoring/staging owners')
        require(kv.get('session_freeze_prompt') and not kv.get('session_capture_terminal') and
            not kv.get('session_capture_decode') and kv.get('session_checkpoint_bytes', 0) > 1,
            'MAB retains only a frozen P prefix between independent questions')
    else:
        require(kv.get('session_checkpoint_bytes') == 1 and not kv.get('session_freeze_prompt') and
            not kv.get('session_capture_decode'), 'Independent cases must not retain cross-case history')


def preflight_reference(rows, sampling, profile, suite):
    """Validate every original input before freezing or launching model work."""
    require(suite in ('memoryagentbench', 'longmemeval_s'), 'Unknown reference suite')
    expected = 1116 if suite == 'memoryagentbench' else 102
    require(len(rows) == expected and len({row['id'] for row in rows}) == expected,
        'Original reference cohort size/identity differs')
    kv = profile['kvmem']
    block = kv['block_tokens']
    budget = kv['cpu_read_cache_bytes'] - 16 * 1024**3
    groups = {}
    output_caps = set()
    graph_capacity = {}
    if suite == 'memoryagentbench':
        per_owner, total = frozen_prefix_graph_capacity(kv['repository_blocks'], kv['session_capacity'])
        require(kv.get('graph_max_bytes_per_session', 0) >= per_owner and
            kv.get('graph_max_total_bytes', 0) >= total,
            'Frozen-P graph capacity must cover live/checkpoint/scoring/staging owners')
        graph_capacity = dict(graph_per_owner_required_bytes=per_owner,
            graph_total_required_bytes=total,
            graph_capacity_policy='8192 bytes per repository block including retained graph owners; cap, not allocation')
    for row in rows:
        config = sampling[row['id']]
        require(config.get('token_sha256') == row['token_sha256'], 'Original request token identity differs')
        require(all(key in config for key in ('temperature','top_p','top_k','min_p',
            'presence_penalty','repetition_penalty','seed','max_tokens')), 'Missing original sampling setting')
        original_output = config['max_tokens']
        require(type(original_output) is int and original_output > 0, 'Invalid original output limit')
        # LME preserves its existing, explicitly documented G-limited serving
        # cap. MAB keeps every original per-question output limit unchanged.
        output = original_output if suite == 'memoryagentbench' else kv['gen_budget']
        require(output <= kv['gen_budget'], 'Original output exceeds the unchanged generation budget')
        require(row['input_tokens'] + output <= profile['max_model_len'],
            'Complete original input/output exceeds model context: ' + row['id'])
        extent = row['input_tokens'] + 2 * kv['gen_budget']
        records = (extent + block - 1) // block
        require(records <= kv['repository_blocks'],
            'Complete original history and staging exceed repository blocks: ' + row['id'])
        amount = records * block * HISTORY_BYTES_PER_TOKEN
        require(amount <= budget, 'One complete original history exceeds CPU admission credits: ' + row['id'])
        group = row['id'].rsplit('_q', 1)[0] if suite == 'memoryagentbench' else row['id']
        groups[group] = max(groups.get(group, 0), amount)
        output_caps.add(original_output)
    largest_eight = sum(sorted(groups.values(), reverse=True)[:8])
    return dict(cases=len(rows), independent_owners=len(groups), **graph_capacity,
        input_token_range=[min(row['input_tokens'] for row in rows), max(row['input_tokens'] for row in rows)],
        original_output_caps=sorted(output_caps), cpu_payload_credits_bytes=budget,
        maximum_owner_credit_bytes=max(groups.values()), largest_eight_owner_credit_bytes=largest_eight,
        largest_eight_fit_cpu_credits=largest_eight <= budget,
        admission='Eight client slots; complete histories wait for CPU credits without truncation or recomputation')


def validate_profiles(configs, suite='niah'):
    require(set(configs) == set(ARMS), 'A/C/D must all be present')
    common = []
    for arm in ARMS:
        validate_profile(configs[arm], arm, suite)
        stripped = deepcopy(configs[arm])
        for name in ('graph_mode', 'pressure_policy'):
            stripped['kvmem'].pop(name)
        common.append(stripped)
    require(all(item == common[0] for item in common), 'Non-policy settings differ between A/C/D')


def infer_arm(profile):
    kv = profile['kvmem']
    pair = kv.get('graph_mode'), kv.get('pressure_policy')
    found = [arm for arm, expected in ARM_POLICY.items() if pair == expected]
    require(len(found) == 1, 'Profile is not a declared A/C/D arm')
    return found[0]


def audit_events(refresh, graph, profile, arm, query, *, require_pressure=False):
    """Inspect actual selected phases and seeds, not the requested profile alone."""
    kv = profile['kvmem']
    queries = [event for event in refresh if event.get('selection') == 'query']
    pressure = [event for event in refresh if event.get('selection') == 'pressure']
    require(len(queries) == 1, 'Missing or duplicate public Query selection')
    selected = queries[0]
    require(selected.get('query_span') == list(query) and
        selected.get('mean_k_backend') == 'qw3_cuda' and not selected.get('query_history_enabled') and
        selected.get('guided_query_tokens') == 0, 'Query coordinates/backend/history changed')
    require(selected.get('action_attention_enabled') and selected.get('action_history_pages') == 0,
        'Independent Query inherited output attention or disabled collection')
    require(selected.get('prefill_attention_samples', 0) > 0, 'No accepted prefill attention')
    for event in (*queries, *pressure):
        phase = event['selection']
        require(event.get('phase') == phase and event.get('scoring_version') == 'v2' and
            event.get('score_domain_version') == SCORE_DOMAIN and
            event.get('pressure_policy') == kv['pressure_policy'] and event.get('alpha') == .2,
            'Actual phase/version/domain/pressure policy/alpha differs')
        require(0 < event.get('selected_tokens', 0) <= kv['select_budget'], 'Selection exceeds K')
        require(event.get('seed_source') == 'mean_k' if phase == 'query' else
            event.get('seed_source') in ('output', 'prefill', 'none'), 'Missing actual seed source')
    if require_pressure:
        require(pressure and any(event['seed_source'] == 'prefill' for event in pressure),
            'Long-input qualification must execute attention-seeded pressure')
    query_graph = [event for event in graph if event.get('phase') == 'query']
    pressure_graph = [event for event in graph if event.get('phase') == 'pressure']
    require(len(query_graph) == 1, 'Missing actual Query graph/collection evidence')
    require(query_graph[0].get('mode') == ARM_POLICY[arm][0], 'Wrong actual Query graph mode')
    require(len(pressure_graph) == len(pressure), 'Pressure reports do not match pressure selections')
    for selection, report in zip(pressure, pressure_graph):
        require(all(selection.get(name) == report.get(name) for name in
            ('seed_source', 'selected_tokens', 'pressure_policy', 'score_mask_count', 'transport_mask_count')),
            'Actual pressure selection differs from its scoring report')
    if arm != 'D':
        require(all(event.get('propagation_applied') is False and
            event.get('transport_mask_count') == 0 for event in pressure_graph),
            'Graph propagation leaked into legacy attention pressure')
    for event in graph:
        require(event.get('scoring_version') == 'v2' and event.get('score_domain_version') == SCORE_DOMAIN and
            event.get('alpha') == .2 and event.get('pressure_policy') == kv['pressure_policy'],
            'Graph event version/domain/alpha/policy differs')
        require(0 <= event.get('graph_live_bytes', -1) <= kv['graph_max_bytes_per_session'],
            'Missing or exceeded graph live resource accounting')
        # Legacy attention pressure records the graph's live allocation but
        # creates no graph snapshot and runs no propagation operator.
        if event['phase'] == 'query' or arm == 'D':
            require(event.get('observed_sources', 0) > 0 and event.get('edges', 0) > 0,
                'Graph has no observed attention edges')
            require(0 <= event.get('graph_global_bytes', -1) <= kv['graph_max_total_bytes'] and
                event.get('snapshot_bytes', -1) >= 0, 'Missing or exceeded graph snapshot resource accounting')
        if event['phase'] == 'query':
            require(event.get('seed_source') == 'mean_k' and
                event.get('accepted_through', -1) >= selected['replay_span'][0], 'Stale Query graph snapshot')
        else:
            require(event.get('seed_source') in ('output', 'prefill', 'none') and
                event.get('candidate_count', 0) > 0, 'Missing graph pressure seed/candidate evidence')
    return dict(query=selected, pressure=pressure, graph=graph,
        phases={'query': len(queries), 'pressure': len(pressure)},
        seed_sources=sorted({event['seed_source'] for event in (*queries, *pressure)}),
        score_domain_version=SCORE_DOMAIN,
        boundary_scope='Original frozen token IDs without invented message boundaries; unknown locality fallback.')


def audit_case(capture, response_id, profile, query, *, require_pressure=False,event_reader=None):
    import qualify_block_graph as q
    response = {'response_id': response_id}
    matching = lambda event: q.native_matches(response, event)
    reader = event_reader if event_reader is not None else q.records
    return audit_events([e for e in reader(Path(capture)/'bounded_refresh.jsonl') if matching(e)],
        [e for e in reader(Path(capture)/'block_graph.jsonl') if matching(e)],
        profile, infer_arm(profile), query, require_pressure=require_pressure)


def audit_actual_concurrency(steps, *, require_eight=False):
    measured = [event for event in steps if type(event.get('model_ms')) in (int, float) and
                math.isfinite(event['model_ms']) and event['model_ms'] > 0]
    require(measured, 'No completed native model-step measurements')
    counts = [event.get('prefill_requests', 0) + event.get('decode_requests', 0) for event in measured]
    require(max(counts) <= 8, 'Native executed batch exceeded eight owners')
    if require_eight:
        require(max(counts) == 8, 'Eight admitted clients are not proof of an executed eight-owner batch')
    return dict(executed_request_peak=max(counts), executed_eight_owner_steps=counts.count(8),
        native_eight_owner_qualified=max(counts) == 8,
        executed_prefill_request_peak=max(event.get('prefill_requests', 0) for event in measured),
        executed_decode_request_peak=max(event.get('decode_requests', 0) for event in measured),
        measured_steps=len(measured), proof='Completed native model steps; prefill_requests + decode_requests',
        pure_decode_rate_inferred=False)


def release_owner(profile, capture, session, token_id):
    import qualify_block_graph as q
    import qualify_graph_niah as niah
    response = q.stream_request(profile, session, [token_id], 1, 0, release=True)
    terminal = niah.wait_retired(Path(capture), response['response_id'])
    return niah.wait_owner_released(Path(capture), session, response, terminal)
