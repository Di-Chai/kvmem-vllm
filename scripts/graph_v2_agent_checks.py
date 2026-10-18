"""CPU-checkable contracts and captured evidence for V2 natural agent pairs."""
from copy import deepcopy
import json
import math
from pathlib import Path

import benchmark_capacity_chain as chain
import benchmark_graph_v2_performance as pairs
import qualify_graph_v2_performance as counters

ready, cap, work = chain.ready, chain.cap, chain.work
require, read, sha = cap.require, cap.read, cap.sha
SCHEMA = 'graph-v2-natural-agents-1'
FIXTURE_SHA = '8f049098de7fa13c804adfc3e2d0c9035d8ce883489490e33263a721c9b9a63a'
WORKLOAD = dict(chain.WORKLOAD)
DEFAULT_OFF_SOURCE_EXTENSIONS = frozenset('src/vllm_kvmem/'+name for name in
    ('options.py', 'storage.py', 'elastic_serve.py', 'memory_storage.py'))
QUALIFICATION_FILES = ('manifest.json', 'profile.json', 'fixture.json', 'status.json', 'cleanup.json',
    'qualification/results.json', 'qualification/manifest.json')


def job_order(count):
    require(type(count) is int and count in (1, 3), 'Use exactly one screening pair or three repeated pairs')
    return [dict(id=f'p{index+1}-{role}', pair=index+1, role=role)
            for index in range(count) for role in
            (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline'))]


def profiles(base, port, candidate):
    configs = pairs.profiles(base, port, candidate)
    for profile in configs.values():
        profile.update(max_model_len=262144, served_model_name='kvmem-graph-v2-agents')
        profile['kvmem'].update(session_checkpoint_bytes=4*1024**3, action_profiling=False)
    validate_profiles(configs, candidate)
    return configs


def validate_profiles(configs, candidate):
    normalized = deepcopy(configs)
    for name, profile in configs.items():
        kv = profile['kvmem']
        require(profile['max_model_len'] == 262144 and profile.get('verify_session_restore') is False,
                'Natural agent pairs require full256K and an unprofiled performance run')
        require(kv.get('session_checkpoint_bytes') == 4*1024**3 and
            kv.get('session_capture_terminal') is True and kv.get('session_checkpoint_policy') == 'pm' and
            not kv.get('session_freeze_prompt') and not kv.get('action_profiling'),
            'Agents require retained P/M checkpoints without Action profiling')
        require(kv.get('async_restore') is True and kv.get('cpu_history_coordination') is True and
            kv.get('async_archive_retirement') is True and not kv.get('active_resident_pressure_only'),
            'Require qualified asynchronous lifecycle without temporal scheduling policies')
        # Reuse the exact single-variable D contract, adapting only the declared
        # agent context/checkpoint retention instead of changing the NIAH gate.
        normalized[name]['max_model_len'] = 524288
        normalized[name]['kvmem']['session_checkpoint_bytes'] = 1
    pairs.validate_profiles(normalized, candidate)


def fixture_proof(path):
    require(sha(path) == FIXTURE_SHA, 'Use the unchanged original fixed8 agent fixture')
    fixture = read(path)
    require(len(fixture.get('agent_plans', [])) == 8 and type(fixture.get('eos')) is int,
            'Exactly eight original agents and EOS are required')
    plans = work.validate_agent_plans(fixture, concurrency=8, warmup_turns=2, epoch_turns=16,
                                    gen_budget=16384, enforce_output_budget=False)
    require(all(len(plan['initial_tokens'])+16384 <= 262144 for plan in plans),
            'Complete original initial context cannot fit natural output ceiling')
    return dict(fixture_sha256=FIXTURE_SHA, agents=[dict(agent=i,
        initial_tokens=len(plan['initial_tokens']), initial_tokens_sha256=plan['initial_tokens_sha256'],
        query_plan=plan['query_plan'], source_task=plan['source_task'], source_turn=plan['source_turn'])
        for i, plan in enumerate(plans)],
        scope='Original complete IDs, external tools/waits and early absolute Query plans; no truncation or query relocation')


def host_budget():
    archive = 8*(262144+64)*cap.ARCHIVE_BYTES_PER_TOKEN
    index = 8*cap.INDEX_BYTES_PER_REQUEST
    return dict(history_limits=[262144]*8, archive_bytes=archive, mean_k_index_bytes=index,
        extra_bytes=32*1024**3, required_available_bytes=archive+index+32*1024**3,
        minimum_free_bytes=8*1024**3,
        scope='Conservative eight full256K histories plus partial blocks, MeanK index and host reserve; not old FP8 calibration')


def qualification_proof(root, baseline, runtime_hashes, synchronous):
    manifest, profile = read(root/'manifest.json'), read(root/'profile.json')
    result, cleanup = read(root/'qualification/results.json'), read(root/'cleanup.json')
    require(manifest.get('synchronous') is synchronous and
        manifest.get('qualification_script') == 'qualify_graph_v2_sessions.py' and
        read(root/'status.json')['status'] == result.get('status') == 'passed' and
        result.get('owners') == 2 and result.get('turns') == 6 and not result.get('errors') and
        result.get('all_owned_sessions_released') is True, 'Passed V2 sync/async lifecycle evidence required')
    require(cleanup.get('remaining_owned_process_groups') == {} and cleanup.get('remaining_gpu_pids') == [] and
            cleanup.get('server_returncode') == 0, 'Qualification owned cleanup incomplete')
    require(sha(root/'profile.json') == manifest['profile_sha256'] and
            sha(root/'fixture.json') == manifest['fixture_sha256'], 'Qualification profile or fixture changed')
    for name, expected in manifest['runtime_sha256'].items():
        require(sha(root/'runtime'/name) == expected, 'Qualification runtime changed: '+name)
    evidence = result['evidence']
    require({(e['agent'], e['turn']) for e in evidence} == {(a,t) for a in range(2) for t in range(3)},
            'Qualification omitted an owner/turn')
    for item in evidence:
        require(item.get('query') and (item['turn'] != 0 or item.get('pressure')),
                'Qualification lacks Query/initial-pressure lifecycle evidence')
        for event in item['query']+item.get('pressure', []):
            require(event.get('scoring_version') == 'v2' and event.get('alpha') == .2 and
                event.get('pressure_policy') == 'graph', 'Qualification used another graph policy')
        if item['turn']:
            require(item.get('target_restore') and all(x.get('byte_equal') is True for x in item['target_restore']) and
                item.get('state_restore') and all(x.get('recurrent_byte_equal') is True for x in item['state_restore']),
                'Qualification lacks target/recurrent byte equality')
    before = {k:v for k,v in manifest['runtime_sha256'].items() if k.startswith('src/')}
    after = {k:v for k,v in runtime_hashes.items() if k.startswith('src/')}
    require(before.keys() == after.keys(), 'Qualification production file set differs')
    changed = {k for k in before if before[k] != after[k]}
    require(changed <= DEFAULT_OFF_SOURCE_EXTENSIONS,
            'New lifecycle/native source requires fresh qualification: '+str(sorted(changed)))
    # Normalize only missing default-false candidate flags. The old test did not
    # exercise these enabled paths, nor the new eight-owner profile.
    old, new = deepcopy(profile), deepcopy(baseline)
    for config in (old, new):
        for name in ('cpu_direct_record_seal', 'async_action_metadata', 'native_action_metadata'):
            config['kvmem'].setdefault(name, False)
    for name, value in (('gpu_raw_k_cache_bytes',0), ('cpu_direct_record_seal',False), ('async_action_metadata',False)):
        require(old['kvmem'].get(name) == new['kvmem'].get(name) == value, 'Qualification must describe the disabled baseline')
    top_allowed = {'port','served_model_name','max_model_len','verify_session_restore'}
    kv_allowed = {'session_checkpoint_bytes','action_profiling','action_profile_interval','action_profile_max_events',
                  'async_refresh','async_restore','cpu_history_coordination','maintenance_max_steps'}
    deltas = {}
    for key in old.keys() | new.keys():
        if key == 'kvmem': continue
        if old.get(key) != new.get(key):
            require(key in top_allowed, 'Qualification model/profile differs: '+key)
            deltas[key] = dict(qualified=old.get(key), proposed=new.get(key))
    for key in old['kvmem'].keys() | new['kvmem'].keys():
        if old['kvmem'].get(key) != new['kvmem'].get(key):
            require(key in kv_allowed, 'Qualification runtime policy differs: '+key)
            deltas['kvmem.'+key] = dict(qualified=old['kvmem'].get(key), proposed=new['kvmem'].get(key))
    require(profile['kvmem']['async_restore'] is (not synchronous), 'Qualification async flag differs')
    return dict(passed_prior_lifecycle=True, synchronous=synchronous, tested_owners=2, tested_turns=6,
        exact_source_match=not changed, source_extensions={k:dict(qualified=before[k], proposed=after[k]) for k in sorted(changed)},
        profile_deltas=deltas, candidate_prequalified=False, current_profile_prequalified=False,
        baseline_semantics='Missing new boolean options resolve to false; proposed source extensions remain disabled in baseline.',
        scope='Prior sync/async byte-restore and release evidence only. Fresh baseline and candidate runs each require their own history/config/resource checks.')


def graph_evidence(capture, fixture, rows):
    graphs, refresh = ready.records(capture/'block_graph.jsonl'), ready.records(capture/'bounded_refresh.jsonl')
    require(graphs and refresh, 'Actual graph and selection evidence required')
    for event in graphs:
        require(event.get('scoring_version') == 'v2' and event.get('score_domain_version') == 'qw3-native-v2' and
            event.get('alpha') == .2 and event.get('pressure_policy') == 'graph' and event.get('mode') == 'multi',
            'Actual graph policy/version differs')
        require(event.get('phase') in ('query','pressure') and
            event.get('seed_source') in (('mean_k',) if event['phase'] == 'query' else ('prefill','output','none')),
            'Actual graph seed/phase differs')
        require(0 <= event.get('graph_global_bytes', -1) <= 268435456 and
            0 <= event.get('graph_bytes', -1) <= 33554432, 'Observed graph budget exceeded or unmeasured')
    verified = []
    restores = ready.records(capture/'async_restore.jsonl')
    for row in rows:
        if row['kind'] == 'cleanup': continue
        expected = fixture['agent_plans'][row['agent']]['query_plan']
        actual = row.get('query_plan') or {}
        require(all(actual.get(k) == value for k,value in expected.items()) and
            set(actual)-set(expected) <= {'kvmem_locality_spans','kvmem_locality_provenance'},
            'Original initial Query/mandatory positions changed')
        match = lambda event: counters.q.native_matches(row, event)
        selections = [event for event in refresh if match(event) and event.get('selection') == 'query']
        # A continuation may reuse its already applied Query. Re-executed
        # selections must preserve the original coordinates; all requests must
        # retain the full original Query plan in transport and restore history.
        for event in selections:
            require(event.get('query_span') == expected['kvmem_query_span'], 'Executed Query coordinates changed')
        if row['turn'] == 0:
            require(selections, 'Initial epoch lacks an actual Query selection')
        else:
            restored = [event for event in restores if match(event)]
            require(restored and all(not e.get('cancelled') and e.get('history_tokens',0) > expected['kvmem_query_span'][1]
                for e in restored), 'Continuation did not restore the original Query-containing history')
        selected = [dict(selection=e['selection'],ranges=e['selected_history_ranges']) for e in refresh if match(e)]
        verified.append(dict(request_id=row['request_id'], agent=row['agent'],epoch=row['epoch'],turn=row['turn'],
            query_executions=len(selections),selection_sha256=counters.digest(selected),
            query_plan_sha256=counters.digest(expected), input_sha256=row['input_sha256'], output_sha256=row['output_sha256']))
    return dict(requests=verified, query_events=sum(e['phase']=='query' for e in graphs),
        pressure_events=sum(e['phase']=='pressure' for e in graphs), original_query_positions_preserved=True)


def actual_profile(directory, expected):
    launch = read(directory/'server/launch.json')
    require(launch['profile'] == expected['profile'] and launch['vllm_argv'] == expected['vllm_argv'],
            'Actual launched profile or worker payload differs from frozen expectation')
    geometry = read(directory/'server/capture/cache-geometry.json')
    require((geometry.get('select_budget'),geometry.get('gen_budget'),geometry.get('historical_tokens'),
        geometry.get('cpu_archive_bytes_per_token'),geometry.get('cpu_offload_bytes')) ==
        (36864,16384,262144,34816,100000000000), 'Actual cache/archive geometry differs')
    return dict(launch_sha256=sha(directory/'server/launch.json'), geometry=geometry)


def measured_total(rows, field):
    result = counters.total(rows, field)
    result['observed'] = result['observed_rows'] > 0
    if not result['observed']: result['value'] = None
    return result


def measurements(directory, profile, result):
    capture = directory/'server/capture'
    window = read(directory/'tools/window.json')
    lo, hi = window['begin'], window['end']
    require(hi-lo == 300 and window['agents'] == 8 and window['warmup_turns'] == 2,
            'Fixed measurement window or warmup changed')
    steps = [r for r in ready.records(capture/'model_step_latency.jsonl') if lo <= r['started_at'] < hi and
        type(r.get('model_ms')) in (int,float) and math.isfinite(r['model_ms']) and r['model_ms']>0]
    require(steps, 'No actual model steps in measurement window')
    pure = [r for r in steps if r['decode_requests'] and not r['prefill_requests'] and not r['prefill_tokens']]
    refresh = [r for r in ready.records(capture/'bounded_refresh.jsonl') if lo <= r['refresh_started_at'] < hi]
    restore = [r for r in ready.records(capture/'async_restore.jsonl') if lo <= r['started'] < hi]
    require(all(not r.get('cancelled') for r in refresh+restore), 'Cancelled maintenance in measured interval')
    samples = ready.records(directory/'agent_counters.jsonl')
    observed = [r for r in samples if lo <= r['at'] < hi]
    require(observed and all(r['nvml']['gpu_free_mib'] >= 4096 for r in samples), 'Device headroom below4GiB or missing NVML evidence')
    kv = profile['kvmem']
    hits = sum(r.get('raw_cache_hit_tokens', 0) for r in refresh+restore)
    require(kv['gpu_raw_k_cache_bytes'] or hits == 0, 'Disabled raw-K cache reported hits')
    archive = result['archive_final']
    rows = read(directory/'tools/traces.json')
    offered = [r for r in rows if r['kind']=='workload' and lo <= r['begin'] < hi]
    require({r['agent'] for r in offered} == set(range(8)), 'All eight owners must enter the fixed window')
    points = sorted((at,delta) for r in offered for at,delta in ((r['begin'],1),(r['end'],-1)))
    concurrent = peak_http = 0
    for _,delta in points:
        concurrent += delta
        peak_http = max(peak_http,concurrent)
    latency = dict(request_p50_seconds=work.percentile([r['end']-r['begin'] for r in offered],.5),
        request_p95_seconds=work.percentile([r['end']-r['begin'] for r in offered],.95),
        ttft_p50_seconds=result.get('ttft_observed_p50_seconds'),ttft_p95_seconds=result.get('ttft_observed_p95_seconds'),
        intertoken_event_gap_p50_seconds=result.get('output_event_gap_p50_seconds'),
        intertoken_event_gap_p95_seconds=result.get('output_event_gap_p95_seconds'),
        ttft_right_censored=result.get('ttft_right_censored'),
        output_gap_right_censored=result.get('output_gap_right_censored'),
        scope='Full latency of requests offered in window (including drain); TTFT/event gaps observed inside window with censoring retained. An MTP event can contain multiple accepted tokens.')
    latency['measurement_complete'] = all(type(latency[k]) in (int,float) and math.isfinite(latency[k]) and latency[k]>=0
        for k in ('request_p50_seconds','request_p95_seconds','ttft_p50_seconds','ttft_p95_seconds',
                  'intertoken_event_gap_p50_seconds','intertoken_event_gap_p95_seconds'))
    gather = [r['assembly_read_stats'] for r in refresh if 'assembly_read_stats' in r]
    peak = max(r['decode_requests']+r['prefill_requests'] for r in steps)
    require(peak<=8, 'Actual native batch exceeded eight owners')
    return dict(output_tokens=result['output_tokens'], output_tokens_per_second=result['output_tokens_per_second'],
        offered_new_input_tokens_per_second=result['offered_new_input_tokens']/300,
        latency=latency,
        rates_scope='Accepted workload token events in fixed [T0,T1); external new input offered in window. Excludes warmup/cleanup and drain output.',
        raw_cache_hit_tokens=hits, raw_cache_exercised=hits>0,
        configured_total_raw_k_bytes=kv['gpu_raw_k_cache_bytes'], per_arena_cap_bytes=kv['gpu_raw_k_cache_bytes']//2,
        refresh=dict(events=len(refresh), uploaded_bytes=measured_total(refresh,'assembly_uploaded_bytes'),
            host_prepare_seconds=measured_total(refresh,'assembly_host_prepare_seconds'),
            service=counters.interval_evidence(refresh,'refresh_started_at','refresh_finished_at')),
        restore=dict(events=len(restore), uploaded_bytes=measured_total(restore,'uploaded_bytes'),
            model_bytes=measured_total(restore,'model_bytes'), service=counters.interval_evidence(restore,'started','finished')),
        cpu_gather=dict(observed=any('cache_copy_seconds' in row for row in gather),
            copy_seconds=measured_total(gather,'cache_copy_seconds')),
        seal=counters.seal_evidence(archive,kv['cpu_direct_record_seal']),
        action_metadata=counters.metadata_evidence(capture,profile),
        counter_scope='Transfer/service events started in fixed window may finish after T1. Seal/metadata are fresh-worker cumulative across all phases; never divide these totals by300. Timestamped archive snapshots retained separately.',
        independent_owners=dict(offered=8,completed=8,http_inflight_peak=peak_http,
            scope='Eight persistent owner loops without serial waves; HTTP overlap does not imply a native batch of eight.'),
        native=dict(step_count=len(steps), actual_batch_peak=peak, actual_c8_compute_observed=peak>=8,
            prefill_peak=max(r['prefill_requests'] for r in steps), decode_peak=max(r['decode_requests'] for r in steps),
            pure_decode_steps=len(pure), pure_decode_batch_histogram={str(n):sum(r['decode_requests']==n for r in pure) for n in range(1,9)},
            scope='Completed public GPU steps; accepted workload TPS is not pure decode TPS or speculative proposed-token throughput.'),
        nvml=dict(samples=len(observed), peak_used_mib=max(r['nvml']['gpu_used_mib'] for r in observed),
            minimum_free_mib=min(r['nvml']['gpu_free_mib'] for r in observed),
            mean_utilization_percent=sum(r['nvml']['gpu_utilization_percent'] for r in observed)/len(observed),
            mean_power_w=sum(r['nvml']['gpu_power_w'] for r in observed)/len(observed)),
        gpu_idle_inferred=False, gpu_idle_seconds=None)
