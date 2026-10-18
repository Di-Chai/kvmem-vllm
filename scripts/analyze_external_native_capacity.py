"""Read-only fixed12 capacity evidence against an explicit external native run.

Existing step telemetry has counts, not request IDs. The N shortest possible
HTTP input histories give a conservative decode-subset bound, including mixed
batches. CUDA durations are not an absolute GPU wall timeline.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


# These frozen revisions were reviewed together: the second adds opt-in natural
# generation without changing fixed quotas, temperature=0, seed=1000, ignore_eos.
FIXED_WORKLOAD_HASHES = {
    'abcba7c31f1e6187adbd74a7c31cc6eb49cc98c1d6b665ac9449581e895367b4',
    '108c1142e7e5071d17164b095aac69ca366b3bf009cfbcf4956163d3699e83de',
}
MODEL_CONFIG_SHA = '71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8'
NATIVE_SPEC_SHA = '1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb'
NATIVE_POOL_SHA = '34c2fd67148cecb7763ff49ad694a57073759b0fa67572a4270dc995b4acf8fc'
NATIVE_MANAGER_SHA = '0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf'
PROTOCOL_FILES = ('scripts/timed_agent_workload.py', 'scripts/capacity_scaling_workload.py',
                  'scripts/benchmark_timed_agents.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def records(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def positive_int(value):
    return type(value) is int and value > 0


def arguments(argv):
    """Parse recorded effective argv; no launch or installed-module import."""
    result = {}
    for i, value in enumerate(argv):
        if value.startswith('--'):
            require(value not in result, 'Duplicate effective CLI option: '+value)
            result[value] = argv[i+1] if i+1 < len(argv) and not argv[i+1].startswith('--') else True
    require('serve' in argv and argv.index('serve')+1 < len(argv), 'Missing effective model argument')
    result['model'] = argv[argv.index('serve')+1]
    return result


def load_run(root, arm):
    manifest = read(root/'manifest.json')
    require(manifest.get('kind') == 'ready_agents_v1' and arm in manifest['arms'], 'Wrong fixed-window run/arm')
    require(read(root/'status.json')['status'] == 'completed_pending_review', 'Run must have completed normal drain')
    require(sha(root/'fixture.json') == manifest['fixture_sha256'], 'Frozen fixture hash mismatch')
    require(sha(root/(arm+'.json')) == manifest['profiles_sha256'][arm], 'Frozen profile hash mismatch')
    for name in PROTOCOL_FILES:
        require(sha(root/'frozen'/name) == manifest['source_sha256'][name], 'Frozen protocol source changed: '+name)
    require(manifest['source_sha256'][PROTOCOL_FILES[0]] in FIXED_WORKLOAD_HASHES,
            'Unknown fixed workload implementation; review protocol before comparison')
    directory = root/arm
    result = read(directory/'results.json')
    require(result.get('status') == 'completed' and result.get('drained') is True and
            result.get('native_client_tokens_equal') is True, 'Incomplete or unreconciled run')
    launch = read(directory/'launch.json' if arm == 'native' else directory/'server/launch.json')
    effective = arguments(launch['argv'] if arm == 'native' else launch['vllm_argv'])
    return dict(root=root, arm=arm, manifest=manifest, profile=read(root/(arm+'.json')),
        fixture=read(root/'fixture.json'), result=result, effective=effective,
        window=read(directory/'tools/window.json'), traces=read(directory/'tools/traces.json'),
        samples=records(directory/'native_samples.jsonl'))


def compare_controls(k, n):
    require(k['arm'] != 'native' and n['arm'] == 'native', 'Explicit K and native arms required')
    km, nm = k['manifest'], n['manifest']
    require(km['fixture_sha256'] == nm['fixture_sha256'], 'Different frozen fixture/sample/token IDs')
    kw, nw = km['workload'], nm['workload']
    require(kw == nw and kw.get('concurrency') == 12 and kw.get('mode') == 'tools' and
            kw.get('generation_mode', 'fixed') == 'fixed', 'Require identical fixed12 protocol')
    require(km.get('generation_sampling') == nm.get('generation_sampling'), 'Sampling manifest differs')
    require(km['native_source'] == nm['native_source'], 'Native implementation identity differs')
    for name in PROTOCOL_FILES[1:]:
        require(km['source_sha256'][name] == nm['source_sha256'][name], 'Different protocol helper: '+name)
    for key in ('model', 'tokenizer', 'hf_config_path', 'model_format', 'execution_mode',
                'draft_method', 'draft_tokens', 'draft_model', 'draft_quantization', 'max_model_len',
                'reasoning_end_str', 'linear_backend_per_quant', 'profiler_config'):
        require(k['profile'].get(key) == n['profile'].get(key), 'Model/sampling configuration differs: '+key)
    require(not k['profile'].get('verify_session_restore') and not n['profile'].get('verify_session_restore'),
            'Correctness instrumentation is not a performance control')
    fields = ('model', '--tokenizer', '--hf-config-path', '--dtype', '--kv-cache-dtype',
              '--max-model-len', '--max-num-batched-tokens', '--speculative-config',
              '--generation-config', '--compilation-config', '--load-format', '--kernel-config',
              '--reasoning-config')
    for key in fields:
        a, b = k['effective'].get(key), n['effective'].get(key)
        if key.endswith('-config') and a is not None and b is not None and a != 'vllm':
            a, b = json.loads(a), json.loads(b)
        require(a is not None and a == b, 'Effective launch differs: '+key)
    for run in (k, n):
        window = run['window']
        require(numeric(window['begin']) and numeric(window['end']) and
                math.isclose(window['end']-window['begin'], kw['duration_seconds'], abs_tol=1e-6) and
                window.get('agents') == 12 and window.get('warmup_turns') == kw['warmup_turns'],
                'Formal window disagrees with declared protocol')
        require(run['effective']['model'] == run['profile']['model'] and
                int(run['effective']['--max-model-len']) == run['profile']['max_model_len'],
                'Effective model/context differs from profile')
    return dict(passed=True, fixture_sha256=km['fixture_sha256'], workload=kw,
        sampling=dict(mode='fixed', temperature=0, seed=1000, ignore_eos=True,
                      source_sha256=[km['source_sha256'][PROTOCOL_FILES[0]],nm['source_sha256'][PROTOCOL_FILES[0]]]),
        actual_batch_tokens=int(k['effective']['--max-num-batched-tokens']),
        deployment={name:{key:run['effective'].get(key) for key in
            ('--gpu-memory-utilization','--max-num-seqs','--port','--async-scheduling',
             '--no-async-scheduling','--enable-prefix-caching','--no-enable-prefix-caching')}
            for name,run in (('kvmem',k),('native',n))},
        scope='Same fixed12 fixture/protocol/model/sampling and actual token batch; GPU fraction and active '
              'slots may differ. K connector/scheduler/APC/async integration differences remain explicit. '
              'Model path/config identity is checked; historical weight-file byte identity was not recorded.')


def cache_geometry(samples):
    variants = []
    for sample in samples:
        lines = [line for line in sample['raw_metrics'].splitlines() if line.startswith('vllm:cache_config_info{')]
        if not lines:
            continue
        require(len(lines) == 1, 'Only one native engine/cache pool is supported')
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', lines[0]))
        if labels not in variants:
            variants.append(labels)
    require(len(variants) == 1, 'Missing or changing native cache_info geometry')
    labels = variants[0]
    require(all(labels.get(key, '').isdigit() and int(labels[key]) > 0 for key in ('block_size','num_gpu_blocks')),
            'Invalid actual native pool/block geometry')
    require(labels.get('is_attention_free') == 'False' and labels.get('sliding_window') == 'None',
            'Full-attention capacity bound is unsupported for this cache geometry')
    return dict(block_tokens=int(labels['block_size']), total_pool_pages=int(labels['num_gpu_blocks']), labels=labels)


def resolve_state_contract(k, n, geometry):
    """Unknown deployments get FA-only evidence, never an invented15/18 charge."""
    reasons = []
    config = Path(n['profile']['model'])/'config.json'
    if not config.is_file() or sha(config) != MODEL_CONFIG_SHA:
        reasons.append('unreviewed_model_config')
    spec = json.loads(n['effective']['--speculative-config'])
    if spec != dict(method='mtp',num_speculative_tokens=4):
        reasons.append('not_native_MTP4')
    if n['effective']['--kv-cache-dtype'] != 'fp8_e4m3' or geometry['labels'].get('cache_dtype') != 'fp8_e4m3':
        reasons.append('not_FP8_KV')
    if geometry['labels'].get('mamba_cache_mode') != 'align':
        reasons.append('not_native_align')
    source = n['manifest']['native_source']
    interfaces = [Path(path) for path,value in source.get('files',{}).items()
                  if path.endswith('/v1/kv_cache_interface.py') and value == NATIVE_SPEC_SHA]
    if source.get('version') != '0.30.0' or len(interfaces) != 1:
        reasons.append('unreviewed_native_cache_spec')
    manager_proof = None
    if len(interfaces) == 1:
        manager_source = interfaces[0].parent/'core/single_type_kv_cache_manager.py'
        if manager_source.is_file() and sha(manager_source) == NATIVE_MANAGER_SHA:
            manager_proof = dict(source=str(manager_source),sha256=NATIVE_MANAGER_SHA,
                scope='Audited current MambaManager retains1+spec scratch per group and retires boundary '
                      'state, lines1646-1678/1836-1888; not a retroactive full-package hash claim.')
    if manager_proof is None:
        reasons.append('unverified_instantaneous_state_lifetime')
    null_proof = None
    if len(interfaces) == 1:
        pool_source = interfaces[0].parent/'core/block_pool.py'
        if pool_source.is_file() and sha(pool_source) == NATIVE_POOL_SHA:
            null_proof = dict(reserved_null_pages=1, source=str(pool_source), sha256=NATIVE_POOL_SHA,
                scope='Audited installed BlockPool reserves one shared null page, lines191-194/877; '
                      'current source verified, not a retroactive full-package hash claim.')
    groups_path = k['root']/k['arm']/'server/capture/cache-geometry.json'
    groups = read(groups_path).get('groups',[]) if groups_path.exists() else []
    recurrent = [g for g in groups if g.get('recurrent') is True]
    full = [g for g in groups if g.get('recurrent') is False]
    if (len(recurrent) != 3 or any(g.get('speculative_blocks') != 4 for g in recurrent) or
            len(full) != 1 or full[0].get('block_tokens') != geometry['block_tokens'] or
            len(full[0].get('layers',[])) != 17):
        reasons.append('unverified_three_recurrent_plus_full_MTP_attention_groups')
    try:
        declared = geometry['total_pool_pages']/float(geometry['labels']['kv_cache_max_concurrency'])
        expected = math.ceil(int(n['effective']['--max-model-len'])/geometry['block_tokens'])+18
        if not math.isclose(declared,expected,abs_tol=1e-6):
            reasons.append('native_declared_group_geometry_disagrees')
    except (KeyError,ValueError,ZeroDivisionError):
        reasons.append('missing_native_declared_group_geometry')
    geometry.update(state_contract_qualified=not reasons, state_unknown_reasons=reasons,
        recurrent_pages_per_request_lower_bound=15 if not reasons else None,
        native_declared_state_planning_maximum=18 if not reasons else None,
        state_lifetime_proof=manager_proof,
        null_page_proof=null_proof, null_pages=1 if null_proof else None,
        usable_pool_pages=geometry['total_pool_pages']-(1 if null_proof else 0),
        null_scope='Unknown null reservation uses full pool, a deliberately more generous native comparison.'
            if not null_proof else 'One actual source-defined reserved null page excluded.',
        state_scope='15 is a conservative instantaneous3*(1+MTP4) state bound;18 is declared align planning '
                    'maximum only. Unknown model/mode gets attention-only evidence.')
    return geometry


def trace_histories(fixture, traces, publications=None):
    """Reconstruct each full input from original fixture and its own outputs."""
    plans = fixture['agent_plans']
    verified, errors, state = {}, {}, {}
    duplicates = Counter(row.get('request_id') for row in traces)
    for row in sorted(traces,key=lambda r:r.get('begin',0)):
        rid = row.get('request_id')
        try:
            require(isinstance(rid,str) and duplicates[rid] == 1, 'duplicate_or_missing_request_id')
            require(row.get('status') == 'completed' and numeric(row.get('begin')) and
                    numeric(row.get('end')) and row['end'] >= row['begin'], 'incomplete_trace')
            require(row.get('generation_mode','fixed') == 'fixed', 'not_fixed_output')
            ids = row['output_ids']
            require(isinstance(ids,list) and all(type(v) is int and v >= 0 for v in ids), 'invalid_output_ids')
            if row['kind'] == 'cleanup':
                tokens, count = [fixture['eos']], 1
            else:
                require(row['kind'] in ('warmup','workload'), 'unknown_trace_kind')
                agent,epoch,turn = row['agent'],row['epoch'],row['turn']
                require(type(agent) is int and 0 <= agent < 12 and type(epoch) is int and epoch >= 0,
                        'invalid_agent_or_epoch')
                plan = plans[agent]
                require(type(turn) is int and 0 <= turn < len(plan['turns']), 'invalid_turn')
                source = plan['turns'][turn]
                if turn == 0:
                    require((agent,epoch) not in state, 'duplicate_epoch_start')
                    tokens = list(plan['initial_tokens'])
                else:
                    previous = state.get((agent,epoch))
                    require(previous is not None and previous[0] == turn-1, 'missing_previous_history')
                    tokens = previous[1]+source['tool_tokens']
                count = source['output_tokens']
            require(row.get('input_tokens') == len(tokens) and row.get('input_sha256') == digest(tokens),
                    'full_input_history_mismatch')
            require(len(ids) == count and row.get('usage',{}).get('completion_tokens') == count and
                    row.get('usage',{}).get('prompt_tokens') == len(tokens) and
                    row.get('output_sha256') == digest(ids), 'fixed_quota_or_output_hash_mismatch')
            if row['kind'] != 'cleanup':
                state[(agent,epoch)] = (turn,tokens+ids)
            if publications is not None and row['kind'] != 'cleanup':
                response = row.get('response_id')
                matches = [p for p in publications if isinstance(response,str) and
                    re.fullmatch(re.escape(response)+r'-0-[0-9a-f]{8}',p.get('request_id',''))]
                target = len(tokens)+len(ids)-1
                require(matches and all(p.get('history_end') == p.get('checkpoint_tokens') == target and
                    p.get('checkpoint_kind') == 'M' for p in matches), 'missing_full_CPU_history_or_terminal_M')
            verified[rid] = dict(input_tokens=len(tokens), kind=row['kind'], session=row.get('session'))
        except (ValueError,KeyError,TypeError,IndexError) as error:
            errors[str(rid)] = str(error)
    return dict(verified=verified, unknown=errors,
        scope='Verified full original input token hashes and fixed output quotas, including complete prior own '
              'outputs/tool messages; K additionally requires terminal full-history M publication. '
              'Current-turn generated growth is intentionally omitted from the capacity lower bound.')


def capacity_steps(traces, steps, histories, geometry, window, *, margin_seconds=.02):
    lo,hi = window['begin'],window['end']
    require(numeric(lo) and numeric(hi) and hi > lo, 'Invalid window')
    require(numeric(margin_seconds) and margin_seconds >= 0, 'Invalid HTTP timing margin')
    state_pages = geometry.get('recurrent_pages_per_request_lower_bound')
    require(state_pages in (None,15), 'Only audited15 lower bound permitted;18 is a planning maximum')
    block,pool = geometry['block_tokens'],geometry['usable_pool_pages']
    require(positive_int(block) and positive_int(pool), 'Invalid pool')
    groups = {name:dict(steps=0,history_known_steps=0,unknown_steps=0,exceeding_steps=0,
        cuda_event_span_seconds=0.,known_history_event_span_seconds=0.,exceeding_event_span_seconds=0.,
        host_anchored_clipped_event_estimate_seconds=0.,exceeding_clipped_event_estimate_seconds=0.,
        unknown_reasons={},decode_batch_histogram={}) for name in ('pure','mixed')}
    evidence=[]
    prefill_only_steps = 0
    unbounded_http = any(not numeric(t.get('begin')) or not numeric(t.get('end')) or
                        t['end'] < t['begin'] for t in traces)
    for row in steps:
        at,n = row.get('started_at'),row.get('decode_requests')
        require(numeric(at), 'Invalid step submission timestamp')
        if not lo <= at < hi:
            continue
        if not n:
            prefill_only_steps += int(bool(row.get('prefill_requests') or row.get('prefill_tokens')))
            continue
        require(positive_int(n) and n <= 12 and numeric(row.get('step_ms')) and row['step_ms'] >= 0,
                'Invalid actual decode count/event span')
        require(type(row.get('prefill_requests')) is int and row['prefill_requests'] >= 0 and
                type(row.get('prefill_tokens')) is int and row['prefill_tokens'] >= 0, 'Unknown step phase')
        phase = 'mixed' if row['prefill_requests'] or row['prefill_tokens'] else 'pure'
        total = groups[phase]
        seconds = row['step_ms']/1000
        clipped = min(seconds,hi-at)
        total['steps'] += 1
        total['cuda_event_span_seconds'] += seconds
        total['host_anchored_clipped_event_estimate_seconds'] += clipped
        total['decode_batch_histogram'][str(n)] = total['decode_batch_histogram'].get(str(n),0)+1
        possible = [t for t in traces if numeric(t.get('begin')) and numeric(t.get('end')) and
                    t['begin']-margin_seconds <= at < t['end']+margin_seconds]
        reason = None
        if unbounded_http:
            reason = 'unknown_HTTP_time_extent'
        elif len(possible) < n:
            reason = 'insufficient_HTTP_superset'
        elif len({t.get('session') for t in possible}) != len(possible) or any(not t.get('session') for t in possible):
            reason = 'ambiguous_or_duplicate_session'
        elif any(t.get('request_id') not in histories['verified'] for t in possible):
            reason = 'unverified_full_history_in_possible_set'
        event = dict(sequence=row.get('sequence'),started_at=at,phase=phase,decode_requests=n,
            possible_http_requests=len(possible),membership='unknown_N_shortest_superset',
            cuda_event_span_seconds=seconds,host_anchored_clipped_event_estimate_seconds=clipped)
        if reason:
            total['unknown_steps'] += 1
            total['unknown_reasons'][reason] = total['unknown_reasons'].get(reason,0)+1
            event.update(status='unknown',reason=reason)
        else:
            shortest = sorted((histories['verified'][t['request_id']]['input_tokens'],t['request_id'])
                              for t in possible)[:n]
            attention = sum((length+block-1)//block for length,_ in shortest)
            need = attention+(state_pages or 0)*n
            exceeds = need > pool
            total['history_known_steps'] += 1
            total['known_history_event_span_seconds'] += seconds
            if exceeds:
                total['exceeding_steps'] += 1
                total['exceeding_event_span_seconds'] += seconds
                total['exceeding_clipped_event_estimate_seconds'] += clipped
            event.update(status='bound_available',hypothetical_shortest_input_histories=[
                dict(request_id=rid,input_tokens=length) for length,rid in shortest],
                native_attention_pages_lower_bound=attention,
                native_state_pages_lower_bound=state_pages*n if state_pages is not None else None,
                native_total_pages_lower_bound=need,native_usable_pool_pages=pool,
                exceeds_native_pool=exceeds,bound_kind='full_attention_plus_state' if state_pages is not None else 'attention_only')
        evidence.append(event)
    for total in groups.values():
        total.update(history_step_coverage=total['history_known_steps']/total['steps'] if total['steps'] else None,
            exceeding_known_step_fraction=total['exceeding_steps']/total['history_known_steps'] if total['history_known_steps'] else None,
            exceeding_known_event_span_fraction=total['exceeding_event_span_seconds']/total['known_history_event_span_seconds']
                if total['known_history_event_span_seconds'] else None,
            host_anchored_clipped_event_estimate_fraction_of_window=
                total['host_anchored_clipped_event_estimate_seconds']/(hi-lo),
            exceeding_clipped_event_estimate_fraction_of_window=
                total['exceeding_clipped_event_estimate_seconds']/(hi-lo),
            actual_gpu_overlap_seconds=None,
            exceeding_cohort_gpu_overlap_bounds_seconds=[0.,min(hi-lo,total['exceeding_event_span_seconds'])])
    return dict(window=window,telemetry_available=bool(steps),groups=groups,steps=evidence,
        formal_prefill_only_recorded_steps=prefill_only_steps,http_boundary_margin_seconds=margin_seconds,
        membership='No per-step request IDs in source telemetry; never inferred as exact.',
        scope='Counts cover submissions with host started_at in [T0,T1). For each actual decode countN, '
              'take N shortest verified full input histories among all overlapping HTTP requests with20ms '
              'boundary expansion. Mixed steps ignore all prefill demand. Current-turn output growth is ignored. '
              'A lower bound exceeding pool proves inability to hold that full-history decode subset; '
              'a bound below pool is inconclusive, not proof it fits. Hypothetical chosen IDs are not measured membership.',
        timing_scope='step_ms is a CUDA event interval including maintenance and possible host submission gaps, '
            'not SM-active time. Host-start clipping is an explicitly labeled estimate, not absolute GPU overlap. '
            'Without GPU/host clock correlation only [0,min(window,sum qualifying event spans)] is guaranteed '
            'for this submitted cohort. Sampling/drop coverage and unlogged steps remain unknown.')


def review(kroot, native_root, arm='kvmem-ready'):
    k,n = load_run(kroot,arm),load_run(native_root,'native')
    controls = compare_controls(k,n)
    geometry = resolve_state_contract(k,n,cache_geometry(n['samples']))
    for run in (k,n):
        require(sum(len(r.get('output_ids',[])) for r in run['traces']) ==
                run['result'].get('total_stream_output_tokens'), 'Trace set does not reconcile with full run outputs')
    native_histories = trace_histories(n['fixture'],n['traces'])
    require(not native_histories['unknown'], 'Native trace protocol/history reconciliation failed')
    publications = records(kroot/arm/'history_publications.jsonl')
    histories = trace_histories(k['fixture'],k['traces'],publications)
    report = capacity_steps(k['traces'],records(kroot/arm/'server/capture/model_step_latency.jsonl'),
                            histories,geometry,k['window'])
    report['configured_step_sampling_interval'] = k['profile'].get('kvmem',{}).get('step_latency_interval')
    report['full_execution_coverage_verified'] = False
    paths = [Path(__file__).resolve(),kroot/'manifest.json',native_root/'manifest.json',kroot/'fixture.json',
        kroot/(arm+'.json'),native_root/'native.json',kroot/arm/'tools/traces.json',
        kroot/arm/'tools/window.json',kroot/arm/'history_publications.jsonl',
        kroot/arm/'server/capture/model_step_latency.jsonl',native_root/'native/native_samples.jsonl',
        kroot/arm/'server/capture/cache-geometry.json',
        native_root/'native/tools/traces.json',kroot/arm/'server/launch.json',native_root/'native/launch.json']
    return dict(kind='external_native_fixed12_capacity_readonly_v1',kvmem_run=str(kroot),native_run=str(native_root),
        kvmem_arm=arm,controls=controls,native_geometry=geometry,full_history_evidence=dict(
            verified_requests=len(histories['verified']),unknown_requests=histories['unknown'],scope=histories['scope']),
        capacity=report,source_sha256={str(p):sha(p) if p.exists() else None for p in paths},
        interpretation='Mechanism/capacity evidence only; not exact native executed cohort, throughput gain, '
            'uniform256K capacity, GPU idle saved, or task quality. External native control is explicitly selected; '
            'the script does not select the slowest baseline or claim it is the strongest without other runs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kvmem-run',required=True,type=Path)
    parser.add_argument('--native-run',required=True,type=Path)
    parser.add_argument('--kvmem-arm',default='kvmem-ready')
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    for root in (args.kvmem_run.resolve(),args.native_run.resolve()):
        require(not args.output.resolve().is_relative_to(root), 'Write analysis outside frozen experiment roots')
    require(not args.output.exists(), 'Use a new standalone analysis output')
    report = review(args.kvmem_run.resolve(),args.native_run.resolve(),args.kvmem_arm)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),native_pool=report['native_geometry']['total_pool_pages'],
        state_lower_bound=report['native_geometry']['recurrent_pages_per_request_lower_bound'],
        groups=report['capacity']['groups']),indent=2))


if __name__ == '__main__':
    main()
