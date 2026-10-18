"""Read-only accounting for live DeepSWE capacity windows (no model requests)."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics


def records(path):
    if not path.exists():return []
    result=[]
    with path.open() as stream:
        for line in stream:
            try:result.append(json.loads(line))
            except json.JSONDecodeError:
                if line.endswith('\n'):raise
    return result


def metrics(body):
    result={}
    for line in body.splitlines():
        if not line or line.startswith('#'):continue
        full,value=line.rsplit(' ',1)
        name=full.split('{',1)[0].strip()
        if name=='vllm:prompt_tokens_by_source_total':
            source=re.search(r'(?:\{|,)source="([^"]+)"',full)
            if source is None:raise ValueError('Prompt source missing')
            name+=':'+source.group(1)
        result[name]=result.get(name,0.)+float(value)
    return result


def prefill_count(row):
    if 'vllm:prompt_tokens_by_source_total:local_compute' in row:
        return row['vllm:prompt_tokens_by_source_total:local_compute']
    if {'vllm:prompt_tokens_total','vllm:prompt_tokens_cached_total'}<=row.keys():
        return row['vllm:prompt_tokens_total']-row['vllm:prompt_tokens_cached_total']
    raise ValueError('No computed-prefill accounting; never treat missing cached count as zero')


def delta(first,last,name):
    if name not in first or name not in last:return None
    value=last[name]-first[name]
    if not math.isfinite(value) or value<0:raise ValueError('Counter reset or invalid delta: '+name)
    return value


def distribution(values):
    missing=sum(v is None for v in values)
    values=sorted(v for v in values if v is not None)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Invalid measured distribution value')
    def percentile(q):
        if not values:return None
        x=(len(values)-1)*q;i=int(x)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)
    return dict(count=len(values),missing_count=missing,mean=statistics.mean(values) if values else None,
                p50=percentile(.5),p95=percentile(.95),maximum=max(values,default=None))


def segment(samples,begin,end):
    chosen=[s for s in samples if begin<=s['at']<=end and 'raw_metrics' in s]
    if len(chosen)<2:return dict(available=False,sample_count=len(chosen))
    first,last=chosen[0],chosen[-1];seconds=last['at']-first['at']
    parsed=[metrics(s['raw_metrics']) for s in chosen]
    a,b=parsed[0],parsed[-1]
    pf=prefill_count(b)-prefill_count(a)
    if pf<0:raise ValueError('Computed-prefill counter reset')
    output=delta(a,b,'vllm:generation_tokens_total')
    if output is None:raise ValueError('Generation counter missing')
    gauges={key:distribution([s.get(key) for s in parsed]) for key in (
        'vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc')}
    drafts=delta(a,b,'vllm:spec_decode_num_drafts_total')
    drafted=delta(a,b,'vllm:spec_decode_num_draft_tokens_total')
    accepted=delta(a,b,'vllm:spec_decode_num_accepted_tokens_total')
    return dict(available=True,begin=first['at'],end=last['at'],seconds=seconds,sample_count=len(chosen),
        computed_public_prefill_tokens=pf,prefill_tokens_per_second=pf/seconds,
        accepted_generation_tokens=output,generation_tokens_per_second=output/seconds,
        telemetry_interval_seconds=distribution([b['at']-a['at'] for a,b in zip(chosen,chosen[1:])]),
        preemptions=delta(a,b,'vllm:num_preemptions_total'),native_gauges=gauges,
        speculation=dict(sequence_verify_steps=drafts,drafted_tokens=drafted,accepted_draft_tokens=accepted,
            draft_acceptance_rate=accepted/drafted if drafted and accepted is not None else None,
            mean_acceptance_length=1+accepted/drafts if drafts and accepted is not None else None),
        scope='Native counter deltas / observed wall seconds, including tool waits and all interference; generation includes first tokens. Private KVMem Query replay is additional work.')


def decode_interiors(starts,finishes,margin=.02):
    """Bound decode interiors despite unknown ingress/egress transport costs.

    Let a/f be bridge admission/completion and q,p,d native queue/prefill/decode
    durations. Unknown nonnegative ingress+egress is e=f-a-q-p-d. Every possible
    decode interval then contains [f-d, a+q+p+d]. These are engine request-state
    intervals, not proof that the request ran in every GPU batch.
    """
    index={};duplicates=set();reasons=Counter();intervals=[];tokens=seconds=0.
    for row in starts:
        key=row['task'],row['turn']
        if key in index:duplicates.add(key)
        index[key]=row
    for row in finishes:
        key=row['task'],row['turn'];native=row.get('metrics') or {}
        values=[native.get(k) for k in ('queue_time_ms','time_to_first_token_ms','generation_time_ms')]
        if key not in index or key in duplicates:
            reasons['missing_or_ambiguous_start']+=1;continue
        if any(v is None or not math.isfinite(v) or v<0 for v in values):
            reasons['missing_or_invalid_native_timing']+=1;continue
        q,p,d=[v/1000 for v in values];a=index[key]['at'];f=row['at']
        if d>0:
            tokens+=max(0,row['usage']['completion_tokens']-1);seconds+=d
        slack=f-a-q-p-d
        if slack<0:
            reasons['inconsistent_clock_or_transport_bounds']+=1;continue
        left,right=f-d+margin,a+q+p+d-margin
        if left>=right:
            reasons['no_certified_interior']+=1;continue
        intervals.append(dict(begin=left,end=right,task=row['task'],turn=row['turn'],
                              unassigned_transport_seconds=slack))
    return dict(intervals=intervals,rejected=dict(reasons),weighted_per_decoder_tokens_per_second=tokens/seconds if seconds else None,
        decoded_tokens_excluding_first=tokens,summed_per_request_decode_seconds=seconds,
        scope='Per-decoder weighted rate excludes first tokens; interval interiors are conservative after20ms margins. They describe completed request readiness, not same-step GPU concurrency.')


def overlap_windows(intervals,target):
    events=Counter()
    for r in intervals:events[r['begin']]+=1;events[r['end']]-=1
    points=sorted(events);active=0;windows=[]
    for start,end in zip(points,points[1:]):
        active+=events[start]
        if active>=target:
            if windows and windows[-1][1]==start:windows[-1][1]=end
            else:windows.append([start,end])
    return windows


def possible_prefill_envelopes(starts,finishes,end,margin=.02):
    # Prompt counters are credited on a request's first output, not on every
    # chunk. A zero prompt-counter delta alone cannot exclude in-flight prefill.
    index={};duplicates=set();result=[]
    for row in finishes:
        key=row['task'],row['turn']
        if key in index:duplicates.add(key)
        index[key]=row
    for row in starts:
        key=row['task'],row['turn'];finish=index.get(key);left=row['at']-margin;right=end+margin
        if finish is not None and key not in duplicates:
            native=finish.get('metrics') or {}
            values=[native.get(k) for k in ('queue_time_ms','time_to_first_token_ms','generation_time_ms')]
            if all(v is not None and math.isfinite(v) and v>=0 for v in values):
                q,p,d=[v/1000 for v in values]
                if finish['at']-row['at']>=q+p+d:
                    left=row['at']+q-margin;right=finish['at']-d+margin
        if left<right:result.append([left,right])
    return result


def decode_accounting(samples,starts,finishes,workers):
    interior=decode_interiors(starts,finishes)
    # A sample's timestamp precedes its metrics read. In this bounded harness
    # each cycle ends with Event.wait(1), so the next cycle's timestamp minus1
    # bounds the preceding read from above. Require the complete acquisition
    # envelope to lie inside a known decode window; omit the unbounded tail.
    rows=[s for s in samples if 'raw_metrics' in s and 'gpu' in s]
    possible_prefill=possible_prefill_envelopes(starts,finishes,max((s['at'] for s in samples),default=0.))
    parsed=[metrics(s['raw_metrics']) for s in rows];reports={}
    for target in sorted({1,workers}):
        spans=overlap_windows(interior['intervals'],target);count=0.;duration=0.;accepted=0
        for i in range(len(rows)-2):
            a,b,c=rows[i:i+3];x,y=parsed[i:i+2]
            if c['at']-b['at']<1:continue
            pf=prefill_count(y)-prefill_count(x)
            if pf<0:raise ValueError('Computed-prefill counter reset')
            output=delta(x,y,'vllm:generation_tokens_total')
            if output is None:continue
            left,right=a['at']-.02,c['at']-1+.02
            excluded=any(start<right and left<end for start,end in possible_prefill)
            if pf==0 and not excluded and any(start<=left and right<=end for start,end in spans):
                count+=output;duration+=b['at']-a['at'];accepted+=1
        reports[str(target)]=dict(minimum_known_ready_decoders=target,
            certified_overlap_seconds=sum(end-start for start,end in spans),
            eligible_sample_intervals=accepted,seconds=duration,generation_tokens=count,
            generation_tokens_per_second=count/duration if duration else None)
    return dict(completed_request_timing=interior,no_public_prefill_windows=reports,
        possible_prefill_envelopes=len(possible_prefill),
        scope='Conservative completed-request service proxy. Excludes all possible queue-to-first-token envelopes, including unfinished requests; zero prompt-counter deltas alone do not rule out chunked prefill. Zero-output stalls inside eligible windows remain in the denominator. Recompute after preemption and private KVMem work may remain; this is not pure GPU-kernel throughput.')


def iteration_histogram(body):
    result={}
    for line in body.splitlines():
        if not line.startswith('vllm:iteration_tokens_total_bucket{'):continue
        label=re.search(r'(?:\{|,)le="([^"]+)"',line)
        if label is None:raise ValueError('Iteration bucket boundary missing')
        boundary=float(label.group(1))
        if boundary in result:raise ValueError('One engine/model series required for iteration proof')
        result[boundary]=float(line.rsplit(' ',1)[1])
    return result


def native_batch_lower_bound(samples,request_count,workers,draft_tokens,quiescent_baseline,
                             *,separated_baseline=False):
    """Lower-bound generation-bearing request rows in an actual native step.

    Native iteration histograms mix accepted generation with the entire prompt
    credit emitted at first output. Pessimistically remove one high observation
    per HTTP request (plus2 for initialization). For scrape races, use the
    penultimate total count minus the latest <=bucket count, a lower bound on
    the latest true tail because the scrapes do not overlap. Never infer an
    exact batch size from a running gauge or from an uncorrected high bucket.
    """
    rows=[s for s in samples if 'raw_metrics' in s]
    if (not quiescent_baseline and not separated_baseline) or len(rows)<(4 if separated_baseline else 3):
        return dict(available=False,reason='Need quiescent baseline and separated observed scrapes')
    # A recomputed request can publish another prompt credit. The one-credit
    # per HTTP request bound below is valid only without native preemptions.
    if any(metrics(r['raw_metrics']).get('vllm:num_preemptions_total') != 0
           for r in (rows[0], rows[1], rows[-2], rows[-1])):
        return dict(available=False,reason='Need recorded zero native preemptions; recompute can repeat prompt credits')
    baseline,previous,last=[metrics(r['raw_metrics']) for r in (rows[1] if separated_baseline else rows[0],rows[-2],rows[-1])]
    first_buckets=iteration_histogram(rows[0]['raw_metrics']);last_buckets=iteration_histogram(rows[-1]['raw_metrics'])
    key='vllm:iteration_tokens_total_count'
    if key not in baseline or key not in previous or not first_buckets or first_buckets.keys()!=last_buckets.keys():
        return dict(available=False,reason='Iteration histogram unavailable or changed')
    per_request=draft_tokens+1;contamination=request_count+2;bounds=[];violations=[];best=0
    for boundary in sorted(k for k in first_buckets if math.isfinite(k)):
        initial_tail=baseline[key]-first_buckets[boundary]
        tail_lower=max(0.,previous[key]-last_buckets[boundary]-initial_tail)
        uncredited_lower=max(0.,tail_lower-contamination)
        minimum=math.ceil((math.floor(boundary)+1)/per_request)
        row=dict(bucket_upper=boundary,new_high_iteration_count_lower_bound=tail_lower,
            possible_prompt_credit_iterations_upper_bound=contamination,
            high_generation_only_iteration_count_lower_bound=uncredited_lower,
            per_step_request_lower_bound=minimum)
        bounds.append(row)
        if uncredited_lower>0:
            if minimum>workers:violations.append(row)
            else:best=max(best,minimum)
    return dict(available=not bool(violations),maximum_observed_request_cohort_lower_bound=best,
        per_request_maximum_new_tokens=per_request,bounds=bounds,assumption_violations=violations,
        baseline_scope='Later initial count minus earlier initial bucket bounds old tail from above' if separated_baseline else 'Quiescent initialization baseline',
        scope='Derived native-step lower bound for single-engine MTP with one output sequence per request. Native source emits one histogram observation per EngineCoreOutputs/model step. Includes generation-bearing requests, not an exact pure-decode batch count or kernel-overlap proof; request history lengths within the identified steps are not recovered.')


def native_verification_cohort_bound(samples,workers):
    """Mean verified sequence count lower-bounds the largest actual cohort.

    Audited normal-MTP scheduler calls observe_draft once per scheduled request
    in update_from_output; one EngineCoreOutputs carries the resulting stats.
    The single frontend observes one iteration histogram value for that output.
    Use inner draft counters and outer iteration counters, with two additional
    iterations for logger records straddling either metric update boundary.
    This contract requires separated scrapes and one native engine/frontend.
    """
    rows=[r for r in samples if 'raw_metrics' in r]
    if len(rows)<4:return dict(available=False,reason='Need four separated scrapes')
    chosen=[rows[0],rows[1],rows[-2],rows[-1]]
    for row in chosen:
        if not iteration_histogram(row['raw_metrics']):
            return dict(available=False,reason='Single-engine iteration histogram required')
        spec=[line for line in row['raw_metrics'].splitlines()
              if line.startswith(('vllm:spec_decode_num_drafts_total{','vllm:spec_decode_num_drafts_total '))]
        if len(spec)!=1:return dict(available=False,reason='One MTP sequence-counter series required')
    first,inner_first,inner_last,last=[metrics(r['raw_metrics']) for r in chosen]
    # Stale outputs after native preemption may carry scheduler speculation
    # stats without output-bearing IterationStats. Do not use their ratio.
    if any(s.get('vllm:num_preemptions_total')!=0 for s in (first,inner_first,inner_last,last)):
        return dict(available=False,reason='Need zero native preemptions; stale stats can lack output iterations')
    sequences=delta(inner_first,inner_last,'vllm:spec_decode_num_drafts_total')
    iterations=delta(first,last,'vllm:iteration_tokens_total_count')
    if sequences is None or iterations is None:
        return dict(available=False,reason='MTP sequence or model-iteration counter missing')
    upper=iterations+2
    bound=math.ceil(sequences/upper)
    return dict(available=bound<=workers,maximum_verification_request_cohort_lower_bound=bound,
        verified_sequences_lower_bound=sequences,output_iterations_upper_bound=upper,
        extra_boundary_iterations=2,assumption_violated=bound>workers,
        scope='Source-derived normal-MTP target-verification request-cohort lower bound, zero native preemptions, inner sequence counts / outer output-iteration counts. Not a running gauge, an exact pure-decode batch count, a kernel-overlap proof or a request-history mapping. Extra logger boundary records are included conservatively.')


def historical_view_evidence(samples,profile,geometry,archive,run_id,task_names,workers):
    kv=profile['kvmem'];limit=kv['select_budget']+kv['gen_budget'];bytes_per_token=geometry.get('cpu_archive_bytes_per_token',0)
    history_limit=profile.get('max_model_len')
    valid_limit=type(history_limit) is int and limit<=history_limit<=262144
    contract=(valid_limit and geometry.get('select_budget')==65536 and geometry.get('gen_budget')==32768
        and geometry.get('historical_tokens')==history_limit and geometry.get('cpu_offload_bytes')==100_000_000_000
        and geometry.get('archive_codec')=='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1' and bytes_per_token==34816
        and archive.get('archive_backend')=='cpu' and archive.get('repository_gc') is False
        and archive.get('disk_read_bytes')==archive.get('disk_write_bytes')==0)
    allowed={run_id+'-'+name for name in task_names};consistent=0;best=None;violations=[]
    for sample in samples:
        resources=sample.get('scheduler_resources',{});cpu=sample.get('archive',{})
        all_views=resources.get('session_views',[])
        views=[v for v in all_views if v.get('namespace') and v['namespace'][0] in allowed]
        for view in views:
            if not valid_limit or view['window']>limit or view['history']>history_limit:
                violations.append(dict(at=sample['at'],namespace=view['namespace'],history=view['history'],window=view['window']))
        # snapshot() reports idle entries only, after this session's worker
        # retirement acknowledgment. Other running sessions can still have DMA
        # in flight; global pending_downloads must not invalidate this already
        # published session. Separately sampled files must cover the complete
        # sealed records, not just selected pages; no byte tolerance is allowed.
        minimum=sum(v['history']//kv['block_tokens']*kv['block_tokens']*bytes_per_token for v in all_views)
        coherent=(bytes_per_token>0 and cpu.get('archive_backend')=='cpu'
            and cpu.get('cpu_authority_bytes',-1)>=minimum
            and cpu.get('disk_read_bytes')==cpu.get('disk_write_bytes')==0)
        if not coherent:continue
        consistent+=1
        compressed=list({tuple(v['namespace']):v for v in views
                         if v['history']>limit and 0<v['window']<=limit}.values())
        row=dict(at=sample['at'],compressed_users=len(compressed),views=compressed,
            minimum_sealed_full_history_bytes=minimum,cpu_authority_bytes=cpu['cpu_authority_bytes'],
            owned_blocks=resources.get('owned_blocks'),free_blocks=resources.get('free_blocks'),
            total_blocks=resources.get('total_blocks'))
        if best is None or (row['compressed_users'],minimum)>(best['compressed_users'],best['minimum_sealed_full_history_bytes']):best=row
    return dict(contract_matches=contract,historical_token_limit=history_limit,consistent_publication_snapshots=consistent,
        maximum_coherently_compressed_users=best['compressed_users'] if best else 0,
        requested_agent_workers=workers,all_workers_long_history_compression_observed=bool(
            contract and best and best['compressed_users']>=workers and not violations),
        best_snapshot=best,budget_violations=violations,
        scope='Idle-session snapshot publication accounting only: active/lookup sessions are absent from this diagnostic, so it undercounts long-history users while they compute. CPU bytes cover complete sealed records of the reported views, not only selected pages. This is not a byte oracle, a proof of simultaneous GPU residence, or a full-history-limit-per-user peak test. GPU pool allocation and occupied pages are distinct.')


def step_evidence(root,begin,end,clock,workers):
    path=root/'server/capture/model_step_latency.jsonl'
    if not path.exists() or clock is None:
        return dict(available=False,scope='Running gauges are not same-step decode concurrency; an independent native profile is still required.')
    offset=clock['perf_counter_at']-clock['wall_at']
    # The clock pair maps existing monotonic CUDA logs to UTC; omit boundaries
    # instead of claiming subsecond alignment from independently read clocks.
    rows=[r for r in records(path) if begin+offset+2<=r['started_at']<=end+offset-2]
    pure=[r for r in rows if not r.get('prefill_tokens') and r.get('decode_requests',0)>0]
    mixed=[r for r in rows if r.get('prefill_tokens') and r.get('decode_requests',0)>0]
    histogram=Counter(r['decode_requests'] for r in pure)
    return dict(available=bool(rows),requested_agent_workers=workers,
        observed_public_steps=len(rows),pure_decode_steps=len(pure),mixed_steps=len(mixed),
        pure_decode_batch_histogram=dict(histogram),maximum_pure_decode_requests=max(histogram,default=0),
        target_concurrency_steps=histogram.get(workers,0),
        pure_decode_step_ms=distribution([r['step_ms'] for r in pure]),
        pure_decode_model_ms=distribution([r['model_ms'] for r in pure if 'model_ms' in r]),
        pure_decode_maintenance_host_ms=distribution([r['maintenance_host_ms'] for r in pure if 'maintenance_host_ms' in r]),
        scope='Completed public GPU-step records, with2s boundary exclusion; CUDA-event spans include native MTP/capture. This proves batch occupancy, not useful accepted output by itself.')


def analyze_stage(root,clock=None):
    run=root/'run.json'
    if not run.exists():return dict(status='not_started')
    state=json.loads(run.read_text());begin=state.get('model_ready_at')
    if not begin:return dict(status=state['status'],formal_started=False)
    if state.get('model_ready_monotonic') is not None:
        clock=dict(wall_at=begin,perf_counter_at=state['model_ready_monotonic'])
    samples=records(root/'resources.jsonl')
    end=min(state.get('window_deadline') or float('inf'),
            state.get('ended_at') or max((s['at'] for s in samples),default=begin))
    baseline=root/'metrics-before-formal.txt'
    if baseline.exists():samples=[dict(at=begin,raw_metrics=baseline.read_text()),*samples]
    samples=[s for s in samples if begin<=s['at']<=end]
    events=records(root/'bridge/requests.jsonl')
    starts=[r for r in events if r.get('event')=='start' and begin<=r['at']<=end]
    finishes=[r for r in events if r.get('event')=='finish' and begin<=r['at']<=end]
    histogram=Counter()
    for row in starts:
        n=row['prompt_tokens']
        bucket='<=64K' if n<=65536 else '64K-96K' if n<=98304 else '96K-128K' if n<=131072 else '128K-192K' if n<=196608 else '192K-256K'
        histogram[bucket]+=1
    gpu=[]
    for row in samples:
        if row.get('gpu'):
            try:
                memory,util=map(float,row['gpu'].split(','));gpu.append((memory,util))
            except ValueError:pass
    tasks=Counter(t['state'] for t in state['tasks'].values())
    result=dict(status=state['status'],formal_started=True,window_begin=begin,observed_end=end,
        whole_window=segment(samples,begin,end),
        time_segments=[dict(offset_seconds=offset,**segment(samples,begin+offset,min(begin+offset+300,end)))
                       for offset in range(0,int(max(0,end-begin)),300)],
        started_model_requests=len(starts),completed_model_turns=len(finishes),task_states=dict(tasks),
        finished_turn_output_tokens=sum(r['usage']['completion_tokens'] for r in finishes),
        tool_calls=sum(r.get('tool_calls',0) for r in finishes),prompt_length_histogram=dict(histogram),
        prompt_lengths=distribution([r['prompt_tokens'] for r in starts]),
        incremental_input_tokens=distribution([r['new_input_tokens'] for r in starts if 'new_input_tokens' in r]),
        exact_output_prefix_reuse=dict(requests=len(starts),reused=sum(r.get('exact_output_prefix_reused') is True for r in starts),
            fraction=sum(r.get('exact_output_prefix_reused') is True for r in starts)/len(starts) if starts else None),
        prepare_seconds=distribution([r['prepare_seconds'] for r in starts if 'prepare_seconds' in r]),
        native_ttft_ms=distribution([r.get('metrics',{}).get('time_to_first_token_ms') if r.get('metrics') else None for r in finishes]),
        native_queue_ms=distribution([r.get('metrics',{}).get('queue_time_ms') if r.get('metrics') else None for r in finishes]),
        native_generation_ms=distribution([r.get('metrics',{}).get('generation_time_ms') if r.get('metrics') else None for r in finishes]),
        decode=decode_accounting(samples,starts,finishes,state['workers']),
        gpu_utilization_percent=distribution([u for _,u in gpu]),
        gpu_memory_mib_max=max((n for n,_ in gpu),default=None),
        host_available_bytes_min=min((s['available_bytes'] for s in samples if 'available_bytes' in s),default=None),
        cpu_authority_bytes_max=max((s.get('archive',{}).get('cpu_authority_bytes',0) for s in samples),default=None),
        context_ceiling_events=[r for r in events if r.get('event')=='context_limit'],
        errors=[r for r in events if r.get('event')=='error'],
        quality_scope='Only naturally completed verified tasks have scores; window truncation has no score. Divergent agent trajectories require history/phase-matched interpretation.',
        actual_decode_batch_proof=step_evidence(root,begin,end,clock,state['workers']))
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('backend_kind')=='native':
        profile=json.loads((root/'profile.json').read_text())
        warmup=root/'bounded-warmup.json'
        quiescent=baseline.exists() and warmup.exists() and json.loads(warmup.read_text()).get('passed') is True
        result['native_step_request_lower_bound']=native_batch_lower_bound(samples,
            sum(r.get('event')=='start' for r in events),state['workers'],profile['draft_tokens'],quiescent)
        result['native_verification_request_lower_bound']=(native_verification_cohort_bound(samples,state['workers'])
            if profile.get('draft_method')=='mtp' else dict(available=False,reason='Normal MTP required'))
        for phase in result['time_segments']:
            phase_begin=begin+phase['offset_seconds'];phase_end=min(phase_begin+300,end)
            # An active starting baseline can race with logging. A later count
            # minus an earlier bucket overestimates its old tail; subtracting
            # it cannot invent new high observations. Remove every possible
            # prompt credit, including requests admitted before this phase.
            phase['native_step_request_lower_bound']=native_batch_lower_bound(
                [s for s in samples if phase_begin<=s['at']<=phase_end],
                sum(r.get('event')=='start' and r['at']<=phase_end for r in events),
                state['workers'],profile['draft_tokens'],False,separated_baseline=True)
            phase['native_verification_request_lower_bound']=native_verification_cohort_bound(
                [s for s in samples if phase_begin<=s['at']<=phase_end],state['workers'])
    elif (root/'server/capture/cache-geometry.json').exists() and (root/'server/capture/archive.json').exists():
        profile=json.loads((root/'profile.json').read_text())
        result['historical_view']=historical_view_evidence(samples,profile,
            json.loads((root/'server/capture/cache-geometry.json').read_text()),
            json.loads((root/'server/capture/archive.json').read_text()),state['id'],state['tasks'],state['workers'])
    refreshes=[r for r in records(root/'server/capture/bounded_refresh.jsonl')
               if clock and begin+clock['perf_counter_at']-clock['wall_at']+2<=r.get('refresh_started_at',-1)
               and r.get('refresh_finished_at',float('inf'))<=end+clock['perf_counter_at']-clock['wall_at']-2]
    result['kvmem_refresh']=dict(count=len(refreshes),
        seconds=distribution([r['refresh_finished_at']-r['refresh_started_at'] for r in refreshes]),
        assembly_seconds=distribution([r['assembly_seconds'] for r in refreshes if 'assembly_seconds' in r]),
        uploaded_bytes=sum(r.get('assembly_uploaded_bytes',0) for r in refreshes),
        compressed_refreshes=sum(0<r.get('selected_tokens',0)<=65536<r.get('history_begin',0) for r in refreshes),
        selection_fraction=distribution([r['selected_tokens']/r['history_begin'] for r in refreshes if r.get('history_begin')]),
        scope='Per-request durations can overlap; their sum is not a fraction of GPU wall time.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    m=json.loads((a.root/'manifest.json').read_text())
    mapping=a.root/'analysis-clock.json'
    clock=json.loads(mapping.read_text()) if mapping.exists() else None
    carried=m.get('carried_windows',{})
    result=dict(kind='live_deepswe_capacity_analysis_v1',stages={s:analyze_stage(
        Path(carried[s]['path']) if s in carried else a.root/s,clock) for s in m.get('comparison_stages',m['stages'])},
        carried_windows=carried,
        acceptance='Preliminary single-system windows; no universal throughput or quality claim without coverage and concurrency review')
    body=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(body)
    else:print(body)


if __name__=='__main__':main()
