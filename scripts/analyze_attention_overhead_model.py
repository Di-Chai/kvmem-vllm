"""Report all costs of four independent C8 comparisons without causal overclaim."""
import argparse
import json
from pathlib import Path
import statistics
from analyze_graph_cost import read,records,sha


def mtp_window(path,start,end):
    rows=[r for r in records(path) if start+2<=r['started_at']<=r['finished_at']<=end-2]
    if len(rows)<2:raise ValueError('Missing native MTP counter coverage')
    first,last=rows[0],rows[-1]
    delta={k:last['counters'][k]-v for k,v in first['counters'].items()}
    drafts=delta['vllm:spec_decode_num_drafts_total'];proposed=delta['vllm:spec_decode_num_draft_tokens_total']
    accepted=delta['vllm:spec_decode_num_accepted_tokens_total']
    if not drafts or not proposed or not 0<=accepted<=proposed:raise ValueError('Invalid MTP counters')
    return dict(mean_acceptance_length=1+accepted/drafts,draft_acceptance_rate=accepted/proposed,
        proposals=proposed,accepted_draft_tokens=accepted,drafts=drafts,samples=len(rows),
        scope='Counter deltas inside common C8 decode with two-second guards for 1Hz publication; length includes bonus token.')


def query_overlap(left,right):
    views=[];mandatory=[]
    for job in (left,right):
        rows=[r for r in records(job/'server/capture/bounded_refresh.jsonl') if r['selection']=='query']
        views.append([{p for begin,count in r['selected_history_ranges'] for p in range(begin//32,(begin+count-1)//32+1)} for r in rows])
        mandatory.append(sorted(json.dumps(r['mandatory_fit'],sort_keys=True) for r in rows))
    overlap=[len(a&b)/len(a|b) for a in views[0] for b in views[1]]
    return dict(mean=statistics.mean(overlap),minimum=min(overlap),maximum=max(overlap),
        mandatory_fits_equal=mandatory[0]==mandatory[1],scope='All cross-owner history-page Jaccards; first inputs identical for all eight owners.')


def measure(root,name):
    job=root/name;manifest=read(job/'manifest.json');proof=read(root/(name+'-executed-audit.json'))
    if proof['status']!='independently_verified_full_window_C8':raise ValueError('Missing full-window audit')
    for path,digest in proof['evidence_sha256'].items():
        if sha(job/path)!=digest:raise ValueError('Audited evidence changed: '+path)
    capture=job/'server/capture';steps=records(capture/'model_step_latency.jsonl')
    prefill=[r for r in steps if r['prefill_tokens'] and not r['decode_requests']]
    actions=proof['overhead']['actions'];batch=read(job/'qualification/decode/results.json')['batches'][0]
    resources=records(root/(name+'-resources.jsonl'));work=read(capture/'public_work.json')['submitted']
    mtp=mtp_window(root/(name+'-mtp-counters.jsonl'),proof['first_window_at'],proof['last_window_at'])
    graph=records(capture/'block_graph.jsonl');lifetime=resources[-1]['at']-resources[0]['at']
    return dict(job=name,component=manifest['overhead_component'],mode=manifest['overhead_mode'],
        engine_startup_seconds=proof['engine_startup_seconds'],cold_preparation_seconds=proof['cold_preparation_seconds'],
        effective_input_tokens_per_second=proof['effective_input_tokens_per_second'],
        pure_prefill_model_tokens_per_second=sum(r['prefill_tokens'] for r in prefill)/(sum(r['model_ms'] for r in prefill)/1000),
        full_decode_step_mean_ms=proof['full_step_mean_ms'],model_decode_step_mean_ms=proof['model_step_mean_ms'],
        shared_decode_output_tokens_per_second=proof['window']['output_tokens_per_second'],
        whole_batch_seconds=batch['wall_seconds'],whole_batch_output_tokens_per_second=batch['end_to_end']['successful_output_tokens_per_second'],
        observed_process_lifetime_seconds=lifetime,observed_output_tokens_per_second_including_startup=131072/lifetime,
        shared_decode_mtp_mean_acceptance_length=mtp['mean_acceptance_length'],
        shared_decode_draft_acceptance_rate=mtp['draft_acceptance_rate'],mtp=mtp,
        owner_verifications=work['decode_request_batches'],output_per_owner_verification=(131072-8)/work['decode_request_batches'],
        accepted_output_samples=sum(r['samples'] for r in actions),accepted_prefill_samples=sum(r['prefill_samples'] for r in actions),
        candidate_samples=max(r['candidate_samples_total'] for r in actions),
        metadata_early_batches=max(r['metadata_early_batches'] for r in actions),
        metadata_fallback_batches=max(r['metadata_fallback_batches'] for r in actions),
        gpu_peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in actions),
        gpu_peak_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in actions),
        observed_peak_gpu_used_mib=max(r['gpu_used_mib'] for r in resources),
        observed_peak_owned_process_rss_bytes=max(r.get('owned_process_rss_bytes',0) for r in resources),
        observed_peak_owned_process_pss_bytes=max(r.get('owned_process_pss_bytes',0) for r in resources),
        graph_scoring_seconds=sum(r['scoring_seconds'] for r in graph),
        graph_selection_seconds=sum(r['graph_selection_seconds'] for r in graph),
        decode_commit_native_calls=max(r.get('decode_commit_native_calls',0) for r in actions),
        decode_commit_python_calls=max(r.get('decode_commit_python_calls',0) for r in actions),
        download_pool=next((r['download_pool_total'] for r in reversed(actions) if 'download_pool_total' in r),None),
        scoring_backend_counts=actions[-1].get('scoring_backend_counts',{}),
        graph_backends={key:sum(r.get('backend','python_reference')==key for r in graph) for key in ('cpp_v2','python_reference')},
        native_graph_indexes={owner:{key:max(r.get(key,0) for r in graph if r['request_id']==owner)
            for key in ('native_index_bytes','native_index_builds','native_index_hits')}
            for owner in {r['request_id'] for r in graph if r.get('backend')=='cpp_v2'}},
        native_qk_reuse=False,identities=proof['identities'],original_job_status=proof['original_job_status'],
        original_snapshot_status=proof['original_snapshot_status'],original_snapshot_violations=proof['original_snapshot_violations'],
        audit_sha256=sha(root/(name+'-executed-audit.json')))


def analyze_combined(root,manifest):
    """One all-on cohort against the predeclared reused or fresh all-off reference."""
    fresh_reference=manifest.get('fresh_reference',False)
    expected_order=(['combined-reference-1','combined-optimized-1'] if fresh_reference
        else ['combined-optimized-1'])
    if manifest['order']!=expected_order:raise ValueError('Combined cohort order differs from its declared plan')
    name=manifest['order'][-1];job=root/name;reference=Path(manifest['reference'])
    if sha(job/'manifest.json')!=manifest['jobs'][name]:raise ValueError('Frozen all-on job changed')
    if fresh_reference:
        if reference!=root/'combined-reference-1':raise ValueError('Fresh reference differs from the declared cohort')
        if sha(reference/'manifest.json')!=manifest['jobs'][reference.name]:
            raise ValueError('Frozen all-off job changed')
    old=read(reference/'manifest.json');new=read(job/'manifest.json')
    if old.get('overhead_mode')!='reference' or new['overhead_component']!='combined':
        raise ValueError('Require declared all-off and combined all-on modes')
    source=lambda m:{k:v for k,v in m['runtime_sha256'].items() if k.startswith('src/')}
    if source(old)!=source(new):raise ValueError('Serving sources differ')
    for folder,m in ((reference,old),(job,new)):
        for path,digest in m['runtime_sha256'].items():
            if sha(folder/'runtime'/path)!=digest:raise ValueError('Frozen runtime changed: '+path)
    profile=read(reference/'profile.json');expected=json.loads(json.dumps(profile))
    if any(profile['kvmem'].get(flag,False) for flag in manifest['switches'].values()):
        raise ValueError('Reference must have all four switches off')
    expected['kvmem'].update({flag:True for flag in manifest['switches'].values()})
    if expected!=read(job/'profile.json'):raise ValueError('Combined comparison changed other profile settings')
    left=measure(reference.parent,reference.name);right=measure(root,name)
    metrics=[key for key,value in left.items() if type(value) in (int,float) and value]
    # Already completed identical all-off repetitions diagnose trajectory noise;
    # they require no further GPU work and do not replace the declared baseline.
    repeats=[]
    for proof in sorted(reference.parent.glob('*-executed-audit.json')):
        sibling=reference.parent/proof.name.removesuffix('-executed-audit.json')
        sibling_manifest=read(sibling/'manifest.json')
        if (sibling_manifest.get('overhead_mode')=='reference' and source(sibling_manifest)==source(old)
                and read(sibling/'profile.json')==profile):
            value=measure(reference.parent,sibling.name)
            repeats.append({key:value[key] for key in ('job','shared_decode_output_tokens_per_second',
                'full_decode_step_mean_ms','shared_decode_mtp_mean_acceptance_length',
                'original_job_status','original_snapshot_status')})
    return dict(combined=True,fresh_reference=fresh_reference,analyzer_sha256=sha(Path(__file__)),reference=str(reference),
        measurements=dict(all_off=left,all_on=right),
        observed_percent_change={key:100*(right[key]/left[key]-1) for key in metrics},
        changed_switches=list(manifest['switches'].values()),serving_sources_identical=True,native_qk_reuse=False,
        trajectory=dict(identity_equal={key:left['identities'][key]==right['identities'][key]
            for key in left['identities']},query_history_overlap=query_overlap(reference,job),
            verification_count_equal=left['owner_verifications']==right['owner_verifications']),
        existing_all_off_repetitions=repeats,
        limitation='User requested one combined all-off/all-on comparison. '
        +('Both primary cohorts are fresh and their all-off/all-on order was declared before launch. '
          if fresh_reference else
          'Primary all-off reference was declared before all-on launch and reuses a completed strict-passed full C8 run. ')+
        'A single pair cannot establish stable causal throughput improvement. Generation/selection and MTP differences '
        'are reported separately. Step CUDA-event intervals and wall throughput cover different scopes; startup includes '
        'first-use cache/JIT effects. Resource observations are sampled at 1Hz, process RSS/PSS at 5s. '
        'Raw snapshot-gate failures remain unchanged; complete executed-window and terminal/release audits are independent.')


def analyze(root):
    manifest=read(root/'manifest.json')
    if read(root/'status.json')['status']!='completed':raise ValueError('All declared cohorts must finish before comparison')
    if manifest.get('combined'):return analyze_combined(root,manifest)
    values={};sources=None
    for name in manifest['order']:
        job=root/name
        if sha(job/'manifest.json')!=manifest['jobs'][name]:raise ValueError('Frozen job changed')
        m=read(job/'manifest.json');src={k:v for k,v in m['runtime_sha256'].items() if k.startswith('src/')}
        if sources is None:sources=src
        if src!=sources:raise ValueError('Serving sources differ across cohorts')
        values[name]=measure(root,name)
    metrics=('engine_startup_seconds','cold_preparation_seconds','effective_input_tokens_per_second',
        'pure_prefill_model_tokens_per_second','full_decode_step_mean_ms','model_decode_step_mean_ms',
        'shared_decode_output_tokens_per_second','whole_batch_seconds','whole_batch_output_tokens_per_second',
        'observed_process_lifetime_seconds','observed_output_tokens_per_second_including_startup',
        'shared_decode_mtp_mean_acceptance_length','shared_decode_draft_acceptance_rate','output_per_owner_verification',
        'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','observed_peak_gpu_used_mib',
        'observed_peak_owned_process_rss_bytes','observed_peak_owned_process_pss_bytes',
        'graph_scoring_seconds','graph_selection_seconds')
    components={}
    for component,flag in manifest['switches'].items():
        groups={mode:[v for v in values.values() if v['component']==component and v['mode']==mode]
            for mode in ('reference','optimized')}
        if any(len(v)!=2 for v in groups.values()):raise ValueError('Require two complete cohorts per arm')
        medians={mode:{key:statistics.median(v[key] for v in group) for key in metrics} for mode,group in groups.items()}
        pairs=[]
        for reference,optimized in zip(groups['reference'],groups['optimized']):
            old=read(root/reference['job']/'profile.json');new=read(root/optimized['job']/'profile.json')
            expected=json.loads(json.dumps(old));expected['kvmem'][flag]=True
            if expected!=new:raise ValueError('More than the declared switch changed')
            pairs.append(dict(reference=reference['job'],optimized=optimized['job'],
                identity_equal={key:reference['identities'][key]==optimized['identities'][key] for key in reference['identities']},
                query_history_overlap=query_overlap(root/reference['job'],root/optimized['job']),
                verification_count_equal=reference['owner_verifications']==optimized['owner_verifications']))
        components[component]=dict(switch=flag,medians=medians,pairs=pairs,
            within_mode_repeats={mode:dict(
                identity_equal={key:group[0]['identities'][key]==group[1]['identities'][key]
                    for key in group[0]['identities']},
                query_history_overlap=query_overlap(root/group[0]['job'],root/group[1]['job']),
                verification_count_equal=group[0]['owner_verifications']==group[1]['owner_verifications'],
                second_minus_first={key:group[1][key]-group[0][key] for key in
                    ('shared_decode_output_tokens_per_second','full_decode_step_mean_ms',
                     'shared_decode_mtp_mean_acceptance_length')}) for mode,group in groups.items()},
            median_observed_percent_change={key:100*(medians['optimized'][key]/medians['reference'][key]-1)
                for key in metrics if medians['reference'][key]})
    return dict(analyzer_sha256=sha(Path(__file__)),measurements=values,components=components,
        native_qk_reuse=False,limitation='Two repetitions per arm are not a stable causal throughput estimate. '
        'Generation, selections and MTP verification lengths must be reported separately. Step values are CUDA event '
        'intervals and may include submission gaps; shared/whole-batch wall rates include inter-step CPU work. '
        'Startup includes first-use cache/JIT effects. Resource observations are sampled at 1Hz, process RSS/PSS at 5s; '
        'whole-controller endpoints have about two seconds of uncertainty. CSR/cache and pinned-pool budget sizes '
        'are reported separately from allocator/device process peaks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier analyses')
    result=analyze(args.root.resolve());args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result.get('combined'):
        print(json.dumps({mode:{key:value[key] for key in ('pure_prefill_model_tokens_per_second',
            'shared_decode_output_tokens_per_second','full_decode_step_mean_ms',
            'whole_batch_output_tokens_per_second','shared_decode_mtp_mean_acceptance_length')}
            for mode,value in result['measurements'].items()},indent=2))
    else:print(json.dumps({k:v['medians'] for k,v in result['components'].items()},indent=2))


if __name__=='__main__':main()
