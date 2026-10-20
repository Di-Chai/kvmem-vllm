"""Summarize audited, alternating C8 runs without attributing trajectory drift."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_graph_cost import read, records, sha


def mtp_window(path, start, end):
    # The engine publishes these counters once per second. Leave two seconds
    # at both boundaries so delayed publication cannot import mixed prefill.
    observed=[r for r in records(path) if start+2 <= r['started_at'] <= r['finished_at'] <= end-2]
    if len(observed)<2:
        raise ValueError('MTP counter observations do not cover the shared decode window')
    first,last=observed[0],observed[-1]
    delta={key:last['counters'][key]-value for key,value in first['counters'].items()}
    drafts=delta['vllm:spec_decode_num_drafts_total']
    proposed=delta['vllm:spec_decode_num_draft_tokens_total']
    accepted=delta['vllm:spec_decode_num_accepted_tokens_total']
    if not drafts or not proposed or not 0 <= accepted <= proposed:
        raise ValueError('Invalid MTP counters')
    return dict(mean_acceptance_length=1+accepted/drafts,draft_acceptance_rate=accepted/proposed,
        drafts=drafts,drafted_tokens=proposed,accepted_draft_tokens=accepted,
        seconds=last['finished_at']-first['started_at'],samples=len(observed),
        scope='Counter reads inside shared C8 decode, with two-second guards for 1Hz publication; '
              'acceptance length includes bonus token.')


def query_overlap(left,right):
    views=[];mandatory=[]
    for job in (left,right):
        query=[r for r in records(job/'server/capture/bounded_refresh.jsonl') if r['selection']=='query']
        views.append([{p for begin,count in r['selected_history_ranges']
                       for p in range(begin//32,(begin+count-1)//32+1)} for r in query])
        mandatory.append(sorted(json.dumps(r['mandatory_fit'],sort_keys=True) for r in query))
    overlap=[len(a&b)/len(a|b) for a in views[0] for b in views[1]]
    return dict(mean=statistics.mean(overlap),minimum=min(overlap),maximum=max(overlap),
        mandatory_fits_equal=mandatory[0]==mandatory[1],
        scope='All cross-owner Query history-page Jaccards; all first inputs are identical. '
              'No arbitrary client pairing or output-trajectory equality is assumed.')


def analyze(root):
    manifest = read(root/'manifest.json')
    if read(root/'status.json')['status'] != 'completed':
        raise ValueError('All frozen runs must complete before comparison')
    values = {}
    for name in manifest['order']:
        job = root/name
        if sha(job/'manifest.json') != manifest['jobs'][name]:
            raise ValueError('Frozen job manifest changed')
        proof = read(root/(name+'-executed-audit.json'))
        if proof['status'] != 'independently_verified_full_window_C8':
            raise ValueError('Missing full-window C8 audit')
        for path,digest in proof['evidence_sha256'].items():
            if sha(job/path) != digest:
                raise ValueError('Audited evidence changed: '+path)
        capture = job/'server/capture'
        steps = records(capture/'model_step_latency.jsonl')
        actions = records(capture/'action_attention.jsonl')
        batch = read(job/'qualification/decode/results.json')['batches'][0]
        work = read(capture/'public_work.json')['submitted']
        prefill = [r for r in steps if r['prefill_tokens'] and not r['decode_requests']]
        resources = records(root/(name+'-resources.jsonl'))
        native = proof.get('native_stats')
        counts = {} if native is None else native['layer_phase_counts']
        mtp=mtp_window(root/(name+'-mtp-counters.jsonl'),proof['first_window_at'],proof['last_window_at'])
        lifetime=resources[-1]['at']-resources[0]['at']
        values[name] = dict(job=name,mode=read(job/'manifest.json')['native_qk_mode'],
            engine_startup_seconds=proof['engine_startup_seconds'],
            cold_preparation_seconds=proof['cold_preparation_seconds'],
            pure_prefill_model_tokens_per_second=sum(r['prefill_tokens'] for r in prefill)/
                (sum(r['model_ms'] for r in prefill)/1000),
            full_decode_step_mean_ms=proof['full_step_mean_ms'],
            model_decode_step_mean_ms=proof['model_step_mean_ms'],
            shared_decode_output_tokens_per_second=proof['window']['output_tokens_per_second'],
            whole_batch_output_tokens_per_second=batch['end_to_end']['successful_output_tokens_per_second'],
            whole_batch_seconds=batch['wall_seconds'],
            observed_process_lifetime_seconds=lifetime,
            observed_output_tokens_per_second_including_startup=(8*16384)/lifetime,
            lifetime_observation_limit='1Hz resources bracket controller load/run/cleanup to about two seconds.',
            output_per_owner_verification=(8*16384-8)/work['decode_request_batches'],
            owner_verifications=work['decode_request_batches'],
            mtp=mtp,
            shared_decode_mtp_mean_acceptance_length=mtp['mean_acceptance_length'],
            shared_decode_draft_acceptance_rate=mtp['draft_acceptance_rate'],
            gpu_peak_allocated_bytes=max(r.get('gpu_peak_allocated_bytes',0) for r in actions),
            gpu_peak_reserved_bytes=max(r.get('gpu_peak_reserved_bytes',0) for r in actions),
            observed_peak_gpu_used_mib=max(r['gpu_used_mib'] for r in resources),
            accepted_output_samples=sum(r['samples'] for r in actions),
            accepted_prefill_samples=sum(r['prefill_samples'] for r in actions),
            native_fallback_operations=sum(v for k,v in counts.items() if k.startswith('fallback:')),
            native_stats=native,identities=proof['identities'],
            original_job_status=proof['original_job_status'],
            original_snapshot_status=proof['original_snapshot_status'],
            audit_sha256=sha(root/(name+'-executed-audit.json')))
    groups = {mode:[v for v in values.values() if v['mode']==mode] for mode in ('off','reuse')}
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError('Require at least two repetitions per mode')
    metrics = ('cold_preparation_seconds','pure_prefill_model_tokens_per_second','full_decode_step_mean_ms',
        'model_decode_step_mean_ms','shared_decode_output_tokens_per_second','whole_batch_output_tokens_per_second',
        'whole_batch_seconds','observed_process_lifetime_seconds',
        'observed_output_tokens_per_second_including_startup',
        'shared_decode_mtp_mean_acceptance_length','shared_decode_draft_acceptance_rate',
        'output_per_owner_verification','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')
    medians = {mode:{metric:statistics.median(row[metric] for row in group) for metric in metrics}
               for mode,group in groups.items()}
    pairs = []
    for reuse,off in zip(groups['reuse'],groups['off']):
        pairs.append(dict(identity_equal={k:reuse['identities'][k]==off['identities'][k] for k in off['identities']},
            query_history_overlap=query_overlap(root/reuse['job'],root/off['job']),
            verification_count_equal=reuse['owner_verifications']==off['owner_verifications']))
    same = all(all(p['identity_equal'].values()) and p['verification_count_equal'] for p in pairs)
    return dict(analyzer_sha256=sha(Path(__file__)),measurements=values,medians=medians,pairs=pairs,
        median_observed_percent_change={metric:100*(medians['reuse'][metric]/medians['off'][metric]-1)
                                       for metric in metrics if medians['off'][metric]},
        exact_workload_identity_equal=same,
        limitation='Two repetitions per arm; observed changes are not a stable causal estimate. '
            'Report accepted output, Query/pressure selection and MTP verification differences. '
            'Engine startup includes first-use JIT cache effects; request preparation is measured separately. '
        'output_per_owner_verification is complete accepted output divided by actual owner verification '
            'steps. MTP acceptance length is separately computed from native counters inside shared decode.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new analysis output')
    result = analyze(args.root.resolve())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['medians'],indent=2))


if __name__ == '__main__':
    main()
