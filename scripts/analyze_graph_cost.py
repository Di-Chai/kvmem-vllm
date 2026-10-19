"""Read complete immutable cost-screen evidence; never rerun or score answers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def engine_totals(rows):
    # Each owner reports the SAME engine counters at a later instant. Summing
    # the reports would manufacture up to eight times the actual host work.
    return {key:max((row.get(key, 0) for row in rows), default=0) for key in (
        'capture_host_seconds_total', 'metadata_wait_seconds', 'prefill_commit_host_seconds',
        'accept_event_wait_seconds', 'graph_prepare_seconds', 'graph_commit_seconds',
        'graph_prepare_calls', 'graph_commit_calls')}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compare_identity(before, after):
    return dict(input_equal=before['input_sha256'] == after['input_sha256'],
        query_selection_equal=before['query_selections'] == after['query_selections'],
        pressure_selection_equal=before['pressure_selections'] == after['pressure_selections'],
        accepted_output_equal=before['output_sha256'] == after['output_sha256'],
        decode_owner_steps_equal=before['decode_owner_steps'] == after['decode_owner_steps'])


def differences(before, after):
    return {name:100*(after[key]/before[key]-1) for name,key in (
        ('cold_preparation_time_change_percent','cold_preparation_seconds'),
        ('full_decode_step_change_percent','full_step_mean_ms'),
        ('model_decode_step_change_percent','model_step_mean_ms'),
        ('decode_throughput_change_percent','steady_decode_tokens_per_second'),
        ('whole_batch_time_change_percent','wall_seconds'))}


def analyze(root, audit_directory=None):
    manifest = read(root/'manifest.json')
    variants = ('disabled', 'attention', 'collect')
    if set(manifest['order']) != set(variants): raise ValueError('Require all three arms')
    result = dict(schema='graph-cost-comparison-2', single_ordered_cold_screen=True, measurements={}, pairs={})
    for variant in variants:
        job = root/variant
        if sha(job/'manifest.json') != manifest['jobs'][variant]: raise ValueError('Manifest changed')
        audit_file = (audit_directory/(variant+'-executed-audit-r1.json') if audit_directory is not None
                      else root/(variant+'-executed-audit.json'))
        proof = read(audit_file)
        if proof['status'] != 'independently_verified_full_window_C8' or proof['variant'] != variant:
            raise ValueError('Require independent full-window C8 proof')
        for name,digest in proof['evidence_sha256'].items():
            if sha(job/name) != digest: raise ValueError('Audited evidence changed')
        batch = read(job/'qualification/decode/results.json')['batches'][0]
        capture = job/'server/capture'
        steps = records(capture/'model_step_latency.jsonl')
        public = read(capture/'public_work.json')['submitted']
        fixture = read(job/'fixture.json')
        pure_prefill = [r for r in steps if r['prefill_tokens'] and not r['decode_requests']]
        gpu_seconds = sum(r['model_ms'] for r in pure_prefill)/1000
        actions = records(capture/'action_attention.jsonl')
        owner_steps = public['decode_request_batches']
        if owner_steps <= 0 or public['public_non_prefill_tokens'] != 5*owner_steps:
            raise ValueError('MTP4 verification geometry differs')
        result['measurements'][variant] = dict(
            steady_decode_tokens_per_second=proof['window']['output_tokens_per_second'],
            steady_window_seconds=proof['window']['seconds'],
            cold_preparation_seconds=proof['cold_preparation_seconds'],
            effective_input_tokens_per_second=proof['effective_input_tokens_per_second'],
            pure_prefill_model_tokens_per_second=sum(r['prefill_tokens'] for r in pure_prefill)/gpu_seconds,
            model_step_mean_ms=proof['model_step_mean_ms'], full_step_mean_ms=proof['full_step_mean_ms'],
            model_steps_per_second=proof['model_steps_per_second'],
            end_to_end_output_tokens_per_second=batch['end_to_end']['successful_output_tokens_per_second'],
            wall_seconds=batch['wall_seconds'],
            useful_output_per_native_owner_decode_step=(8*16384-8)/owner_steps,
            native_owner_decode_steps=owner_steps,
            mean_accepted_tokens_per_SSE_event=proof['mean_accepted_tokens_per_SSE_event'],
            engine_inclusive_host_totals=proof['engine_totals'],
            accepted_output_attention_samples=sum(r['samples'] for r in actions),
            accepted_prefill_attention_samples=sum(r['prefill_samples'] for r in actions),
            candidate_attention_rows=max((r.get('candidate_samples_total',0) for r in actions),default=0),
            final_per_owner_graph_bytes=[r.get('block_graph',{}).get('total_bytes',0) for r in actions],
            identities=proof['identities'], original_job_status=proof['original_job_status'],
            original_snapshot_status=proof['original_snapshot_status'],
            original_snapshot_violations=proof['original_snapshot_violations'],
            independent_executed_overlap_steps=proof['executed_overlap_steps'],
            native_snapshot_coverage=proof['native_snapshot_coverage'],
            audit_sha256=sha(audit_file),fixture_sha256=sha(job/'fixture.json'))
        if fixture['input_tokens'] != 262101 or fixture['output_tokens'] != 16384:
            raise ValueError('Original input/output changed')
    for before,after,label in (('disabled','attention','attention_statistics'),('attention','collect','graph_collection')):
        a,b = result['measurements'][before],result['measurements'][after]
        same = {k:a['identities'][k] == b['identities'][k] for k in a['identities']}
        if not same['input_sha256']: raise ValueError('Input changed')
        result['pairs'][label] = dict(before=before,after=after,identity=same,
            observed_changes=differences(a,b),
            exact_workload_identity_equal=all(same.values()) and a['native_owner_decode_steps']==b['native_owner_decode_steps'],
            limitation='Single ordered cold screen. Selection/output/verification-step drift prevents pure causal attribution. '
                'Inclusive host timers overlap GPU/other threads and cannot be summed into GPU idle or net overhead.')
    result['status']='completed_initial_cost_screen_with_snapshot_outcomes_preserved'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audits',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve first analysis; choose a new output')
    report = analyze(args.root, args.audits)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],pairs=report['pairs']),indent=2))


if __name__ == '__main__': main()
