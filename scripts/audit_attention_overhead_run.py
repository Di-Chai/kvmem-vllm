"""V2 D overhead execution audit; preserve original snapshot gate outcomes.

Scheduled-query histograms publish on a cadence and count scheduling attempts,
not GPU prefill tokens. A late histogram increment is not accepted as model
work when complete executed steps and public-forward counters prove otherwise.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_graph_cost import read, records, sha, engine_totals, fingerprint
from graph_v2_protocol import validate_profile as validate_v2_profile


def validate_profile(profile, variant):
    if variant != 'D':
        raise ValueError('Overhead comparison requires unchanged V2 D graph/pressure policy')
    validate_v2_profile(profile, 'D', 'niah')
    kv=profile['kvmem']
    if kv.get('cpu_read_cache_bytes') != 100000000000 or kv.get('action_native_scores','off') != 'off':
        raise ValueError('Keep original 100 GB CPU history and native QK reuse disabled')
    for name in ('action_native_decode_commit','action_download_pool','action_fused_scoring','graph_native_propagation'):
        if type(kv.get(name,False)) is not bool:
            raise ValueError('Overhead switch must be boolean: '+name)



def executed_window(rows, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or end-start < 30:
        raise ValueError('Keep original at-least30s shared window')
    overlap = [r for r in rows if r['started_at'] < end and r['submitted_at'] > start]
    if (not overlap or any(r['decode_requests'] != 8 or r['prefill_requests'] or r['prefill_tokens'] for r in overlap)
            or any(b['sequence'] != a['sequence']+1 for a,b in zip(overlap, overlap[1:]))):
        raise ValueError('Actual native model execution does not prove continuous pure C8')
    if overlap[0]['started_at'] > start or overlap[-1]['submitted_at'] < end:
        raise ValueError('Native execution does not bracket the complete SSE window')
    return overlap


def audit(job):
    manifest = read(job/'manifest.json')
    for name,digest in manifest['runtime_sha256'].items():
        if sha(job/'runtime'/name) != digest: raise ValueError('Frozen source changed')
    if sha(job/'profile.json') != manifest['profile_sha256'] or sha(job/'fixture.json') != manifest['fixture_sha256']:
        raise ValueError('Frozen profile/input changed')
    variant = manifest['arm']; profile = read(job/'profile.json'); fixture = read(job/'fixture.json')
    validate_profile(profile, variant)
    if sha(Path(fixture['input_i32'])) != fixture['input_i32_sha256']: raise ValueError('Original input changed')
    data = read(job/'qualification/decode/results.json'); batch = data['batches'][0]
    request_rows = batch['requests']; window = batch['steady_decode_window']
    if (window['status'] != 'passed' or window['active_streams'] != 8 or len(request_rows) != 8
            or {r['client_index'] for r in request_rows} != set(range(8))
            or any(r['outcome'] != 'success' or r['prompt_tokens'] != fixture['input_tokens']
                or r['completion_tokens'] != 16384 or r['observed_tokens'] != 16384 for r in request_rows)):
        raise ValueError('Keep exact full input, accepted output and SSE gates')
    samples = batch['native_samples']
    origins = [s['absolute_monotonic_at']-s['at'] for s in samples if 'absolute_monotonic_at' in s]
    if not origins or max(origins)-min(origins) > 1e-6: raise ValueError('Native/SSE clocks differ')
    start,end = origins[0]+window['start'],origins[0]+window['end']
    capture = job/'server/capture'; steps = records(capture/'model_step_latency.jsonl')
    overlap = executed_window(steps, start, end)
    snapshots = batch['native_confirmation']
    if snapshots['sse_window_coverage_fraction'] < .9: raise ValueError('Retain90% independent snapshot coverage')
    violations = snapshots.get('violations', [])
    if any(v['reasons'] != ['prefill_increment_during_window'] for v in violations):
        raise ValueError('Unexpected native snapshot ownership, read or refresh violation')
    # Public model forward counters count real tokens, unlike the scheduled
    # histogram. Require no extra prefill even in snapshots published midwindow.
    public = [r for r in records(capture/'public_work.jsonl') if start <= r['at'] <= end]
    if not public or any(r['submitted']['public_prefill_tokens'] != 8*fixture['input_tokens'] for r in public):
        raise ValueError('Real public-forward counters do not prove completed full-input prefill')
    refresh = records(capture/'bounded_refresh.jsonl')
    if any(r['refresh_finished_at'] >= start and r['refresh_started_at'] <= end for r in refresh):
        raise ValueError('Real Query/pressure work overlaps shared decode')
    archive = read(capture/'archive.json'); complete = archive['completed']
    if (len(complete) != 8 or len({r['request_id'] for r in complete}) != 8
            or any(r['terminal_prefix']['history_end'] != fixture['input_tokens']+16384-1
                or r['terminal_prefix']['output_tokens'] != 16384 for r in complete)):
        raise ValueError('Complete native accepted history differs')
    cleanup = read(job/'cleanup.json')
    if cleanup['remaining_owned_process_groups'] or cleanup['remaining_gpu_pids']:
        raise ValueError('Owned processes remain')
    if any(archive[k] for k in ('active_requests','cpu_authority_bytes','session_checkpoints','block_graph_bytes',
                               'disk_read_bytes','disk_write_bytes')):
        raise ValueError('CPU history or owned state not released / unexpected SSD use')
    actions = records(capture/'action_attention.jsonl'); graph = records(capture/'block_graph.jsonl')
    if variant == 'disabled':
        if actions or graph: raise ValueError('Unexpected statistics/graph work')
    else:
        if len(actions) != 8 or any(r['samples'] <= 0 or r['prefill_samples'] <= 0 for r in actions):
            raise ValueError('Actual prefill/output statistics not exercised')
        if variant == 'attention' and (graph or any('block_graph' in r for r in actions)):
            raise ValueError('Unexpected graph construction')
        if variant == 'collect' and (len(graph) != 8 or any(not r.get('selection_unchanged') for r in graph)
                or any(r.get('block_graph', {}).get('observed_sources', 0) <= 0 for r in actions)):
            raise ValueError('Collect graph was absent or changed selection')
    query, pressure = {}, {}
    for owner in complete:
        rid = owner['request_id']; own = [r for r in refresh if r['request_id'] == rid]
        q = [r for r in own if r['selection'] == 'query']
        if len(q) != 1: raise ValueError('Missing exact Mean-K Query selection')
        query[rid] = fingerprint(q[0]['selected_history_ranges'])
        pressure[rid] = fingerprint(sorted((r['history_begin'],r['selected_history_ranges']) for r in own if r['selection']=='pressure'))
    measured = [r for r in steps if overlap[0]['started_at'] <= r['started_at'] <= overlap[-1]['started_at']]
    cold = max(r['first_token_at'] for r in request_rows)-min(r['started_at'] for r in request_rows)
    if variant == 'D':
        if not all(r.get('block_graph',{}).get('observed_sources',0)>0 for r in actions):
            raise ValueError('Accepted graph construction was not exercised for every owner')
        if not any(r.get('phase')=='query' for r in graph) or not any(r.get('phase')=='pressure' for r in graph):
            raise ValueError('Both Query and pressure graph selection must execute')
        if any(r.get('scoring_version')!='v2' or r.get('alpha')!=.2 or r.get('mode')!='multi' for r in graph):
            raise ValueError('Actual V2 graph parameters differ')
        launch=read(job/'server/launch.json')
        if '--no-enable-prefix-caching' not in launch['vllm_argv']:
            raise ValueError('Prefix caching must remain disabled')
    return dict(status='independently_verified_full_window_C8', variant=variant,
        original_job_status=read(job/'status.json')['status'], original_snapshot_status=snapshots['status'],
        original_snapshot_violations=violations, snapshot_gate_modified=False,
        executed_overlap_steps=len(overlap), window=window, native_snapshot_coverage=snapshots['sse_window_coverage_fraction'],
        last_actual_prefill_finished_at=max(r['submitted_at'] for r in steps if r['prefill_tokens']),
        first_window_at=start, last_window_at=end,
        cold_preparation_seconds=cold, effective_input_tokens_per_second=8*fixture['input_tokens']/cold,
        end_to_end=batch['end_to_end'], engine_totals=engine_totals(actions),
        model_step_mean_ms=sum(r['model_ms'] for r in measured)/len(measured),
        full_step_mean_ms=sum(r['step_ms'] for r in measured)/len(measured),
        model_steps_per_second=len(measured)/window['seconds'],
        identities=dict(input_sha256=fixture['input_i32_sha256'],
            query_selection_multiset=sorted(query.values()),pressure_selection_multiset=sorted(pressure.values()),
            accepted_output_multiset=sorted(r['token_ids_sha256'] for r in request_rows)),
        mean_accepted_tokens_per_SSE_event=sum(r['observed_tokens'] for r in request_rows)/sum(len(r['timeline']) for r in request_rows),
        evidence_sha256={str(p.relative_to(job)):sha(p) for p in (job/'manifest.json',
            job/'qualification/decode/results.json',capture/'model_step_latency.jsonl',
            capture/'public_work.jsonl',capture/'archive.json',job/'cleanup.json')},
        auditor_sha256=sha(Path(__file__)),
        scope='Independent executed-model audit; preserves failed legacy delayed scheduling-histogram snapshot gate. '
            'All8 HTTP/accepted histories, original full SSE window/30s/90% coverage retained. '
            'Every actual overlapping model submission pure C8 with contiguous sequences; real public prefill counters constant. '
            'All inputs identical, so cross-run output/selection identities compare multisets, not arbitrary local client pairings.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve original audit')
    result=audit(args.job);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','variant','original_job_status','original_snapshot_status','executed_overlap_steps')}))


if __name__=='__main__': main()
