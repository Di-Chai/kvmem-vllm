"""Read-only accounting of finite MAB historical-prefill cohorts."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from statistics import mean

from audit_layerwise_gilopt import read,save_new


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def measure(root):
    capture=root/'run/server/capture'
    gate=lines(capture/'prefill_phase_gate.jsonl')
    release,=[r for r in gate if r['event']=='history_prefill_released']
    arrivals={r['request_id']:r['at'] for r in gate if r['event']=='history_prefill_arrived'}
    expected=release['expected']
    assert len(arrivals)==expected and len(release['owners'])==expected
    work=read(capture/'public_work.json')
    offset=work['created_wall_time']-work['created_at']
    start=min(read(p)['at'] for p in (root/'run/answers/cases').glob('*/attempt.json'))-offset
    end=release['at']
    events=lines(capture/'async_refresh.jsonl')
    assert not any(r['stage']=='query' and r['at']<end for r in events)
    steps=[r for r in lines(capture/'model_step_latency.jsonl') if start<=r['started_at']<end]
    assert steps and not any(r['decode_requests'] for r in steps)
    geometry=[g for r in steps for g in r['batch_geometry']]
    assert all(g['prefill'] and g['request_id'] in arrivals for g in geometry)
    tokens=sum(r['prefill_tokens'] for r in steps)
    query_keys={(r['request_id'],r['command_id'])
                for r in lines(capture/'bounded_refresh.jsonl') if r['selection']=='query'}
    assert len(query_keys)==expected, 'Wait for all final Query records before the complete audit'
    pressure=defaultdict(dict)
    for r in events:
        if (r['request_id'] in arrivals and (r['request_id'],r['command_id']) not in query_keys
                and r['at']<arrivals[r['request_id']]):
            pressure[r['request_id'],r['command_id']][r['stage']]=r
    assert all('suspended' in rows and ('ready' in rows or 'pipeline_dispatch' in rows)
               for rows in pressure.values())
    resumable = {key:rows.get('pipeline_dispatch', rows.get('ready'))
                 for key,rows in pressure.items()}
    pipelined = any('pipeline_dispatch' in rows for rows in pressure.values())
    timers={}
    pairs = [('assembly_queued','assembly_started'),('assembly_started','assembly_submitted')]
    if not pipelined:
        pairs = [('suspended','ready'), *pairs, ('assembly_submitted','ready')]
    for a,b in pairs:
        values=[r[b]['at']-r[a]['at'] for r in pressure.values()]
        timers[a+'_to_'+b]=dict(mean_seconds=mean(values),sum_owner_seconds=sum(values))
    values = [resumable[key]['at']-rows['suspended']['at'] for key,rows in pressure.items()]
    assert all(value >= 0 for value in values)
    timers['suspended_to_resumable'] = dict(mean_seconds=mean(values),sum_owner_seconds=sum(values))
    if pipelined:
        # Guarded dispatch can precede the final upload submission. Treating
        # it as old all-ready would manufacture a negative completion wait.
        values = [rows['pipeline_completed']['at']-rows['pipeline_dispatch']['at']
                  for rows in pressure.values() if 'pipeline_dispatch' in rows]
        assert all(value >= 0 for value in values)
        timers['dispatch_to_materialization_reaped'] = dict(mean_seconds=mean(values),
                                                            sum_owner_seconds=sum(values))
    first_seen={}
    for r in steps:
        for g in r['batch_geometry']:first_seen.setdefault(g['request_id'],r['started_at'])
    common_start=max(first_seen.values());first_done=min(arrivals.values())
    missing=Counter();common_steps=0
    for r in steps:
        t=r['started_at']
        if not common_start<=t<first_done:continue
        common_steps+=1
        present={g['request_id'] for g in r['batch_geometry']}
        for rid in arrivals:
            if rid in present:
                missing['scheduled']+=1
            elif any(owner==rid and p['suspended']['at']<=t<resumable[key]['at']
                     for key,p in pressure.items() for owner in (key[0],)):
                missing['not_scheduled_pressure_in_flight']+=1
            else:
                missing['not_scheduled_other']+=1
    gpu=[r for r in lines(root/'run/gpu.jsonl') if start+offset<=r['at']<end+offset]
    assert gpu and not any(r.get('external_pids') for r in gpu)
    report=dict(root=str(root),history_tokens=release['history_tokens'],
        seconds=end-start,tokens_per_second=release['history_tokens']/(end-start),
        measured_tokens=tokens,measured_steps=len(steps),tokens_per_step=tokens/len(steps),
        model_event_seconds=sum(r['model_ms'] for r in steps)/1000,
        step_event_seconds=sum(r['step_ms'] for r in steps)/1000,
        max_batch_owners=max(r['prefill_requests'] for r in steps),
        batch_counts_by_owners=dict(sorted(Counter(r['prefill_requests'] for r in steps).items())),
        token_weighted_active_kv=sum(g['query_tokens']*g['active_seq_len'] for g in geometry)/tokens,
        gpu_sample_mean_percent=mean(r['devices'][0]['gpu_utilization_percent'] for r in gpu),
        prefill_pressure=len(pressure),pressure_timings=timers,
        pressure_dispatch_protocol='per_layer_guarded' if pipelined else 'all_layers_complete',
        tail_after_first_history_finished_seconds=end-first_done,
        common_phase_step_count=common_steps,common_phase_owner_slots=dict(missing),
        no_query_or_decode_in_phase=True,
        cohort_owners=expected,
        scope='Complete cold histories, first dispatch through last history arrival. Includes pressure/assembly/publication and HTTP submission, excludes loading/Query/decode. Identical measurement gate changes normal service overlap.',
        caveats=['Model CUDA event intervals include host submission gaps and capture/MTP, not just kernels.',
                 'Missing-owner classification is sampled at model-step entry, not additive wall-time attribution.',
                 'Other missing-owner slots are unresolved publication/admission/scheduler work, not proven GPU idle.',
                 'Owner pressure intervals overlap each other and model work; do not add them to phase seconds.'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=measure(args.root)
    save_new(args.output,report)
    print(json.dumps({k:report[k] for k in ['history_tokens','seconds','tokens_per_second','tokens_per_step','max_batch_owners','prefill_pressure','common_phase_owner_slots']}))
