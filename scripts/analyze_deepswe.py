"""Persist native terminal records and report partial/final DeepSWE evidence."""
import argparse
import json
from pathlib import Path
import time


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass  # A writer may not have completed its last line.
    return result


def comparison(root, snapshot):
    source=root/'baseline-comparison.json'
    if not source.exists():return
    baselines=json.loads(source.read_text())
    current={r['task']:r for r in snapshot['tasks']}
    result={'partial':snapshot['partial'],'current_completed':snapshot['completed_tasks'],
            'current_passed':snapshot['passed_tasks'],
            'limits':baselines['limits']+' Current client wall time includes CPU admission queues; queue-adjusted time still includes tools and concurrent inference contention.',
            'comparisons':{}}
    for name in ('complete_historical','recent_attention_partial'):
        reference=baselines[name]
        paired=[]
        for old in reference:
            new=current.get(old['task'])
            if not new or new['state']!='completed':continue
            paired.append({'task':old['task'],'reference_reward':old['reward'],'current_reward':new['reward'],
                'reference_agent_seconds':old['agent_elapsed_sec'],'current_agent_seconds':new['agent_seconds'],
                'current_cpu_queue_seconds':new['cpu_queue_seconds'],
                'current_agent_seconds_excluding_cpu_queue':max(0,new['agent_seconds']-new['cpu_queue_seconds']) if new['agent_seconds'] is not None else None,
                'reference_requests':old['request_count'],'current_requests':new['requests'],
                'reference_output_tokens':old['visible_decoded_tokens'],'current_output_tokens':new['output_tokens'],
                'reference_final_context_tokens':old['final_context_tokens'],'current_final_prompt_tokens':new['last_prompt_tokens'],
                'current_termination':new['termination']})
        result['comparisons'][name]={'reference_tasks':len(reference),
            'reference_passed':sum(r['reward']==1 for r in reference),'paired_tasks':len(paired),
            'paired_reference_passed':sum(r['reference_reward']==1 for r in paired),
            'paired_current_passed':sum(r['current_reward']==1 for r in paired),'tasks':paired}
    temporary=root/'comparison.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(root/'comparison.json')


def analyze(root):
    run=json.loads((root/'run.json').read_text())
    archive=root/'server/capture/archive.json'
    native=json.loads(archive.read_text()) if archive.exists() else {}
    records=rows(root/'native-terminal.jsonl');seen={r['request_id'] for r in records}
    with (root/'native-terminal.jsonl').open('a') as stream:
        for record in native.get('completed',[]):
            if record['request_id'] not in seen:
                stream.write(json.dumps(record)+'\n');records.append(record);seen.add(record['request_id'])
    events=rows(root/'bridge/requests.jsonl')
    starts=[r for r in events if r['event']=='start']
    finishes=[r for r in events if r['event']=='finish']
    pending_waits={}
    for event in events:
        if event['event']=='cpu_admission_wait':
            pending_waits[event['task']]=event['at']
        elif event['event'] in ('start','history_released'):
            pending_waits.pop(event['task'],None)
    now=run.get('ended_at',time.time())
    pending_queue={task:max(0,now-begin) for task,begin in pending_waits.items()}
    model_metrics=[r for r in finishes if r.get('metrics')]
    ttft_values=[r['metrics']['time_to_first_token_ms']/1000 for r in model_metrics
                 if r['metrics'].get('time_to_first_token_ms') is not None]
    decode_seconds=sum(r['metrics'].get('generation_time_ms',0)/1000 for r in model_metrics)
    decode_tokens=sum(max(0,r['usage']['completion_tokens']-1) for r in model_metrics)
    tasks=[]
    for task,value in run['tasks'].items():
        selected=[r for r in finishes if r['task']==task]
        result=value.get('result',{})
        agent_path=Path(value.get('artifact',root/'cases'/task))/'agent_summary.json'
        agent=json.loads(agent_path.read_text()) if agent_path.exists() else {}
        carried=value.get('metrics',{}) if value.get('carried') else {}
        tasks.append({'task':task,'state':value['state'],'reward':(result.get('reward') or {}).get('reward'),
            'agent_seconds':agent.get('elapsed_sec'),'total_seconds':result.get('elapsed_sec'),'requests':carried.get('requests',len(selected)),
            'cpu_queue_seconds':carried.get('cpu_queue_seconds',sum(r.get('cpu_queue_seconds',0) for r in selected)),
            'pending_cpu_queue_seconds':pending_queue.get(task,0),
            'output_tokens':carried.get('output_tokens',sum(r['usage']['completion_tokens'] for r in selected)),
            'last_prompt_tokens':carried.get('last_prompt_tokens',selected[-1]['usage']['prompt_tokens'] if selected else None),
            'carried':bool(value.get('carried')),
            'termination':value.get('termination')})
    elapsed=(run.get('ended_at',time.time())-run.get('model_ready_at',run['started_at']))
    resources=root/'server/capture/scheduler_resources.json'
    scheduler=json.loads(resources.read_text()) if resources.exists() else {}
    completed=[t for t in tasks if t['state']=='completed']
    output=sum(r['usage']['completion_tokens'] for r in finishes)
    snapshot={'at':time.time(),'run_status':run['status'],'planned_tasks':16,
        'completed_tasks':len(completed),'passed_tasks':sum(t['reward']==1 for t in completed),
        'carried_completed_tasks':sum(t['carried'] for t in tasks),
        'infrastructure_errors':sum(t['state']=='infrastructure_error' for t in tasks),
        'completed_model_requests':len(finishes),'output_tokens':output,'elapsed_seconds':elapsed,
        'mixed_end_to_end_output_tokens_per_second':output/elapsed if elapsed>0 else None,
        'peak_simultaneous_model_requests':max([r.get('inflight',0) for r in starts],default=0),
        'exact_output_prefix_reused_requests':sum(r.get('exact_output_prefix_reused',False) for r in starts),
        'long_trajectory_admissions':[r['task'] for r in events if r['event']=='long_trajectory_admitted'],
        'cpu_waits':sum(r['event']=='cpu_admission_wait' for r in events),
        'engine_cpu_admission':scheduler.get('cpu_history_admission'),
        'engine_cpu_admission_scope':'Engine-lifetime counters including preflight; legacy bridge CPU wait counters do not measure engine admission waits',
        'native_session_restored_tokens':native.get('session_restored_tokens',0)-run.get('preflight_native_counters',{}).get('session_restored_tokens',0),
        'native_checkpoint_deferrals':native.get('session_checkpoint_deferrals',0)-run.get('preflight_native_counters',{}).get('session_checkpoint_deferrals',0),
        'cpu_queue_seconds':sum(r.get('cpu_queue_seconds',0) for r in finishes),
        'pending_cpu_waits':len(pending_queue),
        'pending_cpu_queue_seconds':sum(pending_queue.values()),
        'request_weighted_decode_tokens_per_second':decode_tokens/decode_seconds if decode_seconds else None,
        'mean_native_ttft_seconds':sum(ttft_values)/len(ttft_values) if ttft_values else None,
        'cpu_authority_bytes':native.get('cpu_authority_bytes'),'archive_backend':native.get('archive_backend'),
        'disk_read_bytes':native.get('disk_read_bytes'),'disk_write_bytes':native.get('disk_write_bytes'),
        'tasks':tasks,'partial':run['status']!='completed',
        'throughput_scope':'Throughput covers this execution only; carried completed results contribute scores, not this execution token counts or wall time. Mixed end-to-end output includes tools, prefill and CPU waits. Request-weighted decode is per-stream, not aggregate GPU throughput. Pending queue time is separate.'}
    temporary=root/'live-summary.tmp';temporary.write_text(json.dumps(snapshot,indent=2)+'\n');temporary.replace(root/'live-summary.json')
    comparison(root,snapshot)
    return snapshot


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        value=analyze(a.root)
        if not a.watch or value['run_status'] in ('completed','needs_attention'):
            print(json.dumps({k:v for k,v in value.items() if k!='tasks'}));break
        time.sleep(10)
