"""Read-only, sanitized summaries of explicit Harbor GC comparison jobs."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from diagnostics.dsh_cloud.lifecycle_probe import artifact_gc_evidence


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def elapsed(value):
    if not value or not value.get('started_at') or not value.get('finished_at'):
        return None
    return (datetime.fromisoformat(value['finished_at'].replace('Z','+00:00'))-
            datetime.fromisoformat(value['started_at'].replace('Z','+00:00'))).total_seconds()


def summarize(job, *, task, mode):
    paths=list(job.glob('*/result.json'))
    if len(paths)!=1:
        raise ValueError('Each comparison job must contain exactly one official task trial')
    result=read(paths[0])
    if not result.get('finished_at'):
        raise ValueError('Cannot score a running trial')
    agent=paths[0].parent/'agent'
    bridge=agent/'bridge'
    evidence=artifact_gc_evidence(bridge) if bridge.exists() else {}
    owners={p.name for p in bridge.iterdir() if p.is_dir()} if bridge.exists() else set()
    final=read(agent/'gc-before-cleanup.json',{})
    released=set()
    spans=[]
    for worker in final.get('results',[]):
        for state in worker:
            plan=state.get('plan',{})
            namespace=plan.get('namespace',[])
            if len(namespace)!=2 or namespace[0] not in owners or namespace[0]!=namespace[1]:
                continue
            if plan.get('source')!='verified_dsh_write_artifact':
                continue
            handles={tuple(h) for h in state.get('candidate_handles',[])}
            events=[]
            if state.get('flushed') and state.get('observed') and state.get('complete') and state.get('error') is None:
                events=[e for e in state.get('events',[]) if e.get('kind')=='source_record_released'
                    and e.get('during_live_scope') is True and tuple(e.get('handle',[])) in handles
                    and plan['source_begin']<=e.get('history_begin',-1)
                    and e.get('history_begin',-1)+32<=plan['source_end']]
            for event in events:
                released.add((namespace[0],state.get('archive_request_id'),*event['handle']))
            spans.append(dict(relative_path=plan.get('relative_path'),artifact_sha256=plan.get('artifact_sha256'),
                source_begin=plan.get('source_begin'),source_end=plan.get('source_end'),
                candidate_blocks=len(handles),live_source_releases=len(events),
                remaining_source_records=state.get('remaining_source_records'),
                working_window_source_records=state.get('window_source_records'),error=state.get('error')))
    native=[read(p) for p in sorted(bridge.glob('*/*/native.json'))] if bridge.exists() else []
    replies=[]
    if (bridge/'requests.jsonl').exists():
        replies=[json.loads(line) for line in (bridge/'requests.jsonl').read_text().splitlines() if line.strip()]
        replies=[r for r in replies if r.get('transport')=='native_sse']
    tools=Counter()
    for p in bridge.glob('*/*/response.json') if bridge.exists() else ():
        for call in read(p)['choices'][0]['message'].get('tool_calls',[]):
            tools[call['function']['name']]+=1
    plans=[read(p) for p in sorted(bridge.glob('*/*/plan.json'))] if bridge.exists() else []
    maintenance=[p['artifact_gc_maintenance'] for p in plans if p.get('artifact_gc_maintenance')]
    seconds=sum(r['seconds'] for r in replies)
    turn=read(agent/'dsh/task/turns/000001.result.json')
    outcome=read(agent/'dsh/task/status.json',{})
    verifier_stdout=paths[0].parent/'verifier/test-stdout.txt'
    test_summary=None
    if verifier_stdout.exists():
        for line in verifier_stdout.read_text(errors='replace').splitlines():
            if line.startswith('=') and ' in ' in line and any(s in line for s in ('passed','failed','error')):
                test_summary=line.strip('= ')
    reason=(turn or {}).get('reason')
    reason_kind=reason.get('kind') if isinstance(reason,dict) else reason
    exception=result.get('exception_info')
    return dict(task=task,gc_mode=mode,job=job.name,
        reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward'),
        verifier_summary=test_summary,exception=(dict(type=exception.get('exception_type')) if isinstance(exception,dict) else None),
        dsh_status=outcome.get('state'),dsh_reason_kind=reason_kind,
        operator_stopped=(agent/'operator-stop.json').is_file(),
        dsh_turn_seconds=elapsed(turn),
        agent_execution_seconds=elapsed(result.get('agent_execution')),
        measured_task_seconds=((result.get('agent_result') or {}).get('metadata') or {}).get('task_seconds'),
        generated_model_replies=len(native),committed_model_replies=len(replies),
        input_token_usage=sum(n['usage']['prompt_tokens'] for n in native),
        output_tokens=sum(n['usage']['completion_tokens'] for n in native),
        model_api_reply_seconds=seconds,
        committed_output_tokens_per_api_second=(sum(r['output_tokens'] for r in replies)/seconds if seconds else None),
        model_issued_tool_calls=dict(tools),maintenance_handoffs=len(maintenance),
        maintenance_replay_tokens=sum(m['replay_tokens'] for m in maintenance),
        maximum_logical_prompt_tokens=max((n['usage']['prompt_tokens'] for n in native),default=0),
        candidate_source_spans=spans,live_source_cpu_completions=len(released),
        retired_source_tokens=len(released)*32,
        committed_snapshot_evidence=evidence,
        effective_gc_task_continuation=(mode=='delete' and bool(released) and evidence.get('later_committed_model_replies',0)>0),
        owner_cleanup=[dict(status=r['status']) for r in read(agent/'owner-cleanup.json',[])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--mode',choices=('off','delete'),required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=summarize(args.job,task=args.task,mode=args.mode)
    # Raw DSH result text and release-model output are not publication evidence.
    report.pop('dsh_turn',None)
    report['owner_cleanup']=[dict(status=r['status']) for r in report['owner_cleanup']]
    value=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(value)
    else:print(value,end='')


if __name__=='__main__':
    main()
