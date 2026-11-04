"""Minute monitor with same-thread, user-authorized repair wakeups.

The observer never retries tasks or kills processes. Its saved repair mandate
authorizes this conversation's agent to diagnose and repair infrastructure,
validate it, and continue only never-started work. Stable states stay quiet.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.monitor_deepswe_delivery import wake

PREFIX='【DeepSWE 自动诊断与恢复巡检】'

def read(path):return json.loads(path.read_text()) if path.exists() else {}
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def live(owner):
    try:
        p=psutil.Process(owner['pid'])
        return p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-owner['created'])<.01
    except (psutil.NoSuchProcess,KeyError):return False

def notices(root, state, queue, errors, dead):
    """Stable event identities deliberately exclude heartbeat timestamps."""
    arm=state.get('arm');out=[]
    if state.get('status')=='running' and state.get('stage')=='formal':
        out.append(('started:'+str(arm),dict(kind='arm_started',arm=arm,expected=queue.get('expected'))))
        for name in dead:
            out.append(('dead:'+str(arm)+':'+name,dict(kind='process_failed',arm=arm,process=name)))
        for task,value in queue.get('tasks',{}).items():
            if value.get('state')=='infrastructure_error':
                out.append(('task-error:'+str(arm)+':'+task,dict(kind='task_infrastructure_error',arm=arm,task=task,error=value.get('error'),source=str(root/str(arm)/'worker/cases'/task))))
        for error in errors:
            key='transport:'+str(arm)+':'+str(error.get('task'))+':'+str(error.get('event'))
            out.append((key,dict(kind='transport_failure',arm=arm,event=error)))
    if state.get('status') in ('needs_attention','completed','completed_with_transport_errors'):
        out.append(('terminal:'+str(arm)+':'+state['status'],dict(kind='terminal_state',state=state)))
    return out

def main(root, config, once=False):
    cfg=read(config);assert cfg['thread_id']==os.environ['CODEX_THREAD_ID']
    lock=(root/'repair-watch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    journal=read(root/'repair-watch-state.json');seen=set(journal.get('seen',[]))
    while True:
        try:
            # Explicit incident acknowledgements carry evidence of the prepared
            # fix; they suppress only those exact phase/task events.
            acknowledged=read(root/'repair-acknowledgements.json')
            seen.update(acknowledged.get('events',[]))
            state=read(root/'status.json');arm=state.get('arm');stage=root/str(arm)
            queue=read(stage/'worker/queue-status.json');dead=[];errors=[]
            if state.get('status')=='running' and state.get('stage')=='formal' and queue.get('status')!='completed':
                for name,path in [('controller',root/'sequence-process.json'),('model',stage/'model/server-process.json'),
                    ('bridge',stage/'bridge-process.json'),('tunnel',stage/'tunnel-process.json')]:
                    if path.exists() and not live(read(path)):dead.append(name)
                path=stage/'bridge/requests.jsonl'
                if path.exists():
                    # Tail only: a long run must not make each heartbeat scan all history.
                    with path.open('rb') as f:
                        f.seek(max(0,path.stat().st_size-512*1024));data=f.read()
                    for line in data.splitlines():
                        try:value=json.loads(line)
                        except ValueError:continue
                        if value.get('event') in ('error','context_limit') and value.get('task') not in ('preflight',None):errors.append(value)
            events=notices(root,state,queue,errors,dead)
            if state.get('status')=='waiting_for_previous':
                handoff=root/'handoff-process.json'
                if handoff.exists() and not live(read(handoff)):
                    events.append(('handoff-controller-exited',dict(kind='handoff_controller_failed',
                        required_action='Inspect preserved work; recover the waiting controller without duplicating task attempts')))
            if state.get('status')=='running' and state.get('stage')=='formal':
                capture=stage/'model/server/capture'
                progress=read(capture/'progress.json');resources=read(capture/'scheduler_resources.json')
                gc=progress.get('historical_gc',{})
                counters=[arm,progress.get('batches_submitted'),gc.get('retired'),gc.get('cpu_released')]
                if counters!=journal.get('progress_counters') or not resources.get('requests'):
                    journal.update(progress_counters=counters,progress_at=time.time())
                if resources.get('requests') and time.time()-journal.get('progress_at',time.time())>180:
                    # Investigate an engine barrier; never infer a task deadline
                    # from elapsed time or interrupt tools which are still running.
                    events.append(('model-no-progress:'+str(arm),dict(kind='model_has_requests_without_compute_or_gc_progress',
                        seconds=time.time()-journal['progress_at'],requests=resources['requests'],
                        required_action='Diagnose the waiting barrier; do not cancel tasks merely for elapsed time')))
                observer=read(stage/'observer-error.json')
                if observer and time.time()-state.get('at',0)>180:
                    events.append(('observer:'+str(arm),dict(kind='remote_observation_failed',error=observer)))
                if psutil.disk_usage(root).free<10*1024**3:
                    events.append(('disk-low',dict(kind='disk_capacity_low',free_bytes=psutil.disk_usage(root).free)))
                if psutil.virtual_memory().available<4*1024**3:
                    events.append(('memory-low',dict(kind='memory_capacity_low',available_bytes=psutil.virtual_memory().available)))
            for key,snapshot in events:
                if key in seen:continue
                mandate=(root/'repair-mandate.txt').read_text()
                message=PREFIX+'\n'+mandate+'\n快照：\n'+json.dumps(dict(at=time.time(),root=str(root),**snapshot),ensure_ascii=False)
                try:
                    receipt=asyncio.run(wake(cfg['thread_id'],cfg['socket'],message,
                        str(uuid.uuid5(uuid.NAMESPACE_URL,str(root)+':repair:'+key)),message_prefix=PREFIX))
                except RuntimeError as error:
                    if 'Earlier monitor event is queued' not in str(error):raise
                    journal['notification_pending']=key
                    break  # Keep this event pending while the current chat is busy.
                journal['last_receipt']=receipt;seen.add(key)
                journal.pop('notification_pending',None)
                journal['seen']=sorted(seen)
                save(root/'repair-watch-state.json',journal)
            journal.update(at=time.time(),seen=sorted(seen),status=state.get('status'),arm=arm,
                completed=queue.get('completed'),active=queue.get('active',[]),interval_seconds=60)
            journal.pop('observer_error',None);save(root/'repair-watch-state.json',journal)
            if state.get('status') in ('completed','completed_with_transport_errors') and all(k in seen for k,_ in events):return
        except Exception as error:
            journal.update(at=time.time(),observer_error=repr(error));save(root/'repair-watch-state.json',journal)
        if once:return
        time.sleep(60)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    main(args.root.resolve(),args.config.resolve(),args.once)
