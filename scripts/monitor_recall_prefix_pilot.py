"""Quiet ten-minute checks for the paired official-answer-prefix pilot."""
import argparse
from functools import partial
import json
import time
from pathlib import Path

import monitor_ruler_pooling_queue as common
from monitor_deepswe_delivery import wake

PREFIX='【DuoKV 官方回答前缀小批巡检】'
METHODS=('mean_k','block_value_norm','block_deletion_error')


def inspect(root, *, now=None, is_alive=common.alive,methods=METHODS,expected=26,campaign_name='campaign-v2'):
    now=time.time() if now is None else now
    queue=common.read(root/'queue-status.json',{})
    proc=common.read(root/'process.json',{})
    campaign=root/campaign_name
    state=common.read(campaign/'status.json',{})
    issues=[];arms={};work={}
    status=queue.get('status','missing')
    if status in ('failed','missing'):issues.append('小批队列异常：'+status+' '+queue.get('error',''))
    if status in ('running','waiting') and not is_alive(proc.get('pid')):issues.append('实验队列进程已退出')
    for method in methods:
        phase=campaign/'arms'/method/'qualification'
        arms[method]=dict(completed=len(list((phase/'cases').glob('*/result.json'))),expected=expected,
            done=common.read(phase/'completed.json',{}).get('status')=='completed')
    method=state.get('method')
    if status=='running' and method in methods:
        attempts=sorted((campaign/'arms'/method/'qualification/server-attempts').glob('*'))
        if attempts:
            attempt=attempts[-1];server=common.read(attempt/'process.json',{})
            if server and not is_alive(server.get('pid')):issues.append('当前模型服务进程已退出')
            fatal=common.FATAL.search(common.log_tail(attempt/'server.log'))
            if fatal:issues.append('当前服务日志出现异常：'+fatal.group())
            public=common.read(attempt/'server/capture/public_work.json',{})
            scheduler=common.read(attempt/'server/capture/scheduler_resources.json',{})
            history=scheduler.get('cpu_history_admission',{})
            work=dict(counter_id=public.get('counter_id'),completed=public.get('completed',{}),
                active_requests=scheduler.get('requests'),history_waiting=history.get('waiting'))
            if history.get('oldest_wait_seconds',0)>900:issues.append('历史缓存准入等待超过 15 分钟')
        if state.get('status')=='failed':issues.append('当前方法失败：'+state.get('error',''))
        if state.get('status')=='loading' and now-state.get('at',now)>1200:issues.append('模型加载超过 20 分钟')
    resume=common.read(root/'previous-resume.json',{})
    if resume.get('status')=='failed':issues.append('原 260 题实验自动续跑失败：'+resume.get('error',''))
    complete=status=='completed' and all(v['done'] and v['completed']==expected for v in arms.values())
    if status=='completed' and not complete:issues.append('队列显示完成，但三种方法的 26 题结果不齐')
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','MemTotal'):memory[key]=int(value.strip().split()[0])*1024
    if memory.get('MemAvailable',10*2**30)<4*2**30:issues.append('可用系统内存不足 4 GiB')
    gpu={};pointer=common.read(root/'gpu-pointer.json',{})
    if path:=pointer.get('path'):
        for line in reversed(common.log_tail(Path(path)).splitlines()):
            try:gpu=json.loads(line);break
            except ValueError:continue
    return dict(at=now,queue_status=status,stage=queue.get('stage'),active_arm=method,
        engine_state=state.get('status'),arms=arms,work=work,issues=issues,complete=complete,
        previous_resume=resume,gpu=gpu,memory=memory,expected_cases=expected)


def findings(snapshot,previous):
    moved=(snapshot['arms']!=previous.get('arms') or snapshot.get('work')!=previous.get('work') or
           snapshot.get('active_arm')!=previous.get('active_arm') or snapshot.get('engine_state')!=previous.get('engine_state'))
    snapshot['last_progress_at']=snapshot['at'] if moved else previous.get('last_progress_at',snapshot['at'])
    if (snapshot['queue_status']=='running' and snapshot.get('engine_state')=='running' and
            snapshot['at']-snapshot['last_progress_at']>=900):
        snapshot['issues'].append('连续 15 分钟没有确认的模型计算或新答案，需要排查')
    events=[v for v in snapshot['issues'] if v not in previous.get('issues',[])]
    if snapshot['complete'] and not previous.get('complete'):
        events.append(f"本轮 {len(snapshot['arms'])} 种配置各 {snapshot.get('expected_cases',26)} 题已完成；请核验总体/分题型得分、截断率并按本轮协议汇总。")
    return events


async def notify(thread,socket,message,key):
    return await wake(thread,socket,message,key,message_prefix=PREFIX)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','state-dir','instructions','socket'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--thread',required=True);parser.add_argument('--interval',type=int,default=600)
    parser.add_argument('--campaign-name',default='campaign-v2')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--expected',type=int,default=26)
    parser.add_argument('--message-prefix',default=PREFIX)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.interval<60:parser.error('Minimum check interval is 60 seconds')
    if args.expected<1:parser.error('Expected case count must be positive')
    PREFIX=args.message_prefix
    common.wake=notify
    common.run(args,inspector=partial(inspect,methods=args.methods,expected=args.expected,
        campaign_name=args.campaign_name),event_finder=findings)
