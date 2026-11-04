"""Quiet checks for the owned 13x2 layerwise recall campaign."""
import argparse
import json
from pathlib import Path
import time

import monitor_ruler_pooling_queue as common
from monitor_deepswe_delivery import wake

PREFIX='【DuoKV 逐层 Mean-K 26题巡检】'


def alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False


def inspect(root):
    now=time.time()
    active=common.read(root.parents[1]/'value-recall-active.json',{})
    if active and Path(active.get('root','')).resolve()!=root.resolve():
        return dict(at=now,queue_status='superseded',arms={},issues=[],complete=True,
                    superseded=True,work={})
    state=common.read(root/'status.json',{})
    process=common.read(root/'process.json',{})
    files=list((root/'formal/sequential/cases').glob('*/result.json'))
    issues=[];status=state.get('status','missing')
    if status in ('failed','missing'):
        issues.append('实验状态异常：'+status+' '+state.get('error',''))
    if status in ('loading','running') and not alive(process.get('pid')):
        issues.append('实验控制进程已退出；先保留已有结果并排查')
    work={}
    attempts=sorted((root/'formal/sequential/server-attempts').glob('*'))
    if attempts and status in ('loading','running'):
        attempt=attempts[-1]
        if common.FATAL.search(common.log_tail(attempt/'server.log')):
            issues.append('本轮模型服务日志出现新的严重错误')
        if len(files)<26 and not alive(process.get('server_pid')):
            issues.append('本轮模型服务进程已退出')
        public=common.read(attempt/'server/capture/public_work.json',{})
        work=dict(counter_id=public.get('counter_id'),completed=public.get('completed',{}))
        if status=='loading' and now-state.get('at',now)>1200:
            issues.append('模型加载超过20分钟')
    analysis=common.read(root/'analysis/comparison.json',{})
    audit=common.read(root/'completion-audit.json',{})
    complete=(status=='completed' and len(files)==26 and analysis.get('cases')==26
              and len(analysis.get('per_task',{}))==13 and audit.get('status')=='passed')
    if status=='completed' and not complete:
        issues.append('实验显示完成，但26题、13类或最终评分核验不齐全')
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','MemTotal'):memory[key]=int(value.strip().split()[0])*1024
    if memory.get('MemAvailable',8*2**30)<4*2**30:
        issues.append('系统可用内存不足4GiB')
    return dict(at=now,queue_status=status,active_arm='sequential',current_sample=state.get('sample_id'),
        arms=dict(sequential=dict(completed=len(files),expected=26,status=status)),work=work,
        issues=issues,complete=complete,memory=memory,
        macro_score=analysis.get('macro_score'),truncated=analysis.get('truncated'))


def findings(snapshot,previous):
    if snapshot.get('superseded'):return []
    moved=(snapshot['arms']!=previous.get('arms') or snapshot['work']!=previous.get('work'))
    snapshot['last_progress_at']=snapshot['at'] if moved else previous.get('last_progress_at',snapshot['at'])
    if snapshot['queue_status']=='running' and snapshot['at']-snapshot['last_progress_at']>=900:
        snapshot['issues'].append('连续15分钟没有已确认模型计算或新增结果，需排查')
    events=[issue for issue in snapshot['issues'] if issue not in previous.get('issues',[])]
    if snapshot['complete'] and not previous.get('complete'):
        events.append('逐层 Mean-K：13类各2题、共26题已完成，官方评分已生成；请核验并汇报总体、分题型、截断率。')
    return events


async def notify(thread,socket,message,key):
    return await wake(thread,socket,message,key,message_prefix=PREFIX)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','state-dir','instructions','socket'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--thread',required=True)
    parser.add_argument('--interval',type=int,default=600)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<60:parser.error('Minimum check interval is 60 seconds')
    common.wake=notify
    common.run(args,inspector=inspect,event_finder=findings)
