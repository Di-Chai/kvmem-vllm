"""Quiet ten-minute inspection of the owned layerwise benchmark suite."""
import argparse
from pathlib import Path
import time

import monitor_ruler_pooling_queue as common
from monitor_deepswe_delivery import wake
import gpu_efficiency_monitor as telemetry

PREFIX='【DuoKV 逐层召回效率与完整评测巡检】'


def inspect(root):
    now=time.time();state=common.read(root/'status.json',{})
    pointer=common.read(root.parent/'value-recall-active.json',{})
    if pointer and Path(pointer.get('queue_root','')).resolve()!=root.resolve():
        return dict(at=now,queue_status='superseded',superseded=True,arms={},work={},issues=[],complete=True)
    plan=common.read(root/'queue-plan.json',{});arms={};issues=[];work={}
    for stage in plan.get('stages',[]):
        campaign=Path(stage['root']);status=common.read(campaign/'status.json',{})
        expected=stage.get('expected',0)
        cases=(campaign/'formal/sequential/cases' if stage.get('kind')=='ruler' else campaign/'run/answers/cases')
        count=sum(1 for _ in cases.glob('*/result.json'))
        arms[stage['id']]=dict(completed=count,expected=expected,status=status.get('status'))
    active=Path(state.get('active_root',str(root)))
    current=common.read(active/'status.json',{})
    process=common.read(active/'process.json',{})
    attempts=sorted((active/'formal/sequential/server-attempts').glob('*'))
    server_dir=attempts[-1]/'server' if attempts else active/'run/server'
    public=common.read(server_dir/'capture/public_work.json',{})
    work=dict(counter_id=public.get('counter_id'),completed=public.get('completed',{}))
    owned=telemetry.process_tree_pids([pid for pid in (process.get('pid'),process.get('server_pid')) if isinstance(pid,int)])
    gpu=telemetry.snapshot(owned)
    if state.get('status')=='running' and gpu.get('external_pids'):
        issues.append('GPU出现不属于本实验的进程；排查前不得终止其他任务')
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','MemTotal'):memory[key]=int(value.strip().split()[0])*1024
    if memory.get('MemAvailable',8*2**30)<4*2**30:issues.append('系统可用内存不足4GiB')
    if common.shutil.disk_usage(root).free<10*2**30:issues.append('实验磁盘剩余空间不足10GiB')
    compiler=[]
    for pid in owned:
        try:name=(Path('/proc')/str(pid)/'comm').read_text().strip()
        except OSError:continue
        if name in ('ninja','nvcc','cicc','ptxas','c++','cc1plus'):compiler.append(dict(pid=pid,name=name))
    if compiler:work['kernel_compilers']=sorted(compiler,key=lambda r:r['pid'])
    for grade in root.glob('*-grader.json'):
        record=common.read(grade,{})
        verdict=common.read(Path(record['output'])/'status.json',{})
        if verdict.get('status')=='failed':issues.append('原评分流程失败：'+grade.stem)
    if state.get('status')=='failed':issues.append('队列异常：'+state.get('error',''))
    if state.get('status')=='running' and current.get('status') in ('loading','running'):
        if not common.alive(process.get('pid')):issues.append('当前实验控制进程已退出')
        log=server_dir.parent/'server.log'
        if common.FATAL.search(common.log_tail(log)):issues.append('当前实验服务日志出现严重错误')
    return dict(at=now,queue_status=state.get('status','missing'),active_arm=state.get('stage'),
        active_root=str(active),arms=arms,work=work,gpu=gpu,memory=memory,issues=issues,complete=state.get('status')=='completed')


def findings(snapshot,previous):
    if snapshot.get('superseded'):return []
    progress=snapshot['arms']!=previous.get('arms') or snapshot['work']!=previous.get('work')
    snapshot['last_progress_at']=snapshot['at'] if progress else previous.get('last_progress_at',snapshot['at'])
    if snapshot['queue_status']=='running' and snapshot['at']-snapshot['last_progress_at']>1200:
        snapshot['issues'].append('连续20分钟没有确认的模型计算或新增答案，需检查当前归属和进度')
    result=[i for i in snapshot['issues'] if i not in previous.get('issues',[])]
    for name,record in snapshot['arms'].items():
        old=previous.get('arms',{}).get(name,{})
        if record['status']=='completed' and old.get('status')!='completed':result.append(name+' 已完成，请核验结果并确认后续阶段继续')
    if snapshot['complete'] and not previous.get('complete'):result.append('完整套件及原评分流程均完成，请核验汇总')
    return result


async def notify(thread,socket,message,key):
    return await wake(thread,socket,message,key,message_prefix=PREFIX)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','state-dir','instructions','socket'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--thread',required=True);parser.add_argument('--interval',type=int,default=600)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.interval<60:parser.error('Minimum interval is 60 seconds')
    common.wake=notify;common.run(args,inspector=inspect,event_finder=findings)
