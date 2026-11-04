"""Inspect the owned value-recall queue and wake its existing chat on findings."""
import argparse
from pathlib import Path
import time

import monitor_ruler_pooling_queue as common
from monitor_deepswe_delivery import wake

PREFIX = '【DuoKV 八路召回实验定时巡检】'
METHODS = ('block_value_norm', 'block_deletion_error')


def inspect(root, *, now=None, is_alive=common.alive, disk_usage=common.shutil.disk_usage):
    now = time.time() if now is None else now
    root = root.resolve()
    queue = common.read(root/'eight-way-queue-status.json', {})
    process = common.read(root/'eight-way-queue-process.json', {})
    campaign = root/'eight-way-campaign-v1'
    issues, arms = [], {}
    state = common.read(campaign/'status.json', {})
    queue_status = queue.get('status', 'missing')
    if queue_status in ('failed', 'missing'):
        issues.append('实验队列状态：'+queue_status+' '+queue.get('error',''))
    if queue_status == 'running' and not is_alive(process.get('pid')):
        issues.append('实验队列进程已退出')
    if queue_status == 'running' and state.get('status') == 'failed':
        issues.append('当前方法运行失败：'+state.get('error',''))
    for method in METHODS:
        arm = campaign/'arms'/method
        phases = {}
        for phase in ('qualification','formal'):
            completed = list((arm/phase/'cases').glob('*/result.json'))
            marker = common.read(arm/phase/'completed.json', {})
            phases[phase] = dict(completed=len(completed), status=marker.get('status','pending'))
        arms[method] = phases
    stage, method, phase = queue.get('stage'), state.get('method'), state.get('phase')
    work = {}
    if queue_status == 'running' and method in METHODS and phase in ('qualification','formal'):
        attempts = sorted((campaign/'arms'/method/phase/'server-attempts').glob('*'))
        if attempts:
            attempt = attempts[-1]
            child = common.read(attempt/'process.json', {})
            if child and not is_alive(child.get('pid')):
                issues.append('当前模型服务进程已退出')
            fatal = common.FATAL.search(common.log_tail(attempt/'server.log'))
            if fatal:
                issues.append('当前服务日志出现异常：'+fatal.group())
            capture = attempt/'server/capture'
            public = common.read(capture/'public_work.json', {})
            scheduler = common.read(capture/'scheduler_resources.json', {})
            progress = common.read(capture/'progress.json', {})
            work = dict(counter_id=public.get('counter_id'),
                completed=public.get('completed',{}),requests=scheduler.get('requests'),
                history=scheduler.get('cpu_history_admission',{}),progress=progress,
                model_pid=public.get('pid'),attempt=str(attempt),
                public_wall_time=public.get('wall_time'))
            if state.get('status') == 'loading' and now-state.get('at',now)>1200:
                issues.append('模型加载超过 20 分钟')
            history = work['history']
            if history.get('oldest_wait_seconds',0)>900:
                issues.append('历史缓存准入等待超过 15 分钟')
    if disk_usage(root).free < 10*2**30:
        issues.append('实验磁盘剩余空间不足 10 GiB')
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':',1)
        if key in ('MemAvailable','MemTotal'):
            memory[key] = int(value.strip().split()[0])*1024
    if memory.get('MemAvailable',10*2**30) < 4*2**30:
        issues.append('系统可用内存不足 4 GiB')
    complete = queue_status=='completed' and all(a['formal']['completed']==260 and
        a['formal']['status']=='completed' for a in arms.values())
    if queue_status=='completed' and not complete:
        issues.append('队列显示完成，但两种方法的 260 题结果尚未齐全')
    gpu = {}
    telemetry = root/'eight-way-gpu.jsonl'
    if telemetry.exists():
        for line in reversed(common.log_tail(telemetry).splitlines()):
            try:
                import json
                gpu=json.loads(line)
                break
            except (ValueError,TypeError):
                continue
    return dict(at=now,queue_status=queue_status,stage=stage,active_arm=method,phase=phase,
        engine_state=state.get('status'),arms=arms,issues=issues,complete=complete,
        work=work,gpu=gpu,memory=memory)


def findings(snapshot, previous):
    # An eight-request 256K prefill wave may finish no answers for several
    # minutes. Compare confirmed model work, not completed-answer count alone.
    current = snapshot.get('work',{}).get('completed',{})
    old = previous.get('work',{}).get('completed',{})
    identity = (snapshot.get('work',{}).get('counter_id'),snapshot.get('engine_state'),
                snapshot.get('active_arm'),snapshot.get('phase'))
    old_identity = (previous.get('work',{}).get('counter_id'),previous.get('engine_state'),
                    previous.get('active_arm'),previous.get('phase'))
    moved = identity != old_identity or current != old or snapshot['arms'] != previous.get('arms')
    snapshot['last_progress_at'] = snapshot['at'] if moved else previous.get('last_progress_at',snapshot['at'])
    if (snapshot['queue_status']=='running' and snapshot.get('engine_state') in ('running','functional-qualification')
            and snapshot['at']-snapshot['last_progress_at']>=900):
        snapshot['issues'].append('连续 15 分钟没有确认的模型计算进展或新结果，需检查停滞')
    events = [issue for issue in snapshot['issues'] if issue not in previous.get('issues',[])]
    for method, arm in snapshot['arms'].items():
        old_arm = previous.get('arms',{}).get(method,{})
        if arm['formal']['status']=='completed' and old_arm.get('formal',{}).get('status')!='completed':
            events.append(method+'：完整 260 题已完成，请核验评分并确认后续方法继续运行')
    if snapshot['complete'] and not previous.get('complete'):
        events.append('两种方法共 520 次正式运行和汇总已完成，核验结果后停止巡检')
    return events


async def notify(thread, socket, message, submission_key):
    return await wake(thread,socket,message,submission_key,message_prefix=PREFIX)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','state-dir','instructions','socket'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--thread',required=True)
    parser.add_argument('--interval',type=int,default=600)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<60:parser.error('The interval must be at least 60 seconds')
    common.wake=notify
    common.run(args,inspector=inspect,event_finder=findings)
