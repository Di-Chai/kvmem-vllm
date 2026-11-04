"""Observe a remote worker, retain original scores, and publish a combined view.

This process never starts, stops, retries, or alters a benchmark task. The only
remote operations are metadata reads and result-file synchronization. Full
agent transcripts remain on the worker; copying them every poll would waste the
same network needed by model requests.
"""
import argparse
import asyncio
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

import psutil

try:
    from .prepare_terminalbench_worker import read, sha
    from .run_terminalbench_job import classify
    from .compare_terminalbench_scored_pairs import quality_analysis
    from .compare_terminalbench_policies import exact_mcnemar
    from .monitor_deepswe_delivery import wake
except ImportError:
    from prepare_terminalbench_worker import read, sha
    from run_terminalbench_job import classify
    from compare_terminalbench_scored_pairs import quality_analysis
    from compare_terminalbench_policies import exact_mcnemar
    from monitor_deepswe_delivery import wake


PREFIX = '【Terminal Bench 远端 LRU 巡检】'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def optional(path):
    return read(path) if path.exists() else {}


def index(rows):
    result = {}
    for row in rows:
        name = row['task'].removeprefix('terminal-bench/')
        if name in result:
            raise ValueError('Duplicate task result: '+name)
        result[name] = row
    return result


def combine(preserved, remote, selected):
    left, right = index(preserved), index(remote)
    if set(left) & set(selected) or set(left) & set(right) or not set(right) <= set(selected):
        raise ValueError('Remote task result would replace an existing result or exceed its handoff')
    return [*left.values(), *right.values()]


def validate_remote_results(mirror, rows):
    proofs = {}
    for row in rows:
        trial = row['trial']
        if Path(trial).name != trial or trial in ('.', '..'):
            raise ValueError('Invalid trial identifier')
        path = mirror/'jobs/formal-remote'/trial/'result.json'
        event_path = mirror/'formal-remote-events'/(trial+'.json')
        raw = read(path)
        if classify(raw) != row or raw != read(event_path):
            raise ValueError('Remote raw result and lifecycle disagree: '+trial)
        proofs[trial] = dict(raw=str(path), raw_sha256=sha(path), event_sha256=sha(event_path))
    return proofs


def initialize_view(root, config, authorization):
    view, source = root/'view', Path(config['source'])
    view.mkdir(exist_ok=True)
    if not (view/'thinking').exists():
        (view/'thinking').symlink_to(source.parent/'thinking', target_is_directory=True)
    lru = view/'lru'
    lru.mkdir(exist_ok=True)
    manifest = read(source/'manifest.json')
    manifest['worker_restart'] = dict(preserved_completed=72, restarted_active=6, previously_unexecuted=11)
    manifest['container_memory_admission_bytes'] = read(root/'worker/manifest.json')['container_memory_admission_bytes']
    save(lru/'manifest.json', manifest)
    for name in ('profile.json', 'bridge', 'gpu-utilization.jsonl'):
        if not (lru/name).exists():
            (lru/name).symlink_to(source/name, target_is_directory=(source/name).is_dir())
    (lru/'model/server').mkdir(parents=True, exist_ok=True)
    if not (lru/'model/server/capture').exists():
        (lru/'model/server/capture').symlink_to(source/'model/server/capture', target_is_directory=True)
    (lru/'formal-events').mkdir(exist_ok=True)
    events = lru/'formal-events/events.jsonl'
    if not events.is_symlink():
        events.symlink_to(root/'mirror/formal-remote-events/events.jsonl')
    save(view/'experiment-plan.json', read(source.parent/'experiment-plan.json'))
    save(lru/'preserved-outcomes.json', read(root/'transition/before/formal-events/status.json')['outcomes'])


def sync(root, config):
    mirror = root/'mirror'
    mirror.mkdir(exist_ok=True)
    patterns = ['formal-remote-events/***', 'worker-state.json', 'worker-exit.json',
                'formal-launch.json', 'formal-process.json', 'worker-driver-error.json',
                'memory-admission-control.json', 'memory-admission-change-48g.json',
                'jobs/', 'jobs/formal-remote/', 'jobs/formal-remote/*/',
                'jobs/formal-remote/*/result.json', 'jobs/formal-remote/*/config.json',
                'jobs/formal-remote/result.json', 'jobs/formal-remote/config.json']
    args = ['rsync', '-a', '--timeout=30', '-e',
            'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10']
    args += ['--include=/'+pattern for pattern in patterns]
    args += ['--exclude=*', config['ssh_host']+':'+config['remote_root']+'/', str(mirror)+'/']
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45)
    code = '''from pathlib import Path
import json,psutil,shutil,time
r=Path(REMOTE)
f=r/'formal-process.json'
p=json.loads(f.read_text()) if f.exists() else {}
state='not_started'
if p:
 try:
  v=psutil.Process(p['pid'])
  state=('alive' if abs(v.create_time()-p['created'])<.02 and v.status()!=psutil.STATUS_ZOMBIE else 'exited')
 except psutil.NoSuchProcess:state='exited'
print(json.dumps(dict(at=time.time(),process=state,pid=p.get('pid'),memory_available_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(r).free,system_disk_free_bytes=shutil.disk_usage('/').free)))
'''.replace('REMOTE', repr(config['remote_root']))
    command = shlex.quote(config['remote_python'])+' -B -c '+shlex.quote(code)
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10', config['ssh_host'], command], text=True, stderr=subprocess.PIPE, timeout=25)
    return json.loads(raw)


def sample(root, config, preserved, authorization):
    remote_probe = sync(root, config)
    mirror = root/'mirror'
    state = optional(mirror/'worker-state.json')
    formal = optional(mirror/'formal-remote-events/status.json')
    remote_rows = formal.get('outcomes', [])
    selected = authorization['remaining_tasks']
    raw_proofs = validate_remote_results(mirror, remote_rows)
    rows = combine(preserved, remote_rows, selected)
    launch = optional(mirror/'formal-launch.json')
    at = time.time()
    issues = []
    for row in remote_rows:
        if row['status'] != 'scored':
            issues.append(dict(code='task:'+row['trial'], message=row['task']+' / '+str(row['exception'])))
    for key in ('release_errors', 'network_errors', 'verifier_cache_errors'):
        if formal.get(key):
            issues.append(dict(code=key, message=json.dumps(formal[key], ensure_ascii=False)))
    exit_record = optional(mirror/'worker-exit.json')
    clean_finish = (len(remote_rows)==len(selected) and formal.get('released')==len(selected)
                    and not formal.get('active') and exit_record.get('returncode')==0
                    and not any(formal.get(k) for k in ('release_errors','network_errors','verifier_cache_errors')))
    if state.get('status')=='failed' or optional(mirror/'worker-driver-error.json'):
        issues.append(dict(code='worker_failed', message='Remote runner exited with an error; preserve all attempts'))
    elif remote_probe['process']=='exited' and not exit_record:
        # Two files may be observed across a normal exit boundary; next poll
        # confirms it before the notification cadence.
        issues.append(dict(code='worker_exit_pending', message='Remote runner exited; exit receipt not synchronized yet'))
    for key, limit in [('memory_available_bytes',6*1024**3),('disk_free_bytes',30*1024**3),('system_disk_free_bytes',20*1024**3)]:
        if remote_probe[key]<limit:
            issues.append(dict(code='resource:'+key,message='Remote resource below reserve: '+key))
    for key, identity in config['serving_processes'].items():
        try:
            p = psutil.Process(identity['pid'])
            good = abs(p.create_time()-identity['created'])<.02 and p.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            good = False
        if not good:
            issues.append(dict(code='serving:'+key,message='Serving process unavailable: '+key))
    if clean_finish:
        for path,digest in authorization['preserved_files'].items():
            if sha(path)!=digest:
                raise ValueError('Original completed evidence changed before final comparison')
    view = root/'view'
    save(view/'lru/formal-events/status.json', dict(formal, expected=89, completed=len(rows),
        passed=sum(r['passed'] for r in rows), outcomes=rows, remote_completed=len(remote_rows), preserved_completed=len(preserved)))
    if launch:
        # No pre-launch GC counter snapshot was taken. Keep GC delta unknown;
        # this must not pretend to measure all seventeen tasks' retirement.
        save(view/'lru/formal-boundary.json', dict(at=launch['at'], gc_baseline_unavailable=True))
    phase = 'completed' if clean_finish else 'failed' if state.get('status')=='failed' else state.get('status','prepared')
    save(view/'lru/status.json', dict(status=phase,at=at))
    save(view/'pair-status.json', dict(status='completed' if clean_finish else 'running', arm='lru',
                                    at=at,started_at=launch.get('at')))
    worker = dict(host=config['ssh_host'],preserved=72,restarted=6,new=11,remote_total=17,
                  remote_completed=len(remote_rows),last_sync_at=at,resources=remote_probe,
                  container_admission_bytes=optional(mirror/'memory-admission-control.json').get(
                      'budget_bytes',read(root/'worker/manifest.json')['container_memory_admission_bytes']))
    save(view/'worker-status.json', worker)
    candidate = index(read(Path(config['source']).parent/'thinking/formal-events/status.json')['outcomes'])
    baseline = index(rows);common = set(candidate)&set(baseline)
    quality, excluded = quality_analysis({n:baseline[n] for n in common}, {n:candidate[n] for n in common}, exact_mcnemar)
    report = dict(at=at,complete=clean_finish,quality=quality,excluded=excluded,
        lru=dict(completed=len(rows),scored=sum(r['status']=='scored' for r in rows),passed=sum(r['passed'] for r in rows)),
        remote_completed=len(remote_rows),raw_remote_evidence=raw_proofs,
        authorization_sha256=sha(root/'transition/authorization.json'),
        caveats=['LRU combines 72 retained outcomes with 17 remote outcomes; six interrupted tasks were restarted by explicit user authorization.',
                 'The original stopped attempts and all completed scores remain unchanged. This is not an all-first-attempt campaign.',
                 'Tool host and stopping/restart conditions differ between policies; elapsed-time differences cannot be attributed solely to the eviction policy.'])
    save(root/'accuracy-comparison.json',report)
    snapshot = dict(at=at,root=str(root),scope='remote_lru',complete=clean_finish,
                    lru=report['lru'],worker=worker,issues=issues,report=str(root/'accuracy-comparison.json'))
    save(root/'snapshot.json',snapshot)
    return snapshot


def notice(root, config, snapshot, state):
    current = {item['code']:item['message'] for item in snapshot['issues']}
    previous = state.get('issues')
    changed = [dict(code=k,message=v) for k,v in current.items() if (previous or {}).get(k)!=v]
    if snapshot['complete'] and not state.get('complete'):
        changed.append(dict(code='completed',message='Remote 17-task cohort completed; combined LRU coverage and scored-pair report are ready. Six interrupted tasks were restarted by user authorization.'))
    if changed:
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, str(root)+json.dumps(changed,sort_keys=True)))
        if key not in state.setdefault('delivered',[]) and not any(p['key']==key for p in state.setdefault('outbox',[])):
            message=(PREFIX+'\n只读巡检：本轮为保留本地 72 题 + 远端 17 题，其中 6 题经用户授权重跑。'
                     '不要因耗时或无进展自动取消，不要恢复 Agent 总时限，不要再重试或替换成绩。当前忙碌时只排队。'
                     '\n发现：\n'+json.dumps(changed,ensure_ascii=False)+'\n快照：\n'+json.dumps(snapshot,ensure_ascii=False))
            state['outbox'].append(dict(key=key,message=message))
    state.update(issues=current,complete=snapshot['complete'],at=time.time())
    save(root/'monitor-state.json',state)
    if state.get('outbox'):
        entry=state['outbox'][0]
        try:
            receipt=asyncio.run(wake(config['thread_id'],Path(config['socket']),entry['message'],entry['key'],message_prefix=PREFIX))
            state['delivered'].append(entry['key']);state['outbox'].pop(0)
            state['last_delivery']=dict(at=time.time(),receipt=receipt)
            state.pop('delivery_error',None)
        except Exception as error:
            state['delivery_error']=repr(error)
        save(root/'monitor-state.json',state)


def run(root):
    config=read(root/'coordinator-config.json');authorization=read(root/'transition/authorization.json')
    for path,digest in authorization['preserved_files'].items():
        if sha(path)!=digest:raise ValueError('Original completed evidence changed')
    preserved=read(root/'transition/before/formal-events/status.json')['outcomes']
    initialize_view(root,config,authorization)
    state=optional(root/'monitor-state.json');next_notice=0
    while True:
        try:
            snapshot=sample(root,config,preserved,authorization)
        except Exception as error:
            snapshot=dict(at=time.time(),complete=False,issues=[dict(code='inspection_error',message=str(error)[:1000])],root=str(root))
            save(root/'inspection-error.json',snapshot)
        now=time.time()
        if now<config['monitor_expires_at'] and (now>=next_notice or snapshot['complete']):
            notice(root,config,snapshot,state);next_notice=now+1200
        if snapshot['complete'] and (not state.get('outbox') or now>=config['monitor_expires_at']):
            return
        if now>=config['monitor_expires_at']:
            if not (root/'monitor-expired.json').exists():
                save(root/'monitor-expired.json',dict(at=now,benchmark_cancelled=False))
            # The data mirror may continue; only scheduled chat notifications expire.
            next_notice=float('inf')
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    with (root/'.coordinator.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(root)
