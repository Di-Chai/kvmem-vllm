"""Frozen correctness -> paired policy comparison -> original DeepSWE first16.

CPU preparation does not acquire the GPU. Execution waits for the existing
benchmark lock and for an idle device. Failures preserve evidence and stop;
there are no automatic benchmark retries or foreign-process termination.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.suffix != '.pyc'}


def environment(root):
    return {**os.environ, 'PYTHONPATH':str(root/'frozen/src'), 'TOKENIZERS_PARALLELISM':'false'}


def commands(root):
    base = [sys.executable, str(root/'frozen/scripts/qualify_agent_scheduling.py'),
            '--config', str(root/'input-profile.json'), '--arms', 'A0', 'P1']
    return {
        'correctness':base[:-3] + ['--arms','A0','Pttl','--output', str(root/'correctness'), '--cases', 'agents',
                             '--pairs', '1', '--turns', '3', '--output-tokens', '128', '--verify-restores',
                             '--lifecycle-checks'],
        'isolation':base[:-3] + ['--arms', 'P1', '--output', str(root/'isolation'), '--cases', 'agents',
                             '--pairs', '1', '--turns', '3', '--output-tokens', '128', '--context-tokens', '32768',
                             '--verify-restores', '--distinct-users', '--lifecycle-checks'],
        'pressure':base[:-3] + ['--arms', 'Ppressure', '--output', str(root/'pressure'), '--cases', 'agents',
                             '--pairs', '1', '--turns', '3', '--output-tokens', '128', '--context-tokens', '32768',
                             '--verify-restores', '--retirement-barrier'],
        'performance':base + ['--output', str(root/'performance'), '--cases', 'decode', 'agents', 'mixed',
                             '--pairs', '3', '--turns', '5', '--output-tokens', '1024'],
    }


def prepare(root, baseline):
    root.mkdir(parents=True, exist_ok=False)
    profile = read(baseline/'profile.json')
    if (profile['max_num_seqs'] != 4 or profile['kvmem']['select_budget'] != 131072 or
            profile['kvmem']['gen_budget'] != 131072 or profile['kvmem']['archive_backend'] != 'cpu'):
        raise ValueError('Require the original four-agent K128K/G128K CPU-history baseline')
    save(root/'input-profile.json', profile)
    frozen = root/'frozen'
    for name in ('src', 'scripts', 'benchmarks/deepswe'):
        shutil.copytree(ROOT/name, frozen/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(baseline/'protocol', root/'deepswe/protocol')
    source = hashes(frozen)
    save(root/'source-hashes.json', source)
    save(root/'protocol-hashes.json', hashes(root/'deepswe/protocol'))
    jobs = commands(root)
    for stage, command in jobs.items():
        with (root/f'prepare-{stage}.log').open('w') as log:
            subprocess.run(command + ['--prepare-only'], cwd=frozen, env=environment(root),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(root/'correctness/manifest.json')['fixture_sha256'] == read(root/'performance/manifest.json')['fixture_sha256']
    save(root/'sequence.json', dict(baseline=str(baseline), snapshot=str(frozen), commands=jobs,
        benchmark_reference=read(baseline/'manifest.json')['reference'],
        policy='A0 vs P1 only; exact restores, paired efficiency, then fresh first16 for both arms with identical bridge',
        formal_arms=['A0','P1'], timing_baseline='deepswe-A0',
        formal_workers=4, formal_bash_repeat_limit=32, created_at=time.time()))
    save(root/'status.json', dict(status='prepared', gpu_started=False, at=time.time()))


def efficiency_gate(comparison):
    agents, mixed, decode = [comparison['P1/'+case] for case in ('agents','mixed','decode')]
    common = decode.get('common_decode_ratios', [])
    checks = {
        'three_complete_pairs':all(r['pairs'] >= 3 for r in (agents,mixed,decode)),
        'agents_gain_at_least_ten_percent':agents['throughput_ratio_median'] >= 1.10,
        'agents_improve_each_pair':all(r > 1. for r in agents['throughput_ratios']),
        'common_decode_no_repeatable_regression':len(common) >= 3 and decode['common_decode_ratio_median'] >= .98,
        'mixed_batch_no_repeatable_regression':mixed['throughput_ratio_median'] >= .98,
    }
    return dict(passed=all(checks.values()), checks=checks, comparison=comparison)


def formal_comparison(root):
    """Require two complete identical-protocol first16 runs before timing ratios."""
    paths=[root/('deepswe-'+arm) for arm in ('A0','P1')]
    summaries=[read(path/'live-summary.json') for path in paths]
    manifests=[read(path/'manifest.json') for path in paths]
    evidence=[read(path/'formal-evidence.json') for path in paths]
    for summary in summaries:
        if (summary['run_status']!='completed' or summary['completed_tasks']!=16 or
                summary['infrastructure_errors'] or summary['carried_completed_tasks'] or
                summary['disk_read_bytes'] or summary['disk_write_bytes']):
            raise ValueError('Formal comparison requires all 16 fresh CPU-only tasks in both arms')
    for key in ('reference_sha256','tasks','claude_sha256','workers','sampling','seed','attempts_per_task',
                'setup_sha256','protocol_provenance','cpu_admission','termination_policies','carried_results'):
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Formal protocols differ: '+key)
    if (manifests[0]['workers']!=4 or manifests[0]['attempts_per_task']!=1 or
            manifests[0]['carried_results'] or len(manifests[0]['tasks'])!=16):
        raise ValueError('Formal workload is not the original four-worker single-attempt first16')
    if evidence[0]!=evidence[1]:raise ValueError('Formal bridge or frozen runtime differs')
    profiles=[json.loads(json.dumps(m['profile'])) for m in manifests]
    if profiles[0]['kvmem']['offline_agent_policy'] or not profiles[1]['kvmem']['offline_agent_policy']:
        raise ValueError('Missing baseline/policy arm distinction')
    for profile in profiles:
        for key in ('offline_agent_policy','active_resident_ttl_ms','active_resident_pressure_only','active_resident_max_fraction'):
            profile['kvmem'].pop(key,None)
    if profiles[0]!=profiles[1]:raise ValueError('Formal model settings differ beyond the qualified policy')
    expected={t['task_id'] for t in manifests[0]['tasks']}
    by_task=[{t['task']:t for t in s['tasks']} for s in summaries]
    if len(expected)!=16 or any(set(tasks)!=expected or any(t['state']!='completed' for t in tasks.values()) for tasks in by_task):
        raise ValueError('Formal result identities differ from original first16')
    baseline,current=summaries
    return dict(baseline=baseline,current=current,baseline_complete=True,identical_bridge_and_protocol=True,
        source=evidence[0],
        output_throughput_ratio=current['mixed_end_to_end_output_tokens_per_second']/baseline['mixed_end_to_end_output_tokens_per_second'],
        makespan_ratio=current['elapsed_seconds']/baseline['elapsed_seconds'],
        passed_task_delta=current['passed_tasks']-baseline['passed_tasks'],
        tasks=[dict(task=task,baseline=by_task[0][task],current=by_task[1][task]) for task in sorted(expected)],
        scope='Fresh A0/P1 first16 using identical fixed bridge and protocol. Actual sampled trajectories '
              'can differ; inspect task quality and generated work alongside timing. Old r16 is excluded.')


async def formal(root, manifest, stopped, arm):
    spec = importlib.util.spec_from_file_location('owned_qualification', root/'frozen/scripts/qualify_coordination.py')
    q = importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    if arm not in ('A0','P1'):raise ValueError('Unknown formal arm')
    out = root/('deepswe-'+arm)
    out.mkdir(exist_ok=False)
    shutil.copytree(root/'deepswe/protocol',out/'protocol')
    save(out/'profile.json', read(root/f'performance/{arm}.json'))
    child, monitors, streams = None, [], []
    def spawn(command, name):
        stream = (out/name).open('w');streams.append(stream)
        return subprocess.Popen(command, cwd=root/'frozen', env=environment(root),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    with open('/tmp/kvmem-coordination-qualification.lock','w') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                await asyncio.sleep(10)
        try:
            await q.wait_idle(out,120)
            if hashes(root/'frozen') != read(root/'source-hashes.json'):
                raise RuntimeError('Frozen source changed before formal benchmark')
            if hashes(out/'protocol') != read(root/'protocol-hashes.json'):
                raise RuntimeError('Original tool protocol changed before formal benchmark')
            save(out/'formal-evidence.json',dict(source_sha256=read(root/'source-hashes.json'),
                protocol_sha256=read(root/'protocol-hashes.json')))
            command = [sys.executable,str(root/'frozen/scripts/benchmark_deepswe_first16.py'),
                '--reference',manifest['benchmark_reference'],'--config',str(out/'profile.json'),
                '--output',str(out),'--workers','4','--bash-repeat-limit','32']
            child = spawn(command,'controller.log')
            save(out/'controller-process.json',dict(pid=child.pid,command=command,at=time.time()))
            while child.poll() is None:
                if stopped():raise asyncio.CancelledError()
                gpu=await asyncio.to_thread(q.gpu_state)
                foreign=set(gpu['pids'])-q.owned_pids(child)
                if foreign:raise RuntimeError('Foreign GPU work appeared; stopping only this owned run')
                if (out/'run.json').exists():
                    run=read(out/'run.json')
                    if not monitors:
                        monitors=[spawn([sys.executable,str(root/'frozen/scripts/analyze_deepswe.py'),str(out),'--watch'],'analysis.log'),
                                  spawn([sys.executable,str(root/'frozen/scripts/monitor_deepswe_metrics.py'),str(out)],'metrics.log')]
                    save(out/'status.json',dict(status='running',phase=run['status'],controller_pid=child.pid,at=time.time(),gpu=gpu))
                await asyncio.sleep(2)
            run=read(out/'run.json')
            if child.returncode or run['status']!='completed' or len(run.get('tasks',{}))!=16:
                raise RuntimeError('Formal first16 did not complete; preserve all attempts')
            subprocess.run([sys.executable,str(root/'frozen/scripts/analyze_deepswe.py'),str(out)],
                           cwd=root/'frozen',env=environment(root),check=True,stdout=subprocess.DEVNULL)
            current=read(out/'live-summary.json')
            if current['completed_tasks']!=16 or current['infrastructure_errors'] or current['disk_read_bytes'] or current['disk_write_bytes']:
                raise RuntimeError('Formal run failed complete-task, infrastructure or CPU-only invariants')
            save(out/'status.json',dict(status='completed',tasks=16,at=time.time()))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                await asyncio.to_thread(child.wait,timeout=120)
            for monitor in monitors:
                if monitor.poll() is None:
                    monitor.terminate()
                    await asyncio.to_thread(monitor.wait,timeout=30)
            for stream in streams:stream.close()


async def run(root):
    if read(root/'status.json')['status']!='prepared':
        raise ValueError('Use a fresh prepared sequence; never implicitly restart an attempt')
    manifest=read(root/'sequence.json')
    if hashes(root/'frozen')!=read(root/'source-hashes.json'):
        raise RuntimeError('Prepared frozen source changed')
    child=None;stopped=False;stage='starting'
    task=asyncio.current_task()
    def stop():
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.terminate()
        task.cancel()
    loop=asyncio.get_running_loop()
    for signum in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(signum,stop)
    with open('/tmp/kvmem-agent-policy-sequence.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for stage,command in manifest['commands'].items():
                with (root/(stage+'.log')).open('w') as log:
                    child=subprocess.Popen(command,cwd=root/'frozen',env=environment(root),
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    save(root/'status.json',dict(status='running',stage=stage,pid=child.pid,at=time.time()))
                    while child.poll() is None:await asyncio.sleep(2)
                    if child.returncode:raise RuntimeError(stage+' failed; inspect its preserved log')
                if stage in ('correctness','isolation','pressure') and read(root/stage/'restore-gate.json').get('passed') is not True:
                    raise RuntimeError('Exact restore gate did not pass')
            gate=efficiency_gate(read(root/'performance/comparison.json'))
            save(root/'efficiency-gate.json',gate)
            if not gate['passed']:raise RuntimeError('Policy efficiency gate needs investigation; formal run not started')
            child=None
            for arm in manifest['formal_arms']:
                stage='deepswe-'+arm
                save(root/'status.json',dict(status='running',stage=stage,at=time.time()))
                await formal(root,manifest,lambda:stopped,arm)
            save(root/'formal-comparison.json',formal_comparison(root))
            save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='interrupted' if stopped else 'failed',stage=stage,error=repr(error),at=time.time()))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                await asyncio.to_thread(child.wait,timeout=120)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--run-prepared',action='store_true')
    args=p.parse_args();root=args.output.resolve()
    if args.run_prepared:
        if args.prepare_only or args.baseline:p.error('Prepared execution uses its frozen manifest only')
    else:
        if not args.baseline:p.error('Preparation requires the original baseline directory')
        prepare(root,args.baseline.resolve())
    if not args.prepare_only:asyncio.run(run(root))
