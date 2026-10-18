"""Collect one delayed CUDA timeline from a fresh frozen restore diagnostic.

Collector completion is independent of target completion. No retries or GPU
queueing; only the proven harness receives cancellation, and export follows
normal harness/server cleanup. All results are diagnostic, not performance wins.
"""
import argparse
import asyncio
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

import psutil


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nsys_restore_diagnostic',
    ROOT / 'scripts/diagnose_restore_stages.py')
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)
q = diagnostic.benchmark.q
TERMINAL = {'diagnostic_completed', 'failed', 'interrupted'}


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def identity(process):
    return dict(pid=process.pid, create_time=process.create_time())


def live(owner):
    if owner is None:
        return None
    try:
        process = psutil.Process(owner['pid'])
        if process.create_time() == owner['create_time'] and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def proven_target(row, root, output, claim):
    if (row.get('nonce') != claim['nonce'] or row.get('diagnostic_output') != str(root) or
            row.get('capture_output') != str(output) or
            row.get('manifest_sha256') != claim['manifest_sha256']):
        raise ValueError('Target identity does not match the collector claim')
    process = live(row)
    if process is None:
        raise RuntimeError('Target exited before ownership could be verified')
    command = process.cmdline()
    helper = str(root / 'frozen/scripts/nsys_diagnostic_target.py')
    entry = str(root / 'frozen/scripts/diagnose_restore_stages.py')
    helper_command = [helper, '--diagnostic-output', str(root), '--capture-output', str(output),
                      '--nonce', claim['nonce']]
    entry_command = [entry, '--run-prepared', '--output', str(root)]
    if command[1:] not in (helper_command, entry_command):
        raise ValueError('Target PID no longer runs the claimed frozen diagnostic')
    return row


def signal_target(target, root, output, claim):
    process = live(target)
    if process is None:
        return False
    proven_target(target, root, output, claim)
    # psutil checks PID reuse again before sending. Never signal a process group.
    process.send_signal(signal.SIGTERM)
    return True


def track_server(target, status, owned):
    pid = status.get('server_pid')
    if pid is not None and pid not in owned:
        try:
            process = psutil.Process(pid)
            if not any(p.pid == target['pid'] and p.create_time() == target['create_time']
                       for p in process.parents()):
                raise ValueError('Reported server is not a child of the proven harness')
            if os.getpgid(pid) != pid:
                raise ValueError('Harness server lacks its dedicated process group')
            owned[pid] = dict(**identity(process), server_group=pid)
        except psutil.NoSuchProcess:
            return
    for owner in list(owned.values()):
        process = live(owner)
        if process is not None:
            try:
                children = process.children(recursive=True)
            except psutil.NoSuchProcess:
                # Compiler helpers can exit between the identity check and
                # descendant enumeration. Retain their recorded identity for
                # cleanup checks; their exit is not a model-server failure.
                continue
            for child in children:
                try:
                    owned[child.pid] = dict(**identity(child), server_group=owner['server_group'])
                except psutil.NoSuchProcess:
                    pass


def owned_clean(target, owned):
    if live(target) is not None or any(live(owner) is not None for owner in owned.values()):
        return False
    groups = {row['server_group'] for row in owned.values()}
    # Reparenting must not hide a server worker that escaped a previous snapshot.
    for process in psutil.process_iter(['pid', 'status']):
        try:
            if process.info['status'] != psutil.STATUS_ZOMBIE and os.getpgid(process.pid) in groups:
                return False
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
    return True


def profile_command(nsys, root, output, claim, seconds):
    return [str(nsys), 'profile', '--session-new=' + claim['session'],
        '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
        '--cuda-graph-trace=graph:host-only', '--cuda-event-trace=false', '--osrt-threshold=100000',
        '--start-later=true', '--duration=' + str(seconds), '--kill=none', '--wait=all',
        '--stats=false', '--export=none', '--force-overwrite=false', '--output=' + str(output / 'timeline'),
        sys.executable, str(root / 'frozen/scripts/nsys_diagnostic_target.py'),
        '--diagnostic-output', str(root), '--capture-output', str(output), '--nonce', claim['nonce']]


def validate_prepared(root, output, seconds, offset):
    if ROOT != root / 'frozen' or output.parent != root:
        raise ValueError('Run the frozen wrapper with a new capture directory directly under its diagnostic output')
    manifest, _, _ = diagnostic.validate(root)
    if (read(root / 'status.json').get('status') != 'prepared' or
            (root / 'nsys-claim.json').exists() or any(root.glob('pair-*'))):
        raise ValueError('Require a fresh unclaimed prepared diagnostic; never retry')
    if (not math.isfinite(seconds) or not 30 <= seconds <= 60 or not math.isfinite(offset) or
            offset < 0 or offset + seconds >= manifest['workload']['duration_seconds']):
        raise ValueError('Capture must fit a 30-60 second interval inside the formal window')
    for name in ('profile_restore_diagnostic.py', 'nsys_diagnostic_target.py'):
        if 'scripts/' + name not in manifest['source_sha256']:
            raise ValueError('Prepare a new frozen runtime containing both Nsight entry points')
    return manifest


def require_idle():
    state = q.gpu_state()
    if state['pids'] or state['memory_mib'] >= 512 or state['utilization'] >= 5:
        raise RuntimeError('GPU is not idle; this diagnostic does not queue or stop other work')
    return state


def lock_gpu():
    lock = open(diagnostic.LOCK_PATH, 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def observed_engines(root, owned, begin, end):
    """Reuse the harness's foreign-process guard; do not add GPU polling."""
    pids, samples = set(), 0
    with (root / 'pair-00-P1/gpu_samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if begin <= row['at'] <= end:
                samples += 1
                pids.update(row['pids'])
    if not samples or not pids:
        raise RuntimeError('No engine GPU PID samples inside the requested capture interval')
    if pids - owned.keys():
        raise RuntimeError('Capture GPU PID was not proven to belong to the owned server')
    return dict(begin=begin, end=end, samples=samples,
                identities=[owned[pid] for pid in sorted(pids)])


def validate_export(path, engine_pids):
    engine_pids = set(engine_pids)
    if not engine_pids:
        raise RuntimeError('CUDA export requires observed owned engine PIDs')
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as connection:
        if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
            raise RuntimeError('Nsight export failed SQLite integrity check')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        activities = [name for name in tables if name.startswith('CUPTI_ACTIVITY_KIND_') and
                      any(kind in name for kind in ('KERNEL', 'MEMCPY', 'GRAPH'))]
        counts, extents, processes = {}, {}, {}
        compute_pids = set()
        for name in activities:
            quoted = '"' + name.replace('"', '""') + '"'
            counts[name] = connection.execute('SELECT count(*) FROM ' + quoted).fetchone()[0]
            columns = {row[1] for row in connection.execute('PRAGMA table_info(' + quoted + ')')}
            if {'start', 'end'} <= columns:
                extents[name] = connection.execute('SELECT min(start),max(end) FROM ' + quoted).fetchone()
            if {'globalPid', 'start', 'end'} <= columns:
                # Same serialized GlobalId decoding as this installation's
                # cuda_kern_exec_trace report; host API presence is insufficient.
                rows = connection.execute('SELECT (globalPid >> 24) & 0x00FFFFFF, '
                    'count(*),min(start),max(end) FROM ' + quoted +
                    ' WHERE end > start GROUP BY 1').fetchall()
                processes[name] = [dict(pid=pid, rows=count, start_ns=start, end_ns=end)
                                   for pid, count, start, end in rows]
                if 'KERNEL' in name or 'GRAPH' in name:
                    compute_pids.update(row[0] for row in rows)
        if not any(counts.values()):
            raise RuntimeError('Report has no CUDA device activities; cannot infer GPU idle')
        if engine_pids - compute_pids:
            raise RuntimeError('Missing CUDA device compute activity for observed engine PIDs: ' +
                               str(sorted(engine_pids - compute_pids)))
        return dict(sqlite_integrity='ok', device_activity_rows=counts, device_activity_extent_ns=extents,
                    device_activity_processes=processes, verified_engine_pids=sorted(engine_pids),
                    coverage_scope='Owned engine device activity is present. Complete device/stream and '
                    'interval coverage still require review. CPU-launched Graph intervals include internal '
                    'kernel gaps: their union with kernels/memcpy is a conservative busy envelope, '
                    'not exact GPU execution time or SM occupancy. Only gaps outside that envelope, '
                    'after complete coverage is verified, establish device idle.')


async def run(args):
    root, output = args.diagnostic_output.resolve(), args.output.resolve()
    manifest = validate_prepared(root, output, args.duration_seconds, args.offset_seconds)
    with lock_gpu():
        initial_gpu = require_idle()
    output.mkdir(exist_ok=False)
    claim = dict(nonce=uuid.uuid4().hex, session='kvmem-restore-' + uuid.uuid4().hex,
        capture_output=str(output), manifest_sha256=diagnostic.sha256(root / 'manifest.json'),
        wrapper=identity(psutil.Process()), at=time.monotonic(), wall=time.time())
    with (root / 'nsys-claim.json').open('x') as stream:
        json.dump(claim, stream)
    events, owned, target = [], {}, None
    collector, triggered, reported_exit = None, False, False
    capture_begin = None
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)

    def emit(event, **fields):
        row = dict(event=event, at=time.monotonic(), wall=time.time(), **fields)
        events.append(row)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    async def tool(arguments, timeout=30, *, cleaning=False, check=True):
        if stopped.is_set() and not cleaning:
            raise asyncio.CancelledError('Cancelled before Nsight control/export')
        emit('tool_requested', argv=arguments)
        with (output / 'nsys-control.log').open('a') as stream:
            result = await asyncio.to_thread(subprocess.run, arguments, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        emit('tool_returned', argv=arguments, returncode=result.returncode)
        if stopped.is_set() and not cleaning:
            raise asyncio.CancelledError('Cancelled during Nsight control/export')
        if result.returncode and check:
            raise RuntimeError('Nsight control command failed; inspect nsys-control.log')
        return result.returncode

    async def session_present():
        result = await asyncio.to_thread(subprocess.run,
            [str(args.nsys), 'sessions', 'list', '--show-header=false'],
            capture_output=True, text=True, timeout=15, check=True)
        return claim['session'] in result.stdout.split()

    async def finish_collector():
        if collector is None:
            return
        # Session lifetime is independent of the short-lived nsys launcher.
        for verb in (('stop', 'shutdown') if triggered else ('shutdown',)):
            if not await session_present():
                break
            command = [str(args.nsys), verb, '--session=' + claim['session']]
            if verb == 'shutdown':
                command.append('--kill=none')
            result = await tool(command, cleaning=True, check=False)
            if result and await session_present():
                raise RuntimeError('Owned Nsight session could not be closed')
        deadline = time.monotonic() + 120
        while collector.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Nsight launcher did not finish after its owned session closed')
            await asyncio.sleep(.2)
        if await session_present():
            raise RuntimeError('Owned Nsight session still exists after launcher exit')

    async def cleanup():
        nonlocal target
        errors = []
        try:
            if target is None and (output / 'target.json').exists():
                candidate = read(output / 'target.json')
                # A dead pre-ACK helper cannot start a model and needs no signal.
                if live(candidate) is not None:
                    target = proven_target(candidate, root, output, claim)
            if target is not None:
                emit('target_cancel_requested', sent=signal_target(target, root, output, claim))
                deadline = time.monotonic() + 180
                while not owned_clean(target, owned):
                    track_server(target, read(root / 'status.json'), owned)
                    if time.monotonic() > deadline:
                        raise TimeoutError('Owned harness/server cleanup incomplete; no broad kill attempted')
                    await asyncio.sleep(.5)
        except BaseException as error:
            errors.append(repr(error))
        try:
            await finish_collector()
        except BaseException as error:
            errors.append(repr(error))
        # A missing ACK prevents any late helper from exec'ing a GPU workload.
        # Do not falsely certify its unseen process lifetime as fully verified.
        if collector is not None and target is None:
            emit('unobserved_target_cleanup', exec_authorized=(output / 'allow-exec.json').exists())
            if (output / 'allow-exec.json').exists():
                errors.append('Authorized target identity was lost')
        if errors:
            raise RuntimeError('; '.join(errors))

    try:
        version = await asyncio.to_thread(subprocess.check_output, [str(args.nsys), '--version'], text=True, timeout=15)
        command = profile_command(args.nsys, root, output, claim, args.duration_seconds)
        q.save(output / 'observer.json', dict(diagnostic_only=True, performance_claim_allowed=False,
            nsys_version=version.strip(), command=command, claim=claim,
            source_sha256=manifest['source_sha256'], profile_sha256=manifest['profile_sha256'],
            fixture_sha256=manifest['fixture_sha256'], initial_gpu=initial_gpu,
            capture_seconds=args.duration_seconds, offset_seconds=args.offset_seconds,
            scope='One CUDA timeline. Collector exit is not target completion; profiler and export '
                  'overhead invalidate performance-win claims. graph:host-only means CPU-launched '
                  'graphs, not host API only. Graph intervals include internal kernel gaps, so '
                  'Graph/kernel/memcpy union is a conservative busy envelope, not exact GPU '
                  'execution time or SM occupancy. Complete process/stream/interval coverage '
                  'must be verified before labeling gaps outside that envelope as device idle.'))
        with (output / 'collector.log').open('w') as log:
            if stopped.is_set():
                raise asyncio.CancelledError('Cancelled before collector launch')
            collector = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            emit('collector_launched', identity=identity(psutil.Process(collector.pid)))
            startup_deadline = time.monotonic() + 45
            cleanup_deadline = None
            while True:
                if stopped.is_set():
                    raise asyncio.CancelledError('Wrapper cancelled; stop only the owned harness')
                if target is None and (output / 'target.json').exists():
                    target = proven_target(read(output / 'target.json'), root, output, claim)
                    if diagnostic.sha256(root / 'manifest.json') != claim['manifest_sha256']:
                        raise ValueError('Manifest changed before target authorization')
                    emit('target_proven', identity=target)
                    q.save(output / 'allow-exec.json', dict(nonce=claim['nonce'], **identity(live(target))))
                status = read(root / 'status.json')
                if target is not None:
                    track_server(target, status, owned)
                if status.get('status') == 'waiting_for_gpu':
                    state = status.get('gpu', {})
                    if state.get('pids') or state.get('memory_mib', 0) >= 512 or state.get('utilization', 0) >= 5:
                        raise RuntimeError('GPU became busy before launch; this wrapper will not queue')
                if status.get('status') in ('failed', 'interrupted'):
                    raise RuntimeError('Diagnostic harness failed: ' + str(status.get('error')))
                if collector.poll() is not None and not reported_exit:
                    emit('collector_exited', returncode=collector.returncode, target_alive=live(target) is not None)
                    reported_exit = True
                    if collector.returncode or not triggered:
                        raise RuntimeError('Collector failed or exited before capture trigger')
                if target is None and time.monotonic() > startup_deadline:
                    raise TimeoutError('Collector target did not publish a valid identity')
                window = read(root / 'pair-00-P1/saturated/window.json')
                if window and not triggered and time.monotonic() >= window['begin'] + args.offset_seconds:
                    if time.monotonic() + args.duration_seconds >= window['end']:
                        raise RuntimeError('Missed the bounded formal capture window; do not retry')
                    emit('capture_trigger_requested', window=window)
                    capture_begin = time.monotonic()
                    await tool([str(args.nsys), 'start', '--session=' + claim['session']])
                    triggered = True
                    emit('capture_trigger_returned')
                if target is not None and live(target) is None:
                    if status.get('status') not in TERMINAL:
                        raise RuntimeError('Target exited without a terminal diagnostic status')
                    cleanup_deadline = cleanup_deadline or time.monotonic() + 180
                    if owned_clean(target, owned):
                        break
                    if time.monotonic() > cleanup_deadline:
                        raise TimeoutError('Target exited but owned server group remains')
                await asyncio.sleep(.5)
            emit('target_and_server_finished', status=status, owned=list(owned.values()))
            if stopped.is_set():
                raise asyncio.CancelledError('Cancelled after target cleanup')
            await finish_collector()
        if not triggered or collector.returncode or status['status'] != 'diagnostic_completed':
            raise RuntimeError('Diagnostic or collector did not complete successfully')
        report, export = output / 'timeline.nsys-rep', output / 'timeline.sqlite'
        if not report.is_file() or not report.stat().st_size:
            raise RuntimeError('Collector completed without a report')
        # Hold the same cooperative lock during export so the next formal arm
        # cannot start while report conversion consumes CPU and storage.
        if stopped.is_set():
            raise asyncio.CancelledError('Cancelled before export')
        with lock_gpu():
            require_idle()
            emit('export_started', report=str(report))
            await tool([str(args.nsys), 'export', '--type=sqlite', '--force-overwrite=false',
                        '--output=' + str(export), str(report)], timeout=300)
            engines = observed_engines(root, owned, capture_begin, capture_begin + args.duration_seconds)
            validation = validate_export(export, [row['pid'] for row in engines['identities']])
            validation['engine_observations'] = engines
            if stopped.is_set():
                raise asyncio.CancelledError('Cancelled during export validation')
            emit('export_completed', validation=validation)
        q.save(output / 'result.json', dict(status='completed_diagnostic', diagnostic_only=True,
            performance_claim_allowed=False, target=target, owned=list(owned.values()),
            collector_returncode=collector.returncode, validation=validation, events=events))
    except BaseException as error:
        cleanup_error = None
        try:
            await cleanup()
        except BaseException as failure:
            cleanup_error = repr(failure)
        q.save(output / 'result.json', dict(status='interrupted' if isinstance(error, asyncio.CancelledError)
            else 'failed', diagnostic_only=True, error=repr(error), cleanup_error=cleanup_error,
            target=target, owned=list(owned.values()), events=events, retry_allowed=False))
        raise
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nsys', type=Path, default=Path('/usr/local/bin/nsys'))
    parser.add_argument('--duration-seconds', type=float, default=45.)
    parser.add_argument('--offset-seconds', type=float, default=30.)
    asyncio.run(run(parser.parse_args()))
