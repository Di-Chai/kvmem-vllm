"""Freeze and run the small graph qualification in independent GPU instances.

Owns only process groups it starts and holds the existing GPU qualification
lock. Every attempt gets its own immutable runtime/profile and retained logs.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class QualificationTerminated(KeyboardInterrupt):
    """Do not let HTTP retry handlers swallow the controller's SIGTERM."""


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}


@contextmanager
def cleanup_on_termination():
    """Convert the first SIGTERM into unwinding while cleanup still owns the lock."""
    previous = signal.getsignal(signal.SIGTERM)
    terminating = False

    def terminate(signum, frame):
        nonlocal terminating
        if not terminating:
            terminating = True
            raise QualificationTerminated('Qualification controller received SIGTERM')

    def begin_cleanup():
        nonlocal terminating
        terminating = True

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield begin_cleanup
    finally:
        signal.signal(signal.SIGTERM, previous)


def live_group_members(pgid):
    """Read live members, including workers whose original leader already exited."""
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        # A dead, unreaped child cannot use the GPU or respond to SIGKILL.
        if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
            members.append(int(path.name))
    return members


def stop_owned_groups(processes, grace_seconds=30, kill_seconds=15):
    processes = tuple(p for p in processes if p is not None)

    def signal_groups(groups, signum):
        for pgid in groups:
            if pgid == os.getpgrp():
                raise RuntimeError('Refuse to signal the controller process group')
            try:
                os.killpg(pgid, signum)
            except ProcessLookupError:
                pass

    def remaining():
        for process in processes:
            process.poll()  # Reap a leader even when its workers have not exited.
        return {p.pid: members for p in processes if (members := live_group_members(p.pid))}

    def drain(seconds):
        deadline = time.monotonic()+seconds
        while (groups := remaining()) and time.monotonic() < deadline:
            time.sleep(min(.1, max(0., deadline-time.monotonic())))
        return groups

    signal_groups((p.pid for p in processes), signal.SIGTERM)
    groups = drain(grace_seconds)
    if groups:
        signal_groups(groups, signal.SIGKILL)
        groups = drain(kill_seconds)
    return groups


def prepare(args):
    if not args.fixture.is_file():
        raise ValueError('Prepare a shared fixture first; qualification inputs must be frozen')
    qualifier = getattr(args, 'qualifier', None) or ROOT/'scripts/qualify_block_graph.py'
    qualifier = qualifier.resolve()
    if qualifier.parent != (ROOT/'scripts').resolve() or qualifier.suffix != '.py' or not qualifier.is_file():
        raise ValueError('Qualification client must be a Python file in this repository scripts directory')
    custom_profile = getattr(args, 'profile', None)
    if custom_profile is not None:
        profile = json.loads(custom_profile.read_text())
        if not isinstance(profile.get('kvmem'), dict):
            raise ValueError('Custom qualification profile requires a kvmem configuration')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = root / 'runtime'
    shutil.copytree(ROOT/'src', runtime/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (runtime/'scripts').mkdir()
    shutil.copyfile(ROOT/'scripts/qualify_block_graph.py', runtime/'scripts/qualify_block_graph.py')
    shutil.copyfile(ROOT/'scripts/benchmark_block_graph.py', runtime/'scripts/benchmark_block_graph.py')
    if qualifier.name != 'qualify_block_graph.py':
        shutil.copyfile(qualifier, runtime/'scripts'/qualifier.name)
    if custom_profile is None:
        profile = json.loads((ROOT/'artifacts/capacity-chain-g16-release-performance-20261017-r3/rolling-candidate-v2.json').read_text())
        profile.update(served_model_name='kvmem-graph-qualification',
            verify_session_restore=True, max_model_len=32768, max_num_seqs=2,
            max_num_batched_tokens=1024, long_prefill_token_threshold=1024,
            gpu_memory_utilization=.65, prefill_chunk_tokens=1024, query_replay_budget=1024)
        profile['kvmem'].update(select_budget=4096, gen_budget=4096, repository_blocks=1024,
            cpu_read_cache_bytes=4294967296, cpu_history_cache_bytes_per_request=2147483648,
            session_capacity=4, session_checkpoint_bytes=1073741824,
            query_history=False, action_pressure=False, guided_query_tokens=0,
            action_fraction=.25 if args.arm == 'legacy' else 0.,
            graph_mode=args.arm if args.arm in ('collect','locality','attention','multi') else 'off',
            graph_trace=True, rolling_active_window=False, offline_agent_policy=False,
            active_resident_pressure_only=False, resident_restore_slots=1,
            prefill_max_tokens=2048)
    profile['port'] = args.port
    if args.synchronous:
        profile['kvmem'].update(async_refresh=False, async_restore=False,
            maintenance_max_steps=0, cpu_history_coordination=False)
    write(root/'profile.json', profile)
    shutil.copyfile(args.fixture, root/'fixture.json')
    hashes = {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in runtime.rglob('*') if path.is_file()}
    write(root/'manifest.json', dict(arm=args.arm, synchronous=args.synchronous,
        qualification_script=qualifier.name,
        created_at=time.time(), runtime_sha256=hashes, python=sys.executable,
        fixture=str(root/'fixture.json'), original_fixture=str(args.fixture.resolve()),
        fixture_sha256=hashlib.sha256((root/'fixture.json').read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256((root/'profile.json').read_bytes()).hexdigest()))
    write(root/'status.json', dict(status='prepared', at=time.time()))


def run(args):
    root = args.output.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    runtime = root/'runtime'
    qualifier = manifest.get('qualification_script', 'qualify_block_graph.py')
    if Path(qualifier).name != qualifier or not qualifier.endswith('.py'):
        raise ValueError('Frozen qualification client must be a Python filename')
    if 'qualification_script' in manifest and 'scripts/'+qualifier not in manifest['runtime_sha256']:
        raise ValueError('Frozen qualification client is missing its runtime hash')
    for field, path in (('profile_sha256', root/'profile.json'),
            ('fixture_sha256', Path(manifest['fixture']))):
        if field not in manifest:
            raise ValueError('Unfrozen legacy manifest: prepare a new attempt with profile and fixture hashes')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f'Frozen qualification {field.removesuffix("_sha256")} changed')
    if Path(manifest['python']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Frozen qualification Python interpreter changed')
    for name, sha in manifest['runtime_sha256'].items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen qualification runtime changed')
    if json.loads((root/'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Use a fresh attempt; failed logs are never overwritten')
    profile = json.loads((root/'profile.json').read_text())
    with cleanup_on_termination() as begin_cleanup, open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_processes():
            raise RuntimeError('GPU is occupied; no foreign process will be stopped')
        environment = {k:v for k,v in os.environ.items() if not k.upper().startswith(('VLLM_', 'KVMEM_'))}
        environment.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
            TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1',
            VLLM_LOG_STATS_INTERVAL='1')
        for toolkit in ('/usr/local/cuda-13','/usr/local/cuda'):
            if (Path(toolkit)/'bin/nvcc').exists():
                environment['CUDA_HOME'] = toolkit
                break
        command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve',
            '--config', str(root/'profile.json'), '--run-dir', str(root/'server')]
        write(root/'launch.json', dict(argv=command, cwd=str(runtime)))
        server = client = None
        try:
            with (root/'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=runtime, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root/'status.json', dict(status='loading', pid=server.pid, at=time.time()))
            http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(1200):
                if server.poll() is not None:
                    raise RuntimeError('Model server exited during startup')
                try:
                    with http.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Independent model startup exceeded 20 minutes')
            write(root/'status.json', dict(status='testing', pid=server.pid, at=time.time()))
            client_command = [sys.executable, '-B', str(runtime/'scripts'/qualifier),
                '--config', str(root/'profile.json'), '--capture', str(root/'server/capture'),
                '--output', str(root/'qualification'), '--fixture', manifest['fixture'],
                '--arm', manifest['arm']]
            with (root/'client.log').open('x') as log:
                client = subprocess.Popen(client_command, cwd=runtime, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+1200
            while client.poll() is None:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Model or qualification client failed or timed out')
                time.sleep(1)
            if client.returncode:
                raise RuntimeError('Qualification failed; see retained client.log')
            result = json.loads((root/'qualification/results.json').read_text())
            if result.get('status') != 'passed' or not result.get('all_owned_sessions_released'):
                raise RuntimeError(f"Qualification did not pass: {result.get('status', 'missing status')}")
        except BaseException as error:
            write(root/'status.json', dict(status='failed', error=str(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining_groups = stop_owned_groups((client, server))
            remaining_gpu = sorted(gpu_processes())
            write(root/'cleanup.json', dict(controller_pid=os.getpid(),
                server_pid=None if server is None else server.pid,
                server_returncode=None if server is None else server.poll(),
                remaining_owned_process_groups=remaining_groups,
                remaining_gpu_pids=remaining_gpu, at=time.time()))
            if remaining_groups or remaining_gpu:
                error = 'Qualification cleanup left live process groups or GPU processes'
                write(root/'status.json', dict(status='failed', error=error, at=time.time()))
                raise RuntimeError(error)
        write(root/'status.json', dict(status='passed', at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline','collect','legacy','locality','attention','multi'), required=True)
    parser.add_argument('--port', type=int, default=18618)
    parser.add_argument('--synchronous', action='store_true')
    parser.add_argument('--profile', type=Path,
        help='Prepare using a complete custom profile; only port and --synchronous override it')
    parser.add_argument('--qualifier', type=Path,
        help='Prepare with a repository scripts client accepting the standard qualification arguments')
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
