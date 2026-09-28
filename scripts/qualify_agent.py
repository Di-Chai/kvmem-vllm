"""Run official DeepSWE agent/verifier attempts against one existing vLLM API.

This runner owns only its output directory and Pier child processes. It never
starts/stops the inference service, changes prompts, compacts histories, retries
model failures, or substitutes frozen completions for executed tasks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'benchmarks/deepswe'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_runtime(python, lock):
    code = '''import importlib.metadata as m,json,platform
p=m.distribution('datacurve-pier')
print(json.dumps({'pier_version':p.version,'pier_commit':json.loads(p.read_text('direct_url.json'))['vcs_info']['commit_id'],'mini_swe_agent':m.version('mini-swe-agent'),'python':platform.python_version()}))'''
    actual = json.loads(subprocess.check_output([str(python), '-c', code], text=True))
    expected = {'pier_version': lock['pier']['version'], 'pier_commit': lock['pier']['commit'],
                'mini_swe_agent': lock['mini_swe_agent']['version'], 'python': lock['python']}
    if actual != expected:
        raise RuntimeError(f'Unqualified agent runtime: {actual}; expected {expected}')
    return actual


def scored_result(task_output):
    paths = sorted((task_output / 'pier_jobs').glob('*/*/result.json'))
    trials = []
    for path in paths:
        data = json.loads(path.read_text())
        reward = ((data.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        if type(reward) in (int, float) and math.isfinite(reward):
            trials.append((path, data, reward))
    if len(trials) != 1 or len(paths) != 1:
        return {'status': 'unscored', 'numeric_trials': len(trials), 'result_paths': [str(p) for p in paths]}
    path, data, reward = trials[0]
    exception = data.get('exception_info')
    terminal = exception is None or (isinstance(exception, dict) and exception.get('exception_type') == 'AgentTimeoutError')
    return {'status': 'scored' if terminal else 'scored_with_unclassified_exception',
            'reward': reward, 'result_path': str(path), 'exception_info': exception,
            'trajectory_paths': [str(p) for p in task_output.glob('pier_jobs/*/*/agent/mini-swe-agent.trajectory.json')]}


def make_command(pier, task_id, task_path, task_output, api_base, model):
    model_kwargs = {'drop_params': True, 'extra_headers': {
        'x-kvmem-harness': 'mini-swe-agent', 'x-kvmem-task-id': task_id}}
    return [str(pier), 'run', '--jobs-dir', str(task_output / 'pier_jobs'),
            '--job-name', task_id, '--n-attempts', '1', '--n-concurrent', '1',
            '--max-retries', '0', '--timeout-multiplier', '4', '--yes',
            '--agent-timeout-multiplier', '1.0', '--path', str(task_path),
            '--agent-import-path', 'agent_adapter:ReliableMiniSweAgent',
            '--model', 'openai/' + model, '--agent-kwarg', 'version=2.4.6',
            '--agent-kwarg', 'model_class=litellm', '--agent-kwarg', 'cost_limit=0',
            '--agent-kwarg', 'model_kwargs=' + json.dumps(model_kwargs),
            '--agent-env', 'OPENAI_API_KEY=local-kvmem', '--agent-env', 'OPENAI_BASE_URL=' + api_base,
            '--agent-env', 'MSWEA_COST_TRACKING=ignore_errors']


def snapshot_task(source, destination):
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Task snapshot must not follow external symlinks')
    shutil.copytree(source, destination)
    return {str(p.relative_to(destination)): digest(p)
            for p in sorted(destination.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pier-python', type=Path, required=True)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--task', action='append', required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--api-base', required=True, help='Container-reachable API URL; Pier policy permits only ports 80/443')
    parser.add_argument('--health-api-base', help='Optional host-reachable URL of the same API')
    parser.add_argument('--model', default='qualification')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine-evidence', type=Path, required=True,
                        help='Exact server command/config/source manifest for this running service')
    parser.add_argument('--relay-evidence', type=Path,
                        help='Required for execution: ready managed relay resources.json and its native request journal')
    parser.add_argument('--preflight-evidence', type=Path,
                        help='Required for execution: passed tool/capture/retirement check for this backend')
    parser.add_argument('--prepare-only', action='store_true', help='Validate/snapshot without invoking Pier; not a quality result')
    args = parser.parse_args()
    # Resolving the interpreter symlink would select the base Python and lose
    # the caller's pinned virtualenv (and its sibling pier executable).
    args.pier_python = args.pier_python.expanduser().absolute()
    if len(args.task) != len(set(args.task)):
        parser.error('Each task gets one attempt; duplicate task IDs are not allowed')
    url = urllib.parse.urlparse(args.api_base)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.port not in (None, 80, 443):
        parser.error('Use an HTTP(S) container-reachable API on port 80/443 without URL credentials')
    lock = json.loads((POLICY / 'versions.lock.json').read_text())
    task_root = args.task_root.resolve()
    if digest(task_root / 'manifest.json') != lock['deep_swe_task_manifest_sha256']:
        raise RuntimeError('Task manifest differs from the locked DeepSWE set')
    plan = json.loads((POLICY / 'requestplan10.json').read_text())
    allowed = {item['task_id'] for item in plan['tasks']}
    if not set(args.task) <= allowed:
        parser.error('Requested task is not in the fixed requestplan10 set')
    actual = check_runtime(args.pier_python, lock)
    # Read and freeze evidence; never alter an existing inference service.
    evidence = json.loads(args.engine_evidence.read_text())
    relay = None
    preflight = None
    if not args.prepare_only:
        if args.relay_evidence is None:
            parser.error('Executed qualification requires relay-evidence for native timing and backend identity')
        relay = json.loads(args.relay_evidence.read_text())
        if (evidence.get('state') != 'ready' or relay.get('state') != 'ready'
                or relay.get('upstream_port') != evidence.get('port')
                or relay.get('api_base') != args.api_base
                or (args.health_api_base or '').rstrip('/') != evidence.get('api_base')):
            raise RuntimeError('Ready backend, relay address/port and host health endpoint must describe the same service')
        if args.preflight_evidence is None:
            parser.error('Executed qualification requires preflight-evidence for native tools and KVMem retirement')
        preflight = json.loads(args.preflight_evidence.read_text())
        if (preflight.get('status') != 'passed' or not preflight.get('archive_completion')
                or preflight.get('engine_manager_pid') != evidence.get('manager_pid')
                or preflight.get('engine_api_base') != evidence.get('api_base')
                or preflight.get('engine_source_commit') != evidence.get('source_commit')):
            raise RuntimeError('Tool/capture/retirement preflight must qualify this same backend instance')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), out / 'qualify_agent.py')
    shutil.copytree(POLICY, out / 'agent_source', ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'engine_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if relay is not None:
        (out / 'relay_evidence.json').write_text(json.dumps(relay, indent=2) + '\n')
    if preflight is not None:
        (out / 'preflight_evidence.json').write_text(json.dumps(preflight, indent=2) + '\n')
    tasks = []
    for task_id in args.task:
        task_out = out / task_id
        task_out.mkdir()
        snapshot = task_out / 'task'
        files = snapshot_task(task_root / task_id, snapshot)
        command = make_command(args.pier_python.with_name('pier'), task_id, snapshot,
                               task_out, args.api_base, args.model)
        tasks.append({'task_id': task_id, 'task_files': files, 'command': command})
    report = {'scope': 'official executed DeepSWE tasks on one shared existing API; no persistent-session claim',
              'created_at': datetime.now(timezone.utc).isoformat(), 'runtime': actual,
              'harness_sha256': digest(out / 'qualify_agent.py'),
              'engine_evidence_sha256': digest(args.engine_evidence), 'concurrency': args.concurrency,
              'tasks': tasks, 'prepared_only': args.prepare_only, 'status': 'prepared'}
    (out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.prepare_only:
        print(out / 'plan.json')
        return
    health = (args.health_api_base or args.api_base).rstrip('/') + '/models'
    with urllib.request.urlopen(health, timeout=10) as response:
        models = json.load(response)
    if args.model not in {item['id'] for item in models['data']}:
        raise RuntimeError('Expected model is not served by the requested API')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(out / 'agent_source')
    # Keep the historical verified uv cache optional; no dependency on an old
    # QW3 relay/cache service. Installer falls back to its pinned HTTP/1.1 URL.
    env['QW3_PIER_UV_CACHE_URL'] = 'http://127.0.0.1:9/uv-0.7.13'
    stopped = threading.Event()
    children, guard = {}, threading.Lock()

    def stop_children(*unused):
        stopped.set()
        with guard:
            for child in children.values():
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)

    def run(task):
        task_out = out / task['task_id']
        with (task_out / 'pier_console.log').open('wb') as log:
            with guard:
                if stopped.is_set():
                    return {'task_id': task['task_id'], 'status': 'interrupted_before_start'}
                process = subprocess.Popen(task['command'], cwd=task_out, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                children[process.pid] = process
            (task_out / 'process.json').write_text(json.dumps({'pid': process.pid, 'owned_process_group': process.pid}))
            try:
                stop_at = None
                while True:
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if stopped.is_set():
                            stop_at = stop_at or time.monotonic()
                            if time.monotonic() - stop_at > 30:
                                os.killpg(process.pid, signal.SIGKILL)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                with guard:
                    children.pop(process.pid, None)
        result = {'task_id': task['task_id'], 'pier_exit_code': code, **scored_result(task_out)}
        (task_out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        try:
            results = list(pool.map(run, tasks))
        except BaseException:
            stop_children()
            raise
    report.update(status='interrupted' if stopped.is_set() else 'finished', results=results,
                  all_scored=all(r['status'] == 'scored' for r in results))
    journal = args.relay_evidence.parent / 'requests.jsonl'
    lines = journal.read_text().splitlines() if journal.exists() else []
    requests = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # The live relay may be appending its next record.
        if record.get('task_id') in set(args.task) and record.get('path') in ('/v1/chat/completions', '/v1/completions'):
            requests.append(record)
    (out / 'native_requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    report['native_metrics'] = {}
    for task in args.task:
        records = [r for r in requests if r['task_id'] == task]
        timed = [r for r in records if r.get('status') == 200 and r.get('native_decode_tokens_per_second') is not None]
        seconds = sum(r['native_metrics']['generation_time_ms'] for r in timed) / 1000
        tokens = sum(r['usage']['completion_tokens'] - 1 for r in timed)
        report['native_metrics'][task] = {
            'recorded_requests': len(records), 'timed_decode_requests': len(timed),
            'http_error_requests': sum(r.get('status') != 200 for r in records),
            'weighted_per_request_decode_tokens_per_second': tokens / seconds if seconds else None,
            'summed_decode_seconds': seconds, 'decode_tokens': tokens,
            'scope': 'sum(tokens-1)/sum(native decode intervals); not aggregate throughput under overlap'}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'results': results, 'all_scored': report['all_scored']}))
    if not report['all_scored']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
