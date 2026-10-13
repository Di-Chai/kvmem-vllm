"""Collect separate bounded prefill and C8 decode traces from a local service.

The service must enable the native torch profiler with a finite max_iterations.
The child uses the ordinary authenticated qualification harness. Its throughput
is diagnostic only: profiler overhead and trace export pauses remain included.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import httpx


def validate_collection(cfg, args):
    """Validate finite two-phase collection before starting its workload."""
    profiler = cfg.get('profiler_config')
    if (not isinstance(profiler, dict) or profiler.get('profiler') != 'torch'
            or profiler.get('ignore_frontend') is not True):
        raise ValueError('Require a torch worker profiler with ignore_frontend=true')
    maximum, delay = profiler.get('max_iterations'), profiler.get('delay_iterations', 0)
    if type(maximum) is not int or maximum <= 0 or type(delay) is not int or delay < 0:
        raise ValueError('Require positive max_iterations and nonnegative delay_iterations')
    if any(type(profiler.get(name, 0)) is not int or profiler.get(name, 0) != 0
           for name in ('warmup_iterations', 'wait_iterations')):
        # Native TorchProfilerWrapper retains the same repeat=1 schedule across
        # start/stop. Its second start does not reset torch's schedule position.
        raise ValueError('Two-phase collection requires warmup_iterations=wait_iterations=0')
    directory = profiler.get('torch_profiler_dir')
    if not isinstance(directory, str) or not directory or '\x00' in directory or '://' in directory:
        raise ValueError('Require a local torch_profiler_dir')
    if cfg.get('context_mode') != 'meank' or cfg.get('draft_tokens') not in (1, 3, 7):
        raise ValueError('Two-phase collection requires the MeanK DFlash profile')
    if type(args.concurrency) is not int or args.concurrency <= 0:
        raise ValueError('concurrency must be positive')
    # Reserve every possible accepted candidate plus margin for control RPCs
    # and progress sampling. The trace is validated again after collection.
    headroom = 512 + (maximum + delay + 2) * (cfg['draft_tokens'] + 1)
    if type(args.output_tokens) is not int or args.output_tokens <= 512 + headroom:
        raise ValueError(f'output_tokens must exceed {512 + headroom} for the configured profile window')
    return profiler, headroom


def all_decoding(progress, concurrency, output_tokens, headroom):
    counts = progress.get('output_tokens', [])
    return (progress.get('running') == concurrency and len(counts) == concurrency
            and all(type(count) is int and count >= 512 for count in counts)
            and max(counts) < output_tokens - headroom)


def validate_trace(path, *, concurrency=None, iterations=None):
    """Require a complete worker trace and verify its actual decode batches."""
    try:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as source:
            trace = json.load(source)
    except (OSError, EOFError, ValueError) as error:
        raise RuntimeError(f'Profiler trace is missing, incomplete or invalid: {path.name}') from error
    events = trace.get('traceEvents') if isinstance(trace, dict) else None
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise RuntimeError('Profiler trace has no valid event list')
    native = [event for event in events if event.get('cat') == 'user_annotation'
              and event.get('name', '').startswith('execute_')]
    if not native or not any(event.get('cat') == 'kernel' and event.get('ph') == 'X' for event in events):
        raise RuntimeError('Profiler trace requires native CPU iterations and CUDA kernels')
    pattern = re.compile(r'^execute_(?:\d+_)?context_(\d+)\([^)]*\)_generation_(\d+)\([^)]*\)$')
    for event in native:
        match = pattern.fullmatch(event['name'])
        if (not match or event.get('ph') != 'X'
                or any(type(event.get(key)) not in (int, float) or not math.isfinite(event[key])
                       for key in ('ts', 'dur')) or event['dur'] <= 0):
            raise RuntimeError('Profiler trace contains incomplete native CPU iterations')
        if concurrency is not None and (int(match[1]) != 0 or int(match[2]) != concurrency):
            raise RuntimeError('Decode trace contains prefill or a different concurrency')
    if iterations is not None and len(native) != iterations:
        raise RuntimeError(f'Decode trace has {len(native)} iterations; expected {iterations}')
    return {'native_cpu_iterations': len(native), 'decode_concurrency': concurrency,
            'complete_json': True, 'cuda_kernels_present': True}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_refreshes(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # The worker may still be writing the last line.
    return rows


def run(args):
    cfg = read_json(args.launch)['config']
    profiler, headroom = validate_collection(cfg, args)
    args.output.mkdir(parents=True, exist_ok=False)
    batch = args.output / 'batch'
    command = [sys.executable, str(Path(__file__).with_name('qualify_elastic_long_output.py')),
        '--launch', str(args.launch), '--input', str(args.input), '--output', str(batch),
        '--concurrency', str(args.concurrency), '--output-tokens', str(args.output_tokens),
        '--port', str(args.port)]
    capture = args.capture
    traces = Path(profiler['torch_profiler_dir'])
    existing_requests = {r['request_id'] for r in read_refreshes(capture / 'bounded_refresh.jsonl')}
    deadline = time.monotonic() + args.timeout
    reports = []
    with (args.output / 'batch.log').open('w') as log, httpx.Client(
            base_url=cfg['backend_url'], timeout=180, trust_env=False) as client:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def wait_until(predicate):
            while True:
                value = predicate()
                if value:
                    return value
                if child.poll() is not None:
                    raise RuntimeError('Workload finished before profiling condition was met')
                if time.monotonic() > deadline:
                    raise TimeoutError('Profiling workload deadline exceeded')
                time.sleep(.25)

        def fresh_refreshes():
            return [r for r in read_refreshes(capture / 'bounded_refresh.jsonl')
                if r['request_id'] not in existing_requests]

        def start(phase):
            report = {'phase': phase, 'started_monotonic': time.monotonic(),
                'progress_before': read_json(batch / 'progress.json'),
                'capture_before': read_json(capture / 'progress.json')}
            before = set(traces.glob('*.pt.trace.json*'))
            response = client.post('/start_profile'); response.raise_for_status()
            report['start_response_monotonic'] = time.monotonic()
            return report, before

        def stop(report, before):
            response = client.post('/stop_profile'); response.raise_for_status()
            files = list(set(traces.glob('*.pt.trace.json*')) - before)
            if not files:
                raise RuntimeError('Profiler returned no worker trace')
            validation = {source.name: validate_trace(source,
                concurrency=args.concurrency if report['phase'] == 'decode' else None,
                iterations=profiler['max_iterations'] if report['phase'] == 'decode' else None)
                for source in files}
            destination = args.output / report['phase']; destination.mkdir()
            for source in files:
                shutil.move(str(source), destination / source.name)
            for source in traces.glob('profiler_out*'):
                shutil.copy2(source, destination / source.name)
            report.update(finished_monotonic=time.monotonic(),
                progress_after=read_json(batch / 'progress.json'),
                capture_after=read_json(capture / 'progress.json'),
                files=[str(p) for p in destination.iterdir()], trace_validation=validation)
            reports.append(report)
            (args.output / 'phases.json').write_text(json.dumps(reports, indent=2) + '\n')
            print(json.dumps({'phase': report['phase'], 'trace_files': len(files)}), flush=True)

        try:
            # Capture actual scoring/reassembly, beyond the all-history bootstrap.
            wait_until(lambda: any(r['history_begin'] > cfg['active_context_tokens']
                                   for r in fresh_refreshes()))
            report, before = start('prefill')
            initial = len(fresh_refreshes())
            wait_until(lambda: len(fresh_refreshes()) >= initial + args.concurrency)
            stop(report, before)

            def ready_for_decode():
                path = batch / 'progress.json'
                progress = read_json(path)
                try:
                    # The qualification harness publishes this snapshot every
                    # 20 seconds. Do not start from an old snapshot after prefill.
                    fresh = 0 <= time.time() - path.stat().st_mtime <= 2
                except FileNotFoundError:
                    return False
                return fresh and all_decoding(progress, args.concurrency, args.output_tokens, headroom)

            wait_until(ready_for_decode)
            report, before = start('decode')
            # Native max_iterations ends collection and exports the trace.
            wait_until(lambda: set(traces.glob('*.pt.trace.json*')) - before)
            stop(report, before)
            child.wait(timeout=max(1, deadline - time.monotonic()))
            if child.returncode:
                raise RuntimeError(f'Qualification workload failed: {child.returncode}')
            summary = read_json(batch / 'summary.json')
            if summary.get('status') != 'passed':
                raise RuntimeError('Profiled workload did not pass concurrency checks')
            print(json.dumps({'status': 'passed', 'scope': 'diagnostic, instrumented throughput',
                'phases': [p['phase'] for p in reports]}), flush=True)
        finally:
            try:
                client.post('/stop_profile')
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=4096)
    parser.add_argument('--port', type=int, default=18384)
    parser.add_argument('--timeout', type=float, default=1200)
    run(parser.parse_args())
