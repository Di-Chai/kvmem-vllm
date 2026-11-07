"""Finite matched CPU/tiered HTTP experiment; counts only completed public tokens.

Synthetic capacity/performance qualification, not a task-quality benchmark.
Arms freeze their source, including an optional older baseline. K/G, picker and refresh settings are inherited
unchanged from the supplied profile; only storage/admission/checkpoint controls vary.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TIMING_BASIS = 'First HTTP request start through last completed public response; includes queueing, excludes startup and retirement drain.'


class RequestFailure(RuntimeError):
    """Keep partial public output even when transport/accounting validation fails."""
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def input_digest(payload):
    return hashlib.sha256(json.dumps(payload['prompt'], separators=(',', ':')).encode()).hexdigest()


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError('Invalid percentile samples or fraction')
    values = sorted(values)
    at = (len(values)-1)*fraction
    left, right = math.floor(at), math.ceil(at)
    return values[left]+(values[right]-values[left])*(at-left)


def client_peak(responses):
    edges = [(r['started'], 1) for r in responses]
    edges += [(r['started']+r['seconds'], -1) for r in responses]
    active = peak = 0
    for _, change in sorted(edges):
        active += change
        peak = max(peak, active)
    return peak


def resource_sample(scheduler, archive, elapsed):
    """Compact observations, keeping exact payload and charged bytes distinct."""
    gpu = scheduler.get('gpu_capacity', {})
    owners = gpu.get('requests')
    swaps = scheduler.get('request_swaps', {})
    host = archive.get('process_memory', {})
    worker, cgroup = host.get('worker', {}), host.get('cgroup', {})
    checkpoints = archive.get('request_checkpoints', {}).get('storage') or {}
    return dict(elapsed_seconds=elapsed, scheduler_at=scheduler.get('at'), archive_at=archive.get('at'),
        scheduler_requests=scheduler.get('requests'),
        gpu_owned_pages=gpu.get('request_owned_pages', scheduler.get('owned_blocks')),
        simultaneous_gpu_owners=(sum(row.get('owned_pages', 0) > 0 for row in owners.values())
                                 if isinstance(owners, dict) else None),
        gpu_free_pages=gpu.get('physical_free_pages', scheduler.get('free_blocks')),
        gpu_future_reserved_pages=gpu.get('future_reserved_pages'),
        swap_pauses=swaps.get('pauses'), swap_restores=swaps.get('restores'),
        swap_restore_deferrals=swaps.get('restore_deferrals'),
        checkpoint_cpu_resident_bytes=checkpoints.get('cpu_resident_bytes'),
        checkpoint_cpu_charged_bytes=checkpoints.get('cpu_charged_bytes'),
        checkpoint_cpu_peak_bytes=checkpoints.get('cpu_peak_bytes'),
        checkpoint_cpu_charge_peak_bytes=checkpoints.get('cpu_charge_peak_bytes'),
        checkpoint_ssd_used_bytes=checkpoints.get('ssd_used_bytes'),
        checkpoint_ssd_peak_bytes=checkpoints.get('ssd_peak_bytes'),
        checkpoint_disk_write_bytes=checkpoints.get('disk_write_bytes'),
        checkpoint_disk_read_bytes=checkpoints.get('disk_read_bytes'),
        checkpoint_io_pending=checkpoints.get('io_pending'),
        paused_requests=(sum(row.get('phase') == 'paused' for row in swaps['requests'].values())
                         if isinstance(swaps.get('requests'), dict) else None),
        cpu_resident_bytes=archive.get('cpu_resident_bytes',
            archive.get('cpu_authority_bytes') if archive.get('cpu_native_payloads') else None),
        cpu_authority_charged_bytes=archive.get('cpu_authority_bytes'),
        cpu_reported_process_peak_bytes=archive.get('cpu_peak_bytes'),
        cpu_archive_lifetime_peak_bytes=archive.get('cpu_peak_bytes'),
        worker_rss_bytes=worker.get('rss_bytes'),
        worker_anon_rss_bytes=worker.get('anon_rss_bytes'),
        worker_file_rss_bytes=worker.get('file_rss_bytes'),
        worker_shmem_rss_bytes=worker.get('shmem_rss_bytes'),
        worker_locked_bytes=worker.get('locked_bytes'),
        worker_lifetime_peak_rss_bytes=worker.get('lifetime_peak_rss_bytes'),
        cgroup_current_bytes=cgroup.get('current_bytes'),
        cgroup_max_bytes=cgroup.get('max_bytes'),
        cgroup_anon_bytes=cgroup.get('anon_bytes'),
        cgroup_file_bytes=cgroup.get('file_bytes'),
        cgroup_shmem_bytes=cgroup.get('shmem_bytes'),
        cgroup_file_dirty_bytes=cgroup.get('file_dirty_bytes'),
        cgroup_file_writeback_bytes=cgroup.get('file_writeback_bytes'),
        cpu_history_reserved_bytes=scheduler.get('cpu_history_admission', {}).get('reserved_bytes'),
        cpu_history_waiting=scheduler.get('cpu_history_admission', {}).get('waiting'),
        archive_io_pending_bytes=archive.get('archive_io_pending_bytes'),
        archive_active_requests=archive.get('active_requests'),
        archive_retiring_requests=archive.get('retiring_requests'),
        ssd_used_bytes=archive.get('ssd_used_bytes'),
        ssd_file_bytes=archive.get('ssd_file_bytes'),
        disk_read_bytes=archive.get('disk_read_bytes'),
        disk_write_bytes=archive.get('disk_write_bytes'))


class PhaseSampler:
    """Bounded downsampled timeline; extrema use every observed snapshot."""
    def __init__(self, capture, *, interval=.05, max_samples=4096, clock=time.perf_counter):
        if (type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0 or
                type(max_samples) is not int or max_samples < 4):
            raise ValueError('Invalid phase sampler bounds')
        self.capture, self.interval, self.max_samples = Path(capture), interval, max_samples
        self.clock, self.started = clock, clock()
        self.samples, self.peaks, self.first, self.last = [], {}, {}, {}
        self.total_samples, self.stride = 0, 1
        self.errors = {'scheduler': 0, 'archive': 0}
        self.updates = {'scheduler': 0, 'archive': 0}
        self._versions = {}
        self.source_files = {}
        self._stopped = threading.Event()
        self._thread = None

    def observe(self, scheduler, archive):
        row = resource_sample(scheduler, archive, self.clock()-self.started)
        number = self.total_samples
        self.total_samples += 1
        for key, value in row.items():
            if key in ('elapsed_seconds', 'scheduler_at', 'archive_at') or value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid resource metric '+key)
            self.first.setdefault(key, value)
            self.last[key] = value
            self.peaks[key] = max(self.peaks.get(key, value), value)
        if number % self.stride == 0:
            self.samples.append(row)
            if len(self.samples) >= self.max_samples:
                self.samples = self.samples[::2]
                self.stride *= 2

    def sample(self):
        values = {}
        for key, filename in (('scheduler', 'scheduler_resources.json'), ('archive', 'archive.json')):
            try:
                if key == 'archive' and (self.capture/'archive_resources.json').is_file():
                    filename = 'archive_resources.json'
                path = self.capture/filename
                value = json.loads(path.read_text())
                version = (filename, path.stat().st_mtime_ns)
                if not isinstance(value, dict):
                    raise ValueError('Resource snapshot must be an object')
                self.source_files[key] = filename
                if self._versions.get(key) != version:
                    self.updates[key] += 1
                    self._versions[key] = version
                values[key] = value
            except (OSError, ValueError):
                self.errors[key] += 1
                values[key] = {}
        self.observe(values['scheduler'], values['archive'])

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Phase sampler already started')
        self.sample()
        def poll():
            while not self._stopped.wait(self.interval):
                try:
                    self.sample()
                except Exception as error:
                    self._failure = repr(error)
                    return
        self._thread = threading.Thread(target=poll, name='kvmem-phase-sampler', daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError('Phase sampler did not stop')
        self.sample()
        return self.snapshot()

    def snapshot(self):
        return dict(total_samples=self.total_samples, retained_samples=len(self.samples),
            stride=self.stride, interval_seconds=self.interval, max_samples=self.max_samples,
            source_updates=dict(self.updates), read_errors=dict(self.errors),
            source_files=dict(self.source_files),
            failure=getattr(self, '_failure', None), samples=list(self.samples),
            observed_phase_peaks=dict(self.peaks), first_observed=dict(self.first),
            last_observed=dict(self.last),
            disk_counter_deltas={key:self.last[key]-self.first[key] for key in
                ('disk_read_bytes', 'disk_write_bytes', 'checkpoint_disk_read_bytes',
                 'checkpoint_disk_write_bytes') if key in self.first},
            swap_counter_deltas={key:self.last[key]-self.first[key] for key in
                ('swap_pauses', 'swap_restores', 'swap_restore_deferrals') if key in self.first},
            scope='Sampled phase peaks are lower bounds at published snapshot cadence. cpu_reported_process_peak_bytes is the cumulative archive payload peak, not total process RAM. CPU resident payload excludes transfer, index, model and allocator memory. worker_rss_bytes covers the reporting worker only and omits unmapped file cache; worker_lifetime_peak_rss_bytes is not reset per phase. worker_locked_bytes is not complete CUDA pinned accounting. Cgroup counters cover all charged processes/file cache in that group, overlap RSS and must not be added to it. No total host-memory hard limit is enforced.')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def request(base, payload, barrier):
    req = urllib.request.Request(base+'/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    barrier.wait(timeout=60)
    start = time.perf_counter()
    ids, text_parts, first, usage, finished, reasons = [], [], None, None, False, []
    result = dict(started=start, input_digest=input_digest(payload))
    try:
        with opener.open(req, timeout=1200) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    finished = True
                    break
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise RuntimeError(chunk['error'])
                usage = chunk.get('usage') or usage
                for choice in chunk.get('choices', []):
                    tokens = choice.get('token_ids') or []
                    if tokens and first is None:
                        first = time.perf_counter()-start
                    ids.extend(tokens)
                    text_parts.append(choice.get('text') or '')
                    if choice.get('finish_reason') is not None:
                        reasons.append(choice['finish_reason'])
        if (not finished or not ids or any(type(token) is not int or token < 0 for token in ids) or
                not usage or first is None or
                usage.get('completion_tokens') != len(ids) or usage.get('prompt_tokens') != len(payload['prompt']) or
                len(ids) != payload['max_tokens']):
            raise RuntimeError('Incomplete stream, truncated output or public token accounting mismatch')
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=repr(error))
    finally:
        result.update(seconds=time.perf_counter()-start, ttft_seconds=first,
            output_ids=ids, output_text=''.join(text_parts), usage=usage,
            stream_done=finished, finish_reasons=reasons)
    if result['status'] != 'completed':
        raise RequestFailure(result['error'], result)
    return result


def summarize(responses):
    if not responses:
        raise ValueError('No completed responses')
    for response in responses:
        if (response.get('status', 'completed') != 'completed' or not response.get('output_ids') or
                any(type(token) is not int or token < 0 for token in response['output_ids']) or
                any(type(response.get(key)) not in (int, float) or
                    not math.isfinite(response[key]) for key in ('started', 'seconds', 'ttft_seconds')) or
                response['seconds'] <= 0 or not 0 <= response['ttft_seconds'] <= response['seconds']):
            raise ValueError('Invalid or incomplete public response')
        if response.get('usage', {}).get('completion_tokens') != len(response['output_ids']):
            raise ValueError('Public token usage mismatch')
    span = max(r['started']+r['seconds'] for r in responses)-min(r['started'] for r in responses)
    if span <= 0:
        raise ValueError('Invalid request timing')
    tokens = sum(len(r['output_ids']) for r in responses)
    return dict(requests=len(responses), output_tokens=tokens, wall_seconds=span,
                requests_per_second=len(responses)/span, output_tokens_per_second=tokens/span,
                ttft_median_seconds=statistics.median(r['ttft_seconds'] for r in responses),
                ttft_p95_seconds=percentile([r['ttft_seconds'] for r in responses], .95),
                latency_p50_seconds=percentile([r['seconds'] for r in responses], .5),
                latency_p95_seconds=percentile([r['seconds'] for r in responses], .95),
                latency_max_seconds=max(r['seconds'] for r in responses),
                observed_client_peak=client_peak(responses), timing_basis=TIMING_BASIS)


def semantic_profile(profile):
    value = json.loads(json.dumps(profile))
    value.pop('port', None)
    value.setdefault('verify_request_swap', False)
    for key in ('archive_backend', 'nvme_max_bytes', 'nvme_preallocate',
                'gpu_memory_admission', 'request_checkpoint_bytes', 'request_checkpoint_ssd_bytes',
                'request_swap_min_resident_steps'):
        value.get('kvmem', {}).pop(key, None)
    return value


def arm_profile(config, backend, *, disk_bytes, checkpoint_bytes, port,
                checkpoint_ssd_bytes=4*1024**3):
    if backend not in ('cpu', 'tiered', 'incremental', 'baseline', 'ssd'):
        raise ValueError('Unknown benchmark arm')
    profile = json.loads(json.dumps(config))
    if backend == 'baseline' and profile.pop('verify_request_swap', False):
        raise ValueError('The frozen baseline does not implement request swap byte verification')
    profile['port'] = port
    kv = profile['kvmem']
    incremental = backend in ('incremental', 'baseline', 'ssd')
    kv.update(archive_backend='tiered' if incremental else backend,
              nvme_max_bytes=disk_bytes if backend != 'cpu' else 0, nvme_preallocate=False,
              gpu_memory_admission='incremental' if incremental else 'static',
              request_checkpoint_bytes=(kv.get('request_checkpoint_bytes') or checkpoint_bytes) if incremental else 0,
              request_swap_min_resident_steps=kv.get('request_swap_min_resident_steps', 32))
    # The frozen older parser must not receive even a zero-valued new option.
    kv.pop('request_checkpoint_ssd_bytes', None)
    if backend == 'ssd':
        kv['request_checkpoint_ssd_bytes'] = checkpoint_ssd_bytes
    return profile


def _compare_outputs(left_index, left, right_index, right, phase_index):
    a, b = left['phases'][phase_index], right['phases'][phase_index]
    mismatches = []
    for owner, (x, y) in enumerate(zip(a['responses'], b['responses'])):
        xt, yt = x['output_ids'], y['output_ids']
        text_equal = x.get('output_text') == y.get('output_text') if (
            'output_text' in x and 'output_text' in y) else None
        if xt != yt or text_equal is False:
            first = next((i for i, (u, v) in enumerate(zip(xt, yt)) if u != v),
                         min(len(xt), len(yt)) if len(xt) != len(yt) else None)
            mismatches.append(dict(owner=owner, first_different_token=first,
                                   left_tokens=len(xt), right_tokens=len(yt), text_equal=text_equal))
    return dict(left_arm=left_index, right_arm=right_index, phase_index=phase_index,
                concurrency=a['concurrency'], requests=len(a['responses']),
                matching_outputs=len(a['responses'])-len(mismatches),
                exact=not mismatches, mismatches=mismatches)


def analyze(arms, plan=None):
    """Pure validation and comparison; never turns output drift into a pass."""
    plan = plan or {}
    issues, valid = [], []
    if any(key not in plan for key in ('arms', 'levels', 'prompt_tokens', 'output_tokens')):
        issues.append('Saved plan must declare arms, levels and complete input/output counts')
    expected_levels = plan.get('levels')
    if plan.get('arms') is not None and [a.get('backend') for a in arms] != plan['arms']:
        issues.append('Arm sequence is incomplete or differs from the saved plan')
    if not arms:
        issues.append('No experiment arms')
    profiles = [semantic_profile(a['profile']) for a in arms if 'profile' in a]
    if len(profiles) != len(arms):
        issues.append('Missing per-arm profiles; semantic equality is unverified')
    if profiles and any(p != profiles[0] for p in profiles[1:]):
        issues.append('Profiles differ outside storage/admission/checkpoint controls and port')
    diagnostics = [arm.get('diagnostics', {}) for arm in arms]
    if diagnostics and any(item != diagnostics[0] for item in diagnostics[1:]):
        issues.append('Diagnostic settings differ between arms')
    measured_shape = None
    resource_rows = []
    for arm_index, arm in enumerate(arms):
        if arm.get('status') != 'completed':
            issues.append(f'Arm {arm_index} did not complete: {arm.get("error", arm.get("status"))}')
            continue
        phases = arm.get('phases', [])
        levels = [p.get('concurrency') for p in phases if not p.get('warmup')]
        if not phases or (expected_levels is not None and levels != expected_levels):
            issues.append(f'Arm {arm_index} has incomplete measurement phases')
            continue
        shape = [(p.get('concurrency'), bool(p.get('warmup'))) for p in phases]
        if measured_shape is None:
            measured_shape = shape
        if shape != measured_shape:
            issues.append(f'Arm {arm_index} phase order differs')
            continue
        good = True
        for phase_index, phase in enumerate(phases):
            responses = phase.get('responses', [])
            if phase.get('status', 'completed') != 'completed' or len(responses) != phase.get('concurrency'):
                issues.append(f'Arm {arm_index} phase {phase_index} is incomplete')
                good = False
                continue
            try:
                summary = summarize(responses)
                for owner, response in enumerate(responses):
                    if ('output_tokens' in plan and len(response['output_ids']) != plan['output_tokens'] or
                            'prompt_tokens' in plan and response['usage'].get('prompt_tokens') != plan['prompt_tokens']):
                        raise ValueError('Requested public token counts differ')
                    if 'input_digest' in response and 'prompt_tokens' in plan:
                        expected = input_digest({'prompt': [1000+owner]*plan['prompt_tokens']})
                        if response['input_digest'] != expected:
                            raise ValueError('Input tokens or owner order differ')
                prior = phase.get('summary', {})
                for key in ('wall_seconds', 'output_tokens', 'requests', 'output_tokens_per_second'):
                    if key in prior and not math.isclose(prior[key], summary[key], rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError('Stored timing/token summary differs from complete responses')
            except (ValueError, KeyError, TypeError) as error:
                issues.append(f'Arm {arm_index} phase {phase_index}: {error}')
                good = False
            if not phase.get('warmup'):
                resources = phase.get('resources', {})
                peaks = resources.get('observed_phase_peaks', {})
                cpu_phase_observed = peaks.get('cpu_resident_bytes', 0) > 0
                cpu_lifetime_observed = peaks.get('cpu_archive_lifetime_peak_bytes',
                    peaks.get('cpu_reported_process_peak_bytes', 0)) > 0
                cpu_evidence = ('sampled_phase_residency' if cpu_phase_observed else
                    'archive_lifetime_peak_only' if cpu_lifetime_observed else 'missing_nonzero_payload')
                available = (resources.get('failure') is None and
                    peaks.get('simultaneous_gpu_owners', 0) > 0 and
                    (cpu_phase_observed or cpu_lifetime_observed) and peaks.get('gpu_owned_pages', 0) > 0)
                resource_rows.append(dict(arm=arm_index, phase=phase_index,
                    concurrency=phase['concurrency'], complete=available,
                    cpu_evidence=cpu_evidence, cpu_phase_peak_observed=cpu_phase_observed,
                    observed_phase_peaks=peaks, source_updates=resources.get('source_updates', {}),
                    source_files=resources.get('source_files', {}),
                    failure=resources.get('failure'),
                    disk_counter_deltas=resources.get('disk_counter_deltas', {}),
                    swap_counter_deltas=resources.get('swap_counter_deltas', {})))
        if good:
            valid.append((arm_index, arm))
    within, cross = [], []
    for (i, a), (j, b) in combinations(valid, 2):
        for phase_index, phase in enumerate(a['phases']):
            if not phase.get('warmup'):
                row = _compare_outputs(i, a, j, b, phase_index)
                (within if a['backend'] == b['backend'] else cross).append(row)
    repeatability = {}
    backends = tuple(dict.fromkeys(plan.get('arms', [a.get('backend') for a in arms])))
    for backend in backends:
        comparisons = [row for row in within if arms[row['left_arm']]['backend'] == backend]
        repeatability[backend] = ('unmeasured' if not comparisons else
                                 'stable' if all(row['exact'] for row in comparisons) else 'unstable')
    if not cross:
        output_status = 'unmeasured'
    elif all(row['exact'] for row in cross):
        output_status = 'exact'
    elif 'unstable' in repeatability.values():
        output_status = 'inconclusive_within_backend_variation'
    elif 'unmeasured' in repeatability.values():
        output_status = 'inconclusive_missing_repeats'
    else:
        output_status = 'mismatch_with_stable_repeats'
    performance = []
    for backend, level in product(backends, sorted({p['concurrency'] for _, a in valid
            for p in a['phases'] if not p.get('warmup')})):
        phases = [p for _, a in valid if a['backend'] == backend for p in a['phases']
                  if not p.get('warmup') and p['concurrency'] == level]
        summaries = [summarize(p['responses']) for p in phases]
        if summaries:
            wall = sum(s['wall_seconds'] for s in summaries)
            performance.append(dict(backend=backend, concurrency=level, repeats=len(summaries),
                weighted_output_tokens_per_second=sum(s['output_tokens'] for s in summaries)/wall,
                weighted_requests_per_second=sum(s['requests'] for s in summaries)/wall,
                output_tokens_per_second_samples=[s['output_tokens_per_second'] for s in summaries],
                ttft_median_seconds_samples=[s['ttft_median_seconds'] for s in summaries],
                latency_p95_seconds_samples=[s['latency_p95_seconds'] for s in summaries]))
    complete_resources = bool(resource_rows) and all(r['complete'] for r in resource_rows)
    status = ('failed' if issues else 'output_mismatch' if output_status == 'mismatch_with_stable_repeats' else
              'passed' if output_status == 'exact' and all(s == 'stable' for s in repeatability.values())
              and complete_resources else 'needs_review')
    return dict(status=status, issues=issues, timing_basis=TIMING_BASIS,
        output_comparison=dict(status=output_status, repeatability=repeatability,
            within_backend=within, cross_backend=cross,
            interpretation='Within-backend variation prevents attributing cross-backend drift to storage alone; it never proves storage equivalence.'),
        resource_evidence=dict(complete=complete_resources, phases=resource_rows,
            all_cpu_phase_peaks_observed=bool(resource_rows) and all(
                row['cpu_phase_peak_observed'] for row in resource_rows),
            scope='Observed live owners/pages and positive payload evidence; client concurrency is not GPU residency. Lifetime-only CPU peaks are explicitly labelled and cannot reconstruct a per-phase peak. Zero residency with no positive lifetime peak is missing evidence, not proof of zero use. Sampled phase peaks include retirement drain and are lower bounds.'),
        performance=performance,
        qualification='Synthetic HTTP experiment only; no task-quality claim. GPU residency comes from sampled page owners, and active swap claims require observed swap transactions plus separate state-equivalence tests.')


def analyze_directory(directory):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text())
    results = json.loads((directory/'results.json').read_text())
    arms = results['arms']
    for index, arm in enumerate(arms):
        profile = directory/f'{index+1:02d}-{arm["backend"]}'/'profile.json'
        if profile.exists():
            arm['profile'] = json.loads(profile.read_text())
    analysis = analyze(arms, plan)
    save(directory/'analysis.json', analysis)
    return analysis


def wait_ready(server, base, timeout=600):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('Server exited during startup; inspect server.log')
        try:
            with opener.open(base+'/health', timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            pass
        time.sleep(.5)
    raise TimeoutError('Server startup timed out')


def settle(capture, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            archive = json.loads((capture/'archive.json').read_text())
            sched = json.loads((capture/'scheduler_resources.json').read_text())
            if not any(archive.get(k, 0) for k in (
                    'active_requests', 'retiring_requests', 'pending_downloads')) and not sched['requests']:
                return dict(archive=archive, scheduler=sched)
        except (OSError, ValueError):
            pass
        time.sleep(.05)
    raise TimeoutError('Request retirement did not drain')


def run(args):
    if (type(args.sample_interval) not in (int, float) or not math.isfinite(args.sample_interval) or
            args.sample_interval <= 0 or type(args.max_samples) is not int or args.max_samples < 4):
        raise ValueError('Invalid phase sampling settings')
    cfg = json.loads(args.profile.read_text())
    kv = cfg['kvmem']
    if set(args.arms) & {'incremental', 'baseline', 'ssd'} and (type(cfg.get('kv_cache_memory_bytes')) is not int or
                                      cfg['kv_cache_memory_bytes'] <= 0):
        raise ValueError('Incremental comparisons require the same explicit kv_cache_memory_bytes in every arm')
    if not kv.get('cpu_history_coordination') or not kv.get('request_planning'):
        raise ValueError('Profile must explicitly enable history coordination and request planning')
    if (kv.get('historical_gc') or kv.get('repository_gc') or kv.get('cpu_direct_record_seal') or
            cfg.get('full_history_benchmark')):
        raise ValueError('Use a profile supported by both CPU and tiered archives')
    if (max(args.levels) > cfg['max_num_seqs'] or args.prompt_tokens+args.output_tokens > cfg['max_model_len']
            or min(args.levels) < 1 or args.output_tokens < 1 or args.prompt_tokens < 128):
        raise ValueError('Workload exceeds profile or has invalid counts')
    cancel_stress = getattr(args, 'cancel_stress', False)
    session_stress = getattr(args, 'session_stress', False)
    baseline_runtime = getattr(args, 'baseline_runtime', None)
    if 'baseline' in args.arms and (baseline_runtime is None or
            not (baseline_runtime/'src/vllm_kvmem').is_dir()):
        raise ValueError('Baseline arms require a frozen --baseline-runtime containing src/vllm_kvmem')
    if (cancel_stress or session_stress) and (not set(args.arms) & {'incremental', 'ssd'} or cfg['max_num_seqs'] < 4):
        raise ValueError('Cancellation stress requires an incremental arm and four request slots')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output/'runtime'
    shutil.copytree(ROOT/'src', runtime/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (runtime/'scripts').mkdir()
    shutil.copy2(Path(__file__), runtime/'scripts'/Path(__file__).name)
    if cancel_stress or session_stress:
        shutil.copy2(ROOT/'scripts/stress_request_swaps.py', runtime/'scripts/stress_request_swaps.py')
    if session_stress:
        shutil.copy2(ROOT/'scripts/qualify_session_swaps.py', runtime/'scripts/qualify_session_swaps.py')
    manifest = {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in runtime.rglob('*') if p.is_file()}
    save(args.output/'source.json', manifest)
    if 'baseline' in args.arms:
        frozen_baseline = args.output/'runtime-baseline'
        shutil.copytree(baseline_runtime/'src', frozen_baseline/'src',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        save(args.output/'baseline-source.json', {
            str(p.relative_to(frozen_baseline)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen_baseline.rglob('*') if p.is_file()})
    save(args.output/'profile.json', cfg)
    diagnostics = dict(request_swap_bytes=bool(cfg.get('verify_request_swap')),
        session_restore=bool(cfg.get('verify_session_restore')),
        session_restore_digest=bool(cfg.get('verify_session_restore_digest')))
    plan = dict(arms=args.arms, levels=args.levels, prompt_tokens=args.prompt_tokens,
                output_tokens=args.output_tokens, cpu_bytes=kv['cpu_read_cache_bytes'],
                disk_bytes=args.disk_bytes, semantic_settings=kv,
                checkpoint_bytes=kv.get('request_checkpoint_bytes') or args.checkpoint_bytes,
                checkpoint_ssd_bytes=getattr(args, 'checkpoint_ssd_bytes', 4*1024**3),
                baseline_runtime=str(baseline_runtime) if baseline_runtime else None,
                diagnostics=diagnostics,
                kv_cache_memory_bytes=cfg.get('kv_cache_memory_bytes'),
                cancel_stress_last_incremental=cancel_stress,
                session_stress_last_incremental=session_stress,
                session_workload=getattr(args, 'session_workload', 'code') if session_stress else None,
                sampling=dict(interval_seconds=args.sample_interval, max_samples=args.max_samples),
                scope='Synthetic HTTP capacity and timings; output and resource evidence determine qualification; no task-quality claim.')
    save(args.output/'plan.json', plan)
    results = []
    for index, backend in enumerate(args.arms):
        arm = args.output/f'{index+1:02d}-{backend}'
        arm.mkdir()
        profile = arm_profile(cfg, backend, disk_bytes=args.disk_bytes,
                              checkpoint_bytes=args.checkpoint_bytes, port=args.port,
                              checkpoint_ssd_bytes=getattr(args, 'checkpoint_ssd_bytes', 4*1024**3))
        save(arm/'profile.json', profile)
        command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config',
                   str(arm/'profile.json'), '--run-dir', str(arm/'server')]
        arm_runtime = frozen_baseline if backend == 'baseline' else runtime
        env = dict(os.environ, PYTHONPATH=str(arm_runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
                   MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', KVMEM_LIFECYCLE_TRACE='1')
        save(arm/'command.json', command)
        base = f'http://127.0.0.1:{args.port}'
        report = dict(backend=backend, status='starting', phases=[], profile=profile,
                      source_manifest='baseline-source.json' if backend == 'baseline' else 'source.json',
                      diagnostics=diagnostics)
        server = None
        try:
            with (arm/'server.log').open('w') as log:
                server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            wait_ready(server, base)
            print(json.dumps(dict(arm=arm.name, status='ready')), flush=True)
            for count in [1, *args.levels]:
                phase = len(report['phases'])
                warmup = phase == 0
                barrier = threading.Barrier(count)
                payloads = [dict(model=profile['served_model_name'],
                    prompt=([1000+owner]*args.prompt_tokens),
                    kvmem_query_span=[args.prompt_tokens-128, args.prompt_tokens],
                    temperature=0, seed=104729, max_tokens=args.output_tokens,
                    ignore_eos=True, return_token_ids=True, stream=True,
                    stream_options={'include_usage':True}) for owner in range(count)]
                phase_dir = arm/f'phase-{phase:02d}'
                save(phase_dir/'requests.json', payloads)
                item = dict(concurrency=count, warmup=warmup, status='running',
                            responses=[None]*count, input_file=str(phase_dir/'requests.json'))
                report['phases'].append(item)
                report['status'] = 'running'
                save(arm/'results.json', report)
                sampler = PhaseSampler(arm/'server/capture', interval=args.sample_interval,
                                       max_samples=args.max_samples)
                failures = []
                sampler.start()
                try:
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        futures = {pool.submit(request, base, p, barrier):owner for owner, p in enumerate(payloads)}
                        for future in as_completed(futures):
                            owner = futures[future]
                            try:
                                response = future.result()
                            except RequestFailure as error:
                                response = error.response
                                failures.append(dict(owner=owner, error=str(error)))
                            except Exception as error:
                                response = dict(status='failed', error=repr(error), output_ids=[])
                                failures.append(dict(owner=owner, error=repr(error)))
                            response['owner'] = owner
                            item['responses'][owner] = response
                            save(phase_dir/f'response-{owner:04d}.json', response)
                            save(arm/'results.json', report)
                    if failures:
                        raise RuntimeError('One or more public requests failed: '+repr(failures))
                    item['settled'] = settle(arm/'server/capture')
                    item['summary'] = summarize(item['responses'])
                    item['status'] = 'completed'
                except BaseException as error:
                    item.update(status='failed', error=repr(error))
                    raise
                finally:
                    item['resources'] = sampler.stop()
                    save(phase_dir/'resources.json', item['resources'])
                    save(arm/'results.json', report)
                print(json.dumps(dict(arm=arm.name, concurrency=count, warmup=warmup,
                                      **item['summary'])), flush=True)
            last_dynamic = backend in ('incremental', 'ssd') and not set(args.arms[index+1:]) & {'incremental', 'ssd'}
            if cancel_stress and last_dynamic:
                stress_output = arm/'cancel-stress'
                subprocess.run([sys.executable, '-B', str(runtime/'scripts/stress_request_swaps.py'),
                    '--url', base, '--capture-dir', str(arm/'server/capture'),
                    '--output', str(stress_output)], env=env, check=True, timeout=660)
                report['cancel_stress'] = json.loads((stress_output/'result.json').read_text())
            if session_stress and last_dynamic:
                stress_output = arm/'session-stress'
                subprocess.run([sys.executable, '-B', str(runtime/'scripts/qualify_session_swaps.py'),
                    '--url', base, '--probe', str(arm/'server/capture'),
                    '--model', profile['served_model_name'], '--output', str(stress_output),
                    '--timeout', '900', '--workload', getattr(args, 'session_workload', 'code')],
                    env=env, check=True, timeout=960)
                report['session_stress'] = json.loads((stress_output/'result.json').read_text())
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            save(arm/'results.json', report)
            if server is not None:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            results.append(report)
            save(args.output/'results.json', dict(arms=results))
            save(args.output/'analysis.json', analyze(results, plan))
    return analyze_directory(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--analyze', type=Path, help='Analyze saved evidence without starting a model')
    parser.add_argument('--arms', nargs='+', choices=('cpu','tiered','incremental','baseline','ssd'), default=['cpu','tiered','tiered','cpu'])
    parser.add_argument('--baseline-runtime', type=Path,
        help='Prior frozen runtime for matched baseline/current throughput comparisons')
    parser.add_argument('--levels', nargs='+', type=int, default=[1,2,4])
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--disk-bytes', type=int, default=8*1024**3)
    parser.add_argument('--checkpoint-bytes', type=int, default=2*1024**3)
    parser.add_argument('--checkpoint-ssd-bytes', type=int, default=4*1024**3)
    parser.add_argument('--port', type=int, default=19216)
    parser.add_argument('--sample-interval', type=float, default=.05)
    parser.add_argument('--max-samples', type=int, default=4096)
    parser.add_argument('--cancel-stress', action='store_true',
        help='Run finite live HTTP cancellation qualification after the final incremental arm')
    parser.add_argument('--session-stress', action='store_true',
        help='Run session restore, paused cancellation and explicit release qualification after the final dynamic arm')
    parser.add_argument('--session-workload', choices=('code', 'repeat'), default='code',
        help='Session qualification input; exact output gates are identical for both workloads')
    arguments = parser.parse_args()
    if arguments.analyze:
        result = analyze_directory(arguments.analyze.resolve())
    else:
        if arguments.profile is None or arguments.output is None:
            parser.error('--profile and --output are required unless --analyze is used')
        arguments.profile = arguments.profile.resolve()
        arguments.output = arguments.output.resolve()
        result = run(arguments)
    print(json.dumps(dict(analysis_status=result['status'],
                         output_comparison=result['output_comparison']['status'])), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(2)
