"""Read-only live Terminal-Bench observer; never sends inference/control requests."""
import argparse
from collections import Counter, deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.request import ProxyHandler, build_opener

try:
    from .deepswe_dashboard import JsonLines
except ImportError:  # Direct script entry point.
    from deepswe_dashboard import JsonLines

PAGE = Path(__file__).resolve().parents[1] / 'benchmarks/terminalbench/dashboard/index.html'
DEFAULT_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/terminalbench-policy-36k16k-c8-20261029-r1')
GENERATION = 'vllm:generation_tokens_total'
RUNNING = 'vllm:num_requests_running'
WAITING = 'vllm:num_requests_waiting'
STALE_SECONDS = 12


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def task_name(value):
    return value.removeprefix('terminal-bench/') if isinstance(value, str) else ''


def single_metric(raw, name):
    """vLLM counts len(EngineCoreOutput.new_token_ids), after draft rejection.

    Read ONLY the accepted generation counter, never prompt or MTP forward work.
    This campaign has one engine (TP=1); refuse unexpected multiple series rather
    than risk double counting an unverified topology.
    """
    values = []
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if parts and parts[0].split('{', 1)[0] == name:
            try:
                value = float(parts[1])
                if number(value) and value >= 0:
                    values.append(value)
            except (ValueError, IndexError):
                pass
    return values[0] if len(values) == 1 else None


def generation_counter(raw):
    return single_metric(raw, GENERATION)


def parse_metrics(raw):
    return dict(generation=single_metric(raw, GENERATION),
                running=single_metric(raw, RUNNING), waiting=single_metric(raw, WAITING))


def fetch_metrics():
    # Explicit loopback, no ambient proxy, no model or control endpoint.
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:18882/metrics', timeout=1.5) as response:
        return parse_metrics(response.read(2_000_000).decode('utf-8'))


def reconfiguration(root, now):
    row=read_json(root/'user-reconfiguration.json')
    if (row.get('reason')!='user_requested_efficiency_optimization' or
            not number(row.get('at')) or not 0 < row['at'] <= now):
        return None
    return dict(at=row['at'],reason=row['reason'],status=row.get('status','stopping'))


class CounterWindow:
    def __init__(self):
        self.rows = deque(maxlen=240)
        self.identity = None

    def add(self, at, value, identity):
        if not number(at) or not number(value) or value < 0:
            return
        if (self.identity != identity or (self.rows and
                (at < self.rows[-1][0] or value < self.rows[-1][1] or at-self.rows[-1][0] > STALE_SECONDS))):
            self.rows.clear()
        self.identity = identity
        if not self.rows or at > self.rows[-1][0]:
            self.rows.append((at, value))

    def view(self, now, active):
        if not self.rows:
            return None, None, 0, True
        latest = self.rows[-1]
        stale = not active or now-latest[0] > STALE_SECONDS
        rows = [row for row in self.rows if row[0] >= latest[0]-60]
        seconds = latest[0]-rows[0][0]
        rate = (latest[1]-rows[0][1])/seconds if seconds > 0 and not stale else None
        return rate, latest[0], seconds, stale


class GaugeWindow:
    """Observed model-request concurrency, weighted by elapsed time, not count."""
    def __init__(self):
        self.rows=deque(maxlen=240)
        self.identity=None

    def add(self, at, running, waiting, identity):
        if not all(number(value) and value>=0 for value in (at,running,waiting)):
            return
        if (identity!=self.identity or (self.rows and
                (at<self.rows[-1][0] or at-self.rows[-1][0]>STALE_SECONDS))):
            self.rows.clear()
        self.identity=identity
        if not self.rows or at>self.rows[-1][0]:
            self.rows.append((at,running,waiting))

    def view(self, now, active):
        result=dict(running=None,waiting=None,running_mean_60s=None,waiting_mean_60s=None,
                    sample_at=None,sample_window_seconds=0,stale=True)
        if not self.rows or not active or now-self.rows[-1][0]>STALE_SECONDS:
            return result
        weighted_running=weighted_waiting=seconds=0
        for index,row in enumerate(self.rows):
            end=self.rows[index+1][0] if index+1<len(self.rows) else now
            elapsed=max(0,min(end,row[0]+STALE_SECONDS)-max(row[0],now-60))
            weighted_running+=row[1]*elapsed
            weighted_waiting+=row[2]*elapsed
            seconds+=elapsed
        result.update(running=self.rows[-1][1],waiting=self.rows[-1][2],
            running_mean_60s=weighted_running/seconds if seconds else None,
            waiting_mean_60s=weighted_waiting/seconds if seconds else None,
            sample_at=self.rows[-1][0],sample_window_seconds=seconds,stale=False)
        return result


def gpu_view(rows, now, start, active):
    valid = [row for row in rows if start and row.get('at', 0) >= start and
             row.get('at', 0) >= now-60 and row.get('at', 0) <= now and
             not row.get('errors') and row.get('devices')]
    if not valid or not active or now-valid[-1]['at'] > STALE_SECONDS:
        return dict(utilization_percent_60s=None, memory_used_mib=None, sample_at=None, stale=True)
    weighted = duration = 0
    for index, row in enumerate(valid):
        values = [d.get('gpu_utilization_percent') for d in row['devices']]
        if not values or not all(number(value) for value in values):
            continue
        end = valid[index+1]['at'] if index+1 < len(valid) else now
        seconds = max(0, min(end, row['at']+2.5)-row['at'])
        weighted += sum(values)/len(values)*seconds
        duration += seconds
    memory = [device.get('memory_used_mib') for device in valid[-1]['devices']]
    return dict(utilization_percent_60s=weighted/duration if duration else None,
        memory_used_mib=sum(memory) if all(number(value) for value in memory) else None,
        sample_at=valid[-1]['at'], stale=False, observed_seconds=duration,
        scope='whole-device utilization; existing one-second sampler')


class Arm:
    def __init__(self, root, arm_id):
        self.root, self.id = root, arm_id
        self.manifest = read_json(root/'manifest.json')
        self.profile = read_json(root/'profile.json')
        self.events = JsonLines(root/'formal-events/events.jsonl', limit=1)
        self.bridge = JsonLines(root/'bridge/requests.jsonl', limit=1)
        self.gpu = JsonLines(root/'gpu-utilization.jsonl', limit=300, initial_tail_bytes=1_000_000)
        self.tasks = {name:dict(name=name, phase='queued', passed=None, reward=None,
            elapsed_seconds=None, output_tokens=None, gc_retired=None, exception=None)
            for name in self.manifest.get('launched_task_names', [])}
        self.owners, self.outputs, self.requests = {}, Counter(), set()
        self.generation, self.prefill = CounterWindow(), CounterWindow()
        self.engine=GaugeWindow()
        self.series = deque(maxlen=900)

    def sample(self, now, active, metrics_fetcher, stopped=None):
        status = read_json(self.root/'status.json')
        formal = read_json(self.root/'formal-events/status.json')
        interruption = read_json(self.root/'user-interruption.json')
        user_cancelled = set(interruption.get('cancelled_trials', [])) if (
            interruption.get('reason') == 'user_requested_switch_to_lru') else set()
        boundary = read_json(self.root/'formal-boundary.json')
        start = boundary.get('at')
        phase = status.get('status', 'prepared')
        live = active and not stopped and phase == 'formal' and number(start) and now >= start
        stop_at=stopped['at'] if stopped else None
        trial_names = {task.get('trial'):name for name, task in self.tasks.items() if task.get('trial')}
        for row in self.events.update():
            name = task_name(row.get('task')) or trial_names.get(row.get('trial'), '')
            event, at = row.get('event'), row.get('at')
            if name not in self.tasks or not number(at) or not start or at < start:
                continue
            task = self.tasks[name]
            if row.get('trial'):
                task['trial'] = row['trial']
                trial_names[row['trial']] = name
            if event == 'start':
                task.update(phase='setup', started_at=at)
            elif event == 'waiting_for_memory':
                task['phase'] = 'waiting_memory'
                task['memory_wait'] = {key:row[key] for key in ('requested','reserved','available')
                                       if number(row.get(key))}
            elif event == 'environment-start':
                task['phase'] = 'setup'
                task.pop('memory_wait', None)
            elif event == 'agent-start':
                task['phase'] = 'agent'
            elif event in ('agent-end', 'verification-start'):
                task['phase'] = 'verifying'
            elif event == 'end':
                task['ended_at'] = at
        trials = {row.get('trial')+'__agent':name for name, row in self.tasks.items() if row.get('trial')}
        for row in self.bridge.update():
            if row.get('event') == 'session_opened':
                self.owners[row.get('owner')] = row.get('session_id')
            elif (start and row.get('at', 0) >= start and type(row.get('output_tokens')) is int
                  and row.get('request_id') and row['request_id'] not in self.requests):
                self.requests.add(row['request_id'])
                self.outputs[row.get('owner')] += row['output_tokens']
        by_task = Counter()
        for owner, tokens in self.outputs.items():
            if self.owners.get(owner) in trials:
                by_task[trials[self.owners[owner]]] += tokens
        for name, task in self.tasks.items():
            task['output_tokens'] = by_task[name] if task['phase'] not in ('queued','not_started') else None
        outcomes = {task_name(row.get('task')):row for row in formal.get('outcomes', [])}
        for name, row in outcomes.items():
            if name not in self.tasks:
                continue
            reward = (row.get('rewards') or {}).get('reward')
            exception = row.get('exception')
            result_phase = ('timeout' if row.get('status') == 'agent_timeout' else
                'error' if row.get('status') == 'error' else 'passed' if row.get('passed') else 'failed')
            if row.get('trial') in user_cancelled and exception == 'CancelledError':
                result_phase = 'interrupted'
                self.tasks[name]['interruption_reason'] = 'user_requested_switch_to_lru'
            ended=self.tasks[name].get('ended_at')
            if stopped and ((number(ended) and ended>=stop_at) or
                    (ended is None and exception in ('CancelledError','asyncio.CancelledError'))):
                result_phase='interrupted'
            self.tasks[name].update(phase=result_phase, passed=row.get('passed'), reward=reward,
                exception=exception if isinstance(exception, str) else None)
        for task in self.tasks.values():
            if stopped and task['phase']=='queued':
                task['phase']='not_started'
            if stopped and task['phase'] in ('waiting_memory','setup','agent','verifying'):
                task['phase']='interrupted'
            if task['phase']=='interrupted':
                if task.get('interruption_reason') == 'user_requested_switch_to_lru':
                    task.update(passed=None,reward=None)
                else:
                    task.update(passed=None,reward=None,exception=None,
                        interruption_reason='user_requested_efficiency_optimization')
            if number(task.get('started_at')):
                end = task.get('ended_at', now)
                if stopped:end=min(end,stop_at)
                task['elapsed_seconds'] = max(0, end-task['started_at'])
        counts = Counter(row['phase'] for row in self.tasks.values())
        completed = sum(counts[key] for key in ('passed','failed','error','timeout'))
        if user_cancelled:
            completed += counts['interrupted']
        count_view = dict(expected=len(self.tasks), completed=completed, passed=counts['passed'],
            failed=counts['failed'], errors=counts['error']+counts['timeout'], timeouts=counts['timeout'],
            active=counts['waiting_memory']+counts['setup']+counts['agent']+counts['verifying'],
            executing=counts['setup']+counts['agent']+counts['verifying'],
            waiting_memory=counts['waiting_memory'], setup=counts['setup'], agents=counts['agent'],
            verifying=counts['verifying'], queued=counts['queued']+counts['not_started'],interrupted=counts['interrupted'])
        capture = self.root/'model/server/capture'
        if live:
            work = read_json(capture/'public_work.json')
            identity = (self.id, work.get('counter_id'))
            try:
                metrics = metrics_fetcher()
            except Exception:
                metrics = {}
            if not isinstance(metrics,dict):
                metrics=dict(generation=metrics)  # Legacy injected counter readers.
            self.generation.add(now, metrics.get('generation'), identity)
            self.engine.add(now,metrics.get('running'),metrics.get('waiting'),identity)
            at, prefill = work.get('wall_time'), work.get('completed', {}).get('public_prefill_tokens')
            if number(at) and at >= start and now-at <= STALE_SECONDS:
                self.prefill.add(at, prefill, identity)
        rate, sample_at, seconds, stale = self.generation.view(now, live)
        prefill_rate, prefill_at, prefill_seconds, prefill_stale = self.prefill.view(now, live)
        engine=self.engine.view(now,live)
        self.gpu.update()
        gpu = gpu_view(self.gpu.rows, now, start, live)
        progress = read_json(capture/'progress.json')
        current_gc = progress.get('historical_gc', {})
        before_gc = boundary.get('capture', {}).get('archive', {}).get('historical_gc', {})
        gc = {}
        for key in ('retired','redundancy_retired','lru_fallback_retired'):
            before, current = before_gc.get(key), current_gc.get(key)
            gc[key] = current-before if number(before) and number(current) and current >= before else None
        gc['scope'] = 'formal policy-retired blocks; excludes warmup/preflight and ordinary session cleanup'
        if live:
            self.series.append(dict(at=now, generation_tps=rate, prefill_tps=prefill_rate,
                gpu_percent=gpu['utilization_percent_60s'], agents=counts['agent'],
                engine_running=engine['running'],engine_waiting=engine['waiting']))
        end = max((row.get('ended_at', start or now) for row in self.tasks.values()), default=now)
        if stopped:
            phase='stopped' if start else 'not_started'
            end=stop_at
        return dict(id=self.id, label='Thinking 去重 + LRU' if self.id=='thinking' else '纯 LRU',
            phase=phase, counts=count_view, started_at=start,engine=engine,
            results_scope='stopped_run_partial_results' if stopped else 'current_campaign',
            elapsed_seconds=max(0,(now if live else end)-start) if number(start) else None,
            throughput=dict(generation_tps_60s=rate, prefill_tps_60s=prefill_rate,
                output_tokens=sum(by_task.values()) if start else None,
                scope='formal only; accepted native generation, including one-token session-release controls; observer sampling window',
                output_tokens_scope='formal completed-response ledger, excludes in-flight responses and cleanup controls',
                prefill_scope='executed public target prefill positions, including re-prefill; excludes private Query and MTP verification',
                sample_at=sample_at, stale=stale, sample_window_seconds=seconds,
                prefill_sample_at=prefill_at, prefill_stale=prefill_stale,
                prefill_sample_window_seconds=prefill_seconds), gpu=gpu, gc=gc,
            series=list(self.series), tasks=list(self.tasks.values()))


class Dashboard:
    def __init__(self, root, metrics_fetcher=fetch_metrics, clock=time.time):
        self.root, self.metrics_fetcher, self.clock = Path(root), metrics_fetcher, clock
        self.plan = read_json(self.root/'experiment-plan.json')
        self.arms = [Arm(self.root/arm, arm) for arm in ('thinking', 'lru')]
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.cached = dict(generated_at=None, campaign=dict(status='loading'), arms=[])

    def sample(self):
        now = self.clock()
        pair = read_json(self.root/'pair-status.json')
        stopped=reconfiguration(self.root,now)
        active_arm = pair.get('arm') if pair.get('status') == 'running' and not stopped else None
        arms = [arm.sample(now, active_arm==arm.id, self.metrics_fetcher,stopped) for arm in self.arms]
        budgets = self.plan.get('budgets', {})
        timing_modes={arm.manifest.get('agent_time_limit') for arm in self.arms}
        timing_mode=self.plan.get('agent_time_limit')
        if timing_mode is None and len(timing_modes)==1:
            timing_mode=next(iter(timing_modes))
        task_limits={arm.manifest.get('task_concurrency',arm.manifest.get('max_concurrency')) for arm in self.arms}
        model_limits={arm.manifest.get('model_concurrency',arm.profile.get('max_num_seqs')) for arm in self.arms}
        task_concurrency=self.plan.get('task_concurrency',self.plan.get('concurrency'))
        if task_concurrency is None and len(task_limits)==1:task_concurrency=next(iter(task_limits))
        model_concurrency=self.plan.get('model_concurrency')
        if model_concurrency is None and len(model_limits)==1:model_concurrency=next(iter(model_limits))
        value = dict(generated_at=now, campaign=dict(status='efficiency_optimization' if stopped else pair.get('status','prepared'),
            active_arm=active_arm, started_at=pair.get('started_at'),
            stopped_at=stopped['at'] if stopped else None,
            results_scope='stopped_run_partial_results' if stopped else 'current_campaign',
            total_tasks=sum(arm['counts']['expected'] for arm in arms),
            completed=sum(arm['counts']['completed'] for arm in arms),
            passed=sum(arm['counts']['passed'] for arm in arms),
            max_concurrency=task_concurrency, config=dict(k=budgets.get('K'),g=budgets.get('G'),
                cpu_history_per_session=budgets.get('CPU_history_tokens_per_session'),
                logical_max_tokens=budgets.get('logical_max_tokens'),
                shared_cpu_cache_bytes=budgets.get('CPU_shared_cache_bytes'),
                max_concurrency=task_concurrency,task_concurrency=task_concurrency,
                model_concurrency=model_concurrency,agent_time_limit=timing_mode)), arms=arms)
        worker=read_json(self.root/'worker-status.json')
        if worker:
            value['campaign']['remote_worker']={k:worker.get(k) for k in
                ('host','preserved','restarted','new','remote_total','remote_completed','last_sync_at','container_admission_bytes')}
        # Enforce a JSON-safe allowlisted presentation object before publication.
        json.dumps(value, allow_nan=False)
        with self.lock:
            self.cached = value
        return value

    def snapshot(self):
        with self.lock:
            value = deepcopy(self.cached)
        at = value.get('generated_at')
        value['stale'] = not number(at) or self.clock()-at > STALE_SECONDS
        worker=value.get('campaign',{}).get('remote_worker')
        if worker and value['campaign'].get('status')!='completed' and (
                not number(worker.get('last_sync_at')) or self.clock()-worker['last_sync_at']>75):
            value['stale']=True
        if value['stale']:
            for arm in value['arms']:
                arm['throughput'].update(stale=True, generation_tps_60s=None, prefill_tps_60s=None)
                arm['gpu'].update(stale=True, utilization_percent_60s=None)
                arm['engine'].update(stale=True,running=None,waiting=None,running_mean_60s=None,waiting_mean_60s=None)
        return value

    def run(self, interval=2):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception:
                # Last successful view remains available, with age/stale marker.
                pass
            self.stop.wait(interval)


def make_handler(dashboard, page=PAGE):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/api/status':
                payload = json.dumps(dashboard.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/':
                try:
                    payload = page.read_bytes()
                except OSError:
                    self.send_error(503, 'Dashboard page is not ready')
                    return
                content_type = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18400)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second')
    dashboard = Dashboard(args.root)
    thread = threading.Thread(target=dashboard.run, args=(args.interval,), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host,args.port), make_handler(dashboard))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
