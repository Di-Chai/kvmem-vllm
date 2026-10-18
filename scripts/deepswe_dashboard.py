"""Local, read-only DeepSWE dashboard backed by the existing benchmark monitors."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

PAGE = Path(__file__).resolve().parents[1] / 'benchmarks/deepswe/dashboard/index.html'
PROMPT = 'vllm:prompt_tokens_total'
CACHED = 'vllm:prompt_tokens_cached_total'
OUTPUT = 'vllm:generation_tokens_total'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class JsonLines:
    """Incremental reads; an unfinished final line remains unread until published."""
    def __init__(self, path, limit=2200):
        self.path, self.offset, self.identity = path, 0, None
        self.rows = deque(maxlen=limit)

    def update(self):
        added = []
        try:
            stat = self.path.stat()
            if self.identity != stat.st_ino or stat.st_size < self.offset:
                self.offset, self.identity = 0, stat.st_ino
                self.rows.clear()
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                while line := stream.readline():
                    if not line.endswith(b'\n'):
                        break
                    self.offset = stream.tell()
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    self.rows.append(row)
                    added.append(row)
        except OSError:
            pass
        return added


def delta_rate(before, after, keys):
    seconds = after['at'] - before['at']
    if not 0 < seconds <= 45:
        return None
    changes = []
    for key, sign in keys:
        a, b = before.get('counters', {}).get(key), after.get('counters', {}).get(key)
        if a is None or b is None or b < a:
            return None  # Missing metrics or a restarted counter are not zero throughput.
        changes.append(sign * (b - a))
    total = sum(changes)
    return total / seconds if total >= 0 else None


def throughput_series(rows, formal_start, phase='running'):
    result, previous = [], None
    for row in rows:
        if (not formal_start or row.get('run_phase') != phase
                or row.get('at', 0) < formal_start or 'error' in row):
            previous = None
            continue
        counters = row.get('counters', {})
        point = dict(at=row['at'], prefill=None, decode=None,
                     running=counters.get('vllm:num_requests_running'),
                     waiting=counters.get('vllm:num_requests_waiting'))
        if previous:
            point['prefill'] = delta_rate(previous, row, [(PROMPT, 1), (CACHED, -1)])
            point['decode'] = delta_rate(previous, row, [(OUTPUT, 1)])
        result.append(point)
        previous = row
    return result


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.metrics = JsonLines(root / 'native-metrics.jsonl')
        self.events = JsonLines(root / 'bridge/requests.jsonl', limit=1)
        self.activity = {}
        self.lock, self.cached_at, self.cached = threading.Lock(), 0, None

    def snapshot(self):
        with self.lock:
            now = time.time()
            if self.cached and now - self.cached_at < 1:
                return self.cached
            root = self.root
            manifest = read_json(root / 'manifest.json')
            run = read_json(root / 'run.json')
            status = read_json(root / 'status.json')
            summary = read_json(root / 'live-summary.json')
            resources = read_json(root / 'server/capture/scheduler_resources.json')
            self.metrics.update()
            for event in self.events.update():
                task = event.get('task')
                if not task:
                    continue
                activity = self.activity.setdefault(task, {})
                kind = event.get('event')
                if kind == 'start':
                    activity.update(pending=True, at=event['at'], turn=event.get('turn'),
                                    prompt_tokens=event.get('prompt_tokens'))
                elif kind in ('finish', 'error', 'cancelled', 'history_released'):
                    activity.update(pending=False, at=event['at'])
            formal_start = run.get('model_ready_at')
            measurement_phase = 'formal' if formal_start else 'preflight'
            series = throughput_series(self.metrics.rows, formal_start or run.get('started_at'),
                                       'running' if formal_start else 'starting')
            latest = series[-1] if series else {}
            metric_age = now - latest['at'] if latest else None
            state = run.get('status', status.get('status', 'starting'))
            active = state in ('starting', 'running', 'prepared', 'waiting_for_gpu')
            stale = active and bool(formal_start or series) and (metric_age is None or metric_age > 35)
            statistics = {r['task']: r for r in summary.get('tasks', [])}
            tasks = []
            for task in manifest.get('tasks', []):
                tid = task['task_id']
                record = run.get('tasks', {}).get(tid, {})
                stats = statistics.get(tid, {})
                case = read_json(root / 'cases' / tid / 'status.json')
                activity = self.activity.get(tid, {})
                task_state = record.get('state', 'queued')
                if task_state == 'running':
                    if case.get('state') in ('verifying', 'finished'):
                        task_state = 'verifying'
                    else:
                        task_state = 'model' if activity.get('pending') else 'tools'
                if not active and task_state in ('model', 'tools', 'verifying'):
                    task_state = 'interrupted'
                elapsed = stats.get('total_seconds') or stats.get('agent_seconds')
                if elapsed is None and record.get('started_at'):
                    elapsed = max(0, run.get('ended_at', now) - record['started_at'])
                tasks.append(dict(id=tid, title=task.get('title', tid), state=task_state,
                    reward=stats.get('reward'), requests=stats.get('requests', 0),
                    output_tokens=stats.get('output_tokens', 0),
                    context_tokens=activity.get('prompt_tokens', stats.get('last_prompt_tokens')),
                    elapsed_seconds=elapsed,
                    pending_seconds=max(0, now-activity['at']) if activity.get('pending') and active else None,
                    termination=stats.get('termination')))
            admission = resources.get('cpu_history_admission', {})
            profile = manifest.get('profile', {})
            preflight = dict(query=bool(read_json(root/'atomic-query-preflight.json').get('passed')),
                short_tail=bool(read_json(root/'short-tail-preflight.json').get('passed')),
                generation_completed=sum(bool(read_json(path).get('cleanup_passed'))
                    for path in root.glob('generation-preflight-*.json')))
            self.cached = dict(at=now, run_name=root.name, state=state,
                measurement_phase=measurement_phase, preflight=preflight,
                formal_start=formal_start, ended_at=run.get('ended_at'),
                elapsed_seconds=max(0, run.get('ended_at', now)-formal_start) if formal_start else 0,
                workers=run.get('workers', manifest.get('workers', 8)),
                planned_tasks=len(tasks), completed_tasks=summary.get('completed_tasks', 0),
                passed_tasks=summary.get('passed_tasks', 0),
                infrastructure_errors=summary.get('infrastructure_errors', 0),
                metric_at=latest.get('at'), metric_age_seconds=metric_age, stale=stale,
                summary_age_seconds=now-summary['at'] if summary.get('at') else None,
                control_age_seconds=now-status['at'] if status.get('at') else None,
                latest=latest, series=series, tasks=tasks,
                gpu=status.get('gpu', {}), gpu_at=status.get('at'), cpu=admission,
                cpu_used_bytes=summary.get('cpu_authority_bytes'),
                refresh_pending=resources.get('refresh_pending'),
                restore_pending=resources.get('connector_pending'),
                output_tokens=summary.get('output_tokens'),
                requests=summary.get('completed_model_requests'),
                e2e_output_rate=summary.get('mixed_end_to_end_output_tokens_per_second'),
                single_stream_decode_rate=summary.get('request_weighted_decode_tokens_per_second'),
                ttft_seconds=summary.get('mean_native_ttft_seconds'),
                restored_tokens=summary.get('native_session_restored_tokens'),
                disk_read_bytes=summary.get('disk_read_bytes'),
                disk_write_bytes=summary.get('disk_write_bytes'),
                model=dict(draft_tokens=profile.get('draft_tokens'),
                    select_budget=profile.get('kvmem', {}).get('select_budget'),
                    gen_budget=profile.get('kvmem', {}).get('gen_budget')))
            self.cached_at = now
            return self.cached


def serve(root, port):
    dashboard = Dashboard(root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                body, content_type = PAGE.read_bytes(), 'text/html; charset=utf-8'
            elif path == '/api/status':
                body = json.dumps(dashboard.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            elif path == '/health':
                body, content_type = b'{"ok":true}', 'application/json'
            elif path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Local: http://localhost:{server.server_address[1]}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--port', type=int, default=50123)
    args = parser.parse_args()
    if not any((args.root / name).is_file() for name in ('run.json', 'manifest.json')):
        parser.error('root must contain a DeepSWE run.json or prepared manifest.json')
    serve(args.root.resolve(), args.port)
