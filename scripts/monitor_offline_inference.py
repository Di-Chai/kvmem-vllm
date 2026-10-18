"""Read-only GPU/prefill/accepted-output telemetry alongside a frozen campaign."""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.request
from zoneinfo import ZoneInfo


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class JsonlCursor:
    def __init__(self, path):
        self.path, self.offset, self.pending = path, 0, b''

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open('rb') as stream:
            if self.path.stat().st_size < self.offset:
                raise RuntimeError('Telemetry source was truncated: '+str(self.path))
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.pending+chunk).split(b'\n')
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line.strip()]


class CounterWindow:
    def __init__(self):
        self.points, self.current, self.resets = deque(), None, 0

    def observe(self, at, value):
        self.current = value
        if value is None:
            return
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid telemetry counter')
        if self.points and value < self.points[-1][1]:
            self.points.clear()
            self.resets += 1
        if self.points and at <= self.points[-1][0]:
            raise ValueError('Non-increasing telemetry time')
        self.points.append((at,value))
        while len(self.points)>2 and self.points[1][0] < at-120:
            self.points.popleft()

    def rate(self, seconds):
        if self.current is None or len(self.points)<2:
            return None, 0
        end,total = self.points[-1]
        anchor = self.points[0]
        for point in self.points:
            if point[0] > end-seconds:
                break
            anchor = point
        elapsed = end-anchor[0]
        return ((total-anchor[1])/elapsed,elapsed) if elapsed>0 else (None,0)


METRICS = {
    'vllm:generation_tokens_total':'accepted_output_total',
    'vllm:num_requests_running':'native_running',
    'vllm:num_requests_waiting':'native_waiting',
}


def parse_metrics(body, model):
    result = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        series,_,number = line.partition(' ')
        name = series.partition('{')[0]
        if name not in METRICS:
            continue
        label = re.search(r'model_name=("(?:\\.|[^"\\])*")',series)
        if label is None or json.loads(label.group(1)) != model:
            continue
        value = float(number.split()[0])
        if math.isfinite(value):
            key = METRICS[name]
            result[key] = result.get(key,0)+value
    return result


def summarize_steps(rows, begin, end):
    rows = [r for r in rows if begin <= r['started_at'] < end]
    kinds = Counter('mixed' if r['prefill_requests'] and r['decode_requests'] else
                    'prefill' if r['prefill_requests'] else
                    'decode' if r['decode_requests'] else 'maintenance' for r in rows)
    return dict(steps=len(rows),step_kinds=dict(kinds),
                max_decode_batch=max((r['decode_requests'] for r in rows),default=0),
                max_prefill_batch=max((r['prefill_requests'] for r in rows),default=0),
                maintenance_host_ms=sum(r.get('maintenance_host_ms',0) for r in rows),
                model_step_ms_p50=statistics.median(r['model_ms'] for r in rows) if rows else None)


def run(args):
    root, output = args.campaign.resolve(), args.output.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    begin = previous = time.monotonic()
    save(output/'manifest.json',dict(campaign=str(root),started_wall_time=time.time(),
        observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval_seconds=args.interval,source_git_commit=manifest['source_git_commit'],
        definitions=dict(prefill='Completed public target prefill tokens per wall second; excludes restored KV and private Query forwards.',
            decode='vLLM accepted output tokens per wall second, including prefill/maintenance interference; not pure GPU decode.',
            gpu='Existing campaign NVML utilization samples; no additional GPU workload.',
            windows='1, 10 and 60 seconds; actual observed spans are recorded during warmup.',
            model_ms='CUDA-event span including submission effects, not hardware GPU-busy time.'),
        limitations=['Monitoring begins now; past accepted-output rates are not reconstructed.',
                     'Missing or stale data is null, not zero.',
                     'Native running count is not actual decode batch size.']))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    current = None
    latest_resource = None
    cursors = {}
    windows = {}
    epoch = None
    reason = 'duration_limit'
    fields = ['local_time','arm','phase','gpu_utilization','gpu_free_mib','gpu_power_w',
              'prefill_tps_1s','prefill_tps_10s','prefill_tps_60s',
              'decode_tps_1s','decode_tps_10s','decode_tps_60s',
              'native_running','native_waiting','outer_running','outer_retiring','outer_waiting',
              'max_decode_batch','max_prefill_batch','completed_cases']
    with (output/'samples.jsonl').open('x') as raw, (output/'samples.csv').open('x',newline='') as table:
        writer = csv.DictWriter(table,fieldnames=fields)
        writer.writeheader()
        due = begin
        while not args.duration or time.monotonic()-begin < args.duration:
            campaign = read_json(root/'status.json') or {}
            if campaign.get('status') in ('completed','failed'):
                reason = 'campaign_'+campaign['status']
                break
            states = {arm:read_json(root/arm/'status.json') or {} for arm in manifest['arms']}
            arm = next((a for a in manifest['arms'] if states[a].get('status') in ('loading','testing')),None)
            arm_changed = arm != current
            if arm_changed:
                current, latest_resource, epoch = arm, None, None
                windows = {name:CounterWindow() for name in ('prefill','decode')}
                cursors = {} if arm is None else dict(
                    resources=JsonlCursor(root/(arm+'-resources.jsonl')),
                    steps=JsonlCursor(root/arm/'server/capture/model_step_latency.jsonl'))
            metrics, error, public, queue, progress = {}, None, None, {}, {}
            step_rows = []
            if arm is not None:
                base = root/arm
                resource_rows = cursors['resources'].read()
                if resource_rows:
                    latest_resource = resource_rows[-1]
                step_rows = cursors['steps'].read()
                public = read_json(base/'server/capture/public_work.json')
                queue = read_json(base/'qualification/queue.json') or {}
                progress = read_json(base/'qualification/progress.json') or {}
                if states[arm].get('status') == 'testing':
                    profile = json.loads((base/'profile.json').read_text())
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/metrics",timeout=2) as response:
                            metrics = parse_metrics(response.read().decode(),profile['served_model_name'])
                    except (OSError,ValueError) as failure:
                        error = type(failure).__name__+': '+str(failure)
            at = time.monotonic()
            row = dict(at=at,wall_time=time.time(),arm=arm,
                       phase=states.get(arm,{}).get('status','between_arms'),metrics_error=error)
            row['local_time'] = datetime.fromtimestamp(row['wall_time'],ZoneInfo('Asia/Shanghai')).isoformat()
            resource_age = at-latest_resource['at'] if latest_resource else None
            row['gpu_source_age_seconds'] = resource_age
            for key in ('gpu_utilization','gpu_free_mib','gpu_power_w'):
                row[key] = latest_resource.get(key) if resource_age is not None and resource_age<3 else None
            public_age = at-public['at'] if public else None
            row['prefill_source_age_seconds'] = public_age
            source_epoch = (arm,public.get('counter_id')) if public else None
            if source_epoch is not None and source_epoch != epoch:
                epoch = source_epoch
                windows = {name:CounterWindow() for name in ('prefill','decode')}
            prefill = public['completed']['public_prefill_tokens'] if public_age is not None and public_age<3 else None
            row['prefill_completed_total'],row['accepted_output_total'] = prefill,metrics.get('accepted_output_total')
            for name,value in [('prefill',prefill),('decode',metrics.get('accepted_output_total'))]:
                if name not in windows:
                    windows[name] = CounterWindow()
                windows[name].observe(at,value)
                for horizon in (1,10,60):
                    rate,span = windows[name].rate(horizon)
                    row[f'{name}_tps_{horizon}s'] = rate
                    row[f'{name}_span_{horizon}s'] = span
                row[name+'_counter_resets'] = windows[name].resets
            row.update({k:metrics.get(k) for k in ('native_running','native_waiting')})
            row.update({f'outer_{stage}':queue.get('counts',{}).get(stage,0) if queue else None
                        for stage in ('running','retiring','waiting')})
            row['completed_cases'] = progress.get('completed',0) if queue else None
            # Async CUDA-event reports can arrive after their start interval.
            # Keep all newly observed completed reports; discard historical
            # rows only when first attaching to an already-running arm.
            row.update(summarize_steps(step_rows,previous if arm_changed else -float('inf'),at))
            row['step_summary_scope'] = 'Newly observed completed step reports; publication may lag execution.'
            raw.write(json.dumps(row,allow_nan=False)+'\n')
            raw.flush()
            writer.writerow({k:row.get(k) for k in fields})
            table.flush()
            save(output/'latest.json',row)
            previous = at
            due += args.interval
            time.sleep(max(0,due-time.monotonic()))
    save(output/'status.json',dict(status='stopped',reason=reason,at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=1.)
    parser.add_argument('--duration',type=float,default=0.,help='Seconds; zero follows the campaign to completion.')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval<.5 or not math.isfinite(args.duration) or args.duration<0:
        parser.error('Use an interval >=0.5 seconds and a nonnegative duration.')
    run(args)


if __name__ == '__main__':
    main()
