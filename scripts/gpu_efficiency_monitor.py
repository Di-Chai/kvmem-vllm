"""Timestamped device GPU telemetry; utilization is not process-attributed.

Examples:
  python scripts/gpu_efficiency_monitor.py sample --output gpu.jsonl --owned-pid 123
  python scripts/gpu_efficiency_monitor.py summarize --input gpu.jsonl --start 10 --end 20

An unavailable metric is null. Failed queries remain in the journal and reduce
summary coverage instead of being converted into zero utilization.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import threading
import time


DEVICE_FIELDS = ('index', 'uuid', 'utilization.gpu', 'utilization.memory',
                 'memory.used', 'memory.total', 'power.draw')
DEVICE_KEYS = ('index', 'uuid', 'gpu_utilization_percent', 'memory_utilization_percent',
               'memory_used_mib', 'memory_total_mib', 'power_watts')
PROCESS_FIELDS = ('gpu_uuid', 'pid', 'used_gpu_memory')


def number(value, integer=False):
    value = value.strip()
    if value in ('', 'N/A', '[N/A]', 'Not Supported', '[Not Supported]'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite GPU metric')
    if integer and not result.is_integer():
        raise ValueError('Non-integer GPU identifier')
    return int(result) if integer else result


def query_rows(kind, fields, runner=subprocess.run):
    result = runner(['nvidia-smi', '--query-' + kind + '=' + ','.join(fields),
                     '--format=csv,noheader,nounits'], capture_output=True,
                    text=True, timeout=5, check=True)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(fields):
            raise ValueError('Unexpected nvidia-smi column count')
        rows.append([value.strip() for value in row])
    return rows


def snapshot(owned_pids=(), runner=subprocess.run):
    """Collect both GPU device metrics and process attribution evidence."""
    owned = set(int(pid) for pid in owned_pids)
    row = dict(started_at=time.time(), devices=[], processes=[], errors={},
               owned_pids=sorted(owned), utilization_scope='whole-device')
    begin = time.monotonic()
    try:
        for values in query_rows('gpu', DEVICE_FIELDS, runner):
            device = dict(zip(DEVICE_KEYS, values))
            device['index'] = number(device['index'], integer=True)
            for key in DEVICE_KEYS[2:]:
                device[key] = number(device[key])
            row['devices'].append(device)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        row['devices'] = []
        row['errors']['devices'] = type(exc).__name__ + ': ' + str(exc)
    try:
        for gpu_uuid, pid, memory in query_rows('compute-apps', PROCESS_FIELDS, runner):
            pid = number(pid, integer=True)
            if pid is None:
                raise ValueError('Missing GPU process ID')
            row['processes'].append(dict(gpu_uuid=gpu_uuid, pid=pid,
                memory_used_mib=number(memory), owned=pid in owned))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        row['processes'] = []
        row['errors']['processes'] = type(exc).__name__ + ': ' + str(exc)
    row['external_pids'] = (None if 'processes' in row['errors'] else
                           sorted({p['pid'] for p in row['processes'] if not p['owned']}))
    for device in row['devices']:
        processes = [p for p in row['processes'] if p['gpu_uuid'] == device['uuid']]
        owned_memory = [p['memory_used_mib'] for p in processes if p['owned']]
        device['owned_memory_mib'] = (None if 'processes' in row['errors'] or
                                     any(value is None for value in owned_memory)
                                     else sum(owned_memory))
        device['external_pids'] = (None if 'processes' in row['errors'] else
                                  sorted({p['pid'] for p in processes if not p['owned']}))
    row['at'] = time.time()
    row['sample_seconds'] = time.monotonic() - begin
    return row


def process_tree_pids(roots):
    """Find descendants using /proc, avoiding an additional runtime dependency."""
    if not roots:
        return set()
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            # The command name in parentheses may contain spaces or parentheses.
            fields = path.read_text().rsplit(')', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {pid for pid in roots if pid in parents}
    while True:
        added = {pid for pid, parent in parents.items() if parent in found} - found
        if not added:
            return found
        found.update(added)


def summarize(rows, start=None, end=None, max_hold_seconds=2.5):
    """Time-weight samples over an explicit interval, showing missing coverage.

    Samples are held only until the next journal row or max_hold_seconds. A
    failed next sample therefore stops coverage immediately; long gaps do not
    get filled with an old utilization value.
    """
    rows = sorted(rows, key=lambda row: row['at'])
    if not rows:
        raise ValueError('No GPU samples')
    start = rows[0]['at'] if start is None else start
    end = rows[-1]['at'] if end is None else end
    if (not all(math.isfinite(value) for value in (start, end, max_hold_seconds)) or
            end <= start or max_hold_seconds <= 0):
        raise ValueError('A positive measurement interval and hold limit are required')
    totals = {}
    errors = []
    external = set()
    attribution_seconds = 0.0
    for index, row in enumerate(rows):
        next_at = rows[index + 1]['at'] if index + 1 < len(rows) else end
        lo = max(start, row['at'])
        hi = min(end, next_at, row['at'] + max_hold_seconds)
        if hi <= lo:
            continue
        elapsed = hi - lo
        if row.get('errors'):
            errors.append(dict(at=row['at'], errors=row['errors']))
        if row.get('external_pids') is not None:
            external.update(row['external_pids'])
            attribution_seconds += elapsed
        for device in row['devices']:
            target = totals.setdefault(device['uuid'], dict(index=device['index'], metrics={}))
            for key in (*DEVICE_KEYS[2:], 'owned_memory_mib'):
                value = device.get(key)
                if value is None:
                    continue
                metric = target['metrics'].setdefault(key, dict(weighted_sum=0., coverage_seconds=0.,
                    minimum=value, maximum=value, samples=0))
                metric['weighted_sum'] += elapsed * value
                metric['coverage_seconds'] += elapsed
                metric['minimum'] = min(metric['minimum'], value)
                metric['maximum'] = max(metric['maximum'], value)
                metric['samples'] += 1
    for device in totals.values():
        for metric in device['metrics'].values():
            metric['mean'] = metric.pop('weighted_sum') / metric['coverage_seconds']
            metric['coverage_fraction'] = metric['coverage_seconds'] / (end - start)
    return dict(start=start, end=end, seconds=end-start, utilization_scope='whole-device',
                max_hold_seconds=max_hold_seconds, devices=totals, errors=errors,
                external_pids=sorted(external), attribution_coverage_seconds=attribution_seconds,
                attribution_coverage_fraction=attribution_seconds / (end-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sample = commands.add_parser('sample')
    sample.add_argument('--output', type=Path, required=True)
    sample.add_argument('--interval', type=float, default=1.0)
    sample.add_argument('--duration', type=float)
    sample.add_argument('--owned-pid', type=int, action='append', default=[])
    sample.add_argument('--owned-root-pid', type=int, action='append', default=[])
    summary = commands.add_parser('summarize')
    summary.add_argument('--input', type=Path, required=True)
    summary.add_argument('--output', type=Path)
    summary.add_argument('--start', type=float)
    summary.add_argument('--end', type=float)
    summary.add_argument('--max-hold-seconds', type=float, default=2.5)
    args = parser.parse_args()
    if args.command == 'summarize':
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        result = summarize(rows, args.start, args.end, args.max_hold_seconds)
        value = json.dumps(result, indent=2) + '\n'
        if args.output:
            args.output.write_text(value)
        else:
            print(value, end='')
        return
    if (not math.isfinite(args.interval) or args.interval <= 0 or
            (args.duration is not None and
             (not math.isfinite(args.duration) or args.duration <= 0))):
        parser.error('interval and duration must be positive')
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started
    with args.output.open('x', buffering=1) as output:
        while not stop.is_set():
            if args.duration is not None and time.monotonic()-started >= args.duration:
                break
            pids = set(args.owned_pid) | process_tree_pids(args.owned_root_pid)
            row = snapshot(pids)
            row['interval_seconds'] = args.interval
            output.write(json.dumps(row, allow_nan=False) + '\n')
            deadline += args.interval
            if deadline < time.monotonic():
                deadline = time.monotonic()
            remaining = deadline-time.monotonic()
            if args.duration is not None:
                remaining = min(remaining, started+args.duration-time.monotonic())
            stop.wait(max(0, remaining))


if __name__ == '__main__':
    main()
