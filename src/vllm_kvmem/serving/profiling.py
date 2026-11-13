"""Opt-in native profiler diagnostics, explicitly separate from performance runs."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import urllib.request


def native_profiler_config(directory, steps, delay=0):
    if type(steps) is not int or not 1 <= steps <= 32 or type(delay) is not int or delay < 0:
        raise ValueError('Profiler requires 1..32 active steps and a nonnegative delay')
    return dict(profiler='torch', torch_profiler_dir=str(Path(directory).resolve()),
                ignore_frontend=True, delay_iterations=delay, max_iterations=steps,
                torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
                detailed_trace_annotation=True)


def control_profiler(url, action):
    if action not in ('start', 'stop'):
        raise ValueError('Unknown native profiler action')
    # Native endpoints return an empty HTTP 200 body, not JSON.
    request = urllib.request.Request(url + '/' + action + '_profile', data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
        if response.status != 200:
            raise RuntimeError('Native profiler control did not succeed')


def summarize_trace(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        trace = json.load(stream)
    kernels = defaultdict(lambda: [0, 0.0])
    intervals = defaultdict(list)
    for event in trace.get('traceEvents', []):
        if event.get('ph') != 'X' or event.get('cat') != 'kernel':
            continue
        duration = float(event['dur'])
        if duration < 0:
            raise ValueError('Negative GPU kernel duration')
        kernels[event['name']][0] += 1
        kernels[event['name']][1] += duration
        intervals[event['pid']].append((float(event['ts']), float(event['ts']) + duration))
    if not kernels:
        raise ValueError('Profiler trace contains no GPU kernels; requested window was not exercised')
    timelines = []
    for device, ranges in intervals.items():
        ranges.sort()
        begin, end = ranges[0]
        busy = 0.0
        for next_begin, next_end in ranges[1:]:
            if next_begin > end:
                busy += end - begin
                begin, end = next_begin, next_end
            else:
                end = max(end, next_end)
        busy += end - begin
        span = max(b for _, b in ranges) - ranges[0][0]
        timelines.append(dict(trace_device=device, kernel_interval_union_us=busy,
                              first_to_last_kernel_us=span, non_kernel_interval_us=span-busy))
    ordered = sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)
    return dict(path=str(path.resolve()), performance_eligible=False,
                kernel_count=sum(row[0] for row in kernels.values()),
                summed_kernel_us=sum(row[1] for row in kernels.values()),
                timelines=timelines,
                top_kernels=[dict(name=name, calls=row[0], summed_us=row[1]) for name, row in ordered[:20]],
                limitations='Instrumented window only. Summed durations overlap across streams; '
                            'interval gaps are not proof of GPU idleness or a particular host bottleneck.')
