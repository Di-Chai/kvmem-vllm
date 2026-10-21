"""Summarize Nsight SQLite activities without adding overlapping durations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3


def merged(spans):
    result = []
    for begin, end in sorted((int(a), int(b)) for a, b in spans if b > a):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def duration(spans):
    return sum(b-a for a,b in merged(spans))


def kernel_category(name):
    if any(part in name for part in ('_row_lse', '_segment_scores', '_clear_lse', '_map_pieces')):
        return 'native_stats_auxiliary'
    if '_segment_stats' in name or '_page_attention' in name:
        return 'reference_attention_statistics'
    if 'BatchPrefillWithPagedKVCacheKernel' in name or 'xqa' in name.lower():
        return 'attention_including_any_fused_stats'
    if 'capture_pack' in name or re.search(r'(^|\W)_(header|payload)(\W|$)', name):
        return 'named_capture_pack'
    if any(part in name for part in ('_scatter', '_rotate', '_assemble', '_move_resident')):
        return 'named_kv_assembly'
    # Generic cat/copy/pointwise kernels are shared by the model and capture.
    # Never label all copies as archive or all elementwise kernels as model.
    if any(part in name.lower() for part in ('gemm', 'cutlass', 'matmul', 'fmha', 'attention')):
        return 'other_named_model_kernel'
    return 'unattributed_kernel'


def rows(connection, name, tables):
    if name not in tables:
        return []
    return [dict(row) for row in connection.execute('SELECT * FROM "' + name + '"')]


def process(row):
    value = row.get('globalPid', row.get('globalTid'))
    return int(value) // (1 << 24) if value is not None else None


def text_name(row, strings):
    for key in ('demangledName', 'shortName', 'nameId', 'name', 'textId', 'text'):
        value = row.get(key)
        if value is not None:
            return strings.get(value, str(value))
    return '<unknown>'


def analyze(path):
    connection = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = {r['id']:r['value'] for r in rows(connection, 'StringIds', tables)}
        apis = []
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
            for row in rows(connection, table, tables):
                if row.get('end', 0) <= row.get('start', 0):
                    continue
                row.update(name=text_name(row, strings), process=process(row))
                apis.append(row)
        launches = defaultdict(list)
        for row in apis:
            if any(s in row['name'] for s in ('Launch', 'Memcpy', 'Memset')) and 'correlationId' in row:
                launches[row['process'], row['correlationId']].append(row)
        activities = []
        for kind, table in [('kernel', 'CUPTI_ACTIVITY_KIND_KERNEL'),
                            ('memcpy', 'CUPTI_ACTIVITY_KIND_MEMCPY'),
                            ('memset', 'CUPTI_ACTIVITY_KIND_MEMSET')]:
            for row in rows(connection, table, tables):
                if row.get('end', 0) <= row.get('start', 0):
                    continue
                name = text_name(row, strings) if kind == 'kernel' else kind + ':' + str(row.get('copyKind', ''))
                candidates = launches.get((process(row), row.get('correlationId')), [])
                eligible = [api for api in candidates if api['start'] <= row['start']]
                api = min(eligible, key=lambda value:value['start']) if eligible else None
                row.update(kind=kind, name=name, category=kernel_category(name) if kind == 'kernel' else kind,
                           launch=api, process=process(row))
                activities.append(row)
        if not activities:
            raise ValueError('No CUDA activities in trace; this is not evidence of zero GPU work')
        devices = sorted({r.get('deviceId', 0) for r in activities})
        report = dict(diagnostic_only=True, devices={}, limitations=[
            'Intervals describe traced process activities, not hardware utilization or unprofiled throughput.',
            'Category sums and CPU API durations overlap; only all-activity union is additive against its enclosing span.',
            'Fused attention side-output cost cannot be separated from attention by kernel names.',
            'Generic copies/pointwise kernels remain unattributed; memcpy is not assumed to be archive traffic.',
            'Launch gaps indicate when the next observed activity API was entered, not a proof of CPU root cause.',
            'NVTX sk is logical scheduled context. Active context equality must also be checked against window offsets.'])
        for device in devices:
            data = [r for r in activities if r.get('deviceId', 0) == device]
            union = merged((r['start'],r['end']) for r in data)
            begin, end = union[0][0], union[-1][1]
            categories = defaultdict(list)
            names = defaultdict(list)
            for row in data:
                categories[row['category']].append(row)
                names[row['name']].append(row)
            def summarize(group):
                return dict(count=len(group), summed_ms=sum(r['end']-r['start'] for r in group)/1e6,
                    union_ms=duration((r['start'],r['end']) for r in group)/1e6,
                    bytes=sum(r.get('bytes', 0) for r in group))
            gaps = []
            by_start = defaultdict(list)
            for row in data:
                by_start[row['start']].append(row)
            for (_, left), (right, _) in zip(union, union[1:]):
                next_rows = by_start[right]
                known = [r['launch'] for r in next_rows if r['launch'] is not None]
                first = min(known, key=lambda r:r['start']) if known else None
                not_entered = max(0, min(right, first['start'])-left) if first else None
                gaps.append(dict(begin_ns=left, end_ns=right, duration_ms=(right-left)/1e6,
                    next_kernels=[r['name'] for r in next_rows],
                    next_launch_api=None if first is None else first['name'],
                    next_launch_entered_ns=None if first is None else first['start'],
                    next_launch_not_entered_ms=None if not_entered is None else not_entered/1e6,
                    classification=('unknown_correlation' if first is None else
                        'next_launch_entered_before_gap' if first['start'] <= left else 'next_launch_entered_during_gap')))
            report['devices'][str(device)] = dict(begin_ns=begin, end_ns=end,
                observed_span_ms=(end-begin)/1e6, activity_union_ms=duration(union)/1e6,
                gaps_ms=sum(gap['duration_ms'] for gap in gaps),
                gap_count=len(gaps), correlated_activities=sum(r['launch'] is not None for r in data),
                categories={k:summarize(v) for k,v in categories.items()},
                top_kernels=sorted((dict(name=k, **summarize(v)) for k,v in names.items()),
                                   key=lambda r:r['summed_ms'], reverse=True)[:50],
                next_launch_not_entered_ms=sum(g['next_launch_not_entered_ms'] or 0 for g in gaps),
                largest_gaps=sorted(gaps, key=lambda r:r['duration_ms'], reverse=True)[:100])
        api_groups = defaultdict(list)
        for row in apis:
            api_groups[row['name']].append(row)
        report['cuda_api'] = sorted((dict(name=name, count=len(group),
            summed_ms=sum(r['end']-r['start'] for r in group)/1e6,
            union_ms=duration((r['start'],r['end']) for r in group)/1e6)
            for name,group in api_groups.items()), key=lambda r:r['summed_ms'], reverse=True)[:50]
        nvtx = []
        pattern = re.compile(r'execute_(\d+)_context_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)_generation_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)')
        for row in rows(connection, 'NVTX_EVENTS', tables):
            name = row.get('text') or strings.get(row.get('textId'), '')
            match = pattern.fullmatch(name)
            if match and row.get('end', 0) > row.get('start', 0):
                values = list(map(int, match.groups()))
                nvtx.append(dict(name=name, start_ns=row['start'], end_ns=row['end'],
                    scheduled_tokens=values[0], context_requests=values[1], query_tokens=values[2],
                    logical_seq_len=values[3], generation_requests=values[6], generation_tokens=values[7]))
        report['profiled_worker_shapes'] = sorted(nvtx, key=lambda r:r['start_ns'])
        report['all_traced_steps_single_owner_2k_prefill'] = bool(nvtx) and all(
            r['context_requests'] == 1 and r['query_tokens'] == 2048 and
            r['generation_requests'] == r['generation_tokens'] == 0 for r in nvtx)
        return report
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.sqlite), indent=2) + '\n')
