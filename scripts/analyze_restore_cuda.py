"""Align a completed Nsight diagnostic with restore host intervals, without claiming causality."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time


NS = 1_000_000_000
ACTIVITIES = ('CUPTI_ACTIVITY_KIND_GRAPH_TRACE', 'CUPTI_ACTIVITY_KIND_KERNEL',
              'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET')
METADATA = ('TARGET_INFO_SESSION_START_TIME', 'ANALYSIS_DETAILS', 'META_DATA_CAPTURE',
            'META_DATA_EXPORT', 'PROCESSES', 'TARGET_INFO_GPU', 'TARGET_INFO_CUDA_GPU',
            'TARGET_INFO_CUDA_DEVICE', 'TARGET_INFO_CUDA_CONTEXT_INFO', 'TARGET_INFO_CUDA_STREAM',
            'DIAGNOSTIC_EVENT', 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL',
            'ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE', 'ENUM_DIAGNOSTIC_SOURCE_TYPE')


def ns(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError('Non-finite clock value')
    return int((value * NS).to_integral_value())


def integer_median(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Empty median')
    middle = len(ordered)//2
    return ordered[middle] if len(ordered)%2 else (ordered[middle-1]+ordered[middle])//2


def load(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def json_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line, parse_float=Decimal)


def merge(intervals, begin, end):
    """Half-open clipped union; a nested end must never move the frontier backward."""
    if end < begin:
        raise ValueError('Reversed analysis window')
    out = []
    for a, b in sorted((max(a, begin), min(b, end)) for a, b in intervals if a < end and b > begin):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def intersection(left, right):
    """Both inputs must already be disjoint sorted unions."""
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def complement(busy, begin, end):
    result, cursor = [], begin
    for a, b in busy:
        if cursor < a:
            result.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        result.append((cursor, end))
    return result


def duration(intervals):
    return sum(b-a for a, b in intervals)


def interval_stats(intervals):
    lengths = sorted(b-a for a, b in intervals)
    return dict(count=len(lengths), total_ns=sum(lengths), maximum_ns=max(lengths, default=0),
                median_ns=integer_median(lengths) if lengths else None,
                thresholds=[dict(at_least_ms=t, count=sum(n >= t*1_000_000 for n in lengths),
                                 total_ns=sum(n for n in lengths if n >= t*1_000_000))
                            for t in (.1, 1, 5, 10, 50, 100)])


def bounded_json_samples(path, points=32):
    """Sample byte positions, not all streaming-token rows; never accept a partial line."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        seen = set()
        for offset in sorted({0, max(0, size-65536), *(size*i//points for i in range(1, points))}):
            stream.seek(offset)
            if offset:
                stream.readline()
            for _ in range(2):
                position = stream.tell()
                line = stream.readline()
                if position in seen or not line:
                    continue
                seen.add(position)
                try:
                    yield json.loads(line, parse_float=Decimal)
                except (ValueError, UnicodeDecodeError):
                    continue


def clock_anchors(root):
    specs = [('nsys/events.jsonl', 'at', 'wall'), ('nsys/target.json', 'at', 'wall'),
             ('pair-00-P1/gpu_samples.jsonl', 'at', 'wall_at'),
             ('pair-00-P1/saturated/events.jsonl', 'monotonic', 'wall'),
             ('pair-00-P1/server/capture/public_work.jsonl', 'at', 'wall_time'),
             ('pair-00-P1/server/capture/checkpoint_timing.jsonl', 'at', 'wall_time')]
    anchors = []
    for name, mono, wall in specs:
        path = root/name
        if not path.exists():
            continue
        rows = [load(path)] if path.suffix == '.json' else bounded_json_samples(path)
        for row in rows:
            if mono in row and wall in row:
                anchors.append(dict(source=name, mono_ns=ns(row[mono]), wall_ns=ns(row[wall])))
    window = load(root/'pair-00-P1/saturated/window.json')
    anchors.append(dict(source='window.t0', mono_ns=ns(window['t0_monotonic']),
                        wall_ns=ns(window['t0_wall'])))  # t1_wall is derived, not an anchor.
    return anchors


class ClockMap:
    def __init__(self, origin_ns, anchors):
        if not anchors:
            raise ValueError('No independent wall/monotonic anchors')
        self.origin_ns = int(origin_ns)
        offsets = [a['wall_ns']-a['mono_ns'] for a in anchors]
        self.offset_ns = integer_median(offsets)
        self.report_zero_mono_ns = self.origin_ns-self.offset_ns
        self.anchors = anchors

    def report(self, mono_seconds):
        return ns(mono_seconds)-self.report_zero_mono_ns

    def mono(self, report_ns):
        return report_ns+self.report_zero_mono_ns

    def summary(self):
        groups = defaultdict(list)
        for anchor in self.anchors:
            groups[anchor['source']].append(anchor)
        def describe(rows):
            values = [a['wall_ns']-a['mono_ns'] for a in rows]
            ordered = sorted(rows, key=lambda a:a['mono_ns'])
            span = ordered[-1]['mono_ns']-ordered[0]['mono_ns']
            delta = (ordered[-1]['wall_ns']-ordered[-1]['mono_ns'])-(ordered[0]['wall_ns']-ordered[0]['mono_ns'])
            return dict(count=len(rows), mono_span_ns=span, median_offset_ns=integer_median(values),
                        minimum_offset_ns=min(values), maximum_offset_ns=max(values),
                        maximum_abs_residual_ns=max(abs(v-self.offset_ns) for v in values),
                        endpoint_offset_change_ns=delta,
                        endpoint_drift_ppm=delta/span*1e6 if span else None)
        return dict(utc_epoch_origin_ns=self.origin_ns, wall_minus_monotonic_ns=self.offset_ns,
                    report_zero_monotonic_ns=self.report_zero_mono_ns,
                    formula='report_ns = monotonic_ns + wall_minus_monotonic_ns - utcEpochNs',
                    arithmetic='Decimal parsing and integer nanoseconds; no binary-float epoch subtraction',
                    all_sources=describe(self.anchors), sources={k:describe(v) for k,v in groups.items()},
                    anchors=self.anchors,
                    precision_scope='Unbracketed paired reads: residuals describe observed agreement, not a strict error bound.')


def quote(name):
    return '"'+name.replace('"', '""')+'"'


def metadata(connection):
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    schema, records = {}, {}
    for name in sorted(tables.intersection(set(METADATA) | set(ACTIVITIES) |
                       {'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'})):
        schema[name] = [dict(r) for r in connection.execute('PRAGMA table_info('+quote(name)+')')]
        if name in METADATA:
            where = " WHERE name NOT LIKE '%ENVIRONMENT%'" if name=='META_DATA_CAPTURE' else ''
            rows = [dict(r) for r in connection.execute('SELECT * FROM '+quote(name)+where+' LIMIT 2001')]
            records[name] = dict(rows=rows[:2000], truncated=len(rows)>2000)
    return tables, schema, records


def capture_window(meta):
    export = {r['name']:r['value'] for r in meta['META_DATA_EXPORT']['rows']}
    if (export.get('EXPORT_PARAM_TIME_NORMALIZE') != 'false' or
            export.get('EXPORT_PARAM_TIME_SHIFT') != '0' or
            export.get('EXPORT_CONFIG_TIME_SHIFT') != '0' or
            export.get('EXPORT_PARAM_TIMES_TO_EXPORT')):
        raise ValueError('Require unshifted, non-epoch-normalized, unfiltered export')
    details = meta['ANALYSIS_DETAILS']['rows']
    if len(details)!=1 or details[0]['duration']<=0 or (
            details[0]['stopTime']-details[0]['startTime']!=details[0]['duration']):
        raise ValueError('Ambiguous trace duration')
    # This schema's activities are capture-relative even though ANALYSIS_DETAILS
    # retains the original collection clock. The installed official collective_loader
    # likewise pairs duration with utcEpochNs, not startTime with that UTC origin.
    return 0,details[0]['duration']


def device_activity(connection, tables, metadata_rows):
    device_map = {(r['pid'], r['cudaId']):r['gpuId'] for r in
                  metadata_rows.get('TARGET_INFO_CUDA_DEVICE', {}).get('rows', [])}
    intervals, summaries, groups = defaultdict(list), {}, Counter()
    unmapped = set()
    for name in ACTIVITIES:
        if name not in tables:
            summaries[name] = dict(present=False)
            continue
        columns = {r['name'] for r in connection.execute('PRAGMA table_info('+quote(name)+')')}
        required = {'start','end','deviceId','contextId','streamId','globalPid'}
        if not required <= columns:
            raise ValueError(name+' lacks activity identity/time columns: '+str(required-columns))
        total = invalid = zero = 0
        minimum = maximum = None
        for row in connection.execute('SELECT start,end,deviceId,contextId,streamId,globalPid FROM '+quote(name)):
            a,b,device,context,stream,gpid = row
            total += 1
            if a is None or b is None or b<a:
                invalid += 1
                continue
            if b == a:
                zero += 1
                continue
            pid = (int(gpid)>>24)&0xFFFFFF if gpid is not None else None
            physical = device_map.get((pid,device))
            if physical is None:
                key = f'unmapped_pid_{pid}_cuda_{device}'
                unmapped.add((pid,device))
            else:
                key = f'gpu_{physical}'
            intervals[key].append((a,b))
            groups[(name,key,pid,device,context,stream)] += 1
            minimum = a if minimum is None else min(minimum,a)
            maximum = b if maximum is None else max(maximum,b)
        summaries[name] = dict(present=True, rows=total, positive_duration_rows=total-invalid-zero,
                               invalid_rows=invalid, zero_duration_rows=zero,
                               first_start_ns=minimum,last_end_ns=maximum)
    return intervals, dict(tables=summaries, unmapped_devices=sorted(unmapped,key=str),
        streams=[dict(table=k[0],physical_device=k[1],pid=k[2],cuda_device=k[3],context=k[4],stream=k[5],rows=v)
                 for k,v in sorted(groups.items(),key=lambda p:str(p[0]))])


def restore_intervals(path, clock, begin, end):
    records, by_stage, outers = [], defaultdict(list), []
    outside = 0
    for row in json_rows(path):
        stages = row['timing']['stages']
        totals = [s for s in stages if s['name']=='restore_main_thread_total']
        if len(totals) != 1:
            raise ValueError('Restore observation must contain one outer interval')
        first,last = clock.report(totals[0]['started_at']),clock.report(totals[0]['finished_at'])
        if last < first:
            raise ValueError('Reversed restore interval')
        if last <= begin or first >= end:
            outside += 1
            continue
        converted = []
        for stage in stages:
            a,b = clock.report(stage['started_at']),clock.report(stage['finished_at'])
            if b<a or a<first or b>last:
                raise ValueError('Restore substage outside its outer interval')
            key = (row['phase'],row.get('lane'),stage['name'])
            by_stage[key].append((a,b))
            converted.append(dict(name=stage['name'],start_ns=a,end_ns=b,
                                  wall_ns=b-a,thread_cpu_ns=ns(stage['thread_cpu_seconds'])))
        outers.append((first,last))
        records.append(dict(request_id=row['request_id'],phase=row['phase'],lane=row.get('lane'),
                            start_ns=first,end_ns=last,boundary_clipped=first<begin or last>end,stages=converted))
    return records,outers,by_stage,outside


def describe_device(intervals, outers, stages, begin, end, limit):
    busy = merge(intervals,begin,end)
    gaps = complement(busy,begin,end)
    interior = [(a,b) for a,b in gaps if a>begin and b<end]
    restore = merge(outers,begin,end)
    overlap = duration(intersection(gaps,restore))
    longest = sorted(gaps,key=lambda ab:ab[1]-ab[0],reverse=True)[:limit]
    longest_rows = []
    for a,b in longest:
        overlaps = []
        for (phase,lane,name), ranges in stages.items():
            amount = duration(intersection([(a,b)],merge(ranges,begin,end)))
            if amount:
                overlaps.append(dict(phase=phase,lane=lane,stage=name,overlap_ns=amount))
        longest_rows.append(dict(start_ns=a,end_ns=b,duration_ns=b-a,boundary=a==begin or b==end,
            restore_direct_overlap_upper_bound_ns=duration(intersection([(a,b)],restore)),
            stage_overlaps=sorted(overlaps,key=lambda r:r['overlap_ns'],reverse=True)))
    sensitivity = []
    for delta in (-1_000_000,-100_000,-10_000,-1000,0,1000,10_000,100_000,1_000_000):
        shifted = merge([(a+delta,b+delta) for a,b in outers],begin,end)
        sensitivity.append(dict(host_clock_shift_ns=delta,direct_overlap_ns=duration(intersection(gaps,shifted))))
    return dict(window_ns=end-begin,busy_envelope_ns=duration(busy),busy_union_intervals=len(busy),
        gap_statistics=interval_stats(gaps),interior_gap_statistics=interval_stats(interior),
        restore_host_union_ns=duration(restore),restore_direct_overlap_upper_bound_ns=overlap,
        interior_restore_direct_overlap_upper_bound_ns=duration(intersection(interior,restore)),
        fraction_of_gap_directly_overlapping_restore=overlap/duration(gaps) if duration(gaps) else None,
        gaps_without_direct_restore_overlap_ns=duration(gaps)-overlap,
        boundary_gap_ns=duration(gaps)-duration(interior),clock_sensitivity=sensitivity,
        stage_overlap_totals=[dict(phase=k[0],lane=k[1],stage=k[2],
            host_union_ns=duration(merge(v,begin,end)),
            gap_overlap_ns=duration(intersection(gaps,merge(v,begin,end))))
            for k,v in sorted(stages.items(),key=lambda kv:str(kv[0]))],
        longest_gaps=longest_rows)


def decision_evidence(path, gaps, clock):
    """One streaming scan. Store only nearest-before/after and first/last inside snapshots."""
    targets = [dict(a=clock.mono(g['start_ns']),b=clock.mono(g['end_ns']),before=None,after=None,
                    first_inside=None,last_inside=None,inside_count=0) for g in gaps]
    if not targets or not path.exists():
        return dict(available=path.exists(),lines_scanned=0,bytes_scanned=0)
    pattern = re.compile(r'^\s*\{\s*"at"\s*:\s*([0-9.eE+-]+)')
    lines = scanned = 0
    with path.open() as stream:
        for line in stream:
            lines += 1
            scanned += len(line.encode())
            match = pattern.match(line)
            at = ns(match.group(1)) if match else ns(json.loads(line,parse_float=Decimal)['at'])
            for target in targets:
                if at <= target['a']:
                    target['before'] = line
                elif at < target['b']:
                    target['inside_count'] += 1
                    target['first_inside'] = target['first_inside'] or line
                    target['last_inside'] = line
                elif target['after'] is None:
                    target['after'] = line
            if all(t['after'] is not None for t in targets):
                break
    def snapshot(line):
        if line is None:
            return None
        row = json.loads(line,parse_float=Decimal)
        candidates = row.get('candidates',[])
        return dict(at=row['at'],step=row.get('step'),scheduled=row.get('scheduled'),
            ready_candidate_count=sum(r.get('reason') is None for r in candidates),
            candidates=[{k:r.get(k) for k in ('request_id','phase','reason','remaining_tokens','restore_tokens')}
                        for r in candidates],free_gpu_blocks=row.get('free_gpu_blocks'))
    for gap,target in zip(gaps,targets):
        gap['scheduler_snapshots'] = dict(inside_snapshot_count=target['inside_count'],
            **{key:snapshot(target[key]) for key in ('before','first_inside','last_inside','after')})
    return dict(available=True,lines_scanned=lines,bytes_scanned=scanned,
        scope='Instantaneous candidate/scheduled evidence only. No snapshot is extended across a CUDA gap.')


def host_gap_evidence(root, gaps, restores, clock):
    path = root/'pair-00-P1/server/capture/model_step_latency.jsonl'
    for gap in gaps:
        a,b = gap['start_ns'],gap['end_ns']
        gap['restore_observations'] = [r for r in restores if r['start_ns']<b and r['end_ns']>a]
        gap['model_step_intersections'] = []
        gap['client_pending_requests'] = []
    for row in json_rows(path):
        a,b = clock.report(row['started_at']),clock.report(row['submitted_at'])
        for gap in gaps:
            if a<gap['end_ns'] and b>gap['start_ns']:
                gap['model_step_intersections'].append({k:row.get(k) for k in
                    ('sequence','started_at','model_submitted_at','submitted_at','decode_requests','prefill_tokens',
                     'maintenance_host_ms','host_submit_ms','maintenance_ms','model_ms','step_ms')})
    # Client outstanding work does not establish scheduler-ready or GPU-admitted work.
    pending = {}
    for row in json_rows(root/'pair-00-P1/saturated/events.jsonl'):
        rid = row.get('request_id')
        if row['event']=='request_started' and row.get('kind')=='workload':
            pending[rid] = row
        elif row['event']=='request_finished' and rid in pending:
            first = pending.pop(rid)
            a,b = clock.report(first['monotonic']),clock.report(row['monotonic'])
            for gap in gaps:
                if a<gap['end_ns'] and b>gap['start_ns']:
                    gap['client_pending_requests'].append(dict(request_id=rid,start_ns=a,end_ns=b,
                        overlap_ns=min(b,gap['end_ns'])-max(a,gap['start_ns']),status=row.get('status')))
    return dict(unclosed_client_workload_requests=list(pending),
                scope='Client request lifetimes establish outstanding client demand only; model-step times are host submission times.')


def api_containment(connection, tables, restores, engine_pids):
    """Independent clock containment cross-check, never used to fit the clock offset."""
    intervals = [(s['start_ns'],s['end_ns'],r['request_id']) for r in restores for s in r['stages']
                 if s['name']=='prepare_stream_dependencies']
    checks = [dict(request_id=rid,start_ns=a,end_ns=b,calls=[]) for a,b,rid in intervals]
    if not intervals or 'StringIds' not in tables:
        return dict(intervals=checks,scope='No containment cross-check available')
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        query = ('SELECT a.start,a.end,a.globalTid,s.value FROM '+quote(table)+' a JOIN StringIds s ON s.id=a.nameId '
                 "WHERE (s.value LIKE 'cudaEventRecord%' OR s.value LIKE 'cuEventRecord%') AND a.end>=? AND a.start<=?")
        for a,b,gtid,name in connection.execute(query,(min(x[0] for x in intervals),max(x[1] for x in intervals))):
            pid,tid = (int(gtid)>>24)&0xFFFFFF,int(gtid)&0xFFFFFF
            if pid not in engine_pids:
                continue
            for check in checks:
                if a>=check['start_ns'] and b<=check['end_ns']:
                    check['calls'].append(dict(table=table,name=name,pid=pid,tid=tid,start_ns=a,end_ns=b))
    return dict(interval_count=len(checks),intervals_with_contained_call=sum(bool(c['calls']) for c in checks),
                intervals_with_exactly_one_call=sum(len(c['calls'])==1 for c in checks),intervals=checks,
                scope='Containment only. Host stages have no TID field; multiple calls or missing sequence identity are ambiguous, not clock proof.')


def host_gap_classification(gap, report_zero_mono_ns):
    a,b = gap['start_ns'],gap['end_ns']
    maintenance,forward = [],[]
    for step in gap['model_step_intersections']:
        start,model,end = (ns(step[k])-report_zero_mono_ns for k in
                           ('started_at','model_submitted_at','submitted_at'))
        maintenance.append((start,model))
        forward.append((model,end))
    maintenance,forward = merge(maintenance,a,b),merge(forward,a,b)
    return dict(maintenance_host_ns=duration(maintenance),
                forward_sampling_mtp_capture_host_ns=duration(forward),
                outside_measured_step_ns=(b-a)-duration(merge(maintenance+forward,a,b)),
                scope='StepTiming.model_begin is after attention preparation; end is after sampling, GDN, MTP and capture. Forward bucket includes KVMem hooks.')


def gap_call_analysis(root, analysis_path, limit=10):
    """Narrow CPU API evidence only; reuse saved gaps and never rescan scheduler decisions."""
    started = dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
    previous = load(analysis_path)
    pids = previous['coverage']['verified_engine_pids']
    selected = [dict(g,physical_device=device) for device,rows in previous['devices'].items()
                for g in rows['longest_gaps'] if not g['restore_direct_overlap_upper_bound_ns']]
    selected = sorted(selected,key=lambda g:g['duration_ns'],reverse=True)[:limit]
    if not selected:
        raise ValueError('No saved non-restore gaps')
    margin = 2_000_000
    windows = [(g['start_ns']-margin,g['end_ns']+margin) for g in selected]
    connection = sqlite3.connect((root/'nsys/timeline.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    raw = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER','OSRT_API'):
        if table not in tables:
            continue
        conditions = ' OR '.join('(a.start < ? AND a.end > ?)' for _ in windows)
        query = ('SELECT a.*,s.value AS api_name FROM '+quote(table)+' a JOIN StringIds s ON a.nameId=s.id '
                 'WHERE ((a.globalTid >> 24) & 16777215) IN ('+','.join('?' for _ in pids)+') AND ('+conditions+')')
        params = list(pids)+[v for a,b in windows for v in (b,a)]
        raw[table] = []
        for row in connection.execute(query,params):
            value = dict(row)
            value['pid'],value['tid'] = (value['globalTid']>>24)&0xFFFFFF,value['globalTid']&0xFFFFFF
            value['main_thread'] = value['pid']==value['tid']
            raw[table].append(value)
        raw[table].sort(key=lambda row:(row['start'],row['end']))
    chain_ids = set()
    for gap in selected:
        a,b = gap['start_ns'],gap['end_ns']
        gap['host_classification'] = host_gap_classification(gap,previous['clock']['report_zero_monotonic_ns'])
        gap['api_evidence'] = {}
        for table,rows in raw.items():
            around = [r for r in rows if r['start']<b+margin and r['end']>a-margin]
            overlaps = [dict(r,overlap_ns=max(0,min(b,r['end'])-max(a,r['start']))) for r in around
                        if r['start']<b and r['end']>a]
            main = [r for r in overlaps if r['main_thread']]
            prior = [r for r in around if r['main_thread'] and r['end']<=a]
            after = [r for r in around if r['main_thread'] and r['start']>=b]
            occupied = merge([(r['start'],r['end']) for r in main],a,b)
            by_name = defaultdict(list)
            for row in main:
                by_name[row['api_name']].append((row['start'],row['end']))
            gap['api_evidence'][table] = dict(main_thread_call_count=len(main),
                main_thread_call_union_overlap_ns=duration(occupied),
                main_thread_time_outside_this_api_table_ns=b-a-duration(occupied),
                by_name=[dict(name=k,union_overlap_ns=duration(merge(v,a,b)),calls=len(v)) for k,v in by_name.items()],
                longest_main_thread_calls=sorted(main,key=lambda r:r['overlap_ns'],reverse=True)[:8],
                previous_main_call_within_margin=max(prior,key=lambda r:r['end'],default=None),
                next_main_call_within_margin=min(after,key=lambda r:r['start'],default=None),
                other_engine_thread_calls=sorted([r for r in overlaps if not r['main_thread']],
                                                key=lambda r:r['overlap_ns'],reverse=True)[:8])
            if table=='OSRT_API':
                chain_ids.update(r['callchainId'] for r in main if r.get('callchainId') is not None)
    chains = []
    if chain_ids and 'OSRT_CALLCHAINS' in tables:
        query = ('SELECT c.*,s.value AS symbol_name,m.value AS module_name FROM OSRT_CALLCHAINS c '
                 'LEFT JOIN StringIds s ON c.symbol=s.id LEFT JOIN StringIds m ON c.module=m.id '
                 'WHERE c.id IN ('+','.join('?' for _ in chain_ids)+') ORDER BY c.id,c.stackDepth')
        chains = [dict(r) for r in connection.execute(query,sorted(chain_ids))]
    connection.close()
    return dict(schema_version=1,diagnostic_only=True,analysis_input=str(analysis_path.resolve()),
        execution=dict(start=started,end=dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())),
        engine_pids=pids,adjacent_call_margin_ns=margin,windows=selected,raw_api_rows=raw,
        main_thread_osrt_callchains=chains,
        limitations=[
            'One bounded OR-window query per API table, no scheduler-log reread. Raw calls retain complete intervals, including calls crossing window boundaries.',
            'Forward host bucket includes sampling, GDN, MTP and KVMem capture/hooks. It does not isolate native model compute.',
            'CUDA API and OSRT intervals can nest. Their durations must not be added, and a synchronization name alone does not identify the caller.',
            'OSRT collection threshold was 100us and engine OSRT/CUDA completeness warnings exist. Missing calls are not proof of CPU execution or absence of waits.',
            'No Python stacks or CPU samples were collected. Untraced host spans cannot distinguish KVMem, framework bookkeeping, allocation, Python GC or descheduling.',
            'Thread id equal to engine pid denotes its main Linux thread; other engine threads are reported separately.'])


def analyze(root, database, limit=20):
    analysis_started = dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
    result_path = root/'nsys/result.json'
    wrapper = load(result_path)
    if wrapper.get('status') != 'completed_diagnostic':
        raise ValueError('Require completed wrapper/export validation before analysis')
    connection = sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    tables,schema,meta = metadata(connection)
    origins = {r['utcEpochNs'] for r in meta['TARGET_INFO_SESSION_START_TIME']['rows']}
    if len(origins) != 1:
        raise ValueError('Ambiguous report UTC origin')
    clock = ClockMap(origins.pop(),clock_anchors(root))
    begin,end = capture_window(meta)
    events = list(json_rows(root/'nsys/events.jsonl'))
    requested = next(r for r in events if r['event']=='capture_trigger_requested')
    returned = next(r for r in events if r['event']=='capture_trigger_returned')
    trigger = dict(requested_mono_ns=ns(requested['at']),returned_mono_ns=ns(returned['at']),
                   capture_start_mono_ns=clock.mono(begin),capture_stop_mono_ns=clock.mono(end))
    trigger['capture_start_within_trigger_call'] = trigger['requested_mono_ns']<=clock.mono(begin)<=trigger['returned_mono_ns']
    raw,coverage = device_activity(connection,tables,meta)
    if not raw:
        raise ValueError('No device activity; absence cannot be interpreted as idle')
    capture = root/'pair-00-P1/server/capture'
    restores,outers,stages,outside = restore_intervals(capture/'restore_host_stages.jsonl',clock,begin,end)
    devices = {key:describe_device(value,outers,stages,begin,end,limit) for key,value in raw.items()}
    gaps = [g for d in devices.values() for g in d['longest_gaps']]
    for gap in gaps:
        gap['monotonic_start_ns'],gap['monotonic_end_ns'] = clock.mono(gap['start_ns']),clock.mono(gap['end_ns'])
    host_evidence = host_gap_evidence(root,gaps,restores,clock)
    engine_pids = set(wrapper['validation']['verified_engine_pids'])
    compute_pids = {r['pid'] for r in coverage['streams'] if 'KERNEL' in r['table'] or 'GRAPH' in r['table']}
    observed = set()
    for row in json_rows(root/'pair-00-P1/gpu_samples.jsonl'):
        if clock.mono(begin)<=ns(row['at'])<=clock.mono(end):
            observed.update(row['pids'])
    coverage.update(verified_engine_pids=sorted(engine_pids),sampled_gpu_pids=sorted(observed),
                    missing_engine_compute_pids=sorted(engine_pids-compute_pids),
                    sampled_gpu_pids_without_compute=sorted(observed-compute_pids),
                    wrapper_validation=wrapper['validation'])
    diagnostics = meta.get('DIAGNOSTIC_EVENT',{}).get('rows',[])
    coverage['engine_diagnostics'] = [r for r in diagnostics if r.get('globalPid') is not None and
                                    ((int(r['globalPid'])>>24)&0xFFFFFF) in engine_pids and r['severity'] in (2,3)]
    coverage['diagnostic_groups'] = [dict(severity=k[0],text=k[1],count=v) for k,v in
        Counter((r['severity'],r['text']) for r in diagnostics if r['severity'] in (2,3)).items()]
    coverage['whole_device_completeness_proven'] = False
    containment = api_containment(connection,tables,restores,engine_pids)
    connection.close()
    warnings = []
    if not trigger['capture_start_within_trigger_call']:
        warnings.append('Trace origin/bounds do not place capture start inside the control call; inspect before interpreting overlap.')
    if coverage['unmapped_devices']:
        warnings.append('Some CUDA ordinals could not be mapped to physical GPUs; separate groups must not be called whole-device unions.')
    if coverage['engine_diagnostics']:
        warnings.append('Engine warning/error diagnostics exist, including possible incomplete CUDA collection; gaps cannot establish device idle.')
    if coverage['sampled_gpu_pids_without_compute']:
        warnings.append('Observed GPU processes lack traced compute activity.')
    # Delay the only large decision-log scan until other inputs have validated.
    decisions = decision_evidence(capture/'offline_decisions.jsonl',gaps,clock)
    return dict(schema_version=1,diagnostic_only=True,performance_claim_allowed=False,
        analysis_execution=dict(start=analysis_started,end=dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns()),
                                scope='Post-diagnostic CPU analysis; not a performance measurement'),
        inputs=dict(root=str(root.resolve()),database=str(database.resolve()),database_bytes=database.stat().st_size,
                    wrapper_result=str(result_path.resolve()),
                    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        interpreter_clocks={n:vars(time.get_clock_info(n)) for n in ('perf_counter','monotonic','time')},
        clock=clock.summary(),capture_bounds=dict(start_ns=begin,end_ns=end,trigger_check=trigger,
            source='[0, ANALYSIS_DETAILS.duration]; export normalize=false, shift=0, no time filter',
            raw_analysis_start_stop_are_not_activity_coordinates=True,
            local_reference='/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/python/packages/nsys_recipe/lib/collective_loader.py:21'),
        schema=schema,metadata=meta,coverage=coverage,api_clock_containment=containment,
        restore_observations_in_capture=len(restores),restore_observations_outside_capture=outside,
        devices=devices,scheduler_evidence_scan=decisions,host_evidence=host_evidence,warnings=warnings,
        limits=[
            'Graph/kernel/memcpy/memset union is a conservative busy envelope. Graph intervals contain internal gaps; this is not execution time or SM occupancy.',
            'Complement intervals are gaps in traced device activities. Process polling and host-only graph capture do not prove complete whole-device coverage.',
            'Direct restore overlap is a temporal upper bound on coincident stalls, not a causal estimate or an upper bound on delayed downstream effects.',
            'Nested restore stages overlap and their totals must not be added. Logging/background work outside measured host intervals is not attributed.',
            'Boundary gaps are clipped observations and are reported separately from interior gaps.',
            'Clock sensitivity samples selected constant offsets; these are not certified uncertainty bounds. No API containment event is used to fit the mapping.',
            'Scheduler snapshots establish candidate/admission state only at recorded instants. Client pending demand is not proof of runnable work.',
            'No CPU sampling, context-switch or Python GIL tracing was collected; wall minus thread CPU cannot attribute a gap to GIL contention.',
            'Profiler overhead makes this diagnostic unsuitable for performance-win claims.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Completed diagnostic performance directory')
    parser.add_argument('--sqlite',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-limit',type=int,default=20)
    parser.add_argument('--gap-calls-from',type=Path,help='Reuse a saved analysis to inspect only its ten longest non-restore gap API windows')
    args = parser.parse_args()
    if not 1<=args.evidence_limit<=100:
        parser.error('evidence-limit must be 1..100')
    result = (gap_call_analysis(args.root,args.gap_calls_from) if args.gap_calls_from else
              analyze(args.root,args.sqlite or args.root/'nsys/timeline.sqlite',args.evidence_limit))
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,default=lambda x:float(x) if isinstance(x,Decimal) else str(x))
        stream.write('\n')
    if args.gap_calls_from:
        print(json.dumps(dict(output=str(args.output),windows=len(result['windows']),
                              api_rows={k:len(v) for k,v in result['raw_api_rows'].items()})))
        return
    print(json.dumps(dict(output=str(args.output),devices={k:{n:v[n] for n in
        ('busy_envelope_ns','gap_statistics','restore_direct_overlap_upper_bound_ns')} for k,v in result['devices'].items()},
        warnings=result['warnings'])))


if __name__=='__main__':
    main()
