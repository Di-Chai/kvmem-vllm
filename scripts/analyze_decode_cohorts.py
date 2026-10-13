"""Offline decode cohorts from qualify_decode results; never sends model requests.

HTTP throughput includes the whole queued workload. Decode cohorts use stable
native owners only when every owner has finished prefill and has an ongoing SSE
decode stream. Native archive commits are approximate; SSE counts are exact
arrivals in those sampled intervals, not GPU kernel timings. All qualified zero
progress intervals remain in the denominator. Excluded intervals are reported.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from qualify_decode import count_at, end_to_end_summary
from qualify_multi_user import distribution


def trace_error(trace):
    """Reject absent/ambiguous token evidence rather than counting SSE chunks."""
    if trace.get('outcome') != 'success':
        return 'request_not_successful'
    if type(trace.get('prompt_tokens')) is not int or trace['prompt_tokens'] < 0:
        return 'missing_prompt_usage'
    total, previous_at = 0, -math.inf
    for point in trace.get('timeline', []):
        ids = point.get('token_ids')
        if (not isinstance(ids, list) or not ids or
                any(type(token) is not int or token < 0 for token in ids)):
            return 'missing_or_invalid_token_ids'
        total += len(ids)
        if point.get('tokens') != total or point.get('delta_tokens') != len(ids):
            return 'token_timeline_count_mismatch'
        at = point.get('at')
        if not isinstance(at, (int, float)) or not math.isfinite(at) or at < previous_at:
            return 'invalid_token_timing'
        previous_at = at
    if not total or total != trace.get('observed_tokens') or total != trace.get('completion_tokens'):
        return 'missing_or_inconsistent_usage'
    if trace.get('finish_reason') != 'length':
        return 'unexpected_finish_reason'
    return None


def trace_bounds(trace, trim):
    timeline, total = trace['timeline'], trace['observed_tokens']
    start = next((point['at'] for point in timeline if point['tokens'] >= trim), None)
    ends = [point['at'] for point in timeline if point['tokens'] <= total - trim]
    end = ends[-1] if ends else None
    return start, end


def match_owner(owner, traces):
    # Saved evidence uses aliases; in-memory run_batch evidence has response IDs.
    matches = [trace for trace in traces if owner == f'client_{trace["client_index"]}' or
               (isinstance(trace.get('_response_id'), str) and
                (owner == trace['_response_id'] or owner.startswith(trace['_response_id'] + '-')))]
    return matches[0] if len(matches) == 1 else None


def classify_interval(group, traces, trim):
    """Classify one adjacent interval with one extra sample as a prefill guard."""
    older, before, after = group
    if any(sample.get('error') for sample in group):
        return None, 'sample_read_error'
    try:
        owners = [sample['progress']['accepted_history'] for sample in group]
        if any(not isinstance(owner_set, dict) for owner_set in owners):
            return None, 'invalid_native_sample'
        if any(set(owner_set) != set(owners[0]) for owner_set in owners[1:]):
            return None, 'owner_transition'
        if not owners[0]:
            return None, 'idle'
        # A long prefill can already hold a reservation while it has no
        # accepted-history entry yet. Its scheduled prefill counter increments
        # only at launch and can stay flat throughout the forward. The three
        # sample counter guard alone cannot classify that time as pure decode.
        reservations = [sample['scheduler'].get('reservations') for sample in group]
        if any(value is None for value in reservations):
            return None, 'missing_admission_reservations'
        if any(not isinstance(value, list) or any(not isinstance(key, str) for key in value)
               or len(set(value)) != len(value) for value in reservations):
            return None, 'invalid_admission_reservations'
        if any(len(value) != len(owners[0]) for value in reservations):
            return None, 'admitted_owner_mismatch'
        if any(set(value) != set(reservations[0]) for value in reservations[1:]):
            return None, 'admission_reservation_transition'
        mapped = {owner: match_owner(owner, traces) for owner in owners[0]}
        if any(trace is None for trace in mapped.values()):
            return None, 'unknown_or_ambiguous_owner'
        if len({trace['client_index'] for trace in mapped.values()}) != len(mapped):
            return None, 'duplicate_owner_for_client'
        if any(trace_error(trace) for trace in mapped.values()):
            return None, 'unverified_request_tokens'
        # accepted_history can contain paused partial prefills. Stable ownership
        # and unchanged global prefill counters alone cannot classify them as decode.
        if any(owner_set[owner] < trace['prompt_tokens']
               for owner_set in owners for owner, trace in mapped.items()):
            return None, 'mixed_or_prefill'
        if any(sample['prefill_raced'] for sample in group):
            return None, 'prefill_raced_read'
        if len({sample['prefill'] for sample in group}) != 1:
            return None, 'prefill_activity_or_trailing_commit'
        if any(sample['scheduler']['refresh_pending'] for sample in group):
            return None, 'refresh_activity'
        if any(sample['scheduler']['connector_pending'] for sample in group):
            return None, 'connector_activity'
        if any(sample['progress']['retiring_requests'] for sample in group):
            return None, 'retiring_activity'
        # scheduler.requests also includes engine waiting requests. More
        # requests than owners do not imply more decode concurrency, and their
        # exact state is unknown from these snapshots.
        if any(sample['scheduler']['requests'] < len(mapped) for sample in group):
            return None, 'scheduler_owner_mismatch'
        read_start = min(sample.get('read_started_at', sample['at']) for sample in (before, after))
        for trace in mapped.values():
            first, last = trace['timeline'][0]['at'], trace['timeline'][-1]['at']
            terminal = min(last, trace.get('finish_at', last))
            if read_start < first:
                return None, 'stream_prefill_boundary'
            if after['at'] >= terminal:
                return None, 'stream_terminal_boundary'
            start, end = trace_bounds(trace, trim)
            if start is None or end is None or start >= end:
                return None, 'insufficient_trimmed_output'
            if read_start < start or after['at'] > end:
                return None, 'stream_trim_boundary'
        seconds = after['at'] - before['at']
        deltas = {owner: owners[2][owner] - owners[1][owner] for owner in mapped}
        if (seconds <= 0 or any(delta < 0 for delta in deltas.values()) or
                after['decode'] < before['decode']):
            return None, 'native_counter_or_timing_inconsistent'
        per_request = []
        for owner, trace in mapped.items():
            sse_tokens = count_at(trace, after['at']) - count_at(trace, before['at'])
            events = sum(before['at'] < point['at'] <= after['at'] for point in trace['timeline'])
            per_request.append({'client_index': trace['client_index'],
                                'native_accepted_input_tokens': deltas[owner],
                                'sse_output_tokens': sse_tokens, 'sse_token_events': events})
        return {'start': before['at'], 'end': after['at'], 'seconds': seconds,
                'stable_decode_owner_count': len(mapped),
                'client_indices': sorted(trace['client_index'] for trace in mapped.values()),
                'scheduler_total_requests': [sample['scheduler']['requests'] for sample in group],
                'scheduler_requests_without_captured_owner': [sample['scheduler']['requests'] - len(mapped)
                                                              for sample in group],
                'native_accepted_input_tokens': sum(deltas.values()),
                'scheduled_verifier_queries': after['decode'] - before['decode'],
                'sse_output_tokens': sum(item['sse_output_tokens'] for item in per_request),
                'per_request': per_request}, None
    except (KeyError, TypeError, AttributeError, ValueError):
        return None, 'invalid_native_sample'


def summarize_cohort(intervals, min_seconds, min_intervals):
    seconds = sum(interval['seconds'] for interval in intervals)
    native_tokens = sum(interval['native_accepted_input_tokens'] for interval in intervals)
    sse_tokens = sum(interval['sse_output_tokens'] for interval in intervals)
    per_client = defaultdict(Counter)
    segments = []
    for interval in intervals:
        for request in interval['per_request']:
            per_client[request['client_index']].update({key: request[key] for key in (
                'native_accepted_input_tokens', 'sse_output_tokens', 'sse_token_events')})
        if (not segments or interval['sample_index'] != segments[-1]['last_sample_index'] + 1 or
                interval['client_indices'] != segments[-1]['client_indices']):
            segments.append({'start': interval['start'], 'end': interval['end'],
                             'client_indices': interval['client_indices'], 'seconds': 0,
                             'interval_count': 0, 'native_accepted_input_tokens': 0,
                             'sse_output_tokens': 0})
        segment = segments[-1]
        segment['end'], segment['last_sample_index'] = interval['end'], interval['sample_index']
        for field in ('seconds', 'native_accepted_input_tokens', 'sse_output_tokens'):
            segment[field] += interval[field]
        segment['interval_count'] += 1
    all_progress = all(item['native_accepted_input_tokens'] > 0 and item['sse_output_tokens'] > 0
                       for item in per_client.values())
    sufficient = seconds >= min_seconds and len(intervals) >= min_intervals and all_progress
    return {'status': 'sufficient_evidence' if sufficient else 'insufficient_evidence',
            'stable_decode_owner_count': intervals[0]['stable_decode_owner_count'],
            'interval_count': len(intervals), 'seconds': seconds,
            'minimum_seconds': min_seconds, 'minimum_intervals': min_intervals,
            'all_observed_owners_made_progress': all_progress,
            'native_accepted_input_tokens': native_tokens,
            'native_accepted_input_tokens_per_second': native_tokens / seconds if seconds else None,
            'sse_output_tokens': sse_tokens,
            'sse_output_tokens_per_second': sse_tokens / seconds if seconds else None,
            'zero_native_progress_intervals': sum(not item['native_accepted_input_tokens'] for item in intervals),
            'zero_sse_progress_intervals': sum(not item['sse_output_tokens'] for item in intervals),
            'per_client': [{'client_index': index, **counts} for index, counts in sorted(per_client.items())],
            'segments': segments, 'intervals': intervals}


def analyze_batch(batch, trim, min_seconds=2, min_intervals=3):
    traces, samples = batch['requests'], batch['native_samples']
    service = end_to_end_summary(traces, batch['wall_seconds'])
    intervals, excluded, observed_owners = [], [], Counter()
    for sample in samples:
        owners = sample.get('progress', {}).get('accepted_history')
        if isinstance(owners, dict):
            observed_owners[len(owners)] += 1
    for index in range(1, len(samples)):
        before, after = samples[index - 1:index + 1]
        seconds = after['at'] - before['at']
        if not math.isfinite(seconds) or seconds <= 0:
            excluded.append({'sample_index': index, 'seconds': 0, 'reason': 'invalid_sample_interval'})
            continue
        interval, reason = (classify_interval(samples[index - 2:index + 1], traces, trim)
                            if index >= 2 else (None, 'prefill_guard_warmup'))
        if interval is None:
            excluded.append({'sample_index': index, 'start': before['at'], 'end': after['at'],
                             'seconds': seconds, 'reason': reason})
        else:
            intervals.append(dict(interval, sample_index=index))
    groups = defaultdict(list)
    for interval in intervals:
        groups[interval['stable_decode_owner_count']].append(interval)
    cohorts = [summarize_cohort(items, min_seconds, min_intervals) for _, items in sorted(groups.items())]
    excluded_reasons = {}
    for item in excluded:
        summary = excluded_reasons.setdefault(item['reason'], {'interval_count': 0, 'seconds': 0})
        summary['interval_count'] += 1
        summary['seconds'] += item['seconds']
    seconds = sum(interval['seconds'] for interval in intervals)
    observed = seconds + sum(item['seconds'] for item in excluded)
    requested = next((cohort for cohort in cohorts
                      if cohort['stable_decode_owner_count'] == batch['concurrency']), None)
    errors = [{'client_index': trace['client_index'], 'error': trace_error(trace)}
              for trace in traces if trace_error(trace)]
    return {'requested_clients': batch['concurrency'], 'round_index': batch.get('round_index'),
            'http_service': service,
            'decode_evidence_status': 'available' if any(cohort['status'] == 'sufficient_evidence'
                                                        for cohort in cohorts) else 'insufficient_evidence',
            'full_requested_client_cohort': {'status': requested['status'] if requested else 'not_applicable',
                'reason': None if requested else 'no_native_confirmed_full_client_decode_interval'},
            'trim_tokens_each_end': trim, 'sample_count': len(samples),
            'observed_native_owner_counts': dict(sorted(observed_owners.items())),
            'sample_read_errors': sum(bool(sample.get('error')) for sample in samples),
            'request_evidence_errors': errors, 'qualified_seconds': seconds,
            'observed_sample_seconds': observed,
            'qualified_fraction_of_sample_span': seconds / observed if observed else None,
            'qualified_fraction_of_batch_wall': seconds / batch['wall_seconds'] if batch['wall_seconds'] else None,
            'cohorts': cohorts, 'excluded_by_reason': excluded_reasons, 'excluded_intervals': excluded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Existing qualify_decode results.json')
    parser.add_argument('--output', type=Path, required=True, help='New analysis JSON file; never overwritten')
    parser.add_argument('--trim-tokens', type=int, help='Default: original run configuration')
    parser.add_argument('--min-cohort-seconds', type=float, default=2)
    parser.add_argument('--min-cohort-intervals', type=int, default=3)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.min_cohort_seconds) or args.min_cohort_seconds <= 0 or
            args.min_cohort_intervals < 1 or (args.trim_tokens is not None and args.trim_tokens < 1)):
        parser.error('Need positive finite duration, interval count and token trim')
    raw = args.results.read_bytes()
    source = json.loads(raw)
    trim = args.trim_tokens if args.trim_tokens is not None else source['configuration']['trim_tokens']
    if type(trim) is not int or trim < 1:
        parser.error('The original run must declare a positive token trim')
    report = {'scope': __doc__, 'source_results': str(args.results.resolve()),
              'source_results_sha256': hashlib.sha256(raw).hexdigest(),
              'source_harness_sha256': source.get('harness_sha256'),
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'analysis_dependency_sha256': {Path(function.__code__.co_filename).name:
                  hashlib.sha256(Path(function.__code__.co_filename).read_bytes()).hexdigest()
                  for function in (end_to_end_summary, distribution)},
              'batches': [analyze_batch(batch, trim, args.min_cohort_seconds, args.min_cohort_intervals)
                          for batch in source['batches']],
              'limitations': [
                  'Requested clients can be queued or paused; owner-count snapshots alone are not decode concurrency.',
                  'All native owners must match successful token-verified HTTP requests, have accepted history >= prompt '
                  'usage, and be within their trimmed SSE decode lifetimes; a paused prefill excludes the whole interval.',
                  'Three snapshots must have identical owners and prefill counters, no refresh, retiring or connector '
                  'activity, and scheduler request counts >= captured owners. Extra scheduler requests have unknown '
                  'state and never increase decode concurrency. This also excludes a trailing prefill commit interval.',
                  'Every snapshot must also have a unique reservation list matching the captured owner count, with '
                  'identical reservation sets across all three reads. An admitted request absent from accepted history '
                  'can be a long in-flight prefill even when the prefill counter stays flat, so that whole interval is excluded.',
                  'Separate periodically updated files cannot rule out unsampled refreshes or identify exact GPU timings.',
                  'SSE output counts use exact token IDs in sampled interval boundaries; burst timing and transport '
                  'buffering limit temporal resolution. Native accepted-input commits lag generated output.',
                  'All qualified intervals, including stalls and short segments, contribute to cohort denominators. '
                  'Minimum evidence thresholds label aggregates; they do not drop slow or zero-progress intervals.',
                  'Cohort rates cover selected pure-decode parts of the workload; excluded time and coverage are reported. '
                  'They are distinct from whole-batch HTTP throughput, quality, or production acceptance.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'batches': [
        {'requested_clients': batch['requested_clients'], 'http_status': batch['http_service']['status'],
         'decode_evidence_status': batch['decode_evidence_status'],
         'cohorts': [{key: cohort[key] for key in ('stable_decode_owner_count', 'status', 'seconds',
                     'native_accepted_input_tokens_per_second', 'sse_output_tokens_per_second')}
                    for cohort in batch['cohorts']]}
        for batch in report['batches']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
