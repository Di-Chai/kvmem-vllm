"""Compare fixed eight-input runs; keep decode trajectory changes separate."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from analyze_pressure_profile import analyze, distribution


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.open()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(root, *, expected_cases=8):
    complete = read(root/'run/answers/completed.json')
    if complete['cases'] != expected_cases or not complete['protocol_passed'] or not complete['all_owned_sessions_released']:
        raise ValueError('Require the complete validated and released cohort')
    config = read(root/'profile.json')
    if config['profiler_config'] or config['kvmem']['action_profiling']:
        raise ValueError('Instrumented run cannot enter a performance pair')
    capture = root/'run/server/capture'
    steps = records(capture/'model_step_latency.jsonl')
    refresh = analyze(capture)
    work = records(capture/'public_work.jsonl')
    zero, last = None, None
    for row in work:
        done = row['completed']
        if done['public_non_prefill_tokens']:
            break
        if not done['public_prefill_tokens']:
            zero = row
        else:
            last = row
    prefill = None
    if zero is not None and last is not None:
        elapsed = last['wall_time'] - zero['wall_time']
        tokens = last['completed']['public_prefill_tokens'] - zero['completed']['public_prefill_tokens']
        prefill = dict(seconds=elapsed, executed_tokens=tokens, tokens_per_second=tokens/elapsed,
            scope='Completed public prefill forwards before the first observed decode work; wall time includes '
                  'selection, archive and submission gaps. Excludes cold loading and later mixed batches. '
                  'Counts executed prompt positions, not speculative proposals.')
    action = records(capture/'action_attention.jsonl')
    upload = [r['metadata_upload_total'] for r in action if 'metadata_upload_total' in r]
    per_case = {}
    for result in complete['results']:
        evidence = read(root/'run/answers/cases'/result['id']/'evidence.json')
        events = [*evidence['pressure'], evidence['query']]
        per_case[result['id']] = dict(input_sha256=result['input_sha256'],
            answer_score=result['official_case_score'], output_sha256=digest(result['text']),
            completion_tokens=result['completion_tokens'], ttft_seconds=result['ttft_seconds'],
            selections={str(e['selection'])+':'+str(e['history_begin']):digest(e['selected_history_ranges'])
                        for e in events})
    return dict(profile=config, runtime_files=read(root/'runtime-proof.json')['files'],
        cases=per_case, cold_prepare_seconds=read(root/'status.json')['cold_prepare_seconds'],
        request_ttft_seconds=distribution([r['ttft_seconds'] for r in complete['results']]),
        client_seconds=complete['seconds'], input_tokens=complete['input_tokens'],
        completion_tokens=complete['completion_tokens'], prefill_wall=prefill,
        prefill_model_interval=refresh['prefill_steps'], pressure=refresh['phases']['pressure'],
        query=refresh['phases']['query'],
        actual_prefill_batch_peak=max(r['prefill_requests'] for r in steps),
        actual_decode_batch_peak=max(r['decode_requests'] for r in steps),
        metadata_upload_latest=upload[-1] if upload else None,
        capture_host_seconds=max(r['capture_host_seconds_total'] for r in action))


def compare(root):
    old, new = [summarize(root/name) for name in ('metadata-sync', 'metadata-async')]
    a, b = deepcopy(old['profile']), deepcopy(new['profile'])
    if a['kvmem'].pop('async_action_metadata') is not False or b['kvmem'].pop('async_action_metadata') is not True:
        raise ValueError('Expected only synchronous versus asynchronous metadata')
    if a != b or old['runtime_files'] != new['runtime_files']:
        raise ValueError('Other profile or model code differences')
    uploads = new['metadata_upload_latest']
    if (not uploads or not uploads['submissions'] or uploads['fallbacks'] or uploads['quarantined']
            or any(uploads[k] for k in ('failed_submissions', 'fence_failures', 'query_failures'))):
        raise ValueError('Candidate metadata upload was not exercised cleanly')
    if old['cases'].keys() != new['cases'].keys() or any(
        old['cases'][k]['input_sha256'] != new['cases'][k]['input_sha256'] for k in old['cases']):
        raise ValueError('Inputs changed')
    selection_checks, changed = 0, []
    for key, before in old['cases'].items():
        after = new['cases'][key]
        if before['selections'].keys() != after['selections'].keys():
            changed.append(dict(case=key, reason='different selection boundaries'))
            continue
        for boundary, value in before['selections'].items():
            selection_checks += 1
            if value != after['selections'][boundary]:
                changed.append(dict(case=key, boundary=boundary))
    return dict(before=old, after=new,
        prefill_wall_change_percent=100*(new['prefill_wall']['tokens_per_second']/old['prefill_wall']['tokens_per_second']-1),
        model_interval_rate_change_percent=100*(new['prefill_model_interval']['model_interval_tokens_per_second']/
            old['prefill_model_interval']['model_interval_tokens_per_second']-1),
        pressure_mean_latency_change_percent=100*(new['pressure']['latency_seconds']['mean']/
            old['pressure']['latency_seconds']['mean']-1),
        checked_selections=selection_checks, changed_selections=changed,
        changed_outputs=[k for k in old['cases'] if old['cases'][k]['output_sha256'] != new['cases'][k]['output_sha256']],
        scope='One fixed-input eight-request pair. Report observed rates, not a statistically established '
              'stable speedup. Generation length and scheduling can change; total client seconds are not '
              'a pure prefill or decode comparison. Per-owner pressure waits overlap with other work.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    (args.root/'comparison.json').write_text(json.dumps(compare(args.root), indent=2)+'\n')
