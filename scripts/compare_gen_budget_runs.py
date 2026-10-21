"""Compare a fixed-input G16K/G64K cohort under unchanged resource limits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from compare_pressure_runs import read, records, summarize


def full_prefill_window(work):
    """Include later admission waves, even when earlier requests are decoding."""
    total = max(r['completed']['public_prefill_tokens'] for r in work)
    if not total:
        raise ValueError('No completed public prefill')
    first = next(i for i, r in enumerate(work) if r['completed']['public_prefill_tokens'])
    if first == 0:
        raise ValueError('Missing initial zero counter')
    start = work[first - 1]
    end = next(r for r in work if r['completed']['public_prefill_tokens'] == total)
    seconds = end['wall_time'] - start['wall_time']
    if seconds <= 0:
        raise ValueError('Invalid counter interval')
    return dict(executed_tokens=total, seconds=seconds, tokens_per_second=total / seconds,
        scope='All public prefill across all admission waves, from the last initial zero '
              'counter to the first observation of the final prefill count. Includes '
              'selection, archive, scheduling and overlapping decode; excludes cold loading. '
              'Counter publication is approximately once per second.')


def resources(root):
    rows = records(root / 'run/resources.jsonl')
    memory = [float(r['gpu'].split(',')[0]) * 1024**2 for r in rows]
    used = [r['cpu']['MemTotal'] - r['cpu']['MemAvailable'] for r in rows]
    mtp = {}
    for line in rows[-1]['native_metrics']:
        for key in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
            if line.startswith('vllm:spec_decode_' + key + '_total{'):
                mtp[key] = float(line.rsplit(' ', 1)[-1])
    if mtp.get('num_drafts'):
        mtp['accepted_length'] = 1 + mtp['num_accepted_tokens'] / mtp['num_drafts']
    queue = read(root / 'run/answers/queue.json')
    geometry = read(root / 'run/server/capture/cache-geometry.json')
    return dict(gpu_peak_bytes=max(memory), gpu_initial_bytes=memory[0],
        system_used_memory_initial_bytes=used[0], system_used_memory_peak_bytes=max(used),
        system_available_min_bytes=min(r['cpu']['MemAvailable'] for r in rows),
        queue_peak_inflight=queue['peak_inflight'], queue_peak_bytes=queue['peak_reserved_bytes'],
        queue_capacity_bytes=queue['history_capacity_bytes'],
        per_case_queue_seconds={r['id']:r['admitted_at'] - r['queued_at'] for r in queue['tasks']},
        native_capacity_bytes=geometry['capacity_bytes'], native_blocks=geometry['native_blocks'],
        mtp_last_sample=mtp,
        scope='5-second sampling: GPU device-wide memory; CPU system used memory, not model RSS. '
              'Last MTP snapshot may precede final output; it is not an exact whole-run acceptance count.')


def compare(before, after, *, cases=8):
    old, new = [summarize(root, expected_cases=cases) for root in (before, after)]
    a, b = deepcopy(old['profile']), deepcopy(new['profile'])
    if a['kvmem'].pop('gen_budget') != 16384 or b['kvmem'].pop('gen_budget') != 65536 or a != b:
        raise ValueError('Require only G16K -> G64K profile difference')
    allowed = {'scripts/guided_benchmark_protocol.py', 'scripts/qualify_graph_niah.py',
               'scripts/benchmark_ruler.py', 'scripts/run_ruler_campaign.py'}
    af, bf = old['runtime_files'], new['runtime_files']
    if af.keys() != bf.keys():
        raise ValueError('Runtime inventory changed')
    changed = [name for name in af if af[name] != bf[name]]
    if not set(changed) <= allowed:
        raise ValueError('Model or unrelated code changed: ' + str(changed))
    if old['cases'].keys() != new['cases'].keys() or any(
            old['cases'][k]['input_sha256'] != new['cases'][k]['input_sha256'] for k in old['cases']):
        raise ValueError('Different input cohort')
    fixtures = [read(root / 'fixture.json') for root in (before, after)]
    for f in fixtures:
        f.pop('profile_sha256')
    if fixtures[0] != fixtures[1]:
        raise ValueError('Sampling, inputs or client queue limits changed')
    for root, summary in ((before, old), (after, new)):
        summary['all_prefill_wall'] = full_prefill_window(records(root / 'run/server/capture/public_work.jsonl'))
        summary['resources'] = resources(root)
        summary['execution_audit'] = read(root / 'run/execution-audit.json')
        if cases == 1:
            ttft = next(iter(summary['cases'].values()))['ttft_seconds']
            summary['input_to_first_token'] = dict(seconds=ttft, input_tokens=summary['input_tokens'],
                tokens_per_second=summary['input_tokens'] / ttft,
                scope='Single request: original input tokens divided by client time to first output token. '
                      'Includes input submission, prefill, refreshes, Query selection and first output; '
                      'excludes model loading. This is effective input processing rate, not kernel throughput.')
    return dict(cases=cases, before=old, after=new, changed_harness_files=changed,
        all_prefill_wall_rate_change_percent=100 * (new['all_prefill_wall']['tokens_per_second'] /
                                                   old['all_prefill_wall']['tokens_per_second'] - 1),
        prefill_model_interval_rate_change_percent=100 * (
            new['prefill_model_interval']['model_interval_tokens_per_second'] /
            old['prefill_model_interval']['model_interval_tokens_per_second'] - 1),
        client_wall_change_percent=100 * (new['client_seconds'] / old['client_seconds'] - 1),
        scope='One fixed-input pair under unchanged resource budgets. Admission concurrency can differ; '
              'report it explicitly. G changes attention visibility and refresh boundaries, so this is '
              'a configuration comparison, not a semantics-preserving optimization. Output trajectories '
              'and lengths can differ. Do not treat first-wave prefill as all eight inputs or add '
              'overlapping owner refresh waits into wall time.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.before, args.after, cases=args.cases), indent=2) + '\n')
