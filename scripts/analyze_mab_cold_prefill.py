"""Measure initial MAB history prefill, excluding reuse, Query and decode."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def measure(root):
    fixture = json.loads((root / 'fixture.json').read_text())
    sample_id = fixture['rows'][0]['id']
    result = json.loads((root / 'run/answers/cases' / sample_id / 'result.json').read_text())
    assert not result['prefix_reused'], 'First case must execute the history'
    capture = root / 'run/server/capture'
    request = result['attention']['request_id']
    command = result['attention']['command_id']
    events = [row for row in lines(capture / 'async_refresh.jsonl')
              if row['request_id'] == request]
    query_start = [row['at'] for row in events
                   if row['command_id'] == command and row['stage'] == 'suspended']
    assert len(query_start) == 1
    end = query_start[0]
    steps = [row for row in lines(capture / 'model_step_latency.jsonl')
             if row['started_at'] < end and
             any(g['request_id'] == request for g in row['batch_geometry'])]
    assert steps and not any(row['decode_requests'] for row in steps)
    assert all(all(g['request_id'] == request and g['prefill']
                   for g in row['batch_geometry']) for row in steps), 'Single-owner interval required'
    assert not any(row['stage'] == 'query' and row['at'] < end for row in events)
    tokens = sum(g['query_tokens'] for row in steps for g in row['batch_geometry'])
    begin = steps[0]['started_at']
    history = result['attention']['history_begin']
    public_work = json.loads((capture / 'public_work.json').read_text())
    wall_offset = public_work['created_wall_time'] - public_work['created_at']
    gpu = [row for row in lines(root / 'run/gpu.jsonl')
           if begin + wall_offset <= row['at'] < end + wall_offset]
    assert gpu and not any(row.get('external_pids') for row in gpu)
    utilization = [row['devices'][0]['gpu_utilization_percent'] for row in gpu]
    # Instrumentation begins after the initial public forward; never charge
    # those unmeasured tokens to the measured interval.
    assert steps[0]['batch_geometry'][0]['logical_seq_len'] - steps[0]['batch_geometry'][0]['query_tokens'] == history - tokens
    commands = defaultdict(dict)
    for row in events:
        if row['command_id'] != command:
            assert row['stage'] not in commands[row['command_id']]
            commands[row['command_id']][row['stage']] = row
    timings = {}
    for start, stop in [('suspended', 'ready'), ('assembly_queued', 'assembly_started'),
                        ('assembly_queued', 'assembly_worker_started'),
                        ('assembly_worker_started', 'assembly_started'),
                        ('assembly_started', 'selection_prepared'),
                        ('selection_prepared', 'assembly_submitted'),
                        ('assembly_submitted', 'ready')]:
        values = [stages[stop]['at'] - stages[start]['at']
                  for stages in commands.values() if start in stages and stop in stages]
        if values:
            timings[f'{start}_to_{stop}'] = {'count': len(values), 'mean_seconds': mean(values),
                                            'total_seconds': sum(values)}
    return dict(root=str(root), sample_id=sample_id, history_tokens=history,
                executed_prefill_tokens=tokens, unmeasured_initial_tokens=history-tokens,
                phase_seconds=end-begin, executed_tokens_per_second=tokens/(end-begin),
                pressure_events=len(commands), measured_steps=len(steps),
                decode_requests_in_phase=0, query_forwards_in_phase=0,
                begin_first_model_step=begin, end_query_refresh_dispatch=end,
                inline_pressure_submissions=sum('pressure_assembly_submitted_inline' in s
                                                for s in commands.values()),
                model_event_seconds=sum(row['model_ms'] for row in steps)/1000,
                gpu_utilization_sample_mean_percent=mean(utilization),
                gpu_utilization_samples=len(utilization),
                pressure_timings=timings,
                scope='First timed history model step through final Query refresh dispatch. '
                      'Includes pressure, KV assembly, publication and scheduling. Excludes '
                      'model loading, untimed initial tokens, Query, warm restore and decode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [measure(root) for root in args.roots]
    assert len({(r['sample_id'], r['executed_prefill_tokens']) for r in rows}) == 1
    reference = rows[0]['executed_tokens_per_second']
    for row in rows:
        row['throughput_change_percent'] = 100 * (row['executed_tokens_per_second'] / reference - 1)
    with args.output.open('x') as out:
        json.dump({'rows': rows, 'caveat': 'Single cold-input measurements do not establish a stable gain.'}, out, indent=2)
    for row in rows:
        print(json.dumps({k: row[k] for k in ['root', 'executed_prefill_tokens', 'phase_seconds',
                                             'executed_tokens_per_second', 'throughput_change_percent',
                                             'pressure_events', 'inline_pressure_submissions']}))
