"""Audit a single full-input chunk diagnostic and its unmodified vLLM control."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from compare_pressure_runs import read, records, summarize
from compare_gen_budget_runs import full_prefill_window, resources


def sampled_positions(root, stride):
    rows = records(root / 'run/server/capture/prefill_attention.jsonl')
    # PrefillHistory.rows samples backwards from count - 1 at the fixed stride.
    positions = [r['end'] - 1 - stride * i for r in rows for i in range(r['query_samples'])]
    if len(set(positions)) != len(positions) or len(positions) != rows[-1]['samples']:
        raise ValueError('Duplicate or missing committed prefill samples')
    return sorted(positions), rows


def summarize_kvmem(root):
    result = summarize(root, expected_cases=1)
    result['root'] = str(root.resolve())
    result['all_prefill_wall'] = full_prefill_window(records(root / 'run/server/capture/public_work.jsonl'))
    result['resources'] = resources(root)
    result['execution_audit'] = read(root / 'run/execution-audit.json')
    if not result['execution_audit']['passed']:
        raise ValueError('Execution audit failed')
    case = next(iter(result['cases'].values()))
    result['effective_input_tokens_per_second'] = result['input_tokens'] / case['ttft_seconds']
    result['archive'] = {k: v for k, v in read(root / 'run/server/capture/archive.json').items()
                         if k.startswith('seal_') or k in ('disk_read_bytes', 'disk_write_bytes')}
    return result


def compare(before, after, native):
    old, new = [summarize_kvmem(root) for root in (before, after)]
    profiles = [deepcopy(x['profile']) for x in (old, new)]
    if [p.pop('prefill_chunk_tokens') for p in profiles] != [1024, 2048] or profiles[0] != profiles[1]:
        raise ValueError('Only the ordinary prefill chunk may differ')
    fixtures = [read(root / 'fixture.json') for root in (before, after)]
    for f in fixtures:
        f.pop('profile_sha256')
    if fixtures[0] != fixtures[1]:
        raise ValueError('Sampling or input fixture changed')
    af, bf = old['runtime_files'], new['runtime_files']
    if af.keys() != bf.keys():
        raise ValueError('Frozen runtime inventories differ')
    changed = [name for name in af if af[name] != bf[name]]
    if not set(changed) <= {'scripts/guided_benchmark_protocol.py', 'scripts/qualify_graph_niah.py',
                            'scripts/benchmark_ruler.py', 'scripts/run_ruler_campaign.py'}:
        raise ValueError('Model implementation changed')
    case_id, = old['cases']
    if (list(new['cases']) != [case_id] or old['cases'][case_id]['input_sha256'] !=
            new['cases'][case_id]['input_sha256']):
        raise ValueError('Input identity changed')
    stride = old['execution_audit']['action_config']['prefill_stride']
    if new['execution_audit']['action_config']['prefill_stride'] != stride:
        raise ValueError('Sampling stride changed')
    positions_a, chunks_a = sampled_positions(before, stride)
    positions_b, chunks_b = sampled_positions(after, stride)
    evidence = [read(root / 'run/answers/cases' / case_id / 'evidence.json') for root in (before, after)]
    events = [[*e['pressure'], e['query']] for e in evidence]
    boundaries = [[(e['selection'], e['history_begin']) for e in es] for es in events]
    selections = []
    if boundaries[0] == boundaries[1]:
        for a, b in zip(*events):
            sa, sb = [set(map(tuple, e['selected_history_ranges'])) for e in (a, b)]
            selections.append(dict(phase=a['selection'], history_begin=a['history_begin'],
                identical=sa == sb, page_jaccard=len(sa & sb) / len(sa | sb)))
    n = read(native / 'result.json')
    launch = read(native / 'native-launch.json')
    case = read(native / 'case.json')
    status = read(native / 'status.json')
    if status['status'] != 'completed' or not status['native_idle']:
        raise ValueError('Native baseline did not complete')
    expected_sampling = dict(fixtures[0]['sampling'], seed=case['row']['seed'])
    reference_row = next(r for r in fixtures[0]['rows'] if r['id'] == case_id)
    if (n['input_sha256'] != old['cases'][case_id]['input_sha256'] or
            n['input_tokens'] != old['input_tokens'] or case['sampling'] != expected_sampling or
            case['row']['seed'] != reference_row['seed']):
        raise ValueError('Native input or sampling differs')
    from benchmark_native_prefill import native_argv
    expected = native_argv(read(before / 'run/server/launch.json'), launch['port'],
                           batch_tokens=launch.get('diagnostic_batch_tokens'))
    if launch['argv'][4:] != expected[4:] or not launch['plugins_disabled'] or not launch['kvmem_environment_absent']:
        raise ValueError('Native baseline contains overrides or extensions')
    for root in (after, native):
        cleanup = root / ('run/cleanup.json' if root == after else 'cleanup.json')
        if read(cleanup)['remaining_owned_groups'] or read(root / 'service-restore-status.json')['status'] != 'running':
            raise ValueError('Owned processes or production restoration incomplete')
    final = read(native / 'final-metrics.json')['metrics']
    mtp = {}
    request_metrics = {}
    for line in final:
        key = line.split('{', 1)[0]
        if key in ('vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_draft_tokens_total',
                   'vllm:spec_decode_num_accepted_tokens_total'):
            mtp[key] = float(line.rsplit(' ', 1)[1])
        if key.startswith(('vllm:request_prefill_time_seconds_', 'vllm:time_to_first_token_seconds_',
                           'vllm:request_inference_time_seconds_', 'vllm:request_decode_time_seconds_')) and key.endswith(('_sum', '_count')):
            request_metrics[key] = float(line.rsplit(' ', 1)[1])
    drafts = mtp.get('vllm:spec_decode_num_drafts_total', 0)
    if drafts:
        mtp['accepted_length'] = 1 + mtp['vllm:spec_decode_num_accepted_tokens_total'] / drafts
    telemetry = records(native / 'resources.jsonl')
    server_log = (native / 'server.log').read_text().splitlines()
    n.update(root=str(native.resolve()), cold_prepare_seconds=status['cold_prepare_seconds'],
        batch_tokens=int(launch['argv'][launch['argv'].index('--max-num-batched-tokens') + 1]),
        native_final_mtp=mtp, request_metrics=request_metrics,
        untuned_shape_warnings=[line for line in server_log if 'No tuned config covers' in line],
        inference_jit_warnings=[line for line in server_log if 'during inference:' in line],
        sampled_gpu_peak_bytes=max(float(r['gpu'].split(',')[0]) * 1024**2 for r in telemetry),
        sampled_system_used_memory_peak_bytes=max(r['cpu']['MemTotal'] - r['cpu']['MemAvailable'] for r in telemetry))
    return dict(before=old, after=new, native=n, changed_harness_files=changed,
        chunk_effective_input_rate_change_percent=100 * (new['effective_input_tokens_per_second'] /
                                                        old['effective_input_tokens_per_second'] - 1),
        semantics_audit=dict(sampling_stride=stride, before_samples=len(positions_a), after_samples=len(positions_b),
            sampled_positions_identical=positions_a == positions_b,
            positions_sha256=hashlib.sha256(json.dumps(positions_a).encode()).hexdigest(),
            before_committed_chunks=len(chunks_a), after_committed_chunks=len(chunks_b),
            selection_boundaries_identical=boundaries[0] == boundaries[1], selections=selections,
            mean_page_jaccard=statistics.mean(s['page_jaccard'] for s in selections) if selections else None,
            score_equivalence=False, reason='Per-chunk mass/exposure averaging and decay are chunk-dependent; '
                'identical sampling positions do not preserve score weights. Actual selections differ.'),
        scope='One single-request complete 256K input per arm. Input/TTFT excludes cold loading. '
              'Native vLLM attends to the full context; KVMem uses a bounded active context. '
              'This is not a matched-attention-domain overhead isolation or a quality benchmark. '
              'CUDA model intervals exclude final mixed batches; archive timers may overlap. '
              'Device/system memory is sampled at 5 seconds; KVMem final MTP snapshots can miss short decode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'native'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.before, args.after, args.native), indent=2) + '\n')
