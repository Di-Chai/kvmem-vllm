#!/usr/bin/env python3
"""Summarize completed small graph qualifications without reading token counts as FLOPs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from qualify_block_graph import ARMS, compare_collect_baseline, digest


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_comparison_profiles(directories):
    """Check every arm, including snapshots from older unhashed controllers."""
    controls, fixtures, legacy = {}, set(), []
    sync_values = dict(async_refresh=False, async_restore=False,
        cpu_history_coordination=False, maintenance_max_steps=0)
    for directory in directories:
        manifest = read(directory/'manifest.json')
        key = (manifest['arm'], manifest['synchronous'])
        if key in controls:
            raise ValueError(f'Duplicate qualification arm: {key}')
        result = read(directory/'qualification/results.json')
        if result['arm'] != manifest['arm']:
            raise ValueError('Manifest and result arms differ')
        profile = read(directory/'profile.json')
        if profile != read(directory/'qualification/profile.json'):
            raise ValueError('Executed qualification profile differs from controller profile')
        fixture = read(directory/'qualification/fixture.json')
        actual = digest({k:v for k,v in fixture.items() if k != 'sha256'})
        if actual != fixture.get('sha256') or actual != result.get('fixture_sha256'):
            raise ValueError('Saved qualification fixture changed')
        fixtures.add(actual)
        hashes = ('profile_sha256', 'fixture_sha256')
        if all(field in manifest for field in hashes):
            for field, path in (('profile_sha256', directory/'profile.json'),
                    ('fixture_sha256', Path(manifest['fixture']))):
                if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
                    raise ValueError('Frozen profile or fixture changed')
        elif any(field in manifest for field in hashes):
            raise ValueError('Partial profile/fixture freeze metadata')
        else:
            legacy.append(str(directory))
        cfg = deepcopy(profile)
        cfg.pop('port', None)
        cfg.pop('served_model_name', None)
        kv = cfg['kvmem']
        expected_mode = key[0] if key[0] in ('collect', 'locality', 'attention', 'multi') else 'off'
        if kv.get('graph_mode') != expected_mode or kv.get('graph_trace') is not True:
            raise ValueError('Graph mode or tracing differs from the declared arm')
        if kv.pop('action_fraction', None) != (.25 if key[0] == 'legacy' else 0.):
            raise ValueError('Legacy attention fusion differs from the declared arm')
        if key[1] and any(kv.get(k) != v for k,v in sync_values.items()):
            raise ValueError('Synchronous arm did not disable all four asynchronous controls')
        for name in tuple(kv):
            if name.startswith('graph_'):
                del kv[name]
        controls[key] = cfg
    if {arm for arm, sync in controls if not sync} != set(ARMS):
        raise ValueError('Comparison requires all six asynchronous arms')
    baseline = controls[('baseline', False)]
    for (arm, sync), cfg in controls.items():
        if sync:
            for name in sync_values:
                if name in baseline['kvmem']:
                    cfg['kvmem'][name] = baseline['kvmem'][name]
                else:
                    cfg['kvmem'].pop(name, None)
        if cfg != baseline:
            raise ValueError(f'Non-graph model or inference controls differ: {arm}, synchronous={sync}')
    if len(fixtures) != 1:
        raise ValueError('Fixture inputs or later questions differ between arms')
    return dict(all_arm_controls_verified=True, fixture_sha256=next(iter(fixtures)),
        profile_fixture_hashed_before_run=not legacy, legacy_manifest_runs=legacy,
        verification='Saved executed profiles and complete fixture snapshots compared for every arm; '
            'legacy manifests did not hash profiles/fixtures before launch.')


def summary(directory):
    result = read(directory / 'qualification/results.json')
    cleanup = read(directory / 'qualification/cleanup.json')
    controller = read(directory / 'cleanup.json')
    if (result['status'] != 'passed' or not result['all_owned_sessions_released'] or
            not cleanup['all_owned_sessions_released'] or controller['remaining_gpu_pids']):
        raise ValueError(f'Incomplete qualification: {directory}')
    rows = read(directory / 'qualification/traces.json')
    capture = directory / 'server/capture'
    actions = lines(capture / 'action_attention.jsonl')
    # Omit the explicit release control responses from cumulative observations.
    actions = [a for a in actions if any(a['request_id'].startswith(r['response_id']+'-') for r in rows)]
    graphs = lines(capture / 'block_graph.jsonl') if (capture / 'block_graph.jsonl').exists() else []
    checkpoints = lines(capture / 'checkpoint_timing.jsonl')
    archive = cleanup['state']['archive']
    first = [r for r in rows if r['turn'] == 0]
    pooled_n = sum(len(r['output_ids'])-r['chunks'][0][1] for r in rows)
    pooled_t = sum(r['chunks'][-1][0]-r['chunks'][0][0] for r in rows)
    def maximum(key):
        return max((a.get(key, 0) for a in actions), default=0)
    def median(key, source):
        vals = [r[key] for r in source if key in r]
        return statistics.median(vals) if vals else None
    return dict(artifact=str(directory), arm=result['arm'],
        synchronous=read(directory / 'manifest.json')['synchronous'],
        status=result['status'], quality=result['quality'], seconds=result['seconds'],
        output_tokens=result['output_tokens'], end_to_end_output_tokens_per_second=result['end_to_end_output_tokens_per_second'],
        per_request_generation_period_pooled_tokens_per_second=pooled_n/pooled_t,
        per_request_seconds_median=median('seconds', rows), ttft_seconds_median=median('ttft_seconds', rows),
        first_turn_input_hashes={str(r['agent']):r['input_sha256'] for r in first},
        step_metrics=result['step_metrics'],
        observed_attention_capture_host_seconds=maximum('capture_host_seconds_total'),
        graph_prepare_seconds=maximum('graph_prepare_seconds'),
        graph_prepare_calls=maximum('graph_prepare_calls'),
        graph_commit_seconds=maximum('graph_commit_seconds'),
        graph_propagation_seconds_median=median('scoring_seconds', graphs),
        graph_selection_with_trace_seconds_median=median('graph_selection_seconds', graphs),
        graph_added_blocks=sum(g.get('added_blocks', 0) for g in graphs),
        graph_observed_global_bytes_max=max(
            [g.get('graph_global_bytes', 0) for g in graphs]+
            [a.get('block_graph', {}).get('global_bytes', 0) for a in actions]+[0]),
        checkpoint_host_total_seconds_median=median('host_total_seconds', checkpoints),
        checkpoint_event_count=len(checkpoints),
        graph_checks={k:v for k,v in result['graph_checks'].items() if k != 'graph_terminals'},
        graph_terminal_counts=[{k:term['terminal'].get(k) for k in
            ('accepted_through', 'prefill_samples', 'output_samples', 'observed_sources', 'edges')}
            | {'agent':term['agent'], 'turn':term['turn']} for term in result['graph_checks']['graph_terminals']],
        verified_kv_restores=lines(capture / 'verified_restores.jsonl'),
        verified_recurrent_restores=lines(capture / 'verified_state_restores.jsonl'),
        final_resources={k:archive[k] for k in ('cpu_authority_bytes', 'session_checkpoints',
            'session_checkpoint_payload_bytes', 'block_graph_bytes') if k in archive})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = [read(p / 'manifest.json') for p in args.runs]
    if any(m['runtime_sha256'] != manifests[0]['runtime_sha256'] for m in manifests):
        raise ValueError('The runs do not share one frozen runtime')
    controls = validate_comparison_profiles(args.runs)
    rows = [summary(p) for p in args.runs]
    if any(r['first_turn_input_hashes'] != rows[0]['first_turn_input_hashes'] for r in rows):
        raise ValueError('First-turn prompts differ')
    by_arm = {r['arm']:Path(r['artifact']) for r in rows if not r['synchronous']}
    comparison = compare_collect_baseline(by_arm['baseline']/'qualification', by_arm['collect']/'qualification')
    comparison.pop('selected_history_ranges', None)
    report = dict(version=1, shared_runtime_verified=True, identical_first_inputs=True,
        collect_baseline=comparison, comparison_controls=controls, runs=rows,
        limitations=[
            'Six answers per arm are a functional check, not a statistically powered benchmark.',
            'Later prompts contain each arm\'s own answers; only the first prompts are identical.',
            'Tracing, restore byte verification and per-step profiling are enabled in every arm.',
            'Generation-period rate is per request, includes interference, and is not aggregate pure decode.',
            'Attention host counters include dispatch/synchronization, not isolated GPU kernel duration.',
            'Graph preparation includes aggregation/top-k; checkpoint totals include model state and graph.',
            'Observed byte maxima are sampled records, not a continuously measured memory high-water mark.',
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for r in rows:
        print(r['arm']+('-sync' if r['synchronous'] else ''),
            f"{r['quality']['correct']}/{r['quality']['total']}",
            f"{r['seconds']:.2f}s", f"{r['end_to_end_output_tokens_per_second']:.2f} output/s",
            f"{r['step_metrics']['pure_prefill_model_tokens_per_second']:.0f} prefill/s")


if __name__ == '__main__':
    main()
