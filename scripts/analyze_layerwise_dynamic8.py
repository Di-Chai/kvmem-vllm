"""Read-only GPU-confirmed historical-prefill ABBA analysis.

The finite gate intentionally separates all history prefill from final Query
and decode. Rates include pressure, assembly, publication and scheduling;
they are not continuous-service or isolated-kernel throughput.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from analyze_mab_prefill_concurrency import measure as historical_measure, lines
from audit_layerwise_gilopt import read, save_new


REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_endpoint(release, owners=8):
    if (release.get('expected') != owners or len(release.get('owners',{})) != owners or
            not release.get('gpu_completion_confirmed')):
        raise ValueError('Every cohort owner must reach a GPU-confirmed prefill endpoint')
    work = release['public_work']
    submitted, completed = work['submitted'],work['completed']
    if (not work['all_submitted_completed'] or work.get('pending_fence') is not None or
            submitted != completed):
        raise ValueError('The endpoint includes unfinished public GPU work')
    total = sum(release['owners'].values())
    if (total != release['history_tokens'] or completed['public_prefill_tokens'] != total or
            completed['public_non_prefill_tokens'] or completed['decode_request_batches']):
        raise ValueError('GPU token accounting does not describe only the complete cold histories')
    fence = work.get('completed_fence')
    if not work.get('counter_id') or not fence:
        raise ValueError('The completed GPU work lacks fence provenance')
    if not fence['completion_lower_bound'] <= fence['completion_upper_bound'] <= release['at']:
        raise ValueError('GPU completion was not observed before Query release')
    return work


def controls(root):
    profile,fixture = read(root/'profile.json'),read(root/'fixture.json')
    kv = profile['kvmem']
    if (profile['execution_mode'] != 'compiled_graph' or profile['draft_tokens'] != 4 or
            profile['max_num_seqs'] != 8 or profile['max_num_batched_tokens'] != 16424 or
            [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')]
            != [36864,16384,32,41,0]):
        raise ValueError('Original model execution or Active/Gen/recall geometry changed')
    if fixture['cases'] != 8 or len(fixture['rows']) != 8:
        raise ValueError('Eight complete inputs are required')
    # This is the original memory driver's CPU admission reservation, not an
    # estimate of process RSS. The finite gate needs every owner admitted.
    credit = sum(((row['input_tokens']+2*kv['gen_budget']+31)//32)*32*17*4*256*2
                 for row in fixture['rows'])
    capacity = kv['cpu_read_cache_bytes']-16*1024**3
    if credit >= capacity:
        raise ValueError('The fixed cohort can deadlock behind CPU admission')
    normalized_profile = deepcopy(profile)
    for name in ('gpu_memory_admission','request_swap_layerwise','request_checkpoint_bytes'):
        normalized_profile['kvmem'].pop(name,None)
    normalized_fixture = deepcopy(fixture)
    for row in normalized_fixture['rows']:
        row.pop('source',None)
    for row in normalized_fixture['bindings']:
        row.pop('request',None)
    return profile,fixture,dict(profile=normalized_profile,fixture=normalized_fixture),credit,capacity


def measure(root):
    if read(root/'status.json')['status'] != 'completed':
        raise ValueError('Complete and preserve the first answers before final analysis')
    profile,fixture,_,credit,capacity = controls(root)
    capture = root/'run/server/capture'
    release, = [event for event in lines(capture/'prefill_phase_gate.jsonl')
                if event['event']=='history_prefill_released']
    work = completed_endpoint(release)
    final_work = read(capture/'public_work.json')
    if final_work['counter_id'] != work['counter_id']:
        raise ValueError('GPU work counters restarted during the measured run')
    # The legacy analyzer independently rejects Query/decode in this interval,
    # foreign GPU owners, unpaired pressure events and missing cohort owners.
    result = historical_measure(root)
    if result['history_tokens'] != work['completed']['public_prefill_tokens']:
        raise ValueError('Legacy phase and GPU-completed token counts disagree')
    request_ids = set()
    for case in fixture['case_ids']:
        answer = read(root/'run/answers/cases'/case/'result.json')
        if answer['prefix_reused']:
            raise ValueError('Warm prefix reuse cannot enter a cold-prefill numerator')
        request_ids.add(answer['attention']['request_id'])
    if len(request_ids)!=8 or request_ids != set(release['owners']):
        raise ValueError('All eight completed answers must match measured prefill owners')
    start = min(read(path)['at'] for path in (root/'run/answers/cases').glob('*/attempt.json'))
    offset = work['created_wall_time']-work['created_at']
    steps = [row for row in lines(capture/'model_step_latency.jsonl')
             if start-offset <= row['started_at'] < release['at']]
    first_seen = {}
    for step in steps:
        for geometry in step['batch_geometry']:
            first_seen.setdefault(geometry['request_id'],step['started_at'])
    arrivals = [event['at'] for event in lines(capture/'prefill_phase_gate.jsonl')
                if event['event']=='history_prefill_arrived']
    if set(first_seen)!=request_ids or max(first_seen.values())>=min(arrivals):
        raise ValueError('The run did not have an overlapping eight-owner prefill cohort')
    timed_tokens = sum(row['prefill_tokens'] for row in steps)
    if not 0<timed_tokens<=release['history_tokens']:
        raise ValueError('Timed model steps are inconsistent with completed historical work')
    scheduler,archive = read(capture/'scheduler_resources.json'),read(capture/'archive_resources.json')
    cleanup = read(root/'run/cleanup.json')
    if (cleanup['gpu_processes'] or cleanup['remaining_owned_groups'] or
            scheduler['requests'] or scheduler['owned_blocks'] or scheduler['reservations']):
        raise ValueError('The completed run retained GPU ownership')
    checkpoints = archive.get('request_checkpoints')
    if checkpoints:
        resource = checkpoints['resources']['resources']['checkpoint_bytes']
        if checkpoints['requests'] or resource['live'] or resource['promised']:
            raise ValueError('The completed run retained native checkpoint bytes')
    autotune = read(root/'autotune-audit.json')
    result.update(
        kind=profile['kvmem']['gpu_memory_admission'],
        gpu_completed_history_tokens=work['completed']['public_prefill_tokens'],
        gpu_completion_fence=work['completed_fence'],public_work_counter_id=work['counter_id'],
        gpu_counter_batch_composition=work['completed'].get('batch_composition'),
        untraced_initial_or_dropped_tokens=release['history_tokens']-timed_tokens,
        timed_model_coverage_fraction=timed_tokens/release['history_tokens'],
        timed_model_interval_seconds=release['at']-steps[0]['started_at'],
        timed_model_interval_tokens_per_second=timed_tokens/(release['at']-steps[0]['started_at']),
        eight_owner_overlap_seconds=min(arrivals)-max(first_seen.values()),
        cpu_client_reserved_bytes=credit,cpu_client_credit_bytes=capacity,
        cpu_history_admission=scheduler['cpu_history_admission'],
        archive_capacity_deferrals=scheduler['archive_capacity_deferrals'],
        gpu_capacity_final=scheduler['gpu_capacity'],
        request_swaps=scheduler.get('request_swaps'),request_checkpoints=checkpoints,
        swap_counter_scope='Whole run including later Query/decode. Zero full-run swaps implies zero in prefill.',
        cpu_archive_peak_bytes_full_run=archive.get('cpu_peak_bytes'),
        autotune_unchanged=autotune['unchanged'],autotune_initial=autotune['initial'],
        source_files={str(path):sha(path) for path in (
            root/'fixture.json',root/'profile.json',root/'autotune-audit.json',
            capture/'prefill_phase_gate.jsonl',capture/'public_work.json',
            capture/'model_step_latency.jsonl',capture/'scheduler_resources.json',
            capture/'archive_resources.json',root/'run/cleanup.json')})
    result['scope'] += ' GPU completion is now confirmed before the final gate release. '
    result['caveats'].extend([
        'The primary wall interval starts at first client attempt and includes submission/admission overhead.',
        'The timed-model secondary rate excludes untimed initialization tokens and may lose dropped samples.',
        'CPU/allocator/swap cumulative counters below cover the full run, not only the prefill interval.',
        'This roomy-pool test can have no swaps; it cannot establish a benefit under memory overcommit.'])
    return result


def aggregate(reports):
    kinds = [row['kind'] for row in reports]
    if kinds != ['static','incremental','incremental','static']:
        raise ValueError('Require the predeclared static/dynamic/dynamic/static order')
    if len({row['gpu_completed_history_tokens'] for row in reports}) != 1:
        raise ValueError('ABBA runs must execute exactly the same history token count')
    pooled = {}
    for kind in ('static','incremental'):
        selected = [row for row in reports if row['kind']==kind]
        seconds = sum(row['seconds'] for row in selected)
        tokens = sum(row['gpu_completed_history_tokens'] for row in selected)
        if seconds<=0 or tokens<=0:
            raise ValueError('Throughput requires positive completed work and wall time')
        pooled[kind]=dict(tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
                          per_repeat_tokens_per_second=[row['tokens_per_second'] for row in selected])
    return dict(pooled=pooled,
                dynamic_change_percent=100*(pooled['incremental']['tokens_per_second']/
                                            pooled['static']['tokens_per_second']-1),
                scope='Pooled rate is sum(completed history tokens)/sum(prefill wall seconds); '
                      'two observations per mode, not a confidence interval.')


def analyze(root, output):
    plan = read(root/'plan.json')
    if len(plan['order']) != 4:
        raise ValueError('Four predeclared ABBA arms are required')
    normalized = [controls(root/name)[2] for name in plan['order']]
    if any(row!=normalized[0] for row in normalized[1:]):
        raise ValueError('Input, sampling, recall or execution controls differ beyond allocation mode')
    reports = [dict(name=name,**measure(root/name)) for name in plan['order']]
    autotune_equal = all(row['autotune_initial']==reports[0]['autotune_initial'] for row in reports)
    stable_autotune = autotune_equal and all(row['autotune_unchanged'] for row in reports)
    summary = dict(**aggregate(reports),reports=reports,
        measurement_passed=True,common_unchanged_autotuning=stable_autotune,
        controlled_performance_comparison=stable_autotune,
        source_files={str(Path(__file__).resolve()):sha(__file__),
                      str(REPO/'scripts/analyze_mab_prefill_concurrency.py'):
                          sha(REPO/'scripts/analyze_mab_prefill_concurrency.py'),
                      str(REPO/'scripts/audit_layerwise_gilopt.py'):
                          sha(REPO/'scripts/audit_layerwise_gilopt.py'),
                      str(REPO/'tests/test_analyze_layerwise_dynamic8.py'):
                          sha(REPO/'tests/test_analyze_layerwise_dynamic8.py'),
                      str(root/'plan.json'):sha(root/'plan.json')},
        formal_resume=False,external_grading=False)
    save_new(output,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root.resolve(),args.output.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('reports','source_files')},indent=2))
