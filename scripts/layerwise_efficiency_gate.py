"""Bind Memory launches to exact full-input and matched throughput evidence."""
from pathlib import Path
import json
import math

import recall_comparison_protocol as p


def observed_prefill(arm):
    """Recompute the reported prefill envelope from saved request boundaries."""
    fixture=p.read(arm/'fixture.json');wall=p.read(arm/'concurrency-performance.json')
    if (p.read(arm/'status.json')['status']!='completed' or len(fixture['rows'])!=8 or
            wall['cases']!=8 or wall['workers']!=8):
        raise ValueError('Matched efficiency needs eight completed owners in every arm')
    attempts=list((arm/'formal/sequential/server-attempts').iterdir())
    if len(attempts)!=1:raise ValueError('Do not merge technical attempts into efficiency acceptance')
    capture=attempts[0]/'server/capture'
    refresh=capture/'bounded_refresh.jsonl'
    queries=[json.loads(line) for line in refresh.read_text().splitlines() if line]
    queries=[q for q in queries if q['selection']=='query']
    if len(queries)!=8:raise ValueError('Missing final Query timing boundaries')
    seconds=max(q['refresh_started_at'] for q in queries)-wall['started_monotonic']
    tokens=sum(q['history_begin'] for q in queries)
    if not math.isfinite(seconds) or seconds<=0 or tokens<=0:
        raise ValueError('Invalid prefill time or input extent')
    return dict(tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds),capture


def user_selection(path,pressure_runtime,performance):
    """Bind an explicit user-directed launch to the measured, qualified version."""
    path=Path(path).resolve();record=p.read(path)
    if (record.get('policy')!='user_selected_qualified_runtime' or
            not record.get('user_request','').strip() or
            Path(record['pressure_runtime']).resolve()!=Path(pressure_runtime).resolve() or
            Path(record['performance']).resolve()!=Path(performance).resolve() or
            record['performance_sha256']!=p.sha(Path(performance)/'performance-comparison.json') or
            record.get('preserve_original_memory_controls') is not True):
        raise ValueError('User selection does not match this qualified Memory runtime')
    return record


def create(pressure_runtime,performance,authorization=None):
    from run_layerwise_query_efficiency import source_audit
    from run_layerwise_matched_performance import assert_matched
    from run_layerwise_query_diagnostic import verify as verify_runtime
    pressure_runtime=pressure_runtime.resolve();performance=performance.resolve()
    checks=source_audit(pressure_runtime)
    if len(checks)!=8:raise ValueError('Memory requires the complete original full-input qualification')
    if not p.read(pressure_runtime/'native-reuse-audit.json')['passed']:
        raise ValueError('Qualified runtime did not retain native score execution')
    plan=p.read(performance/'plan.json')
    if (p.read(performance/'status.json')['status']!='completed' or
            Path(plan['source']).resolve()!=pressure_runtime or
            plan['source_equivalence_sha256']!=p.sha(pressure_runtime/'equivalence.json')):
        raise ValueError('Efficiency evidence belongs to an unfinished or different runtime')
    report=p.read(performance/'performance-comparison.json')
    if p.sha(performance/'launch.py')!=plan['launch_sha256']:
        raise ValueError('Matched efficiency launcher changed')
    files={str(performance/name):p.sha(performance/name) for name in
        ('plan.json','launch.py','status.json','performance-comparison.json')}
    files.update({str(pressure_runtime/name):p.sha(pressure_runtime/name) for name in
        ('equivalence.json','native-reuse-audit.json','fixture.json','profile.json','runtime-proof.json','prepared.json')})
    arms=[];rates={}
    for stage in plan['stages']:
        arm=Path(stage['root']);name=stage['name']
        verify_runtime(arm)
        if p.sha(arm/'prepared.json')!=stage['prepared_sha256']:
            raise ValueError('Frozen matched arm changed')
        fixture,profile=p.read(arm/'fixture.json'),p.read(arm/'profile.json')
        if fixture['rows']!=p.read(pressure_runtime/'fixture.json')['rows']:
            raise ValueError('Timed inputs differ from the qualified inputs')
        arms.append((fixture,profile));observed,capture=observed_prefill(arm)
        if fixture.get('cpu_region_gather_min_bytes') is not None or profile['kvmem'].get('cpu_direct_record_seal',False):
            from run_layerwise_matched_performance import archive_backend_audit
            archive=capture/'archive.json'
            actual=archive_backend_audit(profile,p.read(archive),fixture.get('cpu_region_gather_min_bytes'),
                                        require_region_execution=name!='shared_global')
            if actual!=report['reports'][name]['archive_backend']:
                raise ValueError('Region gather performance evidence differs from actual execution')
            files[str(archive)]=p.sha(archive)
        saved=report['reports'][name]['history_prefill_envelope']
        if not math.isclose(observed['tokens_per_second'],saved['tokens_per_second'],rel_tol=1e-12):
            raise ValueError('Saved prefill throughput disagrees with request boundaries')
        gpu=report['reports'][name]['gpu']
        if gpu['external_pids'] or gpu['attribution_coverage_fraction']<.95:
            raise ValueError('Matched GPU use cannot be attributed to this experiment')
        rates[name]=observed['tokens_per_second']
        for path in [*(arm/n for n in ('prepared.json','fixture.json','profile.json','runtime-proof.json','status.json','concurrency-performance.json')),
                     capture/'bounded_refresh.jsonl',capture/'native_attention_stats.json',capture.parent.parent/'gpu.jsonl']:
            files[str(path)]=p.sha(path)
    if set(rates)!={'shared_global','shared_layerwise_query','layerwise_optimized'}:
        raise ValueError('Missing matched recall control')
    assert_matched(arms)
    ratio=rates['layerwise_optimized']/rates['shared_layerwise_query']
    target=plan['target_layerwise_to_shared_throughput_ratio']
    if (not math.isfinite(target) or not 0<target<=1 or
            not math.isclose(ratio,report['ratios']['shared_layerwise_query']['history_prefill_envelope'],rel_tol=1e-12)):
        raise ValueError('Matched efficiency evidence is inconsistent')
    met=ratio>=target
    if bool(report['goal_met'])!=met:
        raise ValueError('Matched efficiency success flag contradicts the measurement')
    selection=None
    if authorization is not None:
        authorization=Path(authorization).resolve()
        selection=user_selection(authorization,pressure_runtime,performance)
        files[str(authorization)]=p.sha(authorization)
    if not met and selection is None:
        raise ValueError('Matched efficiency target is not met')
    return dict(source=str(performance),pressure_runtime=str(pressure_runtime),efficiency_accepted=met,
        memory_run_authorized=met or selection is not None,
        authorization_file=str(authorization) if selection is not None else None,
        qualified_cases=8,measured_prefill_ratio=ratio,target_prefill_ratio=target,files=files,
        scope='Original full-input exact qualification plus same-input/model/capacity/sampling eight-owner throughput. '
              'Prefill envelope recomputed from saved final Query boundaries; not pure GPU kernel speed.')


def verify(gate,pressure_runtime):
    if Path(gate['pressure_runtime']).resolve()!=Path(pressure_runtime).resolve():
        raise ValueError('Memory efficiency acceptance belongs to another runtime')
    for name,digest in gate['files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Bound efficiency evidence changed: '+name)
    if not gate['efficiency_accepted']:
        authorization=gate.get('authorization_file')
        if (not gate.get('memory_run_authorized') or not authorization or
                authorization not in gate['files'] or gate.get('qualified_cases')!=8):
            raise ValueError('Memory requires efficiency acceptance or explicit user selection')
        user_selection(authorization,pressure_runtime,gate['source'])
