"""Eight complete cold histories, varying only pressure copy execution.

Reuse the validated GPU-completion gate and explicit pressure server shim.
Both arms use static allocation and identical original inputs, budgets and
sampling. First answers are preserved even when concurrent trajectories differ.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys


HELPER = '_pressure_stage_driver.py'
helper = Path(__file__).with_name(HELPER)
if not helper.exists():
    helper = Path(__file__).with_name('benchmark_pressure_stage_efficiency.py')
spec = importlib.util.spec_from_file_location('pressure_eight_helper', helper)
pressure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pressure)
base = pressure.base
GATE = base.GATE
ANALYSIS_MODULES = ('analyze_layerwise_dynamic8.py', 'analyze_mab_prefill_concurrency.py')


def cohort_controls(fixture, profile, case_ids):
    base.validate_controls(fixture, profile)
    ids = tuple(case_ids)
    if (len(ids) != 8 or len(set(ids)) != 8 or
            len({cid.rsplit('_q', 1)[0] for cid in ids}) != 8):
        raise ValueError('Eight distinct original cold-prefix owners required')
    rows, _ = base.workload(fixture, ids)
    kv = profile['kvmem']
    credits = sum(((row['input_tokens'] + 2*kv['gen_budget'] + 31)//32)*32*17*4*256*2
                  for row in rows)
    capacity = kv['cpu_read_cache_bytes'] - 16*1024**3
    if credits >= capacity:
        raise ValueError('All gated owners must fit the unchanged CPU history credits')
    if profile['kv_cache_memory_bytes']//56_262_656 <= 8*48:
        raise ValueError('All gated owners must fit the unchanged static GPU pool')
    return dict(cpu_reservation_bytes=credits, cpu_client_credit_bytes=capacity)


def prepare(root, *, source_arm, fixture_path, autotune_source, variants,
            order=('baseline', 'candidate'), port=19468, overlays=(), overlay_source=base.REPO,
            prefill_state_binary=None):
    root, source_arm = Path(root).resolve(), Path(source_arm).resolve()
    fixture = base.read(fixture_path)
    ids = tuple(fixture['case_ids'])
    credits = cohort_controls(fixture, base.read(source_arm/'profile.json'), ids)
    pressure.prepare(root, source_arm=source_arm, fixture_path=fixture_path,
                     autotune_source=autotune_source, case_ids=ids, overlays=overlays,
                     overlay_source=overlay_source,
                     variants=variants, order=order, port=port,
                     prefill_state_binary=prefill_state_binary)
    plan = pressure.verify(root)
    # Still prepared, never launched: install the same already qualified
    # measurement gate in every arm and freeze the resulting proofs anew.
    runtimes = [root/'source-runtime'] + [root/t['name']/'runtime-v1' for t in plan['trials']]
    for runtime in runtimes:
        path = runtime/'src/vllm_kvmem/layerwise_query.py'
        text = path.read_text()
        if base.ungated(text) != text:
            raise ValueError('Unexpected pre-existing measurement gate')
        path.write_text(text + GATE)
    proof = base.tree(runtimes[0])
    if any(base.tree(runtime) != proof for runtime in runtimes[1:]):
        raise ValueError('Pressure cohort runtimes differ')
    for trial in plan['trials']:
        path = root/trial['name']/'runtime-proof.json'
        base.save(path, proof, replace=True)
        plan['files'][str(path.relative_to(root))] = base.sha(path)
    shutil.copy2(helper, root/HELPER)
    shutil.copy2(__file__, root/'launch.py')
    plan['files'].update({HELPER: base.sha(root/HELPER), 'launch.py': base.sha(root/'launch.py')})
    # The qualified inference runtime may predate these read-only analyzers.
    # Bundle them explicitly instead of assuming they were frozen with it.
    analysis = root/'cohort-analysis'
    analysis.mkdir()
    for name in ANALYSIS_MODULES:
        destination = analysis/name
        shutil.copy2(Path(__file__).with_name(name), destination)
        plan['files'][str(destination.relative_to(root))] = base.sha(destination)
    # An inherited native binary is part of the runtime hash. Also check its
    # actual server import instead of relying only on the copied source tree.
    binary, = (runtimes[0]/'src/vllm_kvmem').glob('_native*.so')
    plan['native_extension'] = dict(relative_path=str(binary.relative_to(runtimes[0])),
        sha256=base.sha(binary), scope='Inherited frozen binary; actual server import is hash checked.')
    plan.update(workers=8, source_runtime_sha256=proof, cohort_driver='pressure-eight-v1',
                **credits, gpu_confirmed_gate=True, first_answers_only=True,
                scope=__doc__, measurement_instrumentation_added=True,
                analysis_modules=list(ANALYSIS_MODULES))
    base.save(root/'plan.json', plan, replace=True)
    base.save(root/'prepared-proof.json', dict(plan_sha256=base.sha(root/'plan.json')), replace=True)
    return verify(root)


def verify(root):
    plan = pressure.verify(root)
    if (plan.get('cohort_driver') != 'pressure-eight-v1' or plan['workers'] != 8 or
            not plan.get('gpu_confirmed_gate') or not plan.get('first_answers_only') or
            plan.get('analysis_modules') != list(ANALYSIS_MODULES)):
        raise ValueError('Frozen eight-owner measurement controls changed')
    first_profile = first_fixture = None
    for trial in plan['trials']:
        arm = root/trial['name']
        profile, fixture = base.read(arm/'profile.json'), base.read(arm/'fixture.json')
        cohort_controls(fixture, profile, plan['case_ids'])
        if first_profile is None:
            first_profile, first_fixture = profile, fixture
        elif profile != first_profile or fixture != first_fixture:
            raise ValueError('Pressure cohort changes input, sampling or allocation controls')
        text = (arm/'runtime-v1/src/vllm_kvmem/layerwise_query.py').read_text()
        if not text.endswith(GATE) or text.count(GATE) != 1:
            raise ValueError('Missing exact GPU-confirmed measurement gate')
        native = plan['native_extension']
        if base.sha(arm/'runtime-v1'/native['relative_path']) != native['sha256']:
            raise ValueError('Inherited native binary changed')
    return plan


def summarize(root):
    plan = verify(root)
    # Reuse exact first-answer checks, without retrying nondeterministic
    # multiowner trajectories until they happen to match.
    report = pressure.BASE_SUMMARIZE(root)
    sys.path[:0] = [str(root/'cohort-analysis'), str(root/'source-runtime/scripts')]
    from analyze_layerwise_dynamic8 import measure
    measurements = {trial['name']: measure(root/trial['name']) for trial in plan['trials']}
    first = next(iter(measurements.values()))
    if any(row['gpu_completed_history_tokens'] != first['gpu_completed_history_tokens']
           for row in measurements.values()):
        raise ValueError('Concurrent pressure variants executed different history counts')
    for row in measurements.values():
        row['throughput_change_percent'] = 100*(row['tokens_per_second']/first['tokens_per_second']-1)
    controlled = all(row['autotune_unchanged'] and row['autotune_initial'] == first['autotune_initial']
                     for row in measurements.values())
    base.save(root/'prefill-comparison.json', dict(measurements=measurements,
        full_input_equivalent=report['full_input_equivalent'],
        measurement_valid=True, common_unchanged_autotuning=controlled,
        correctness_gate='Serial full-input equality is audited separately; preserve concurrent first outputs.',
        stable_speedup_claim=False,
        scope='Eight finite full cold prefixes, GPU-confirmed endpoint, Query/decode excluded. '
              'Not a continuous service benchmark. Owner pressure durations overlap; never sum as GPU idle.'))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'verify', 'run', 'arm', 'analyze'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-arm', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--autotune-source', type=Path)
    p.add_argument('--overlay-source', type=Path, default=base.REPO)
    p.add_argument('--overlays-json', type=Path)
    p.add_argument('--prefill-state-binary', type=Path)
    p.add_argument('--variants-json', type=Path)
    p.add_argument('--order', default='baseline,candidate')
    p.add_argument('--port', type=int, default=19468)
    p.add_argument('--arm')
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare':
        prepare(root, source_arm=a.source_arm, fixture_path=a.fixture,
                autotune_source=a.autotune_source, variants=base.read(a.variants_json),
                order=tuple(a.order.split(',')), port=a.port, overlay_source=a.overlay_source,
                overlays=base.read(a.overlays_json) if a.overlays_json else (),
                prefill_state_binary=a.prefill_state_binary)
    elif a.action == 'verify':
        verify(root)
    elif a.action == 'analyze':
        summarize(root)
    else:
        base.verify, base.summarize = verify, summarize
        if a.action == 'run': base.run(root)
        else: base.arm(root, a.arm)


if __name__ == '__main__':
    main()
