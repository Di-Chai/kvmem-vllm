"""Compare a completed external KVMem run with a frozen native shape diagnostic.

This never launches inference, copies a run, or updates comparison.json. The
external request timestamps/results remain those of the previously measured run.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import types


def require(value, message):
    if not value:
        raise ValueError(message)


class Inputs:
    """Hash every file consumed, rejecting mutation during one analysis."""
    def __init__(self):
        self.hashes = {}

    def data(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        value = hashlib.sha256(data).hexdigest()
        require(self.hashes.get(str(path), value) == value, 'Input changed during analysis: '+str(path))
        self.hashes[str(path)] = value
        return data

    def sha(self, path):
        self.data(path)
        return self.hashes[str(Path(path).resolve())]

    def read(self, path):
        return json.loads(self.data(path))

    def rows(self, path):
        return [json.loads(line) for line in self.data(path).splitlines()]

    def verify(self):
        for path in tuple(self.hashes):
            self.sha(path)


def load_module(name, path, inputs):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(inputs.data(path), str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def frozen_driver(root, inputs):
    """Use only this experiment's frozen stdlib helpers; restore import state."""
    plan = inputs.read(root/'plan.json')
    for name, key in (('driver.py', 'driver_sha256'),
                      ('benchmark_prefill_publication_pipeline.py', 'helper_sha256'),
                      ('active_prefill_shapes.py', 'shape_helper_sha256')):
        require(inputs.sha(root/name) == plan[key], 'Frozen analysis source changed: '+name)
    probe = 'scripts/native_prefill_shape_probe.py'
    require(inputs.sha(root/'native/runtime'/probe) ==
            inputs.read(root/'native/runtime-proof.json')['files'][probe], 'Frozen native probe source changed')
    names = ('benchmark_prefill_publication_pipeline', 'active_prefill_shapes',
             'native_prefill_shape_probe', '_external_active_prefill_driver')
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path[:]
    try:
        helper = load_module(names[0], root/'benchmark_prefill_publication_pipeline.py', inputs)
        load_module(names[1], root/'active_prefill_shapes.py', inputs)
        load_module(names[2], root/'native/runtime/scripts/native_prefill_shape_probe.py', inputs)
        driver = load_module(names[3], root/'driver.py', inputs)
        for module in (helper, driver):
            module.read, module.rows, module.sha = inputs.read, inputs.rows, inputs.sha
        yield driver, helper
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


DIAGNOSTIC_DIFFERENCES = frozenset(('scripts/active_prefill_shapes.py',
    'scripts/analyze_logical_prefill.py', 'scripts/benchmark_active_prefill.py',
    # RSS telemetry/reporting only; production runtime still requires exact hashes.
    'scripts/benchmark_matched_prefill.py',
    'scripts/benchmark_prefill_publication_pipeline.py', 'scripts/benchmark_prefill_variant.py'))


def audit_external_reference(root, reference, inputs, driver):
    plan = inputs.read(root/'plan.json')
    require(Path(plan['source_reference']).resolve() == reference,
            'External reference is not the frozen source reference')
    require(plan['preparation_overrides'] == {}, 'Prepared profile overrides prevent an exact external comparison')
    status = inputs.read(reference/'status.json')
    require(status['status'] == 'completed' and status['protocol_passed'] is True,
            'External reference did not complete its protocol')
    require(inputs.read(reference/'cleanup.json') == dict(remaining_owned_groups={}, gpu_processes=[]),
            'External reference cleanup is incomplete')
    manifest = inputs.read(reference/'manifest.json')
    profile_hash = inputs.sha(reference/'profile.json')
    require(profile_hash == manifest['profile_sha256'], 'External profile changed')
    require(profile_hash == inputs.sha(root/'native/profile.json') == inputs.sha(root/'kvmem/profile.json') ==
            inputs.read(root/'kvmem/manifest.json')['profile_sha256'], 'Frozen native/reference profiles differ')
    require(inputs.sha(reference/'suite.json') == manifest['suite_sha256'], 'External suite changed')
    proof = inputs.read(reference/'runtime-proof.json')['files']
    new_proof = inputs.read(root/'native/runtime-proof.json')['files']
    parent_plan = inputs.read(reference.parent/'plan.json')
    require(driver.digest(proof) == parent_plan['runtime_sha256'] and
            inputs.sha(reference/'suite.json') == parent_plan['suite_sha256'] and
            profile_hash == parent_plan['profile_sha256'][reference.name] and
            inputs.sha(reference.parent/'driver.py') == parent_plan['driver_sha256'],
            'External runtime/profile/suite differs from its original frozen experiment')
    for name, value in proof.items():
        require(inputs.sha(reference/'runtime'/name) == value, 'External frozen source changed: '+name)
    differences = {name: dict(external=proof.get(name), native=new_proof.get(name))
                   for name in proof.keys() | new_proof.keys() if proof.get(name) != new_proof.get(name)}
    require(set(differences) <= DIAGNOSTIC_DIFFERENCES,
            'Production or undeclared runtime source differs: '+str(sorted(set(differences)-DIAGNOSTIC_DIFFERENCES)))
    require(any(name.startswith('src/') for name in proof), 'Production source proof is empty')
    shape_plan = inputs.read(root/'native-shape-plan.json')
    require(Path(shape_plan['source_reference']).resolve() == reference, 'Shape plan has another source')
    for name, value in shape_plan['source_evidence_sha256'].items():
        require(inputs.sha(reference/name) == value, 'Frozen reference shape evidence changed: '+name)
    require(shape_plan['source_evidence_sha256'], 'Shape source evidence is empty')
    require(inputs.read(root/'native/reference-launch.json') == inputs.read(reference/'run/server/launch.json'),
            'Original launch differs from the native frozen source')
    return dict(production_runtime_identical=True, diagnostic_source_differences=differences,
                external_runtime_sha256=driver.digest(proof), native_runtime_sha256=driver.digest(new_proof),
                profile_sha256=profile_hash, measured_status_timestamp=status.get('at'),
                cold_prepare_seconds=status['cold_prepare_seconds'])


def completed_result(directory, definition, inputs, *, expected=None):
    """Validate actual input bytes and output accounting, not just a filename."""
    result, response = inputs.read(directory/'result.json'), inputs.read(directory/'response.json')
    count, token_hash = definition['row']['served_input_tokens'], definition['row']['served_sha256']
    data = inputs.data(Path(definition['input_path']))
    require(len(data) == 4*count and hashlib.sha256(data).hexdigest() == token_hash,
            'Input content/length differs from case: '+str(directory))
    require(result['status'] == 'completed' and result['input_tokens'] == count and
            result['input_sha256'] == token_hash, 'Result input accounting differs: '+str(directory))
    require(result['completion_tokens'] == len(response['output_ids']) and
            response['usage']['prompt_tokens'] == count and
            response['usage']['completion_tokens'] == len(response['output_ids']) and
            result['ttft_seconds'] == response['ttft_seconds'], 'Response accounting differs: '+str(directory))
    require(type(result['ttft_seconds']) in (int, float) and math.isfinite(result['ttft_seconds']) and
            result['ttft_seconds'] > 0, 'Invalid TTFT')
    if expected is not None:
        require(result == expected, 'Result differs from completed status receipt')
    return result, response


def aggregate_ttft(samples):
    require(bool(samples), 'No complete TTFT samples')
    elapsed = sum(row['ttft_seconds'] for row in samples)
    return dict(samples=samples, repetitions=len(samples), mean_ttft_seconds=elapsed/len(samples),
                pooled_input_tokens_per_second=sum(row['input_tokens'] for row in samples)/elapsed)


def aggregate_phases(phases):
    require(bool(phases) and all(p['status'] == 'available' for p in phases),
            'Continuous phase timing is missing or incomplete')
    patterns = {tuple(tuple(shape) for shape in p['shape_pattern']) for p in phases}
    require(len(patterns) == 1, 'Different continuous phase shapes must not be pooled')
    tokens = {p['new_input_tokens'] for p in phases}
    require(len(tokens) == 1, 'Continuous phase token domains differ')
    require(next(iter(tokens)) == sum(shape[0] for shape in next(iter(patterns))),
            'Continuous phase tokens differ from its actual query shapes')
    identities = [(p['id'], p['request_id'], p['first_sequence'], p['last_sequence']) for p in phases]
    require(len(set(identities)) == len(identities), 'Duplicate continuous phase evidence')
    for phase in phases:
        numbers = [phase[key] for key in ('model_seconds', 'interior_gap_seconds',
                                          'model_start_to_last_end_seconds')]
        require(all(type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in numbers) and
                numbers[2] > 0 and math.isclose(numbers[0]+numbers[1], numbers[2], abs_tol=1e-12),
                'Continuous phase span accounting differs')
    duration = sum(p['model_start_to_last_end_seconds'] for p in phases)
    return dict(phases=len(phases), shape_pattern=[list(s) for s in next(iter(patterns))],
        tokens_per_phase=next(iter(tokens)), mean_model_seconds=statistics.mean(p['model_seconds'] for p in phases),
        mean_interior_gap_seconds=statistics.mean(p['interior_gap_seconds'] for p in phases),
        mean_span_seconds=duration/len(phases),
        pooled_new_tokens_per_second=sum(p['new_input_tokens'] for p in phases)/duration)


def compare_phases(native, external):
    left, right = aggregate_phases(native), aggregate_phases(external)
    require(left['shape_pattern'] == right['shape_pattern'] and left['tokens_per_phase'] == right['tokens_per_phase'],
            'Native and external continuous phase geometry differs')
    return dict(native=left, previously_measured_kvmem=right,
        kvmem_throughput_difference_percent=100*(right['pooled_new_tokens_per_second']/left['pooled_new_tokens_per_second']-1),
        kvmem_span_extra_percent=100*(right['mean_span_seconds']/left['mean_span_seconds']-1),
        scope='Matched active lengths/chunks, first model start through last step end, including all interior gaps. '
              'Excludes prefix establishment, pressure before/after this phase, HTTP and final generation. '
              'Phases within each long request are correlated, not independent complete-request trials.')


def recheck_native(root, driver, original):
    """Re-run frozen audits in memory, preserving every original report byte."""
    captured = {}
    allowed = (root/'comparison.json', root/'native/probe/audit.json')
    def capture_save(path, value):
        path = Path(path).resolve()
        require(path in allowed, 'Unexpected frozen analyzer write')
        require(path not in captured, 'Duplicate frozen analyzer write')
        captured[path] = value
    old_save = driver.save
    driver.save = capture_save
    try:
        with redirect_stdout(io.StringIO()):
            driver.analyze(root)
    finally:
        driver.save = old_save
    require(set(captured) == set(allowed), 'Frozen analyzer reports are incomplete')
    require(captured[root/'comparison.json'] == original, 'Existing native comparison disagrees with frozen reanalysis')
    return captured[root/'native/probe/audit.json']


def analyze(root, reference):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    inputs = Inputs()
    analyzer_hash = inputs.sha(Path(__file__))
    original = inputs.read(root/'comparison.json')
    require(not (root/'kvmem/status.json').exists(), 'This adapter is for an external old run, not a newly executed local KVMem arm')
    with frozen_driver(root, inputs) as (driver, helper):
        driver.assert_frozen(root, 'native')
        proof = audit_external_reference(root, reference, inputs, driver)
        native = root/'native'
        native_status = inputs.read(native/'status.json')
        require(native_status['status'] == 'completed' and native_status['protocol_passed'] is True,
                'Native experiment is incomplete')
        require(inputs.read(native/'cleanup.json') == dict(remaining_owned_groups={}, gpu_processes=[]),
                'Native cleanup is incomplete')
        launch = inputs.read(native/'native-launch.json')
        require(launch['connector_absent'] is True and launch['prefix_caching'] is False and
                launch['model_kernels_unmodified'] is True, 'Native launch protocol differs')
        native_audit = recheck_native(root, driver, original)
        definitions = inputs.read(native/'suite.json')['cases']
        native_receipts = {row['id']: row['result'] for row in native_status['results']}
        require(len(native_receipts) == len(native_status['results']) == len(definitions) and
                set(native_receipts) == {item['id'] for item in definitions}, 'Native status receipts are duplicated/incomplete')
        plan = inputs.read(root/'plan.json')
        sources = {key: inputs.read(Path(path)) for key, path in plan['case_sources'].items()}
        native32 = []
        for item in definitions:
            directory = native/'requests'/item['id']
            require(inputs.sha(directory/'case.json') == item['case_sha256'], 'Native case changed')
            definition = inputs.read(directory/'case.json')
            result, response = completed_result(directory, definition, inputs, expected=native_receipts[item['id']])
            source = sources['full32k' if item['family'] in ('full32k','logical2k-control') else 'full256k']
            require(inputs.data(Path(definition['input_path'])) == inputs.data(Path(source['input_path']))[:4*result['input_tokens']],
                    'Native prompt is not the declared original-input prefix')
            expected_sampling = dict(source['sampling'])
            if item['family'] not in ('full32k','logical2k-control'):
                expected_sampling['max_tokens'] = 1 if item['family'] == 'flush' else 8
            require(definition['sampling'] == expected_sampling, 'Native sampling differs from its frozen source')
            if not item['warmup'] and item['family'] == 'full32k':
                native32.append(dict(id=item['id'], instrumented=item['instrumented'], output_ids=response['output_ids']))
        suite = inputs.read(reference/'suite.json')['cases']
        require(len({c['id'] for c in suite}) == len(suite), 'Duplicate reference cases')
        groups = helper.prefill_groups(inputs.rows(reference/'run/server/capture/model_step_latency.jsonl'))
        require(len(groups) == len(suite), 'Reference prefill grouping differs from its suite')
        status = inputs.read(reference/'status.json')
        receipts = {row['id']: row['result'] for row in status['results']}
        require(len(receipts) == len(status['results']) == len(suite) and
                set(receipts) == {item['id'] for item in suite}, 'Reference status receipts are duplicated/incomplete')
        shapes = inputs.read(root/'reference-geometry.json')
        shape_plan = inputs.read(root/'native-shape-plan.json')
        shape_bank = {(s['query_tokens'],s['active_seq_len'],s['prefill_terminal']):s for s in original['shapes']}
        samples, external_phases, outputs = [], [], []
        for item, group in zip(suite, groups):
            directory = reference/'requests'/item['id']
            require(inputs.sha(directory/'case.json') == item['case_sha256'], 'External case changed')
            definition = inputs.read(directory/'case.json')
            result, response = completed_result(directory, definition, inputs, expected=receipts[item['id']])
            count = result['input_tokens']
            require(count in (32768, shapes['input_tokens']), 'Unexpected external input length')
            source = sources['full32k' if count == 32768 else 'full256k']
            require(definition == source, 'External request definition differs from frozen source')
            evidence = inputs.read(directory/'evidence.json')
            # Initial warmup events may predate the profiler becoming ready.
            # Validate its completed request but never use its partial timing.
            if item['warmup']:
                continue
            geometry = driver.reconstruct_geometry(group, evidence['refresh'], count)
            require(all(len(step.get('batch_geometry', ())) == 1 for step in group), 'External exact geometry is missing')
            owner = group[0]['batch_geometry'][0]['request_id']
            require(all(row['batch_geometry'][0]['request_id'] == owner for row in group) and
                    all(row['request_id'] == owner for row in evidence['refresh']), 'External owner evidence differs')
            if count == 32768:
                require([row['query_tokens'] for row in geometry] == shape_plan['full32_chunks'],
                        'External full32K chunk geometry differs from the native control')
            else:
                require(geometry == shapes['rows'], 'External long-input physical geometry differs from frozen shape weights')
            continuous = driver.continuous_steady_phases(group, shape_plan['steady_patterns'], evidence['refresh'])
            sample = dict(id=item['id'], **result, timing=helper.timing(group), continuous_steady_prefill=continuous)
            if count != 32768:
                sample['native_shape_estimate_model_seconds'] = sum(shape_bank[
                    (s['query_tokens'],s['active_seq_len'],s['prefill_terminal'])]['native_model_ms'] for s in geometry)/1000
                external_phases.extend(dict(id=item['id'], **phase) for phase in continuous['phases'])
            else:
                outputs.extend(dict(left='native/'+other['id'], right='previously-measured-kvmem/'+item['id'],
                    **driver.compare_output_tokens(other['output_ids'],response['output_ids'])) for other in native32)
            samples.append(sample)
        native_phases = [dict(id=group['id'], **phase) for group in original['continuous_steady_prefill_native'] for phase in group['phases']]
        full32 = aggregate_ttft([s for s in samples if s['input_tokens'] == 32768])
        long = aggregate_ttft([s for s in samples if s['input_tokens'] != 32768])
        require(full32['repetitions'] == plan['kvmem_full32k'] and long['repetitions'] == plan['kvmem_full256k'],
                'Reference formal request count differs from the frozen protocol')
        per_request_phases = {sample['id']: aggregate_phases([
            phase for phase in external_phases if phase['id'] == sample['id']]) for sample in long['samples']}
        mean_model = statistics.mean(s['timing']['model_cuda_seconds'] for s in long['samples'])
        estimate = statistics.mean(s['native_shape_estimate_model_seconds'] for s in long['samples'])
        require(all(s['native_shape_estimate_model_seconds'] == original['weighted_native_model_seconds'] or
                    math.isclose(s['native_shape_estimate_model_seconds'], original['weighted_native_model_seconds'], abs_tol=1e-12)
                    for s in long['samples']), 'External geometry weights disagree with native estimate')
        result = dict(schema=1, scope='Post hoc comparison with a previously measured external KVMem candidate; no new KVMem inference.',
            external_reference=str(reference), native_experiment=str(root), new_kvmem_run=False,
            provenance=proof, native_audit=native_audit,
            original_native_comparison_unchanged=True,
            matched_shape_model_diagnostic=dict(native_weighted_model_seconds=estimate,
                previously_measured_kvmem_mean_model_seconds=mean_model,
                kvmem_model_extra_seconds=mean_model-estimate, kvmem_model_extra_percent=100*(mean_model/estimate-1),
                measured_native_full256k=False,
                scope='Median native model interval per matching (query, active, terminal) shape, weighted by actual external shape counts. '
                      'Includes launch gaps/MTP; excludes reselection, archive lifecycle and HTTP. Not native complete256K throughput.'),
            continuous_phase_comparison=compare_phases(native_phases, external_phases),
            continuous_external_per_request=per_request_phases,
            continuous_native_samples=native_phases, continuous_external_samples=external_phases,
            full32k=dict(native=original['full32k_native'], previously_measured_kvmem=full32,
                comparisons={name:dict(kvmem_throughput_difference_percent=100*(full32['pooled_input_tokens_per_second']/row['pooled_prefill_tokens_per_second']-1),
                    kvmem_ttft_extra_percent=100*(full32['mean_ttft_seconds']/row['mean_ttft_seconds']-1))
                    for name,row in original['full32k_native'].items()}, output_token_comparisons=outputs),
            full256k_previously_measured_kvmem=long,
            caveats=['External KVMem and native were measured at different times; this is not a newly interleaved pair.',
                'Matching shape/profile/input does not equate selected context contents, positions, outputs or MTP trajectories.',
                'Full32K generated token equality is reported separately and may fail; no numerical/quality equivalence is inferred.',
                'Continuous phases include interior gaps but exclude the pressure that starts/ends each phase.',
                'Complete256K TTFT includes all prefill, archive/statistics and pressure/Query work; no native complete256K run exists here.',
                'CUDA model, host-submit and reselection intervals can overlap; do not add them as independent costs.'])
    inputs.verify()
    result['provenance'].update(analyzer_sha256=analyzer_hash, input_sha256=dict(sorted(inputs.hashes.items())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.reference)
    target = args.root/'reference-comparison.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(target)
    print(json.dumps(dict(output=str(target), matched_shape_model_diagnostic=result['matched_shape_model_diagnostic'],
        continuous_phase_comparison=result['continuous_phase_comparison'],
        full32k_comparisons=result['full32k']['comparisons'],
        full256k_mean_ttft_seconds=result['full256k_previously_measured_kvmem']['mean_ttft_seconds']), indent=2))


if __name__ == '__main__':
    main()
