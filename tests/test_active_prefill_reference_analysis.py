"""Read-only analysis must reject changed provenance and incompatible timing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_active_prefill_reference as analysis


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def test_input_ledger_records_raw_files_and_rejects_mid_analysis_changes(tmp_path):
    path = tmp_path/'input.json'
    save(path, {'a': 1})
    ledger = analysis.Inputs()
    assert ledger.read(path) == {'a': 1}
    assert ledger.hashes == {str(path.resolve()): sha(path)}
    save(path, {'a': 2})
    with pytest.raises(ValueError, match='changed during'):
        ledger.verify()


def frozen_sources(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    for name, source in [('driver.py', 'benchmark_active_prefill.py'),
                         ('benchmark_prefill_publication_pipeline.py', 'benchmark_prefill_publication_pipeline.py'),
                         ('active_prefill_shapes.py', 'active_prefill_shapes.py'),
                         ('native/runtime/scripts/native_prefill_shape_probe.py', 'native_prefill_shape_probe.py')]:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/'scripts'/source, target)
    save(tmp_path/'plan.json', dict(driver_sha256=sha(tmp_path/'driver.py'),
        helper_sha256=sha(tmp_path/'benchmark_prefill_publication_pipeline.py'),
        shape_helper_sha256=sha(tmp_path/'active_prefill_shapes.py')))
    save(tmp_path/'native/runtime-proof.json', dict(files={
        'scripts/native_prefill_shape_probe.py': sha(tmp_path/'native/runtime/scripts/native_prefill_shape_probe.py')}))


def test_frozen_loader_does_not_use_live_imports_or_leave_import_state(tmp_path, monkeypatch):
    frozen_sources(tmp_path)
    sentinel = SimpleNamespace(should_not_be_used=True)
    monkeypatch.setitem(sys.modules, 'benchmark_prefill_publication_pipeline', sentinel)
    old_path = sys.path[:]
    with analysis.frozen_driver(tmp_path, analysis.Inputs()) as (driver, helper):
        assert driver.digest is helper.digest
        assert helper is not sentinel
        assert helper.__file__ == str(tmp_path/'benchmark_prefill_publication_pipeline.py')
        assert driver.reconstruct_geometry([dict(prefill_tokens=32)], [], 32)[0]['prefill_terminal']
        sys.path.insert(0, '/test-path-only')
    assert sys.modules['benchmark_prefill_publication_pipeline'] is sentinel
    assert sys.path == old_path
    assert not list(tmp_path.rglob('*.pyc'))


@pytest.mark.parametrize('filename', ['driver.py', 'benchmark_prefill_publication_pipeline.py',
    'active_prefill_shapes.py', 'native/runtime/scripts/native_prefill_shape_probe.py'])
def test_frozen_sources_checked_before_execution(tmp_path, filename):
    frozen_sources(tmp_path)
    (tmp_path/filename).write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match='source changed'):
        with analysis.frozen_driver(tmp_path, analysis.Inputs()):
            pytest.fail('changed source entered')


def provenance_fixture(tmp_path):
    root, reference = tmp_path/'native-root', tmp_path/'old-experiment/candidate'
    save(root/'plan.json', dict(source_reference=str(reference), preparation_overrides={}))
    save(reference/'status.json', dict(status='completed', protocol_passed=True, cold_prepare_seconds=12, at=100))
    save(reference/'cleanup.json', dict(remaining_owned_groups={}, gpu_processes=[]))
    profile = dict(kvmem={'prefill_physical_coalescing': True})
    for path in (reference/'profile.json', root/'native/profile.json', root/'kvmem/profile.json'):
        save(path, profile)
    profile_sha = sha(reference/'profile.json')
    save(reference/'suite.json', dict(cases=[]))
    save(reference/'manifest.json', dict(profile_sha256=profile_sha, suite_sha256=sha(reference/'suite.json')))
    save(root/'kvmem/manifest.json', dict(profile_sha256=profile_sha))
    code = reference/'runtime/src/model.py'
    code.parent.mkdir(parents=True)
    code.write_text('production = 1\n')
    proof = {'src/model.py': sha(code)}
    save(reference/'runtime-proof.json', dict(files=proof))
    save(root/'native/runtime-proof.json', dict(files=proof))
    (reference.parent/'driver.py').write_text('# frozen driver\n')
    save(reference.parent/'plan.json', dict(runtime_sha256=digest(proof),
        suite_sha256=sha(reference/'suite.json'), profile_sha256={'candidate': profile_sha},
        driver_sha256=sha(reference.parent/'driver.py')))
    save(root/'native-shape-plan.json', dict(source_reference=str(reference),
        source_evidence_sha256={'status.json': sha(reference/'status.json')}))
    for path in (root/'native/reference-launch.json', reference/'run/server/launch.json'):
        save(path, {'argv': ['model', 'fp8']})
    return root, reference


def test_same_production_runtime_allows_only_explicit_diagnostic_changes(tmp_path):
    root, reference = provenance_fixture(tmp_path)
    proof = json.loads((root/'native/runtime-proof.json').read_text())
    proof['files']['scripts/active_prefill_shapes.py'] = 'added-diagnostic'
    save(root/'native/runtime-proof.json', proof)
    result = analysis.audit_external_reference(root, reference, analysis.Inputs(), SimpleNamespace(digest=digest))
    assert result['production_runtime_identical']
    assert set(result['diagnostic_source_differences']) == {'scripts/active_prefill_shapes.py'}
    assert result['measured_status_timestamp'] == 100


@pytest.mark.parametrize('fault', ['wrong_source', 'override', 'unfinished', 'cleanup', 'profile',
    'suite', 'source_bytes', 'source_manifest', 'parent_plan', 'diagnostic_unlisted', 'launch',
    'shape_source', 'shape_evidence'])
def test_external_provenance_rejects_changed_or_unrelated_run(tmp_path, fault):
    root, reference = provenance_fixture(tmp_path)
    def change(path, callback):
        obj = json.loads(path.read_text()); callback(obj); save(path, obj)
    if fault == 'wrong_source': change(root/'plan.json', lambda o: o.update(source_reference=str(tmp_path/'other')))
    elif fault == 'override': change(root/'plan.json', lambda o: o.update(preparation_overrides={'flag': True}))
    elif fault == 'unfinished': change(reference/'status.json', lambda o: o.update(status='running'))
    elif fault == 'cleanup': change(reference/'cleanup.json', lambda o: o.update(gpu_processes=[42]))
    elif fault == 'profile': change(reference/'profile.json', lambda o: o.update(changed=True))
    elif fault == 'suite': change(reference/'suite.json', lambda o: o.update(changed=True))
    elif fault == 'source_bytes': (reference/'runtime/src/model.py').write_text('changed=1')
    elif fault == 'source_manifest': change(root/'native/runtime-proof.json', lambda o: o['files'].update({'src/model.py': 'other'}))
    elif fault == 'parent_plan': change(reference.parent/'plan.json', lambda o: o.update(runtime_sha256='other'))
    elif fault == 'diagnostic_unlisted': change(root/'native/runtime-proof.json', lambda o: o['files'].update({'scripts/serving.py': 'other'}))
    elif fault == 'launch': change(reference/'run/server/launch.json', lambda o: o.update(argv=['other']))
    elif fault == 'shape_source': change(root/'native-shape-plan.json', lambda o: o.update(source_reference=str(tmp_path/'other')))
    else: change(root/'native-shape-plan.json', lambda o: o.update(source_evidence_sha256={'status.json': 'other'}))
    with pytest.raises(ValueError):
        analysis.audit_external_reference(root, reference, analysis.Inputs(), SimpleNamespace(digest=digest))


def completed_fixture(tmp_path):
    prompt = tmp_path/'input.i32'
    prompt.write_bytes(bytes(16))
    definition = dict(input_path=str(prompt), row=dict(served_input_tokens=4, served_sha256=sha(prompt)))
    result = dict(status='completed', input_tokens=4, input_sha256=sha(prompt), completion_tokens=2, ttft_seconds=1)
    response = dict(output_ids=[10,11], usage=dict(prompt_tokens=4, completion_tokens=2), ttft_seconds=1)
    save(tmp_path/'result.json', result); save(tmp_path/'response.json', response)
    return definition, result


def test_input_and_response_are_verified_against_status_receipt(tmp_path):
    definition, expected = completed_fixture(tmp_path)
    result, response = analysis.completed_result(tmp_path, definition, analysis.Inputs(), expected=expected)
    assert result == expected and response['output_ids'] == [10,11]


@pytest.mark.parametrize('fault', ['input', 'length', 'receipt', 'response_count', 'prompt_count', 'ttft', 'nan'])
def test_changed_input_or_result_is_rejected(tmp_path, fault):
    definition, expected = completed_fixture(tmp_path)
    if fault in ('input', 'length'): (tmp_path/'input.i32').write_bytes(b'\1'* (16 if fault == 'input' else 12))
    elif fault == 'receipt': expected = dict(expected, ttft_seconds=2)
    else:
        path = tmp_path/'response.json'
        obj = json.loads(path.read_text())
        if fault == 'response_count': obj['output_ids'].append(12)
        elif fault == 'prompt_count': obj['usage']['prompt_tokens'] = 5
        elif fault == 'ttft': obj['ttft_seconds'] = 3
        else:
            path = tmp_path/'result.json'; obj = dict(expected, ttft_seconds=float('nan'))
        save(path, obj)
    with pytest.raises(ValueError):
        analysis.completed_result(tmp_path, definition, analysis.Inputs(), expected=expected)


def phase(ident, start, duration):
    return dict(id=ident, request_id=ident, first_sequence=start, last_sequence=start+1,
        status='available', shape_pattern=[[32,32800],[64,32864]], new_input_tokens=96,
        model_seconds=duration-.1, interior_gap_seconds=.1, model_start_to_last_end_seconds=duration)


def test_continuous_pooled_rate_uses_elapsed_time_and_keeps_correlated_phase_count():
    native = [phase('native-'+str(i), 1, 1) for i in range(3)]
    external = [phase('old-'+str(r), n*2, 2 if r == 0 else 4) for r in range(2) for n in range(13)]
    report = analysis.compare_phases(native, external)
    assert report['native']['phases'] == 3
    assert report['previously_measured_kvmem']['phases'] == 26
    assert report['previously_measured_kvmem']['pooled_new_tokens_per_second'] == 32
    assert report['kvmem_throughput_difference_percent'] == pytest.approx(-100*2/3)
    assert 'correlated' in report['scope']


@pytest.mark.parametrize('fault', ['unavailable', 'mixed_shapes', 'native_shape', 'tokens', 'duplicate', 'span', 'nan'])
def test_phase_pooling_rejects_invalid_or_different_geometry(fault):
    native, external = [phase('native', 1, 1)], [phase('old', 1, 2), phase('old', 3, 2)]
    if fault == 'unavailable': external[0]['status'] = 'unavailable'
    elif fault == 'mixed_shapes': external[1]['shape_pattern'][1][0] = 128
    elif fault == 'native_shape': native[0]['shape_pattern'][1][0] = 128
    elif fault == 'tokens': external[0]['new_input_tokens'] += 1
    elif fault == 'duplicate': external.append(deepcopy(external[0]))
    elif fault == 'span': external[0]['model_seconds'] += 1
    else: external[0]['model_seconds'] = float('nan')
    with pytest.raises(ValueError): analysis.compare_phases(native, external)


def test_full_request_ttft_is_distinct_from_phase_or_weighted_model_rate():
    report = analysis.aggregate_ttft([dict(input_tokens=32768,ttft_seconds=3),
                                      dict(input_tokens=32768,ttft_seconds=5)])
    assert report['pooled_input_tokens_per_second'] == 8192
    assert report['mean_ttft_seconds'] == 4


@pytest.mark.parametrize('fault', [None, 'different', 'unexpected', 'duplicate', 'missing'])
def test_reanalysis_preserves_original_reports_even_when_validation_fails(tmp_path, fault):
    original = {'checks': {'same_output': False}, 'weighted_seconds': 4}
    save(tmp_path/'comparison.json', original)
    save(tmp_path/'native/probe/audit.json', {'original': True})
    before = {path: path.read_bytes() for path in (tmp_path/'comparison.json',tmp_path/'native/probe/audit.json')}
    driver = SimpleNamespace(save=lambda *args: pytest.fail('Must never write original reports'))
    old_save = driver.save
    def frozen_analyze(root):
        driver.save(root/'native/probe/audit.json', {'passed': True})
        if fault == 'duplicate': driver.save(root/'native/probe/audit.json', {'passed': True})
        if fault == 'unexpected': driver.save(root/'new-report.json', {'unexpected': True})
        if fault != 'missing': driver.save(root/'comparison.json', dict(original, changed=True) if fault == 'different' else original)
    driver.analyze = frozen_analyze
    if fault:
        with pytest.raises(ValueError): analysis.recheck_native(tmp_path, driver, original)
    else:
        assert analysis.recheck_native(tmp_path, driver, original) == {'passed': True}
    assert driver.save is old_save
    assert all(path.read_bytes() == value for path,value in before.items())
    assert not (tmp_path/'new-report.json').exists()
