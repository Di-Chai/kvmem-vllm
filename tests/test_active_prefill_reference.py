"""Reference provenance and suite-order guards, without model startup or freezing."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_active_prefill as benchmark


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.save(path, value)


def reference(root, *, latest=True):
    put(root / 'profile.json', {'kvmem': {'action_native_scores': 'reuse',
        'prefill_publication_pipeline': True, 'prefill_native_plan_arm': True,
        'prefill_action_before_mtp': True, 'capture_fused_pack': True}})
    cases = [('warmup-32k', 32768)]
    if latest:
        cases += [('32k-repeat-1', 32768), ('32k-repeat-2', 32768)]
    cases += [('256k-repeat-1', 262045), ('256k-repeat-2', 262045)]
    suite, steps = [], []
    for ident, count in cases:
        case = root / 'requests' / ident / 'case.json'
        put(case, {'row': {'served_input_tokens': count}, 'source': ident})
        put(case.parent / 'evidence.json', {'refresh': []})
        suite.append({'id': ident, 'case_sha256': benchmark.sha(case)})
        steps += [{'prefill_tokens': count}, {'prefill_tokens': 0}]
    put(root / 'suite.json', {'cases': suite})
    path = root / 'run/server/capture/model_step_latency.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in steps))
    put(root / 'run/server/launch.json', {'argv': ['chosen-reference-launch']})


def test_explicit_reference_preserves_optimizations_and_uses_its_case_order(tmp_path):
    arm = tmp_path / 'latest/reference'
    reference(arm)
    loaded = benchmark.reference_inputs(tmp_path, Path('latest/reference'))
    assert loaded['reference'] == arm
    assert loaded['profile'] == benchmark.read(arm / 'profile.json')
    assert loaded['preparation_overrides'] == {}
    assert loaded['cases32']['source'] == '32k-repeat-1'
    assert loaded['geometry_suite_index'] == 3
    assert loaded['geometry'][0]['query_tokens'] == 262045
    assert loaded['launch'] == {'argv': ['chosen-reference-launch']}


def test_default_keeps_original_override_and_legacy_full32k_fixture(tmp_path):
    arm = tmp_path / 'artifacts/prefill-pipeline-20261021-r1/on-a'
    reference(arm, latest=False)
    path = tmp_path / 'artifacts/matched-32k-prefill-20261021-r2/kvmem/requests/32k-repeat-1/case.json'
    put(path, {'row': {'served_input_tokens': 32768}, 'source': 'legacy32'})
    loaded = benchmark.reference_inputs(tmp_path)
    assert loaded['profile']['kvmem']['prefill_native_plan_arm'] is False
    assert loaded['preparation_overrides'] == {'prefill_native_plan_arm': False}
    assert loaded['cases32']['source'] == 'legacy32'
    assert loaded['geometry_suite_index'] == 1
    # Reading a reference must never modify its saved configuration.
    assert benchmark.read(arm / 'profile.json')['kvmem']['prefill_native_plan_arm'] is True


def test_explicit_old_reference_can_reuse_its_full32k_warmup(tmp_path):
    reference(tmp_path / 'old', latest=False)
    loaded = benchmark.reference_inputs(tmp_path, tmp_path / 'old')
    assert loaded['cases32']['source'] == 'warmup-32k'
    assert loaded['profile']['kvmem']['prefill_native_plan_arm'] is True


@pytest.mark.parametrize('corruption', ['duplicate', 'missing_group', 'changed_case', 'short32'])
def test_reference_input_audit_fails_closed(tmp_path, corruption):
    arm = tmp_path / 'reference'
    reference(arm)
    if corruption == 'duplicate':
        suite = benchmark.read(arm / 'suite.json')
        suite['cases'].append(suite['cases'][3])
        put(arm / 'suite.json', suite)
        message = 'exactly one'
    elif corruption == 'missing_group':
        path = arm / 'run/server/capture/model_step_latency.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[2:]) + '\n')
        message = 'grouping differs'
    else:
        case = arm / 'requests/32k-repeat-1/case.json'
        put(case, {'row': {'served_input_tokens': 30720}})
        message = 'case changed'
        if corruption == 'short32':
            suite = benchmark.read(arm / 'suite.json')
            suite['cases'][1]['case_sha256'] = benchmark.sha(case)
            put(arm / 'suite.json', suite)
            message = 'all 32768'
    with pytest.raises(ValueError, match=message):
        benchmark.reference_inputs(tmp_path, arm)
