import json
import subprocess

import pytest

from audit_layerwise_gilopt import (CHANGED, apply_patch, audit, compare_cases, first_difference,
                                   save_new, sha, tree, verify_audit)
from test_layerwise_efficiency_equivalence import evidence


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def case(root, *, tokens=(1, 2), session='one', mutate=None):
    data = evidence()
    pressure = dict(event='pressure_selected', event_type='prefill_pressure',
        query_source='accepted_public_prefill', accepted_history=128,
        actual_method='accumulated_attention_per_layer', selected_tokens=64,
        layers=[dict(layer=i, candidate_starts=[0, 32, 64, 96], selected_starts=[0, 64],
                     charged_tokens=64, accepted_score_end=128, query_samples=4, sampled_chunks=1)
                for i in range(16)])
    data['events'].insert(0, pressure)
    if mutate:
        mutate(data)
    result = dict(id='r', answer='answer', seconds=1, layerwise=data)
    save(root/'result.json', result)
    save(root/'events.json', data['refresh'])
    save(root/'request.json', dict(session_id=session, cache_salt=session, seed=20260722,
        token_sha256='full-input', kvmem_query_span=[128, 130], max_tokens=16384))
    save(root/'response.json', dict(choices=[dict(token_ids=list(tokens), text='text')]))


def test_first_difference_detects_missing_elements_not_just_zipped_prefix():
    assert first_difference([1], [1]) is None
    assert first_difference([1], [1, 2]) == dict(path='[1]', before=None, after=2, missing=True)
    assert first_difference({'a': [1, 2]}, {'a': [1, 3]})['path'] == 'a[1]'


def test_compare_ignores_ownership_uuid_but_requires_exact_outputs(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    case(a)
    case(b, session='two')
    assert compare_cases(a, b)['passed']
    case(b, tokens=(1, 3))
    report = compare_cases(a, b)
    assert not report['passed'] and report['same_pressure'] and report['same_answer']
    assert report['first_output_difference']['path'] == 'token_ids[1]'


@pytest.mark.parametrize('scope', ['selected', 'candidate', 'score_end', 'query', 'stage', 'missing'])
def test_compare_reports_first_pressure_or_query_divergence(tmp_path, scope):
    a, b = tmp_path/'a', tmp_path/'b'
    case(a)
    def mutate(data):
        layer = data['events'][0]['layers'][3]
        if scope == 'selected': layer['selected_starts'] = [0, 96]
        if scope == 'candidate': layer['candidate_starts'] = [0, 64, 96]
        if scope == 'score_end': layer['accepted_score_end'] = 127
        if scope == 'query': data['layers'][0]['selected_starts'] = [0, 96]
        if scope == 'stage': data['events'][0]['event_type'] = 'decode_pressure'
        if scope == 'missing': data['events'].pop(0)
    case(b, mutate=mutate)
    report = compare_cases(a, b)
    assert not report['passed']
    if scope == 'query':
        assert report['first_query_difference']['path'].endswith('selected_starts[1]')
    else:
        assert report['first_pressure_context']['event'] == 0
        if scope == 'selected':
            assert report['first_pressure_context']['layers'][0] == dict(layer=3,
                same_candidates=True, same_score_end=True, selected_symmetric_difference=[64, 96])


def test_compare_rejects_changed_sampling_and_missing_output_ids(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    case(a)
    case(b)
    request = json.loads((b/'request.json').read_text())
    request['seed'] = 42
    save(b/'request.json', request)
    with pytest.raises(ValueError, match='sampling mismatch'):
        compare_cases(a, b)
    case(b, tokens=())
    with pytest.raises(ValueError, match='token IDs'):
        compare_cases(a, b)


def audit_fixture(tmp_path):
    a, b, patch = tmp_path/'a', tmp_path/'b', tmp_path/'opt.patch'
    hunks = []
    for name in CHANGED:
        for root, text in ((a, 'old\n'), (b, 'new\n')):
            path = root/'runtime-v1'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        hunks.append(f'diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-old\n+new\n')
    patch.write_text(''.join(hunks))
    for root in (a, b):
        save(root/'fixture.json', dict(bindings=[]))
        save(root/'profile.json', dict(kvmem=dict(select_budget=16384, gen_budget=4096)))
    save(a/'runtime-proof.json', tree(a/'runtime-v1'))
    save(a/'prepared-proof.json', {n: sha(a/n) for n in ('fixture.json', 'profile.json', 'runtime-proof.json')})
    return a, b, patch


def test_audit_reapplies_patch_and_verifies_all_runtime_files(tmp_path):
    a, b, patch = audit_fixture(tmp_path)
    report = audit(a, b, patch)
    assert report['passed'] and report['patch_reapplied_to_frozen_baseline']
    assert not report['launch_time_verification'] and not report['output_equivalence']
    verify_audit(report)
    (b/'runtime-v1'/CHANGED[0]).write_text('drift\n')
    with pytest.raises(ValueError, match='runtime tree changed'):
        verify_audit(report)
    with pytest.raises(ValueError, match='exactly the approved patch'):
        audit(a, b, patch)


def test_audit_records_exact_backup_and_rejects_other_extra_files(tmp_path):
    a, b, patch = audit_fixture(tmp_path)
    backup = b/'runtime-v1/src/vllm_kvmem/window_assembly.py.orig'
    backup.write_text('old\n')
    assert audit(a, b, patch)['preserved_nonimportable_backups']
    backup.write_text('unexpected\n')
    with pytest.raises(ValueError, match='file set changed'):
        audit(a, b, patch)


def test_patch_inside_a_repository_cannot_silently_skip_snapshot_files(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    a, b, patch = audit_fixture(tmp_path)
    destination = tmp_path/'artifacts/arm/runtime-v1'
    apply_patch(a/'runtime-v1', patch, destination)
    assert all((destination/n).read_text() == 'new\n' for n in CHANGED)
    assert all((a/'runtime-v1'/n).read_text() == 'old\n' for n in CHANGED)


def test_new_reports_never_replace_existing_results(tmp_path):
    output = tmp_path/'report.json'
    save_new(output, {'first': True})
    with pytest.raises(FileExistsError):
        save_new(output, {})
    assert json.loads(output.read_text()) == {'first': True}
