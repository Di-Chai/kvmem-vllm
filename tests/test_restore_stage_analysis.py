import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('restore_stage_analysis',
    Path(__file__).resolve().parents[1] / 'scripts/analyze_restore_stages.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def stage(name, a, b):
    return dict(name=name, started_at=a, finished_at=b, wall_seconds=b-a, thread_cpu_seconds=(b-a)/2)


def observation(a, b):
    return dict(request_id='r', phase='prepare', lane='resident', timing=dict(stages=[
        stage('restore_main_thread_total', a, b),
        stage('claim', a + .1, b - .1), stage('nested_restore', a + .2, b - .2)]))


def test_nested_durations_are_not_counted_twice_and_public_remainder_is_explicit():
    result = analysis.analyze([observation(1., 2.)], [dict(sequence=1, started_at=.5,
        model_submitted_at=2.5, decode_requests=7, prefill_tokens=0)], 0., 3.)
    assert result['unattributed_host_seconds']['sum'] == pytest.approx(.2)
    public = result['overlapping_public_steps'][0]
    assert public['restore_host_overlap_seconds'] == 1.
    assert public['other_maintenance_seconds'] == 1.


def test_boundary_samples_are_excluded_and_crossing_public_steps_are_not_misattributed():
    result = analysis.analyze([observation(1., 2.), observation(2.5, 3.5), observation(4., 5.)],
        [dict(sequence=1, started_at=2., model_submitted_at=4.)], 0., 3.)
    assert result['observations'] == 1
    assert result['boundary_observations_excluded'] == 1
    assert result['outside_observations'] == 1
    assert result['overlapping_public_steps'] == []


def test_overlapping_intervals_are_clipped_and_unioned():
    assert analysis.union_seconds([(0., 2.), (1., 3.), (5., 8.)], 1., 6.) == 3.


def test_no_evidence_is_not_reported_as_zero():
    with pytest.raises(ValueError, match='No complete restore observation'):
        analysis.analyze([], [], 0., 3.)


def test_invalid_nested_time_is_rejected():
    row = observation(1., 2.)
    row['timing']['stages'].append(stage('bad', .5, 1.5))
    with pytest.raises(ValueError, match='outside'):
        analysis.analyze([row], [], 0., 3.)
