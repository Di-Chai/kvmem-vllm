"""Reject misleading GC overlap and unmeasured latency claims without a GPU."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('qualify_historical_gc', SCRIPTS/'qualify_historical_gc.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def sample(mode='delete'):
    events = [dict(kind='published', at=12.)]
    for name in ('retired', 'cpu_released'):
        if mode == 'delete':
            events.append(dict(kind=name, before=0, after=4,
                interval_begin=13., interval_end=14., accepted_history=520,
                following_decode_at=15., following_accepted_history=522))
    fixture = dict(source='synthetic_gc_throughput_fixture', semantic_measurement=False,
        error=None, flushed=True, observations_published=3, events=events,
        plan=dict(prompt_tokens=256, decode_tokens=512),
        candidate_handles=[[i, 1] for i in range(4)],
        baseline=dict(counters=dict(retired=0, cpu_released=0)),
        last_sample=dict(counters=dict(retired=4 if mode=='delete' else 0,
                                       cpu_released=4 if mode=='delete' else 0)))
    result = dict(timeline=[[10., 1], [20., 1]])
    return fixture, result


@pytest.mark.parametrize('mode', ['off', 'delete'])
def test_audit_accepts_only_the_declared_synthetic_scope(mode):
    fixture, result = sample(mode)
    proof = qualification.audit_statistical_fixture(fixture, result, mode,
        [dict(refresh_started_at=1., refresh_finished_at=9.)])
    assert proof['semantic_measurement'] is False
    assert proof['overlapping_refreshes'] == 0
    assert bool(proof['release_witnesses']['cpu_released']) == (mode == 'delete')


@pytest.mark.parametrize('field,value', [
    ('interval_begin', 9.), ('interval_end', 21.),
    ('accepted_history', 768), ('following_decode_at', 21.),
    ('following_decode_at', 13.5), ('following_accepted_history', 520),
    ('following_accepted_history', 768)])
def test_audit_rejects_prefill_finalization_or_no_subsequent_decode(field, value):
    fixture, result = sample()
    fixture['events'][2][field] = value
    with pytest.raises(AssertionError, match='inside decode'):
        qualification.audit_statistical_fixture(fixture, result, 'delete', [])


def test_some_release_inside_decode_cannot_mask_the_rest_after_decode():
    fixture, result = sample()
    fixture['events'][2]['after'] = 3
    fixture['events'].append(dict(kind='cpu_released', before=3, after=4,
        interval_begin=19., interval_end=21., accepted_history=767))
    with pytest.raises(AssertionError, match='inside decode'):
        qualification.audit_statistical_fixture(fixture, result, 'delete', [])


def test_logical_retirement_without_physical_release_is_not_success():
    fixture, result = sample()
    fixture['events'] = [r for r in fixture['events'] if r['kind'] != 'cpu_released']
    fixture['last_sample']['counters']['cpu_released'] = 0
    with pytest.raises(AssertionError, match='cpu_released'):
        qualification.audit_statistical_fixture(fixture, result, 'delete', [])


@pytest.mark.parametrize('row', [dict(refresh_started_at=9., refresh_finished_at=11.),
                                dict(at=15., stage='assembly_started')])
def test_decode_refresh_invalidates_the_stable_window_comparison(row):
    fixture, result = sample()
    with pytest.raises(AssertionError, match='performed a refresh'):
        qualification.audit_statistical_fixture(fixture, result, 'delete', [row])


def test_off_requires_no_native_reclamation():
    fixture, result = sample()
    with pytest.raises(AssertionError, match='Off retired'):
        qualification.audit_statistical_fixture(fixture, result, 'off', [])


@pytest.mark.parametrize('field,value', [('error', 'failure'), ('flushed', False),
    ('semantic_measurement', True), ('observations_published', 2)])
def test_partial_or_mislabeled_producer_fails_closed(field, value):
    fixture, result = sample()
    fixture[field] = value
    with pytest.raises(AssertionError, match='Incomplete'):
        qualification.audit_statistical_fixture(fixture, result, 'delete', [])


def test_unmeasured_tail_latency_stays_unmeasured_and_asymmetric_setup_fails():
    assert qualification.relative_regression(None, None) is None
    assert qualification.relative_regression(10.1, 10.) == pytest.approx(.01)
    with pytest.raises(AssertionError):
        qualification.relative_regression(10., None)


def test_uncertainty_is_reproducible_and_does_not_hide_bad_pair_range():
    pairs = [dict(throughput_loss=v) for v in (-.01, 0., .005, .02, .03)]
    first = qualification.throughput_uncertainty(pairs)
    assert first == qualification.throughput_uncertainty(pairs)
    assert first['minimum'] == -.01 and first['maximum'] == .03
    assert first['median'] == .005  # Old point gate passes, but CI is unresolved.
    assert first['percentile_interval'][1] > .01
    assert first['assessment'] == 'unresolved' and first['more_pairs_recommended']


def test_uncertainty_keeps_single_pair_unresolved():
    value = qualification.throughput_uncertainty([dict(throughput_loss=0.)])
    assert value['percentile_interval'] is None
    assert value['assessment'] == 'unresolved'
