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


def sample(mode='delete', prompt=256):
    events = [dict(kind='published', at=12.)]
    for name in ('retired', 'cpu_released'):
        if mode == 'delete':
            events.append(dict(kind=name, before=0, after=4,
                interval_begin=13., interval_end=14., accepted_history=prompt+264,
                following_decode_at=15., following_accepted_history=prompt+266))
    fixture = dict(source='synthetic_gc_throughput_fixture', semantic_measurement=False,
        error=None, flushed=True, observations_published=3, events=events,
        plan=dict(prompt_tokens=prompt, decode_tokens=512),
        candidate_handles=[[i, 1] for i in range(4)],
        baseline=dict(counters=dict(retired=0, cpu_released=0)),
        last_sample=dict(counters=dict(retired=4 if mode=='delete' else 0,
                                       cpu_released=4 if mode=='delete' else 0)))
    result = dict(timeline=[[10., 1], [20., 1]])
    return fixture, result


@pytest.mark.parametrize('mode', ['off', 'delete'])
@pytest.mark.parametrize('prompt', [256, 36864])
def test_audit_accepts_only_the_declared_synthetic_scope(mode, prompt):
    fixture, result = sample(mode, prompt)
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
    assert first['assessment'] == 'unresolved' and first['additional_evidence_needed']


def test_uncertainty_keeps_single_pair_unresolved():
    value = qualification.throughput_uncertainty([dict(throughput_loss=0.)])
    assert value['percentile_interval'] is None
    assert value['assessment'] == 'unresolved'


def paired(losses, *, output=True, p95=None, p99=None):
    return [dict(throughput_loss=v, p95_regression=p95, p99_regression=p99,
                 identical_output=output) for v in losses]


def test_passing_median_with_crossing_interval_is_inconclusive():
    result = qualification.paired_conclusion(paired([-.01, 0., .005, .02, .03]))
    assert result['paired_medians']['throughput_loss'] < .01
    assert result['throughput_uncertainty']['percentile_interval'][1] > .01
    assert result['status'] == 'inconclusive' and result['exit_code'] == 2


@pytest.mark.parametrize('losses,status,code', [
    ([.002, .003, .004, .005, .006], 'passed', 0),
    ([.01]*5, 'passed', 0),
    ([.02, .021, .022, .023, .024], 'failed', 1),
    ([-.01, .005, .015, .02, .03], 'inconclusive', 2),
    ([0.], 'inconclusive', 2),
    ([0.]*4, 'inconclusive', 2),
])
def test_final_status_follows_noninferiority_bound_and_minimum_pairs(losses, status, code):
    result = qualification.paired_conclusion(paired(losses))
    assert (result['status'], result['exit_code']) == (status, code)
    assert result['conclusion_reasons']
    assert result['throughput_uncertainty']['automatic_retesting'] is False
    if status == 'passed':
        assert result['throughput_uncertainty']['percentile_interval'][1] <= .01


def test_output_mismatch_is_a_failure_even_when_throughput_is_fast():
    result = qualification.paired_conclusion(paired([-.01]*5, output=False))
    assert result['status'] == 'failed'
    assert result['conclusion_reasons'] == ['paired_output_mismatch']


def test_measured_tail_gate_remains_independent_of_throughput():
    result = qualification.paired_conclusion(paired([0.]*5, p95=.03, p99=0.))
    assert result['status'] == 'failed' and result['tail_latency_measured']
    assert result['conclusion_reasons'] == ['measured_tail_latency_gate_exceeded']


def test_clear_regression_does_not_recommend_retesting_until_pass():
    result = qualification.paired_conclusion(paired([.02]*5))
    uncertainty = result['throughput_uncertainty']
    assert uncertainty['assessment'] == 'lower_bound_exceeds_1pct'
    assert not uncertainty['additional_evidence_needed']
    assert not uncertainty['automatic_retesting']


def large_fixture_config():
    return dict(max_num_seqs=1, execution_mode='eager', draft_method='mtp', draft_tokens=1,
        max_model_len=262144, bounded_kv_capacity=True,
        kvmem=dict(historical_gc=dict(policy='statistical'), block_tokens=32,
                   select_budget=36864, gen_budget=16384, repository_blocks=8196,
                   cpu_read_cache_bytes=12*1024**3))


@pytest.mark.parametrize('prompt', [256, 36864, 52672])
def test_fixture_accepts_aligned_long_prompt_through_exact_stable_boundary(prompt):
    import copy
    config = large_fixture_config()
    before = copy.deepcopy(config)
    qualification.validate_statistical_fixture_config(config, prompt, 512)
    assert config == before  # Validation cannot shrink logical/K/G/CPU budgets.


@pytest.mark.parametrize('prompt,decode', [(0,512), (224,512), (255,512),
                                         (257,512), (36865,512), (36864,511)])
def test_fixture_rejects_invalid_prompt_or_decode_shape(prompt, decode):
    with pytest.raises(ValueError, match='aligned prompt'):
        qualification.validate_statistical_fixture_config(large_fixture_config(), prompt, decode)


def test_fixture_rejects_window_overflow_and_inconsistent_logical_limit():
    config = large_fixture_config()
    with pytest.raises(ValueError, match='stable window'):
        qualification.validate_statistical_fixture_config(config, 52704, 512)
    for logical in (53247, 36864, None, True):
        config['max_model_len'] = logical
        with pytest.raises(ValueError, match='logical context'):
            qualification.validate_statistical_fixture_config(config, 36864, 512)


@pytest.mark.parametrize('area,key,value', [
    ('top','max_num_seqs',2), ('top','execution_mode','graph'),
    ('top','draft_method','dflash'), ('top','draft_tokens',2),
    ('kv','block_tokens',16), ('kv','graph_mode','sparse'),
    ('kv','action_native_scores','on'), ('kv','historical_gc',dict(policy='lru')),
])
def test_long_fixture_retains_qualified_execution_guards(area, key, value):
    config = large_fixture_config()
    (config if area == 'top' else config['kvmem'])[key] = value
    with pytest.raises(ValueError, match='eager C1/MTP1'):
        qualification.validate_statistical_fixture_config(config, 36864, 512)
