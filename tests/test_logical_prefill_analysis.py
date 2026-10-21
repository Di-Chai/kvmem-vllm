import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_logical_prefill as module
from analyze_logical_prefill import audit_logical_sampling, expand_logical_geometry


def fixture():
    # Initial 2K must precede Action initialization; later 4K is one physical
    # group. Active offset is deliberately nonzero in the expanded interval.
    shapes = [dict(query_tokens=2048,active_seq_len=2048,logical_seq_len=2048,
                   offset=0,prefill=True,prefill_terminal=False),
              dict(query_tokens=4096,active_seq_len=5120,logical_seq_len=6144,
                   offset=1024,prefill=True,prefill_terminal=False)]
    groups=[dict(history_begin=2048,local_begin=1024,count=4096,retained=4096,
        scheduler_step=2,confirmation_step=3,spans=[
            dict(history_begin=2048,local_begin=1024,count=2048),
            dict(history_begin=4096,local_begin=3072,count=2048)])]
    return shapes,groups


def test_expand_uses_published_logical_coordinates_without_changing_offset():
    shapes,groups=fixture()
    actual=expand_logical_geometry(shapes,groups)
    assert [(x['query_tokens'],x['active_seq_len'],x['logical_seq_len'],x['offset']) for x in actual]==[
        (2048,2048,2048,0),(2048,3072,4096,1024),(2048,5120,6144,1024)]


@pytest.mark.parametrize('kind',['duplicate','unconfirmed','partial','gap','wrong_shape','terminal','unmatched'])
def test_unproven_expansion_is_rejected(kind):
    shapes,groups=fixture()
    if kind=='duplicate': groups*=2
    if kind=='unconfirmed': groups[0]['confirmation_step']=2
    if kind=='partial': groups[0]['retained']=2048
    if kind=='gap': groups[0]['spans'][1]['history_begin']+=32
    if kind=='wrong_shape': shapes[1]['query_tokens']=3072
    if kind=='terminal': shapes[1]['prefill_terminal']=True
    if kind=='unmatched': shapes=shapes[:1]
    with pytest.raises(ValueError): expand_logical_geometry(shapes,groups)


def sampling_fixture():
    _,groups = fixture()
    sampled = list(range(31,2048,32))
    for i,span in enumerate(groups[0]['spans']):
        # A historical page split into two segments remains one page identity.
        # Counts must not be assumed equal to active_length/block or pages.
        span.update(sampled_rows=64,sampled_row_indices=list(sampled),segments=97+64*i)
    trace = [dict(end=end,query_samples=64,samples=64*(i+1),chunks=i+1,pages=pages)
             for i,(end,pages) in enumerate(((2048,64),(4096,96),(6144,160)))]
    profile = dict(kvmem=dict(prefill_attention_stride=32,prefill_attention_interval=1024,
                             block_tokens=32,select_budget=32768,gen_budget=16384))
    return groups,trace,deepcopy(trace),profile


def test_sampling_audit_joins_original_commits_and_allows_partial_reordered_segments():
    args = sampling_fixture()
    result = audit_logical_sampling(*args)
    assert all(result['checks'].values())
    assert result['checked_sampled_rows'] == 128 and result['checked_logical_spans'] == 2


@pytest.mark.parametrize('side',[1,2])
@pytest.mark.parametrize('kind',['empty','duplicate','unordered','missing_middle','bad_total','bad_chunks'])
def test_sampling_reports_require_complete_nonempty_unique_commits(side,kind):
    args = list(sampling_fixture())
    trace = args[side]
    if kind == 'empty': trace.clear()
    if kind == 'duplicate': trace.append(dict(trace[-1]))
    if kind == 'unordered': trace.reverse()
    if kind == 'missing_middle': trace.pop(1)
    if kind == 'bad_total': trace[-1]['samples'] += 1
    if kind == 'bad_chunks': trace[-1]['chunks'] += 1
    with pytest.raises(ValueError): audit_logical_sampling(*args)


def test_same_sample_counts_with_wrong_row_phase_are_not_identical():
    groups,base,candidate,profile = sampling_fixture()
    groups[0]['spans'][0]['sampled_row_indices'] = list(range(30,2048,32))
    checks = audit_logical_sampling(groups,base,candidate,profile)['checks']
    assert checks['sampling_boundaries_identical']
    assert not checks['sampled_row_indices_identical']


@pytest.mark.parametrize('kind',['row_count','duplicate_row','row_outside'])
def test_malformed_published_rows_fail_closed(kind):
    groups,base,candidate,profile = sampling_fixture()
    span = groups[0]['spans'][0]
    if kind == 'row_count': span['sampled_rows'] -= 1
    if kind == 'duplicate_row': span['sampled_row_indices'][1] = 31
    if kind == 'row_outside': span['sampled_row_indices'][-1] = 2048
    with pytest.raises(ValueError): audit_logical_sampling(groups,base,candidate,profile)


def test_omitted_tail_reports_and_empty_rows_cannot_fake_interval_skip():
    groups,base,candidate,profile = sampling_fixture()
    base.pop();candidate.pop()
    groups[0]['spans'][-1].update(sampled_rows=0,sampled_row_indices=[])
    checks = audit_logical_sampling(groups,base,candidate,profile)['checks']
    assert checks['sampling_boundaries_identical']
    assert not checks['sampled_row_indices_identical']
    assert not checks['logical_reports_identical']


def test_legitimate_interval_skip_has_no_report_and_no_sampled_rows():
    groups,base,candidate,profile = sampling_fixture()
    profile['kvmem']['prefill_attention_interval'] = 4096
    groups[0]['spans'][0].update(sampled_rows=0,sampled_row_indices=[])
    for trace in (base,candidate):
        trace.pop(1)
        trace[-1].update(samples=128,chunks=2)
    assert all(audit_logical_sampling(groups,base,candidate,profile)['checks'].values())


def test_near_capacity_forces_sampling_despite_interval():
    groups,base,candidate,profile = sampling_fixture()
    profile['kvmem'].update(prefill_attention_interval=8192,select_budget=3072,gen_budget=2048)
    assert all(audit_logical_sampling(groups,base,candidate,profile)['checks'].values())


def test_zero_stride_keeps_original_last_row_only():
    groups,base,candidate,profile = sampling_fixture()
    profile['kvmem']['prefill_attention_stride'] = 0
    for span in groups[0]['spans']:
        span.update(sampled_rows=1,sampled_row_indices=[2047])
    for trace in (base,candidate):
        for i,row in enumerate(trace):
            row.update(query_samples=1,samples=i+1)
    assert all(audit_logical_sampling(groups,base,candidate,profile)['checks'].values())


@pytest.mark.parametrize('kind',['too_few','too_many','wrong_growth','pages_exceed_segments'])
def test_segment_evidence_rejects_impossible_counts_without_assuming_page_identity(kind):
    groups,base,candidate,profile = sampling_fixture()
    spans = groups[0]['spans']
    if kind == 'too_few': spans[0]['segments'] = 95
    if kind == 'too_many': spans[0]['segments'] = 3073
    if kind == 'wrong_growth': spans[1]['segments'] += 1
    if kind == 'pages_exceed_segments': candidate[1]['pages'] = 98
    assert not audit_logical_sampling(groups,base,candidate,profile)['checks']['segment_evidence_valid']


def test_analyze_keeps_original_failed_exact_checks_and_limits_summary_claims(monkeypatch,tmp_path):
    shapes,publications = fixture()
    groups,baseline,candidate,profile = sampling_fixture()
    expected = expand_logical_geometry(shapes,publications)
    pair = dict(id='case',checks=dict(output_token_ids_identical=False,geometry_identical=False,
        selections_and_mandatory_identical=False,accepted_score_graph_summary_identical=False))
    original_checks = deepcopy(pair['checks'])
    result = dict(pairs=[pair],all_checks_passed=False,summary={})
    monkeypatch.setattr(module.variant,'compare',lambda *_a,**_kw:result)
    def read(path):
        if path.name == 'suite.json': return dict(cases=[dict(id='case',warmup=False)])
        if path.name == 'profile.json':
            value = deepcopy(profile)
            value['kvmem']['prefill_physical_coalescing'] = 'candidate' in path.parts
            return value
        if path.name == 'response.json': return dict(response_id='req')
        if path.name == 'evidence.json': return dict(refresh=[])
        if path.name == 'result.json': return dict(input_tokens=6144)
        raise AssertionError(path)
    def rows(path):
        reference = 'reference' in path.parts
        if path.name == 'model_step_latency.jsonl': return ['base' if reference else 'candidate']
        if path.name == 'logical_prefill_publication.jsonl':
            return [] if reference else [dict(row,request_id='req-0') for row in groups]
        if path.name == 'prefill_attention.jsonl':
            return [dict(row,request_id='req-0') for row in (baseline if reference else candidate)]
        raise AssertionError(path)
    monkeypatch.setattr(module,'read',read)
    monkeypatch.setattr(module,'rows',rows)
    monkeypatch.setattr(module.pipeline,'prefill_groups',lambda value:[value])
    monkeypatch.setattr(module,'reconstruct_geometry',lambda value,*_:expected if value == ['base'] else shapes)
    monkeypatch.setattr(module,'save',lambda *_:None)
    actual = module.analyze(tmp_path)
    assert actual['pairs'][0]['checks'] == original_checks
    assert actual['all_checks_passed'] is False
    assert actual['logical_policy_evidence']['all_checks_passed'] is True
    assert 'No per-page score or graph-edge weight equality' in actual['logical_policy_evidence']['scope']


@pytest.mark.parametrize('number',[3,4,7,8])
def test_expansion_preserves_each_larger_group_boundary(number):
    shapes,groups = fixture()
    record = groups[0]
    for i in range(2,number):
        record['spans'].append(dict(history_begin=2048+2048*i,local_begin=1024+2048*i,count=2048))
    record['count'] = record['retained'] = number*2048
    shapes[-1].update(query_tokens=number*2048,active_seq_len=1024+number*2048,
                      logical_seq_len=2048+number*2048)
    result = expand_logical_geometry(shapes,groups,number)
    assert len(result) == number+1
    assert result[-1]['logical_seq_len'] == 2048+number*2048
    assert all(row['query_tokens'] == 2048 for row in result)
    if number > 2:
        with pytest.raises(ValueError,match='bounded'):
            expand_logical_geometry(shapes,groups,number-1)
