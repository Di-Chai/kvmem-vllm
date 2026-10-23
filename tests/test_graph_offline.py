"""Recorded Query ablations retain native budgets without inventing quality."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1]/'scripts/analyze_block_graph.py'
spec = importlib.util.spec_from_file_location('analyze_block_graph',path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_six_arms_native_budget_and_causal_dependency_fixture():
    result = analysis.analyze_record(analysis.causal_fixture())
    arms = result['arms']
    assert tuple(arms) == analysis.ARMS
    assert arms['mean_k']['selected_starts'] == [8,12,20]
    assert arms['collect']['selected_starts'] == arms['mean_k']['selected_starts']
    assert arms['attention']['selected_starts'] == [4,8,20]
    assert arms['multi']['selected_starts'] == [4,8,20]
    assert all(arm['selected_tokens'] == 12 for arm in arms.values())
    assert arms['mean_k']['quality']['required_evidence_coverage'] == .5
    assert arms['attention']['quality']['required_evidence_coverage'] == 1.
    assert arms['attention']['quality']['required_dependency_coselection'] == 1.
    assert all(arm['quality']['answer_correctness'] is None for arm in arms.values())


def test_unlabeled_record_never_fabricates_quality_or_task_correctness():
    record = analysis.causal_fixture()
    record.pop('evidence')
    result = analysis.analyze_record(record)
    assert not result['labeled_fixture']
    assert all(all(value is None for value in row['quality'].values()) for row in result['arms'].values())
    assert result['arms']['attention']['observed_mass_coselection'] == 1.


def test_full_cpu_history_above_gpu_window_keeps_original_budgets():
    record = analysis.causal_fixture()
    record['starts'],record['counts'] = list(range(0,80,4)),[4]*20
    record['raw_scores'] = [.001]*14+[.01,.10,.15,.12,.02,.60]
    record['eligible'] = [True]*20
    record['config']['repository_blocks'] = 32
    record['graph']['rows'][0]['source'] = 76
    record['graph']['rows'][0]['targets'] = [60]
    record['graph']['accepted_through'] = record['graph']['completed_through'] = 80
    record.pop('evidence')
    original = deepcopy(record)
    result = analysis.analyze_record(record)
    assert record == original
    assert result['budget'] == 12 and result['history_blocks'] == 20
    assert result['arms']['mean_k']['selected_starts'] == [64,68,76]
    assert result['arms']['attention']['selected_starts'] == [60,64,76]
    assert all(row['selected_tokens'] == 12 for row in result['arms'].values())


def test_runtime_budget_sink_recent_mandatory_and_exact_mask():
    record = analysis.causal_fixture()
    record['config'].update(select_budget=16,sink_blocks=1,recent_blocks=1)
    record['mandatory_starts'] = [8]
    record['eligible'] = [False,True,False,True,True,False]
    record['raw_scores'] = [0.,.2,0.,.7,.1,0.]
    record['budget'] = 12
    result = analysis.analyze_record(record)
    assert all(row['selected_starts'] == [0,8,20] for row in result['arms'].values())
    record['eligible'][2] = True
    with pytest.raises(ValueError,match='eligible mask'):
        analysis.analyze_record(record)


def test_legacy_native_output_attention_takes_precedence_over_prefill():
    record = analysis.causal_fixture()
    record['legacy_action_scores'] = {0:1.}
    record['prefill_scores'] = {16:100.}
    result = analysis.analyze_record(record,legacy_fraction=1.)
    legacy = result['arms']['legacy_attention']
    assert 0 in legacy['selected_starts']
    assert 16 not in legacy['selected_starts']
    assert result['arms']['attention']['selected_starts'] == [4,8,20]


@pytest.mark.parametrize('change,match',[
    (lambda r:r.update(starts=[0,4,8,12,16,24]),'contiguous'),
    (lambda r:r['graph']['rows'][0].update(source=4,targets=[20]),'noncausal'),
    (lambda r:r.update(counts=[4,3,4,4,4,4]),'non-tail'),
    (lambda r:r.update(budget=16),'runtime budget'),
    (lambda r:r.update(version=2),'version')])
def test_invalid_records_are_rejected_instead_of_changing_experiment(change,match):
    record = analysis.causal_fixture()
    change(record)
    with pytest.raises(ValueError,match=match):
        analysis.analyze_record(record)


def test_cli_preserves_record_hash_and_writes_bounded_json_csv(tmp_path):
    recorded = tmp_path/'recorded.jsonl'
    recorded.write_text(json.dumps(analysis.causal_fixture())+'\n')
    output = tmp_path/'result'
    result = analysis.main(['--input',str(recorded),'--output-dir',str(output)])
    saved = json.loads((output/'analysis.json').read_text())
    assert saved['records'][0]['input_sha256'] == result['records'][0]['input_sha256']
    assert saved['kind'] == 'fixed_recorded_input_selection_analysis'
    assert 'No model inference' in saved['limitations'][1]
    assert len((output/'summary.csv').read_text().splitlines()) == 7
    assert len(saved['sources']) == 1


def test_record_limit_and_partial_tail_accounting(tmp_path):
    record = analysis.causal_fixture()
    record['counts'][-1] = 2
    record['graph']['rows'] = []
    record['graph']['accepted_through'] = 22
    record['graph']['completed_through'] = 20
    result = analysis.analyze_record(record)
    assert all(row['selected_tokens'] <= 12 for row in result['arms'].values())
    path = tmp_path/'input.jsonl'
    path.write_text((json.dumps(record)+'\n')*3)
    records,_ = analysis.load_records([path],2)
    assert len(records) == 2
    with pytest.raises(ValueError,match='limit'):
        analysis.load_records([path],257)


def fixture_case(change=None):
    """Real qualification schema with exact byte-token needles, without models."""
    spec = importlib.util.spec_from_file_location('offline_qualification',
        Path(__file__).parents[1]/'scripts/qualify_block_graph.py')
    qual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qual)
    class Tokenizer:
        eos_token_id = 0

        def encode(self,text,add_special_tokens=False):
            return list(text.encode())
    tokenizer = Tokenizer()
    fixture = qual.build_fixture(tokenizer,prompt_tokens=2048,concurrency=2)
    if change:
        change(fixture)
        for agent in fixture['agents']:
            agent['first_input_sha256'] = analysis.canonical_digest(agent['initial_tokens'])
        fixture['sha256'] = analysis.canonical_digest({k:v for k,v in fixture.items() if k!='sha256'})
    traces,records = [],[]
    for agent in fixture['agents']:
        index = agent['index']
        output = [900001+index,900003+index]
        second,_,spans,before = qual.next_turn(agent,agent['initial_tokens'],output)
        edited,_,_ = qual.edited_turn(agent,before,spans)
        for turn,prompt in enumerate((agent['initial_tokens'],second,edited)):
            rid = f'cmpl-agent{index}-turn{turn}'
            traces.append(dict(agent=index,turn=turn,response_id=rid,
                input_sha256=analysis.canonical_digest(prompt),input_tokens=len(prompt),output_ids=output))
            end = agent['archive_prefix_tokens']//32*32 if turn==0 else len(before)//32*32
            starts = list(range(0,end,32))
            record = analysis.causal_fixture()
            record.update(request_id=rid+'-0-native',authority_id=f'cmpl-agent{index}-turn0-0-native',
                mode='collect',starts=starts,counts=[32]*len(starts),raw_scores=[1/len(starts)]*len(starts),
                eligible=[True]*len(starts),mandatory_starts=[],budget=128,
                config=dict(block_tokens=32,select_budget=128,gen_budget=128,repository_blocks=512,
                    sink_blocks=0,recent_blocks=0,action_attention=True,action_fraction=0.,action_combine='quota'),
                graph=dict(rows=[],partial=None,message_spans=[],message_end_known=[],
                    accepted_through=end,completed_through=end,revision=1),
                legacy_action_scores={},prefill_scores=None)
            record.pop('evidence')
            records.append(record)
    return records,fixture,traces,tokenizer


def test_frozen_fixture_labels_exact_full_evidence_spans_and_trace_hashes():
    records,fixture,traces,tokenizer = fixture_case()
    originals = deepcopy((records,fixture,traces))
    labeled,proof = analysis.annotate_fixture(records,fixture,traces,tokenizer)
    assert (records,fixture,traces) == originals
    assert len(labeled)==len(proof['joins'])==6 and len(proof['needle_proofs'])==4
    for record,derived in zip(records,labeled):
        assert derived['annotation']['original_record_sha256']==analysis.canonical_digest(record)
        spans = derived['evidence']['spans']
        assert {span['name'] for span in spans}=={'source','dependency'}
        required = set(derived['evidence']['required_starts'])
        for span in spans:
            assert span['block_starts']==list(range(span['begin']//32*32,((span['end']-1)//32+1)*32,32))
        quality = analysis._quality(derived,required,required)
        assert quality['required_evidence_coverage']==quality['required_dependency_coselection']==1.
        assert quality['full_evidence_coselection'] is True
        assert quality['outdated_blocks'] is quality['irrelevant_blocks'] is quality['answer_correctness'] is None
        assert not any(required&set(group['starts']) for group in derived['evidence']['duplicate_groups'])
        prompt = fixture['agents'][derived['annotation']['agent']]['initial_tokens']
        for group in derived['evidence']['duplicate_groups']:
            for start in group['starts']:
                if start+32<=len(prompt):
                    assert prompt[start:start+32]==group['token_ids']
    result = analysis.analyze_record(labeled[0])
    assert result['annotation']==labeled[0]['annotation']
    summary = analysis.summarize_records([result])
    assert summary['multi']['required_evidence_coverage']['labeled_records']==1
    assert summary['multi']['irrelevant_blocks']==dict(labeled_records=0,mean=None,total=None)


def test_duplicate_metric_counts_extra_selected_identical_blocks_only():
    record = dict(evidence=dict(required_starts=[32],
        duplicate_groups=[dict(starts=[0,64,96],token_ids=[7]*32)]))
    single = analysis._quality(record,{96},{96})
    assert single['duplicate_blocks']==0
    assert analysis._quality(record,{64,96},{64,96})['duplicate_blocks']==1
    assert analysis._quality(record,{0,64,96},{0,64,96})['duplicate_blocks']==2
    assert single['outdated_blocks'] is single['irrelevant_blocks'] is None
    record['evidence'].pop('duplicate_groups')
    assert analysis._quality(record,{96},{96})['duplicate_blocks'] is None


@pytest.mark.parametrize('tokens,needle,match',[
    ([1,2,3],[4],'found 0'),([1,2,1,2],[1,2],'found 2'),([1],[],'empty')])
def test_needle_search_rejects_absence_and_ambiguity(tokens,needle,match):
    with pytest.raises(ValueError,match=match):
        analysis.unique_token_span(tokens,needle)


@pytest.mark.parametrize('duplicate',[False,True])
def test_annotation_rejects_missing_or_ambiguous_frozen_statement_even_with_valid_hashes(duplicate):
    def change(fixture):
        agent = fixture['agents'][0]
        tokens,needle = agent['initial_tokens'],agent['source_tokens']
        begin,end = analysis.unique_token_span(tokens,needle)
        if duplicate:
            tokens[:len(needle)] = needle
        else:
            tokens[begin:end] = [900009]*len(needle)
    args = fixture_case(change)
    with pytest.raises(ValueError,match='found 2' if duplicate else 'found 0'):
        analysis.annotate_fixture(*args)


@pytest.mark.parametrize('mutation,match',[
    (lambda r,f,t: t[1].update(input_sha256='changed'),'prompt fingerprint'),
    (lambda r,f,t: r[0].update(request_id='unknown-request'),'joined uniquely'),
    (lambda r,f,t: r[0].update(authority_id=r[3]['authority_id']),'another fixture session'),
    (lambda r,f,t: t[3].update(response_id=t[0]['response_id']),'joined uniquely'),
    (lambda r,f,t: f['agents'][0]['source_tokens'].append(1),'immutable hash')])
def test_annotation_rejects_forged_trace_session_or_fixture(mutation,match):
    records,fixture,traces,tokenizer = fixture_case()
    mutation(records,fixture,traces)
    with pytest.raises(ValueError,match=match):
        analysis.annotate_fixture(records,fixture,traces,tokenizer)


def test_annotation_requires_local_tokenizer_and_complete_recorded_evidence():
    records,fixture,traces,tokenizer = fixture_case()
    class WrongTokenizer:
        def encode(self,*args,**kwargs):
            return [1]
    with pytest.raises(ValueError,match='tokenizer encoding'):
        analysis.annotate_fixture(records,fixture,traces,WrongTokenizer())
    begin,_ = analysis.unique_token_span(fixture['agents'][0]['initial_tokens'],fixture['agents'][0]['source_tokens'])
    records[0]['starts'].remove(begin//32*32)
    with pytest.raises(ValueError,match='missing part'):
        analysis.annotate_fixture(records,fixture,traces,tokenizer)


def v2_query_fixture():
    record = analysis.causal_fixture()
    record.update(version=2, phase='query', scoring_version='v2', alpha=.2,
        score_domain_version='qw3-native-v2', seed_source='mean_k', accepted_boundary=24,
        score_mask=[True]*6, transport_mask=[True]*6,
        # Historical generation-qualified slots need not match offline slots.
        handles=[[slot, 10+i] for i, slot in enumerate([9, 2, 5, 0, 13, 7])],
        baseline_selected_starts=[8, 12, 20])
    record['graph']['locality_provenance'] = {'source': 'renderer-offsets', 'tokenizer': 'fixture'}
    return record


def v2_pressure_fixture():
    record = v2_query_fixture()
    repository = [(slot, 10+i, 4*i, 4) for i, slot in enumerate([9, 2, 5, 0, 13, 7, 10, 11])]
    candidates = [repository[i] + (4*j,) for j, i in enumerate([0, 2, 3, 5])]
    record.update(phase='pressure', seed_source='output', pressure_policy='graph', alpha=.8,
        starts=[r[2] for r in candidates], counts=[r[3] for r in candidates],
        handles=[list(r[:2]) for r in candidates], raw_scores=[0., 0., .1, .9],
        eligible=[True]*4, score_mask=[True]*4, transport_mask=[True]*4,
        candidate_records=candidates, repository_records=repository,
        accepted_boundary=32, budget=8, baseline_selected_starts=[12, 20])
    # Recent Query keep bands must not become hard retention in pressure.
    record['config']['recent_blocks'] = 2
    record['graph'].update(accepted_through=32, completed_through=32,
        rows=[dict(source=20, targets=[4, 8], weights=[.9, .01],
                   samples=1, prefill_samples=0, output_samples=1)])
    record['legacy_action_scores'] = {28: 999.}
    record.pop('evidence')
    return record


def test_v2_query_uses_recorded_version_alpha_domain_and_provenance():
    record = v2_query_fixture()
    original = deepcopy(record)
    result = analysis.analyze_record(record)
    assert record == original
    assert result['phase'] == 'query' and result['scoring_version'] == 'v2'
    assert result['score_domain_version'] == 'qw3-native-v2'
    assert result['alpha'] == .2 and result['input_version'] == 2
    assert result['locality_provenance'] == record['graph']['locality_provenance']
    assert result['recorded_handles'] == record['handles']
    assert result['arms']['multi']['diagnostics']['locality_weight'] == .1
    assert result['arms']['multi']['diagnostics']['attention_weight'] == .1
    assert result['arms']['attention']['diagnostics']['attention_external_mass'] == pytest.approx(.12)
    assert result['baseline_selected_starts'] == [8, 12, 20]


def test_recorded_max_pooling_replays_as_max_and_is_not_merged_with_sum():
    summed=v2_query_fixture()
    pooled=deepcopy(summed)
    pooled['aggregation']='max'
    before=deepcopy(pooled)
    result=analysis.analyze_record(pooled)
    assert pooled==before
    assert result['aggregation']=='max'
    assert result['arms']['multi']['diagnostics']['aggregation']=='max'
    assert result['arms']['multi']['diagnostics']['score_sum_conserved'] is False
    summary=analysis.summarize_records([analysis.analyze_record(summed),result])
    assert summary['query:v2:multi']['records']==1
    assert summary['query:v2:max:multi']['records']==1


def test_v2_export_can_explicitly_replay_v1_operator_with_recorded_weights():
    record = v2_query_fixture()
    record.update(scoring_version='v1', locality_weight=.03, attention_weight=.4,
                  cross_message_weight=.5)
    result = analysis.analyze_record(record)
    assert result['scoring_version'] == 'v1' and result['alpha'] is None
    assert result['arms']['multi']['diagnostics']['locality_weight'] == .03
    assert result['arms']['multi']['diagnostics']['attention_weight'] == .4
    assert result['arms']['attention']['diagnostics']['attention_external_mass'] == pytest.approx(.216)


def test_v2_query_native_domain_includes_extra_mandatory_in_score_mask():
    record = v2_query_fixture()
    record.update(mandatory_starts=[8], baseline_selected_starts=[8, 12, 20])
    record['eligible'][2] = record['transport_mask'][2] = False
    result = analysis.analyze_record(record)
    assert result['arms']['mean_k']['diagnostics']['raw_score_sum'] == pytest.approx(1.)
    assert result['arms']['mean_k']['diagnostics']['raw_eligible_sum'] == pytest.approx(.85)
    record['score_mask'][2] = False
    with pytest.raises(ValueError, match='score mask'):
        analysis.analyze_record(record)


def test_v2_query_small_history_native_transport_is_all_identity():
    record = v2_query_fixture()
    record['config'].update(select_budget=24, gen_budget=8, sink_blocks=1, recent_blocks=1)
    record.update(budget=24, eligible=[False]*6, transport_mask=[False]*6,
                  baseline_selected_starts=record['starts'])
    result = analysis.analyze_record(record)
    assert result['arms']['multi']['selected_starts'] == record['starts']
    assert result['arms']['multi']['diagnostics']['identity']


def test_pressure_replay_uses_original_picker_complete_candidates_and_logical_identity(monkeypatch):
    record = v2_pressure_fixture()
    original = deepcopy(record)
    native_history, calls = analysis._native_history, []
    def tracked_history(*args):
        memory, handles = native_history(*args)
        preview = memory.preview_selection
        def pressure_only(scores, mandatory=(), **kwargs):
            assert scores == {} and kwargs['action'] is False
            values = kwargs['pressure_scores']
            assert [slot for slot, _score in values] == [handles[start].slot for start in record['starts']]
            assert len(values) == len(record['candidate_records'])
            calls.append(values)
            return preview(scores, mandatory, **kwargs)
        memory.preview_selection = pressure_only
        return memory, handles
    monkeypatch.setattr(analysis, '_native_history', tracked_history)
    result = analysis.analyze_record(record)
    assert record == original
    assert len(calls) == len(analysis.PRESSURE_ARMS)
    assert tuple(result['arms']) == analysis.PRESSURE_ARMS
    assert result['arms']['attention_seed']['selected_starts'] == [12, 20]
    assert result['arms']['attention']['selected_starts'] == [8, 20]
    assert result['history_blocks'] == 8 and result['candidate_blocks'] == 4
    assert all(24 not in row['selected_starts'] and 28 not in row['selected_starts']
               for row in result['arms'].values())
    assert result['arms']['attention']['retained_raw_mean_k_mass'] is None
    assert result['arms']['attention']['retained_raw_seed_mass'] == .9


@pytest.mark.parametrize('raw,source', [([0., 0., .1, .9], 'output'),
                                      ([0., 0., 0., 0.], 'none'),
                                      ([0., 0., 1., 0.], 'prefill')])
def test_pressure_alpha_zero_is_exact_baseline_including_zero_candidates(raw, source):
    record = v2_pressure_fixture()
    record.update(raw_scores=raw, seed_source=source, alpha=0.)
    result = analysis.analyze_record(record)
    assert all(row['selected_starts'] == [12, 20] for row in result['arms'].values())
    assert all(row['diagnostics']['identity'] for row in result['arms'].values())


def test_pressure_allows_native_sink_and_mandatory_outside_working_candidates():
    record = v2_pressure_fixture()
    record['config'].update(sink_blocks=1, recent_blocks=0)
    record.update(budget=12, mandatory_starts=[16], baseline_selected_starts=[0, 16, 20])
    for name in ('starts', 'counts', 'raw_scores', 'eligible', 'score_mask', 'transport_mask',
                 'handles', 'candidate_records'):
        record[name] = record[name][1:]
    result = analysis.analyze_record(record)
    assert result['arms']['attention_seed']['selected_starts'] == [0, 16, 20]
    assert all({0, 16} <= set(row['selected_starts']) for row in result['arms'].values())


def test_pressure_partial_tail_and_preserved_future_graph_watermark():
    record = v2_pressure_fixture()
    record['repository_records'][-1] = (*record['repository_records'][-1][:3], 2)
    record.update(accepted_boundary=30)
    record['graph'].update(accepted_through=64, completed_through=64)
    result = analysis.analyze_record(record)
    assert result['arms']['attention_seed']['selected_starts'] == [12, 20]
    assert result['arms']['attention']['diagnostics']['accepted_through'] == 64


@pytest.mark.parametrize('change,match', [
    (lambda r:r.update(version=3), 'version'),
    (lambda r:r.update(scoring_version='v3'), 'scoring version'),
    (lambda r:r.update(score_domain_version='unknown'), 'score-domain version'),
    (lambda r:r.update(alpha=True), 'alpha'),
    (lambda r:r.update(phase='future'), 'phase'),
    (lambda r:r.update(baseline_selected_starts=[4, 8]), 'baseline'),
    (lambda r:r['handles'][0].__setitem__(1, 0), 'handles'),
    (lambda r:r['transport_mask'].__setitem__(0, False), 'masks'),
    (lambda r:r['candidate_records'].__setitem__(0, (9, 999, 0, 4, 0)), 'candidate generation'),
    (lambda r:r['candidate_records'].pop(), 'complete candidate'),
    (lambda r:r['repository_records'].pop(), 'accepted boundary'),
    (lambda r:r['score_mask'].__setitem__(0, False), 'complete scoring'),
])
def test_v2_replay_rejects_incomplete_or_conflicting_evidence(change, match):
    record = v2_pressure_fixture()
    change(record)
    with pytest.raises(ValueError, match=match):
        analysis.analyze_record(record)


def test_cli_mixed_versions_and_pressure_have_separate_summaries(tmp_path):
    recorded = tmp_path/'mixed.jsonl'
    rows = [analysis.causal_fixture(), v2_query_fixture(), v2_pressure_fixture()]
    recorded.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    output = tmp_path/'result'
    result = analysis.main(['--input', str(recorded), '--output-dir', str(output)])
    assert {'query:v1:multi', 'query:v2:multi', 'pressure:v2:multi'} <= set(result['summary'])
    assert all(row['records'] == 1 for row in result['summary'].values())
    assert len((output/'summary.csv').read_text().splitlines()) == 1+6+6+4
    assert 'Pressure arms use native pressure selection' in result['limitations'][-1]


def test_legacy_records_cannot_be_silently_relabelled_as_v2_or_pressure():
    record = analysis.causal_fixture()
    record['scoring_version'] = 'v2'
    with pytest.raises(ValueError, match='version 1'):
        analysis.analyze_record(record)
    record.pop('scoring_version')
    record['phase'] = 'pressure'
    with pytest.raises(ValueError, match='version 1'):
        analysis.analyze_record(record)
    with pytest.raises(ValueError, match='Legacy weight overrides'):
        analysis.analyze_record(v2_query_fixture(), attention_weight=.7)


def test_offline_history_handle_values_do_not_retain_closed_native_store():
    record = analysis.causal_fixture()
    cfg, _ = analysis._validate_record(record)
    memory, handles = analysis._native_history(cfg, record['starts'], record['counts'])
    repository = memory.repository
    assert memory.close()
    assert handles and repository.stats['leases'] == 0
