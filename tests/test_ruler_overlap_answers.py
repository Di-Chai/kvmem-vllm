"""Offline checks for the one-arm overlap answer experiment and its evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

import analyze_block_graph as offline
from vllm_kvmem.graph_scoring import score_graph

_source = Path(__file__).resolve().parents[1]/'scripts/test_ruler_overlap_answers.py'
_spec = importlib.util.spec_from_file_location('overlap_answer_campaign', _source)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def test_overlap_profile_preserves_execution_and_only_changes_requested_query_policy():
    source = Path(__file__).resolve().parents[1]/'artifacts/ruler-retrieval-answer-20261023-r1/mean_k/profile.json'
    if not source.exists():
        pytest.skip('Saved local diagnostic cohort is unavailable')
    original = experiment.read(source); before = deepcopy(original)
    profile = experiment.profile_for_overlap(original)
    assert original == before
    allowed = {'query_score_mode', 'query_window_cache_bytes', 'graph_mode',
        'graph_scoring_version', 'graph_aggregation', 'graph_alpha', 'graph_trace',
        'guided_query_tokens', 'query_history'}
    assert {key for key in original['kvmem'] if original['kvmem'][key] != profile['kvmem'][key]} <= allowed
    assert {key for key in original if original[key] != profile[key]} <= {'port', 'kvmem'}
    assert profile['kvmem']['graph_alpha'] == .2
    assert profile['kvmem']['graph_aggregation'] == 'sum'
    assert profile['kvmem']['graph_trace']


def audit_fixture(alpha=.2):
    record = offline.causal_fixture(); record.pop('evidence')
    record['starts'] = [v*8 for v in record['starts']]
    record['counts'] = [v*8 for v in record['counts']]
    record['budget'] *= 8
    for key in ('block_tokens', 'select_budget', 'gen_budget'):
        record['config'][key] *= 8
    for row in record['graph']['rows']:
        row['source'] *= 8; row['targets'] = [v*8 for v in row['targets']]
    for key in ('accepted_through', 'completed_through'):
        record['graph'][key] *= 8
    record.update(version=2, phase='query', seed_source=experiment.ARM, scoring_version='v2',
        aggregation='sum', mode='multi', alpha=alpha, score_domain_version='qw3-native-v2',
        cross_message_weight=.25,
        accepted_boundary=192, score_mask=[True]*6, transport_mask=[True]*6,
        handles=[[i,1] for i in range(6)])
    cfg,snapshot = offline._validate_record(record)
    memory,handles = offline._native_history(cfg,record['starts'],record['counts'])
    try:
        with score_graph(record['raw_scores'], record['starts'], record['transport_mask'], snapshot,
                mode='multi', scoring_version='v2', aggregation='sum', alpha=alpha,
                cross_message_weight=record['cross_message_weight'], block_tokens=32) as scores:
            slots = memory.preview_selection({handles[start].slot:score for start,score in
                zip(record['starts'],scores.scores)}, (), action=False, max_tokens=record['budget'])
            selected = sorted(start for start,handle in handles.items() if handle.slot in slots)
            identity = scores.diagnostics['identity']
    finally:
        memory.close()
    query = dict(query_scoring_method=experiment.ARM, query_score_tokens=sum(record['raw_scores']),
        selected_history_ranges=[[start,32] for start in selected],
        query_window_scoring=dict(method=experiment.ARM,window_tokens=16,window_stride=8,block_tokens=32,
            valid_windows=23,geometry_windows=24,fallback_blocks=0,
            normalization='block softmax after per-layer/per-Query/per-head window max'))
    graph = {key:record[key] for key in ('seed_source','scoring_version','mode','aggregation','alpha')}
    graph.update(identity=identity, propagation_applied=not identity)
    profile = dict(kvmem=dict(select_budget=record['budget'],sink_blocks=0,graph_alpha=alpha))
    return query,graph,record,profile


def test_execution_audit_replays_the_existing_native_picker():
    query,graph,record,profile = audit_fixture()
    audit = experiment.audit_selection(query,graph,record,profile)
    assert audit['native_picker_replay_equal'] and audit['no_post_picker_neighbors']
    assert audit['selected_tokens'] == record['budget']
    picked = {start for start,_ in query['selected_history_ranges']}
    replacement = next(start for start in record['starts'] if start not in picked)
    query['selected_history_ranges'][0][0] = replacement
    with pytest.raises(ValueError,match='native SUM picker replay'):
        experiment.audit_selection(query,graph,record,profile)


@pytest.mark.parametrize('field,value', [('aggregation','max'), ('alpha',0), ('seed_source','mean_k')])
def test_execution_audit_rejects_a_different_graph_policy(field,value):
    query,graph,record,profile = audit_fixture()
    graph[field] = value
    with pytest.raises(ValueError,match='SUM propagation'):
        experiment.audit_selection(query,graph,record,profile)


def test_overlap_trace_is_not_exhausted_by_previous_pressure_diagnostics(tmp_path):
    from test_graph_v2_ablation import export_fixture,export
    authority,rows,snapshot = export_fixture(tmp_path,nodes=4)
    authority.query_score_mode = experiment.ARM
    authority._graph_input_count = 8
    export(authority,rows,snapshot)
    assert authority._window_query_input_count == 1 and authority._graph_input_count == 8
    assert json.loads((tmp_path/'block_graph_inputs.jsonl').read_text())['seed_source'] == experiment.ARM


def test_overlap_trace_keeps_a_complete_bounded_256k_four_edge_graph(tmp_path):
    from test_graph_v2_ablation import export_fixture
    from vllm_kvmem.block_graph import GraphRow
    from vllm_kvmem.graph_integration import record_query_inputs
    authority,rows,snapshot = export_fixture(tmp_path,nodes=8192)
    authority.query_score_mode = experiment.ARM
    snapshot.rows = tuple(GraphRow(i*32,tuple((i-j)*32 for j in range(1,min(i,4)+1)),
        tuple(.12345678901234567 for _ in range(min(i,4))),1,1,0) for i in range(1,8192))
    snapshot.locality_provenance = (('bounded_extra_metadata','x'*100000),)
    record_query_inputs(authority,rows,[.0000012345678901234567]*len(rows),[True]*len(rows),snapshot,(),None)
    path = tmp_path/'block_graph_inputs.jsonl'
    assert 2*1024**2 < path.stat().st_size < 4*1024**2
    loaded = json.loads(path.read_text())
    assert len(loaded['raw_scores']) == 8192 and len(loaded['graph']['rows']) == 8191
    snapshot.locality_provenance = (('oversized','x'*(4*1024**2)),)
    with pytest.raises(BufferError,match='record byte capacity'):
        record_query_inputs(authority,rows,[1/len(rows)]*len(rows),[True]*len(rows),snapshot,(),None)


def test_controller_cannot_execute_from_mutable_source_tree(tmp_path):
    with pytest.raises(ValueError,match='frozen'):
        experiment.run(tmp_path/'new-root')
    assert not (tmp_path/'new-root').exists()


def test_alpha_zero_replays_original_overlap_seeds_without_graph_transport():
    query,graph,record,profile = audit_fixture(0)
    audit = experiment.audit_selection(query,graph,record,profile)
    assert audit['graph_alpha'] == 0 and audit['propagation_applied'] is False
    assert audit['propagation']['identity'] is True
    assert audit['propagation']['raw_l1_change'] == 0
    assert audit['native_picker_replay_equal']


@pytest.mark.parametrize('change', [dict(propagation_applied=True), dict(identity=False),
    dict(propagation_applied=None)])
def test_alpha_zero_rejects_missing_or_nonidentity_execution_evidence(change):
    query,graph,record,profile = audit_fixture(0)
    graph.update(change)
    with pytest.raises(ValueError,match='explicitly report no graph propagation'):
        experiment.audit_selection(query,graph,record,profile)


def test_paired_overlap_profile_only_changes_graph_alpha_and_port():
    before = dict(port=18864,kvmem=dict(graph_alpha=.2,query_score_mode=experiment.ARM,
        select_budget=36864,gen_budget=16384,graph_trace=True))
    after = deepcopy(before);after['port']=18865;after['kvmem']['graph_alpha']=0.
    assert experiment.profile_difference(before,after) == {
        'port':dict(before=18864,after=18865),'kvmem.graph_alpha':dict(before=.2,after=0.)}
    after['kvmem']['select_budget'] = 32768
    with pytest.raises(ValueError,match='beyond graph_alpha and port'):
        experiment.profile_difference(before,after)


def reference_fixture(tmp_path):
    root = tmp_path/'completed-overlap'
    fixture = dict(rows=[dict(id='case-a',seed=42)], input_directory='/immutable-inputs',
        sampling=dict(max_tokens=8192), starts_in_thinking=True, upstream_path='/immutable-scorer',
        tokenizer_sha256='tokenizer-digest')
    profile = dict(port=18864,kvmem=dict(query_score_mode=experiment.ARM,graph_alpha=.2,
        select_budget=36864,graph_trace=True))
    for name,value in {'fixture.json':fixture,'status.json':dict(status='completed'),
            'execution-audit.json':dict(status='passed'),'target-spans.json':{},
            experiment.ARM+'/profile.json':profile, experiment.ARM+'/scores.json':dict(cases=1),
            experiment.ARM+'/cleanup.json':dict(remaining_owned_groups=[],gpu_processes=[])}.items():
        experiment.save(root/name,value)
    runtime_source = root/'runtime/src/scorer.py';runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text('immutable scorer\n')
    experiment.save(root/'runtime-proof.json',dict(files={'src/scorer.py':experiment.sha(runtime_source)}))
    experiment.save(root/'prepared-proof.json', {'fixture.json':experiment.sha(root/'fixture.json'),
        experiment.ARM+'/profile.json':experiment.sha(root/experiment.ARM/'profile.json')})
    case = root/experiment.ARM/'cases/case-a'
    for name in experiment.CASE_FILES:
        experiment.save(case/name,dict(original=name))
    experiment.save(case/'hashes.json',{name:experiment.sha(case/name) for name in experiment.CASE_FILES})
    candidate = deepcopy(profile);candidate['kvmem']['graph_alpha']=0.;candidate['port']=18865
    return root,fixture,candidate


def test_overlap_reference_hashes_answers_and_rejects_modified_evidence(tmp_path):
    root,fixture,candidate = reference_fixture(tmp_path)
    proof = experiment.overlap_reference_proof(root,fixture,candidate)
    assert proof['candidate_alpha'] == 0 and proof['reference_alpha'] == .2
    assert experiment.ARM+'/cases/case-a/response.json' in proof['files']
    assert experiment.ARM+'/cases/case-a/hashes.json' in proof['files']
    experiment.save(root/experiment.ARM/'cases/case-a/response.json',dict(changed=True))
    with pytest.raises(ValueError,match='Immutable experiment evidence changed'):
        experiment.overlap_reference_proof(root,fixture,candidate)


def test_overlap_reference_requires_identical_samples_and_completed_audit(tmp_path):
    root,fixture,candidate = reference_fixture(tmp_path)
    altered = deepcopy(fixture);altered['rows'][0]['seed']=43
    with pytest.raises(ValueError,match='fixture, seed or sampling'):
        experiment.overlap_reference_proof(root,altered,candidate)
    experiment.save(root/'execution-audit.json',dict(status='failed'))
    with pytest.raises(ValueError,match='completed and audited'):
        experiment.overlap_reference_proof(root,fixture,candidate)


@pytest.mark.parametrize('alpha', [-.2, .1, 1, True, '0'])
def test_only_requested_graph_alpha_choices_are_allowed(alpha):
    with pytest.raises(ValueError,match='Only graph alpha'):
        experiment.validate_alpha(alpha)
