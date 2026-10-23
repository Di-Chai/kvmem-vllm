"""CPU-only protocol and evidence checks for the fixed Key FPS answer cohort."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

import analyze_block_graph as offline
from vllm_kvmem.graph_scoring import score_graph

_source = Path(__file__).resolve().parents[1]/'scripts/test_ruler_fps_answers.py'
_spec = importlib.util.spec_from_file_location('fps_answer_campaign', _source)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


def original_profile():
    return dict(port=18865, execution_mode='compiled_graph', draft_method='mtp', draft_tokens=4,
        prefill_chunk_tokens=2048, max_num_seqs=1, max_num_batched_tokens=16424,
        model='unchanged-nvfp4-model', tokenizer='unchanged-tokenizer',
        gpu_memory_utilization=.72, query_replay_budget=16384,
        kvmem=dict(query_score_mode='overlap_16_8', query_window_cache_bytes=2147483648,
            graph_alpha=0., graph_mode='multi', graph_scoring_version='v2',
            graph_aggregation='sum', graph_trace=True, guided_query_tokens=0, query_history=False,
            select_budget=36864, gen_budget=16384, block_tokens=32, sink_blocks=41, recent_blocks=0,
            pressure_policy='attention', prefill_attention=True, prefill_physical_coalescing=False,
            archive_backend='cpu', cpu_read_cache_bytes=100000000000,
            action_native_scores='reuse', arbitrary_unchanged_setting=[1, 2, 3]))


def test_fps_profile_changes_only_the_query_scorer_and_port():
    before = original_profile(); untouched = deepcopy(before)
    after = experiment.profile_for_fps(before)
    assert before == untouched
    assert experiment.profile_difference(before, after) == {
        'port': dict(before=18865, after=18866),
        'kvmem.query_score_mode': dict(before='overlap_16_8', after='key_fps_4')}
    assert after['kvmem']['graph_alpha'] == 0
    after['kvmem']['arbitrary_unchanged_setting'].append(4)
    assert before == untouched
    with pytest.raises(ValueError, match='beyond query_score_mode and port'):
        experiment.profile_difference(before, after)


def test_actual_saved_overlap_profile_has_only_the_two_requested_changes():
    source = experiment.OVERLAP_ROOT/'overlap_16_8/profile.json'
    if not source.exists():
        pytest.skip('Local completed overlap/no-graph profile is unavailable')
    before = experiment.read(source)
    after = experiment.profile_for_fps(before)
    assert set(experiment.profile_difference(before, after)) == {'port', 'kvmem.query_score_mode'}
    from vllm_kvmem.serve import build_launch
    launch = build_launch(after, _source.parents[1], Path('/tmp/fps-test-no-start'))
    assert launch['vllm_argv']


@pytest.mark.parametrize('location,key,value', [
    ('kvmem','graph_alpha',.2), ('kvmem','query_score_mode','mean_k'),
    ('kvmem','graph_aggregation','max'), ('kvmem','graph_mode','attention'),
    ('kvmem','guided_query_tokens',512), ('kvmem','query_history',True),
    ('kvmem','select_budget',32768), ('kvmem','gen_budget',65536),
    ('kvmem','recent_blocks',1), ('kvmem','pressure_policy','query'),
    ('kvmem','prefill_physical_coalescing',True), ('root','max_num_seqs',8),
    ('root','draft_tokens',1), ('root','prefill_chunk_tokens',1024),
    ('root','execution_mode','eager')])
def test_profile_rejects_an_unqualified_baseline(location,key,value):
    profile = original_profile()
    (profile if location == 'root' else profile[location])[key] = value
    with pytest.raises(ValueError):
        experiment.profile_for_fps(profile)


def audit_fixture():
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
        aggregation='sum', mode='multi', alpha=0., score_domain_version='qw3-native-v2',
        cross_message_weight=.25, accepted_boundary=192, score_mask=[True]*6,
        transport_mask=[True]*6, handles=[[i,1] for i in range(6)])
    cfg,snapshot = offline._validate_record(record)
    memory,handles = offline._native_history(cfg,record['starts'],record['counts'])
    try:
        with score_graph(record['raw_scores'], record['starts'], record['transport_mask'], snapshot,
                mode='multi', scoring_version='v2', aggregation='sum', alpha=0.,
                cross_message_weight=record['cross_message_weight'], block_tokens=32) as result:
            assert list(result.scores) == record['raw_scores']
            slots = memory.preview_selection({handles[start].slot:score for start,score in
                zip(record['starts'],result.scores)}, (), action=False, max_tokens=record['budget'])
            selected = sorted(start for start,handle in handles.items() if handle.slot in slots)
    finally:
        memory.close()
    query = dict(query_scoring_method=experiment.ARM, query_score_tokens=sum(record['raw_scores']),
        selected_history_ranges=[[start,32] for start in selected],
        query_key_fps_scoring=dict(method=experiment.ARM, representatives_per_block=4, block_tokens=32,
            selection='raw_key_farthest_first',
            normalization='block softmax after per-layer/per-Query/per-head representative max'))
    graph = {key:record[key] for key in ('seed_source','scoring_version','mode','aggregation','alpha')}
    graph.update(identity=True, propagation_applied=False)
    profile = dict(kvmem=dict(select_budget=record['budget'],sink_blocks=0,graph_alpha=0.))
    return query,graph,record,profile


def test_fps_audit_replays_identity_seeds_and_the_existing_native_picker():
    query,graph,record,profile = audit_fixture()
    audit = experiment.audit_selection(query,graph,record,profile)
    assert audit['native_picker_replay_equal'] and audit['no_post_picker_neighbors']
    assert audit['graph_alpha'] == 0 and audit['propagation_applied'] is False
    assert audit['propagation']['identity'] is True and audit['propagation']['raw_l1_change'] == 0
    assert audit['selected_tokens'] == record['budget']
    picked = {start for start,_ in query['selected_history_ranges']}
    replacement = next(start for start in record['starts'] if start not in picked)
    query['selected_history_ranges'][0][0] = replacement
    with pytest.raises(ValueError,match='native SUM picker replay'):
        experiment.audit_selection(query,graph,record,profile)


@pytest.mark.parametrize('change', [dict(propagation_applied=True), dict(identity=False),
    dict(propagation_applied=None)])
def test_fps_audit_requires_explicit_no_graph_identity(change):
    query,graph,record,profile = audit_fixture(); graph.update(change)
    with pytest.raises(ValueError,match='explicitly report no graph propagation'):
        experiment.audit_selection(query,graph,record,profile)


@pytest.mark.parametrize('field,value', [('method','overlap_16_8'), ('representatives_per_block',8),
    ('block_tokens',16), ('selection','query_dependent_topk'),
    ('normalization','softmax before representative max')])
def test_fps_audit_rejects_wrong_scorer_or_aggregation_order(field,value):
    query,graph,record,profile = audit_fixture()
    query['query_key_fps_scoring'][field] = value
    with pytest.raises(ValueError):
        experiment.audit_selection(query,graph,record,profile)


def test_fps_audit_rejects_nonzero_alpha_even_if_graph_records_agree():
    query,graph,record,profile = audit_fixture()
    graph['alpha'] = record['alpha'] = profile['kvmem']['graph_alpha'] = .2
    with pytest.raises(ValueError,match='requires graph alpha=0'):
        experiment.audit_selection(query,graph,record,profile)


@pytest.mark.parametrize('change', ['count','budget'])
def test_fps_audit_requires_original_block_extents_and_picker_budget(change):
    query,graph,record,profile = audit_fixture()
    if change == 'count': query['selected_history_ranges'][0][1] -= 1
    else: record['budget'] -= 32
    with pytest.raises(ValueError,match='Native block geometry or selected budget'):
        experiment.audit_selection(query,graph,record,profile)


def reference_fixture(tmp_path):
    root = tmp_path/'completed-overlap'
    fixture = dict(rows=[dict(id='case-a',seed=42)], input_directory='/immutable-inputs',
        sampling=dict(max_tokens=8192), starts_in_thinking=True, upstream_path='/immutable-scorer',
        tokenizer_sha256='tokenizer-digest')
    profile = original_profile()
    arm = experiment.OVERLAP_ARM
    for name,value in {'fixture.json':fixture,'status.json':dict(status='completed'),
            'execution-audit.json':dict(status='passed'),'target-spans.json':{},
            arm+'/profile.json':profile, arm+'/scores.json':dict(cases=1),
            arm+'/cleanup.json':dict(remaining_owned_groups=[],gpu_processes=[])}.items():
        experiment.save(root/name,value)
    runtime_source = root/'runtime/src/scorer.py';runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text('immutable scorer\n')
    experiment.save(root/'runtime-proof.json',dict(files={'src/scorer.py':experiment.sha(runtime_source)}))
    experiment.save(root/'prepared-proof.json', {'fixture.json':experiment.sha(root/'fixture.json'),
        arm+'/profile.json':experiment.sha(root/arm/'profile.json')})
    case = root/arm/'cases/case-a'
    for name in experiment.CASE_FILES:
        experiment.save(case/name,dict(original=name, text='original raw model answer'))
    experiment.save(case/'hashes.json',{name:experiment.sha(case/name) for name in experiment.CASE_FILES})
    return root, fixture, experiment.profile_for_fps(profile)


def test_reference_proof_preserves_raw_answers_hashes_and_only_two_profile_changes(tmp_path):
    root,fixture,candidate = reference_fixture(tmp_path)
    proof,snapshots = experiment.reference_proof(root,fixture,(experiment.OVERLAP_ARM,),candidate=candidate)
    key = experiment.OVERLAP_ARM+'/cases/case-a/response.json'
    assert key in proof['files']
    assert experiment.OVERLAP_ARM+'/cases/case-a/hashes.json' in proof['files']
    saved = snapshots[experiment.OVERLAP_ARM]['case-a']
    assert saved['response'] == experiment.read(root/key)
    assert saved['sha256']['response.json'] == experiment.sha(root/key)
    assert set(proof['profile_difference']) == {'port','kvmem.query_score_mode'}
    experiment.save(root/key,dict(changed=True))
    with pytest.raises(ValueError,match='Immutable experiment evidence changed'):
        experiment.reference_proof(root,fixture,(experiment.OVERLAP_ARM,),candidate=candidate)


@pytest.mark.parametrize('kind', ['seed','failed','gpu','profile','source'])
def test_reference_proof_rejects_mismatched_or_unfinished_reference(tmp_path,kind):
    root,fixture,candidate = reference_fixture(tmp_path)
    if kind == 'seed': fixture['rows'][0]['seed'] = 43
    elif kind == 'failed': experiment.save(root/'execution-audit.json',dict(status='failed'))
    elif kind == 'gpu': experiment.save(root/experiment.OVERLAP_ARM/'cleanup.json',
        dict(remaining_owned_groups=[123],gpu_processes=[123]))
    elif kind == 'profile': candidate['max_num_seqs'] = 8
    elif kind == 'source': (root/'runtime/src/scorer.py').write_text('modified source')
    with pytest.raises(ValueError):
        experiment.reference_proof(root,fixture,(experiment.OVERLAP_ARM,),candidate=candidate)


def test_preparation_cannot_overwrite_an_existing_campaign(tmp_path):
    root = tmp_path/'campaign'; root.mkdir()
    sentinel = root/'response.json'; sentinel.write_bytes(b'original answer')
    with pytest.raises(ValueError,match='new FPS experiment root'):
        experiment.prepare(root,tmp_path/'unused-source',tmp_path/'unused-overlap')
    assert sentinel.read_bytes() == b'original answer'


def test_controller_cannot_execute_from_mutable_source_tree(tmp_path):
    with pytest.raises(ValueError,match='frozen'):
        experiment.run(tmp_path/'new-root')
    assert not (tmp_path/'new-root').exists()


def test_frozen_controller_rejects_repeated_attempt_before_gpu_access(tmp_path,monkeypatch):
    root = tmp_path/'campaign'
    frozen = root/'runtime/scripts/test_ruler_fps_answers.py'
    frozen.parent.mkdir(parents=True); frozen.write_text('frozen-controller-placeholder')
    monkeypatch.setattr(experiment,'__file__',str(frozen))
    experiment.save(root/'runtime-proof.json',dict(files={}))
    experiment.save(root/'prepared-proof.json',{})
    for name in ('reference-proof.json','overlap-reference-proof.json'):
        experiment.save(root/name,dict(root=str(root),files={}))
    (root/'attempt.json').write_text('original attempt\n')
    monkeypatch.setattr(experiment.lifecycle,'gpu_processes',
        lambda: pytest.fail('GPU access must not occur for an already attempted campaign'))
    with pytest.raises(FileExistsError):
        experiment.run(root)
    assert (root/'attempt.json').read_text() == 'original attempt\n'
