"""CPU-only independent group geometry, native budget replay and protocol checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

import analyze_block_graph as offline


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ROOT=Path(__file__).resolve().parents[1]
experiment=load(ROOT/'scripts/test_ruler_fps_halo_answers.py','halo_answer_controller')
fps_tests=load(ROOT/'tests/test_ruler_fps_answers.py','halo_previous_test_fixture')


def profile():
    previous=fps_tests.original_profile();previous['port']=18866
    previous['kvmem']['query_score_mode']='key_fps_4'
    return previous


@pytest.mark.parametrize('optimized',[False,True])
def test_halo_profile_only_changes_explicit_geometry_implementation_and_port(optimized):
    before=profile();copy=deepcopy(before)
    after=experiment.profile_for_halo(before,optimized=optimized)
    assert before==copy and after['kvmem']['query_score_mode']=='key_fps_4'
    assert after['kvmem']['query_fps_optimized'] is optimized
    changes=experiment.profile_difference(before,after)
    assert changes['port']==dict(before=18866,after=18867)
    assert changes['kvmem.query_fps_halo_blocks']==dict(before=0,after=1)
    assert ('kvmem.query_fps_optimized' in changes) is optimized
    after['kvmem']['gen_budget']=65536
    with pytest.raises(ValueError,match='differs beyond'):
        experiment.profile_difference(before,after)


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_implementation_flag_must_be_explicit_boolean(value):
    with pytest.raises(ValueError,match='explicitly'):
        experiment.profile_for_halo(profile(),optimized=value)


def test_independent_groups_preserve_logical_holes_partial_tail_and_all_anchors():
    inputs=dict(starts=[0,32,96,128],counts=[32,7,32,9],raw_scores=[0.,3.,0.,2.],graph={})
    groups=experiment.expected_groups(inputs)
    assert groups['anchor_starts']==[0,32,96,128]
    assert groups['groups']==[[0,39],[0,39],[96,137],[96,137]]
    assert groups['group_members']==[[0,32],[0,32],[96,128],[96,128]]
    assert groups['group_scores']==inputs['raw_scores']


def test_independent_groups_clip_at_message_edges_and_unknown_or_crossing_blocks():
    inputs=dict(starts=[0,32,64,96,128],counts=[32]*5,raw_scores=[1.]*5,
        graph=dict(message_spans=[[0,64,'user'],[64,110,'assistant'],[130,160,'tool']]))
    groups=experiment.expected_groups(inputs)
    assert groups['group_members']==[[0,32],[0,32],[64],[96],[128]]
    assert groups['groups']==[[0,64],[0,64],[64,96],[96,128],[128,160]]


def audit_fixture():
    query,graph,inputs,options=fps_tests.audit_fixture()
    query['query_key_fps_scoring']['optimized']=False
    options['kvmem'].update(query_score_mode='key_fps_4',query_fps_halo_blocks=1,query_fps_optimized=False)
    expected=experiment.expected_groups(inputs)
    cfg,snapshot=offline._validate_record(inputs)
    memory,handles=offline._native_history(cfg,inputs['starts'],inputs['counts'])
    try:
        mandatory=tuple(handles[start].slot for start in inputs['mandatory_starts'])
        baseline=memory.preview_selection({handles[s].slot:v for s,v in zip(inputs['starts'],inputs['raw_scores'])},
            mandatory,action=False,prefill_scores=None,max_tokens=inputs['budget'])
        picked=memory.preview_group_selection(expected['groups'],expected['group_scores'],mandatory,max_tokens=inputs['budget'])
        fixed=memory._store.fixed_selection_blocks(list(mandatory),pressure=False,budget=inputs['budget'])
        convert=lambda slots:{start for start,handle in handles.items() if handle.slot in slots}
        baseline,picked,fixed=map(convert,(baseline,picked,fixed))
    finally:memory.close()
    counts=dict(zip(inputs['starts'],inputs['counts']))
    stats=dict(expected,selected_tokens=sum(counts[s] for s in picked),selected_starts=sorted(picked),
        baseline_selected_starts=sorted(baseline),added_starts=sorted(picked-baseline),displaced_starts=sorted(baseline-picked),
        covered_anchor_starts=[s for s,m in zip(expected['anchor_starts'],expected['group_members']) if set(m)<=picked],
        selected_blocks=len(picked),charged_tokens=32*len(picked),fixed_starts=sorted(fixed),budget=inputs['budget'],
        unused_block_slots=inputs['budget']//32-len(picked),all_fit=len(inputs['starts'])<=inputs['budget']//32)
    inputs['query_group_selection']=stats
    query['query_group_selection']={key:value for key,value in stats.items()
        if key not in ('groups','group_scores','group_members','anchor_starts')}
    query['selected_history_ranges']=[[start,counts[start]] for start in sorted(picked)]
    return query,graph,inputs,options


def test_native_atomic_group_replay_and_baseline_displacement_are_explicit():
    query,graph,inputs,profile=audit_fixture()
    audit=experiment.audit_selection(query,graph,inputs,profile)
    assert audit['native_group_picker_replay_equal'] and audit['independent_group_geometry_verified']
    assert audit['before_picker_expansion'] and audit['post_picker_expansion'] is False
    assert audit['charged_block_tokens']<=profile['kvmem']['select_budget']
    assert audit['groups']['displaced_starts']
    assert audit['groups']['covered_anchor_starts']==inputs['query_group_selection']['covered_anchor_starts']
    assert audit['propagation_applied'] is False


@pytest.mark.parametrize('change',['group_score','group_span','group_members','post_expand','baseline','covered'])
def test_replay_rejects_changed_group_geometry_scores_or_accounting(change):
    query,graph,inputs,profile=audit_fixture()
    full=inputs['query_group_selection']
    if change=='group_score':full['group_scores'][0]+=1
    elif change=='group_span':full['groups'][0][1]+=32
    elif change=='group_members':full['group_members'][0].append(96)
    elif change=='post_expand':full['post_picker_expansion']=True
    elif change=='baseline':full['baseline_selected_starts']=[]
    else:full['covered_anchor_starts']=[]
    # Refresh summary too: the independent oracle/native replay must still reject.
    query['query_group_selection']={key:value for key,value in full.items()
        if key not in ('groups','group_scores','group_members','anchor_starts')}
    with pytest.raises(ValueError):experiment.audit_selection(query,graph,inputs,profile)


def test_old_block_picker_result_is_not_accepted_as_atomic_group_selection():
    query,graph,inputs,profile=audit_fixture()
    query['selected_history_ranges']=[[start,32] for start in inputs['query_group_selection']['baseline_selected_starts']]
    with pytest.raises(ValueError,match='native atomic group picker'):
        experiment.audit_selection(query,graph,inputs,profile)


def test_requested_fps_optimization_must_match_actual_execution_statistics():
    query,graph,inputs,profile=audit_fixture()
    profile['kvmem']['query_fps_optimized']=True
    with pytest.raises(ValueError,match='Actual FPS implementation differs'):
        experiment.audit_selection(query,graph,inputs,profile)


def test_preparation_and_mutable_execution_never_overwrite_existing_data(tmp_path):
    root=tmp_path/'existing';root.mkdir();(root/'answer').write_text('keep')
    with pytest.raises(ValueError,match='new halo experiment root'):
        experiment.prepare(root,tmp_path/'unused',optimized=False)
    with pytest.raises(ValueError,match='frozen'):
        experiment.run(tmp_path/'new')
    assert (root/'answer').read_text()=='keep' and not (tmp_path/'new').exists()
