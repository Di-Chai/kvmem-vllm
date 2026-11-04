from pathlib import Path
import json

import pytest

import recall_comparison_protocol as p
from layerwise_efficiency_gate import create,observed_prefill,verify


def measurement(root):
    capture=root/'formal/sequential/server-attempts/0001/server/capture';capture.mkdir(parents=True)
    p.save(root/'fixture.json',dict(rows=[dict(id=str(i)) for i in range(8)]))
    p.save(root/'status.json',dict(status='completed'))
    p.save(root/'concurrency-performance.json',dict(cases=8,workers=8,started_monotonic=100.))
    rows=[dict(selection='query',history_begin=1000,refresh_started_at=110.+i) for i in range(8)]
    (capture/'bounded_refresh.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return capture


def test_prefill_gate_uses_recorded_boundaries_and_whole_concurrent_envelope(tmp_path):
    capture=measurement(tmp_path)
    actual,source=observed_prefill(tmp_path)
    assert source==capture
    assert actual==dict(tokens=8000,seconds=17.,tokens_per_second=8000/17)


@pytest.mark.parametrize('fault',['unfinished','owners','cases','retry','missing_query','clock'])
def test_incomplete_or_incomparable_performance_cannot_pass(fault,tmp_path):
    capture=measurement(tmp_path)
    if fault=='unfinished':p.save(tmp_path/'status.json',dict(status='running'))
    elif fault in ('owners','cases','clock'):
        wall=p.read(tmp_path/'concurrency-performance.json')
        wall[{'owners':'workers','cases':'cases','clock':'started_monotonic'}[fault]]={'owners':1,'cases':7,'clock':200.}[fault]
        p.save(tmp_path/'concurrency-performance.json',wall)
    elif fault=='retry':(tmp_path/'formal/sequential/server-attempts/0002').mkdir()
    else:(capture/'bounded_refresh.jsonl').write_text('')
    with pytest.raises(ValueError):observed_prefill(tmp_path)


def test_efficiency_acceptance_is_bound_to_runtime_and_immutable_evidence(tmp_path):
    source=tmp_path/'accepted.json';p.save(source,dict(goal_met=True))
    gate=dict(efficiency_accepted=True,pressure_runtime='/qualified',files={str(source):p.sha(source)})
    verify(gate,Path('/qualified'))
    with pytest.raises(ValueError,match='another runtime'):verify(gate,Path('/unqualified'))
    p.save(source,dict(goal_met=False))
    with pytest.raises(ValueError,match='changed'):verify(gate,Path('/qualified'))


@pytest.mark.parametrize('fault',[None,'optimistic_report','below_target','external_gpu','different_runtime',
    'authorized_below_target','authorized_other_runtime','authorized_bad_measurement'])
def test_full_gate_recomputes_target_instead_of_trusting_a_success_flag(tmp_path,monkeypatch,fault):
    import run_layerwise_query_efficiency
    import run_layerwise_query_diagnostic
    from run_layerwise_matched_performance import controls,MODES
    from test_layerwise_matched_performance import base
    # Arithmetic/evidence-binding test; model byte qualification and runtime
    # manifests are independently tested by their original verifier suites.
    monkeypatch.setattr(run_layerwise_query_efficiency,'source_audit',lambda root:[{}]*8)
    monkeypatch.setattr(run_layerwise_query_diagnostic,'verify',lambda root:None)
    source=tmp_path/'qualified';source.mkdir();fixture,profile=base()
    fixture['rows']=[dict(id=str(i)) for i in range(8)]
    for name,value in [('fixture.json',fixture),('profile.json',profile),('equivalence.json',{'passed':True}),
                       ('native-reuse-audit.json',{'passed':True}),('runtime-proof.json',{}),('prepared.json',{})]:
        p.save(source/name,value)
    root=tmp_path/'performance';root.mkdir();(root/'launch.py').write_text('# frozen launcher\n')
    stages=[];reports={}
    for i,mode in enumerate(MODES):
        arm=root/mode;capture=measurement(arm)
        f,config=controls(fixture,profile,mode,i)
        p.save(arm/'fixture.json',f);p.save(arm/'profile.json',config)
        p.save(arm/'prepared.json',{});p.save(arm/'runtime-proof.json',{})
        p.save(capture/'native_attention_stats.json',{})
        (capture.parent.parent/'gpu.jsonl').write_text('')
        if mode=='layerwise_optimized':
            last=140. if fault=='below_target' or (fault or '').startswith('authorized_') else 116.
            rows=[dict(selection='query',history_begin=1000,refresh_started_at=last-7+j) for j in range(8)]
            (capture/'bounded_refresh.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        observed,_=observed_prefill(arm)
        reports[mode]=dict(history_prefill_envelope=observed,
            gpu=dict(external_pids=[42] if fault=='external_gpu' else [],attribution_coverage_fraction=1.))
        stages.append(dict(name=mode,root=str(arm),prepared_sha256=p.sha(arm/'prepared.json')))
    ratio=reports['layerwise_optimized']['history_prefill_envelope']['tokens_per_second']/reports['shared_layerwise_query']['history_prefill_envelope']['tokens_per_second']
    if fault in ('optimistic_report','authorized_bad_measurement'):
        reports['layerwise_optimized']['history_prefill_envelope']['tokens_per_second']*=1.2
    p.save(root/'plan.json',dict(source=str(source if fault!='different_runtime' else tmp_path/'other'),
        source_equivalence_sha256=p.sha(source/'equivalence.json'),launch_sha256=p.sha(root/'launch.py'),
        stages=stages,target_layerwise_to_shared_throughput_ratio=.9))
    p.save(root/'status.json',dict(status='completed'))
    authorized=(fault or '').startswith('authorized_')
    p.save(root/'performance-comparison.json',dict(reports=reports,goal_met=not authorized,
        ratios={'shared_layerwise_query':{'history_prefill_envelope':ratio}}))
    authorization=None
    if authorized:
        authorization=tmp_path/'user-selection.json'
        p.save(authorization,dict(policy='user_selected_qualified_runtime',
            user_request='Run the current fastest qualified version on original Memory benchmarks.',
            pressure_runtime=str(source if fault!='authorized_other_runtime' else tmp_path/'other'),
            performance=str(root),performance_sha256=p.sha(root/'performance-comparison.json'),
            preserve_original_memory_controls=True))
    if fault and fault!='authorized_below_target':
        with pytest.raises(ValueError):create(source,root,authorization)
    else:
        gate=create(source,root,authorization)
        assert gate['efficiency_accepted'] is (not authorized)
        assert gate['memory_run_authorized']
        assert gate['measured_prefill_ratio']==pytest.approx(17/40 if authorized else 17/16)
        verify(gate,source)
        if authorized:
            selection=p.read(authorization);selection['pressure_runtime']='/different'
            p.save(authorization,selection)
            with pytest.raises(ValueError,match='changed'):verify(gate,source)


def test_false_efficiency_flag_without_user_selection_still_rejects():
    gate=dict(efficiency_accepted=False,pressure_runtime='/qualified',files={},
              memory_run_authorized=True,qualified_cases=8)
    with pytest.raises(ValueError,match='explicit user selection'):verify(gate,Path('/qualified'))
