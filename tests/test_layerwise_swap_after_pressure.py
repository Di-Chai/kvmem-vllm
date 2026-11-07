"""Delayed-arrival qualification must use confirmed work and unchanged inputs."""
import importlib.util
import json
from pathlib import Path

import pytest


def driver():
    path = Path(__file__).resolve().parents[1]/'scripts/qualify_layerwise_swap_after_pressure.py'
    spec = importlib.util.spec_from_file_location('swap_after_pressure_driver',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pressure(*, distinct=True):
    return dict(event='pressure_selected',actual_method='accumulated_attention_per_layer',
                fallback_reason=None,accepted_history=52992,
                layers=[dict(layer=i,selected_starts=[0,32+i*32 if distinct else 32]) for i in range(16)])


def test_release_needs_confirmed_work_and_actual_heterogeneous_pressure(tmp_path):
    d = driver()
    work = tmp_path/'public_work.json'
    events = tmp_path/'layerwise_query.jsonl'
    work.write_text(json.dumps({'submitted':{'public_prefill_tokens':200000},
                                'completed':{'public_prefill_tokens':90000}}))
    events.write_text(json.dumps(pressure())+'\n')
    assert d.arrival_evidence(tmp_path,98304) is None
    work.write_text(json.dumps({'completed':{'public_prefill_tokens':98304}}))
    events.write_text(json.dumps(pressure(distinct=False))+'\n')
    assert d.arrival_evidence(tmp_path,98304) is None
    events.write_text(json.dumps(pressure())+'\n{"event":')
    evidence = d.arrival_evidence(tmp_path,98304)
    assert evidence['public_work']['completed']['public_prefill_tokens'] == 98304
    assert d.heterogeneous(evidence['pressure'])


def test_layerwise_evidence_rejects_fallback_and_missing_layers():
    d = driver()
    event = pressure()
    assert d.heterogeneous(event)
    event['fallback_reason'] = 'fallback'
    assert not d.heterogeneous(event)
    event['fallback_reason'] = None
    event['layers'].pop()
    assert not d.heterogeneous(event)


def test_prepare_reuses_frozen_runtime_and_first_answers_read_only(tmp_path):
    d = driver()
    reference = tmp_path/'previous'
    original = reference/'incremental_tight'
    runtime = original/'runtime-v1'
    runtime.mkdir(parents=True)
    (runtime/'source.py').write_text('# immutable runtime\n')
    baseline = reference/'static_serial'
    d.save(baseline/'status.json',{'status':'completed'})
    d.save(reference/'serial-comparison.json',{'passed':True})
    rows,bindings = [],[]
    for i in range(2):
        case = f'case{i}'
        source = reference/(case+'.gz')
        source.write_bytes(bytes([i]))
        binding = reference/(case+'.json')
        binding.write_text('{"seed":1}')
        rows.append(dict(id=case,source=str(source),source_cache_sha256=d.sha(source)))
        bindings.append(dict(id=case,request=str(binding),sha256=d.sha(binding)))
        for name in ('request.json','response.json','result.json','events.json'):
            d.save(baseline/'run/answers/cases'/case/name,{'first_answer':True})
    fixture = dict(cases=2,rows=rows,bindings=bindings,case_ids=['case0','case1'])
    profile = dict(verify_request_swap=True,kvmem=dict(select_budget=36864,gen_budget=16384,
                                                      request_swap_layerwise=True))
    d.save(original/'profile.json',profile)
    d.save(original/'fixture.json',fixture)
    d.save(original/'runtime-proof.json',d.tree(runtime))
    before = d.tree(reference)
    root = tmp_path/'followup'
    d.prepare(root,reference)
    plan = d.verify(root)
    assert plan['workers'] == 2 and plan['threshold'] == 98304
    assert d.read(root/'profile.json') == profile
    assert d.tree(reference) == before
    assert not (root/'run').exists()
    assert d.tree(root/'runtime-v1') == d.tree(runtime)
    with pytest.raises(ValueError,match='Preserve'):
        d.prepare(root,reference)
    (baseline/'run/answers/cases/case0/response.json').write_text('{}')
    with pytest.raises(ValueError,match='reference changed'):
        d.verify(root)
