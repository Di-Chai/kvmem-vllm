"""Observed physical shapes, refresh barriers and matched native schedules."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_active_prefill as benchmark
from active_prefill_shapes import observed_native_plan, apply_observed_native_cap, compare_output_tokens


def fixture_geometry():
    initial = [2048]+[4096]*11+[1984]
    steady = [32]+[4096]*3+[2048,1952]
    chunks = initial+steady+steady+[32,736,22,7]
    boundaries = [49088,65408,81728,82496]
    refresh = [dict(history_begin=n,selected_tokens=32768) for n in boundaries]
    geometry = benchmark.reconstruct_geometry([dict(prefill_tokens=n) for n in chunks],refresh,sum(chunks))
    full32_chunks = [2048]+[4096]*7+[2048]
    full32 = benchmark.reconstruct_geometry([dict(prefill_tokens=n) for n in full32_chunks],[],32768)
    return geometry,refresh,full32


def test_families_come_from_observed_chunks_and_keep_refresh_boundary_guards():
    geometry,refresh,full32 = fixture_geometry()
    plan = observed_native_plan(geometry,refresh,full32)
    assert plan['max_chunk'] == 4096
    assert plan['full32_chunks'] == [2048]+[4096]*7+[2048]
    assert plan['logical2k_control_chunks'] == [2048]*16
    assert len(plan['families']) == 4
    initial,steady,tail,query = plan['families'].values()
    assert initial['chunks'] == [2048]+[4096]*11+[1984,8]
    assert steady['chunks'] == [2048]*16+[32]+[4096]*3+[2048,1952,8]
    assert steady['target_begin_index'] == 16 and len(steady['source_spans']) == 2
    assert tail['chunks'] == [2048]*16+[32,736,8]
    assert query['chunks'] == [2048]*16+[22,7]
    assert query['target_shapes'][-1] == [7,32797,True]
    assert plan['steady_patterns'] == [[[32,32800],[4096,36896],[4096,40992],
                                      [4096,45088],[2048,47136],[1952,49088]]]
    covered = set()
    for family in plan['families'].values():
        end = 0
        for i,count in enumerate(family['chunks']):
            end += count
            covered.add((count,end,i==len(family['chunks'])-1))
    assert all((s['query_tokens'],s['active_seq_len'],s['prefill_terminal']) in covered for s in geometry)


def test_a_different_recorded_chunk_pattern_is_not_replaced_by_the_expected_4k_guess():
    geometry,refresh,full32 = fixture_geometry()
    chunks = [r['query_tokens'] for r in geometry]
    first = next(i for i,r in enumerate(geometry) if r['logical_seq_len'] == 49120)
    chunks[first+1:first+4] = [2048]*6
    changed = benchmark.reconstruct_geometry([dict(prefill_tokens=n) for n in chunks],refresh,sum(chunks))
    plan = observed_native_plan(changed,refresh,full32)
    assert len(plan['steady_patterns']) == 2
    assert [row[0] for row in plan['steady_patterns'][0]] == [32]+[2048]*7+[1952]
    assert [row[0] for row in plan['steady_patterns'][1]] == [32]+[4096]*3+[2048,1952]


@pytest.mark.parametrize('fault', ['duplicate_refresh','crossed_refresh','wrong_offset','missing_terminal',
    'early_terminal','missing_prefill','too_large','missing_refresh','extra_refresh','wrong_full32'])
def test_ambiguous_physical_evidence_is_rejected(fault):
    geometry,refresh,full32 = fixture_geometry()
    if fault == 'duplicate_refresh': refresh.append(refresh[0])
    elif fault == 'crossed_refresh': refresh[0] = dict(history_begin=49100,selected_tokens=32768)
    elif fault == 'wrong_offset': geometry[14]['offset'] += 32
    elif fault == 'missing_terminal': geometry[-1]['prefill_terminal'] = False
    elif fault == 'early_terminal': geometry[2]['prefill_terminal'] = True
    elif fault == 'missing_prefill': geometry[2]['prefill'] = False
    elif fault == 'too_large': geometry[2]['query_tokens'] = 16385
    elif fault == 'missing_refresh': refresh.pop(0)
    elif fault == 'extra_refresh': refresh.append(dict(history_begin=999999,selected_tokens=32768))
    else: full32[-1]['logical_seq_len'] -= 1
    with pytest.raises(ValueError): observed_native_plan(geometry,refresh,full32)


@pytest.mark.parametrize('maximum',[4,7,8])
def test_larger_actual_groups_keep_complete_original_two_k_output_control(maximum):
    def grouped(count):
        full,tail = divmod(count,maximum)
        return [2048*maximum]*full+([2048*tail] if tail else [])
    initial = [2048]+grouped(22)+[1984]
    steady = [32]+grouped(7)+[1952]
    chunks = initial+steady+[32,736,22,7]
    refresh = [dict(history_begin=n,selected_tokens=32768) for n in (49088,65408,66176)]
    geometry = benchmark.reconstruct_geometry([dict(prefill_tokens=n) for n in chunks],refresh,sum(chunks))
    full32_chunks = [2048]+grouped(14)+[2016,25,7]
    full32 = benchmark.reconstruct_geometry([dict(prefill_tokens=n) for n in full32_chunks],[],32768)
    plan = observed_native_plan(geometry,refresh,full32)
    assert plan['max_chunk'] == maximum*2048
    assert plan['logical2k_control_chunks'] == [2048]*15+[2016,25,7]
    assert [row[0] for row in plan['steady_patterns'][0]] == steady
    argv = ['serve','--long-prefill-token-threshold','2048','--max-num-batched-tokens','16424']
    assert apply_observed_native_cap(argv,plan)[2] == str(maximum*2048)


def test_refresh_inside_expected_steady_interval_prevents_false_continuous_phase():
    # Same offset is still a semantic barrier, even if shapes look continuous.
    geometry,refresh,full32 = fixture_geometry()
    for h in (53216,69536):
        row = next(row for row in geometry if row['logical_seq_len'] == h)
        refresh.append(dict(history_begin=h,selected_tokens=row['active_seq_len']))
    with pytest.raises(ValueError,match='No complete observed'):
        observed_native_plan(geometry,refresh,full32)


def test_continuous_matching_uses_six_actual_steps_including_interior_gaps():
    geometry,refresh,full32 = fixture_geometry()
    plan = observed_native_plan(geometry,refresh,full32)
    pattern, = plan['steady_patterns']
    steps = [dict(sequence=i+1,call_index=i+1,previous_sequence=i,previous_call_index=i,
        previous_end_to_model_ms=999 if i==0 else 2,model_ms=10,decode_requests=0,
        batch_geometry=[dict(request_id='r',query_tokens=q,active_seq_len=k,logical_seq_len=k+16320,
                             offset=16320,prefill=True,prefill_terminal=False)])
        for i,(q,k) in enumerate(pattern)]
    result = benchmark.continuous_steady_phases(steps,plan['steady_patterns'])
    assert result['matched_phases'] == 1
    phase, = result['phases']
    assert phase['new_input_tokens'] == 16320 and phase['steps'] == 6
    assert phase['model_start_to_last_end_seconds'] == pytest.approx(.07)
    assert benchmark.continuous_steady_phases(steps)['matched_phases'] == 0
    wrong = deepcopy(steps)
    wrong[2]['previous_call_index'] += 1
    with pytest.raises(ValueError,match='unlinked'):
        benchmark.continuous_steady_phases(wrong,plan['steady_patterns'])
    with pytest.raises(ValueError,match='refresh boundary'):
        benchmark.continuous_steady_phases(steps,plan['steady_patterns'],
            [dict(history_begin=steps[1]['batch_geometry'][0]['logical_seq_len'])])


def test_only_observed_diagnostic_cap_changes_and_original_argv_is_untouched():
    plan = observed_native_plan(*fixture_geometry())
    original = ['serve','--long-prefill-token-threshold','2048','--max-num-batched-tokens','16424','--kv-cache-dtype','fp8']
    result = apply_observed_native_cap(original,plan)
    assert result == ['serve','--long-prefill-token-threshold','4096','--max-num-batched-tokens','16424','--kv-cache-dtype','fp8']
    assert original[2] == '2048'
    for bad in (['--long-prefill-token-threshold','4096','--max-num-batched-tokens','16424'],
                ['--long-prefill-token-threshold','2048','--max-num-batched-tokens','2048']):
        with pytest.raises(ValueError): apply_observed_native_cap(bad,plan)


def recorded_reference(tmp_path):
    geometry,refresh,full32 = fixture_geometry()
    root = tmp_path/'reference'
    root.mkdir()
    benchmark.save(root/'profile.json',dict(kvmem=dict(action_native_scores='reuse',
        prefill_publication_pipeline=True,prefill_physical_coalescing=True)))
    benchmark.save(root/'status.json',dict(status='completed'))
    benchmark.save(root/'runtime-proof.json',dict(files={}))
    definitions,steps = [],[]
    for name,shapes,events in [('32k-repeat-1',full32,[]),('256k-repeat-1',geometry,refresh)]:
        target = root/'requests'/name
        target.mkdir(parents=True)
        prompt = target/'input.i32'
        prompt.write_bytes(b'\0'*4*shapes[-1]['logical_seq_len'])
        benchmark.save(target/'case.json',dict(input_path=str(prompt),sampling=dict(max_tokens=128,temperature=0),
            row=dict(served_input_tokens=shapes[-1]['logical_seq_len'],served_sha256=benchmark.sha(prompt))))
        benchmark.save(target/'evidence.json',dict(refresh=events))
        benchmark.save(target/'response.json',dict(output_ids=[7,8,9]))
        definitions.append(dict(id=name,case_sha256=benchmark.sha(target/'case.json')))
        steps.extend(dict(prefill_tokens=row['query_tokens'],batch_geometry=[dict(request_id=name,**row)]) for row in shapes)
        steps.append(dict(prefill_tokens=0))
    benchmark.save(root/'suite.json',dict(cases=definitions))
    capture = root/'run/server/capture'
    capture.mkdir(parents=True)
    (capture/'model_step_latency.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in steps))
    benchmark.save(root/'run/server/launch.json',dict(vllm_argv=[]))
    return root


@pytest.mark.parametrize('fault',[None,'unfinished','missing_geometry','changed_geometry'])
def test_reference_adapter_requires_completed_actual_rows_and_records_source_proof(tmp_path,fault):
    root = recorded_reference(tmp_path)
    if fault == 'unfinished': benchmark.save(root/'status.json',dict(status='running'))
    elif fault:
        path = root/'run/server/capture/model_step_latency.jsonl'
        rows = benchmark.rows(path)
        if fault == 'missing_geometry': del rows[0]['batch_geometry']
        else: rows[0]['batch_geometry'][0]['active_seq_len'] += 1
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if fault:
        with pytest.raises(ValueError): benchmark.reference_inputs(tmp_path,root)
    else:
        result = benchmark.reference_inputs(tmp_path,root)
        plan = result['native_shape_plan']
        assert plan['source_reference'] == str(root)
        assert plan['source_evidence_sha256']['run/server/capture/model_step_latency.jsonl'] == benchmark.sha(
            root/'run/server/capture/model_step_latency.jsonl')


def prepare_fixture(tmp_path,monkeypatch):
    reference = recorded_reference(tmp_path)
    repo = tmp_path/'repo'
    (repo/'scripts').mkdir(parents=True)
    source_scripts = Path(benchmark.__file__).parent
    names = ('benchmark_active_prefill.py','native_prefill_shape_probe.py',
             'active_prefill_shapes.py','benchmark_prefill_publication_pipeline.py')
    for name in names: shutil.copy2(source_scripts/name,repo/'scripts'/name)
    monkeypatch.setattr(benchmark,'runtime_origins',lambda repo,ref,names:
        ({name:repo/name for name in names},[]))
    monkeypatch.setattr(benchmark.subprocess,'check_output',lambda *a,**k:'test-head\n')
    root = tmp_path/'prepared'
    benchmark.prepare(root,repo,reference)
    return root,reference


def test_prepare_freezes_dynamic_families_full32_schedule_and_helpers(tmp_path,monkeypatch):
    root,reference = prepare_fixture(tmp_path,monkeypatch)
    benchmark.assert_frozen(root,'native')
    plan = benchmark.read(root/'native-shape-plan.json')
    suite = benchmark.read(root/'native/suite.json')
    assert suite['native_prefill_chunk'] == 4096
    for item in suite['cases']:
        if item['family'] in ('full32k','logical2k-control'):
            assert item['chunks'] == plan['full32_chunks' if item['family']=='full32k' else 'logical2k_control_chunks']
            case = benchmark.read(root/'native/requests'/item['id']/'case.json')
            assert case['input_path'] == str(reference/'requests/32k-repeat-1/input.i32')
            assert case['sampling'] == dict(max_tokens=128,temperature=0)
        elif item['family'] != 'flush':
            assert item['chunks'] == plan['families'][item['family']]['chunks']
    controls = [item for item in suite['cases'] if item['family']=='logical2k-control']
    assert len(controls) == 2 and all(item['instrumented'] and not item['warmup'] for item in controls)
    for name in ('active_prefill_shapes.py','native-shape-plan.json','native/suite.json','reference-geometry.json',
                 'output-control-reference.json'):
        path = root/name
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        with pytest.raises(ValueError,match='changed'): benchmark.assert_frozen(root,'native')
        path.write_bytes(original)


@pytest.mark.parametrize('left,right,equal,prefix', [([1,2],[1,2],True,2),([1,2],[1,3],False,1),
    ([1,2],[1,2,3],False,2),([1,2],[7],False,0)])
def test_output_shape_controls_compare_exact_ids_without_accuracy_inference(left,right,equal,prefix):
    result = compare_output_tokens(left,right)
    assert result['equal'] is equal and result['common_prefix_tokens'] == prefix
    assert result['first_difference'] == (None if equal else prefix)


def test_analyzer_keeps_slow_logical2k_controls_outside_all_performance_pools(tmp_path,monkeypatch):
    root,_ = prepare_fixture(tmp_path,monkeypatch)
    suite = benchmark.read(root/'native/suite.json')['cases']
    timings = []
    sequence = 0
    for item in suite:
        control = item['family']=='logical2k-control'
        ident = item['id']
        response = dict(output_ids=[1,3] if control else [1,2])
        benchmark.save(root/'native/requests'/ident/'response.json',response)
        benchmark.save(root/'native/requests'/ident/'result.json',dict(input_tokens=item['input_tokens'],
            ttft_seconds=100000 if control else 2,output_sha256=benchmark.digest(response['output_ids'])))
        if not item['instrumented']:
            continue
        end = 0
        for index,count in enumerate(item['chunks']):
            sequence += 1
            end += count
            timings.append(dict(sequence=sequence,call_index=sequence,previous_sequence=sequence-1,
                previous_call_index=sequence-1,previous_end_to_model_ms=.1,
                model_ms=100000 if control else 1,step_ms=1,host_submit_ms=1,decode_requests=0,
                batch_geometry=[dict(request_id='cmpl-'+item['request_id']+'-uuid',query_tokens=count,
                    active_seq_len=end,logical_seq_len=end,offset=0,prefill=True,
                    prefill_terminal=index==len(item['chunks'])-1)]))
    probe = root/'native/probe'
    probe.mkdir()
    (probe/'model_step_latency.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in timings))
    benchmark.analyze(root)
    result = benchmark.read(root/'comparison.json')
    assert all(row['native_model_ms']==1 for row in result['shapes'])
    assert result['full32k_native']['timed']['pooled_prefill_tokens_per_second'] == 16384
    assert len(result['full32k_native']['timed']['samples']) == 3
    assert len(result['full32k_native']['timing_disabled']['samples']) == 3
    assert len(result['continuous_steady_prefill_native']) == 3
    diag = result['physical_shape_output_diagnostic']
    assert sum(row['label']=='native-logical2k-control' for row in diag['samples']) == 2
    pair = next(row for row in diag['pairs'] if row['left'].startswith('native-full32k/') and
                row['right'].startswith('native-logical2k-control/'))
    assert pair['equal'] is False and pair['first_difference'] == 1
