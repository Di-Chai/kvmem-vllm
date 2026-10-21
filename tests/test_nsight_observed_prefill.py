"""CPU-only provenance/shape qualification for the explicit observed-chunks mode."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import nsight_prefill_shapes as shapes
import profile_prefill_nsight as profile


CHUNKS = [2048, 14336, 14336, 2016, 25, 7]


def geometry(chunks=CHUNKS, response_id='cmpl-source'):
    end, result = 0, []
    for count in chunks:
        end += count
        result.append(dict(decode_requests=0, atomic_prefill_tokens=0,
            batch_geometry=[dict(request_id=response_id+'-0', query_tokens=count,
                logical_seq_len=end, active_seq_len=end, offset=0, prefill=True,
                prefill_terminal=end == 32768)]))
    return result


def write_steps(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def source_fixture(tmp_path):
    source = tmp_path/'source'
    request = source/'requests/32k-repeat-2'
    request.mkdir(parents=True)
    runtime = source/'runtime'
    runtime.mkdir()
    (runtime/'sample.py').write_text('frozen = True\n')
    profile.save(source/'runtime-proof.json', dict(files={'sample.py':profile.sha(runtime/'sample.py')}))
    input_path = tmp_path/'tokens.i32'
    input_path.write_bytes(bytes(32768*4))
    case = dict(input_path=str(input_path), sampling={'max_tokens':128, 'temperature':0},
        row=dict(served_input_tokens=32768, served_sha256=profile.sha(input_path)))
    profile.save(request/'case.json', case)
    profile.save(source/'profile.json', dict(prefill_chunk_tokens=2048,
        max_num_batched_tokens=16424, kvmem={'prefill_physical_coalescing':True,
                                          'prefill_physical_max_chunks':7}))
    profile.save(source/'suite.json', dict(serial=True, prefix_caching=False,
        cases=[dict(id='32k-repeat-2', warmup=False, case_sha256=profile.sha(request/'case.json'))]))
    profile.save(source/'manifest.json', dict(profile_sha256=profile.sha(source/'profile.json'),
                                            suite_sha256=profile.sha(source/'suite.json')))
    result = dict(status='completed', input_tokens=32768, input_sha256=profile.sha(input_path),
                  completion_tokens=3, pressure_count=0, query_count=0)
    profile.save(request/'result.json', result)
    profile.save(request/'response.json', dict(response_id='cmpl-source', output_ids=[7,8,9],
        usage=dict(prompt_tokens=32768, completion_tokens=3)))
    profile.save(request/'evidence.json', {'refresh':[]})
    profile.save(source/'status.json', dict(status='completed', protocol_passed=True,
        results=[dict(id='32k-repeat-2', result=result)]))
    write_steps(source/'run/server/capture/model_step_latency.jsonl', geometry())
    args = Namespace(source=source, case=request/'case.json', root=tmp_path/'probe', mode='native',
        port=18866, delay_iterations=2, max_iterations=2, startup_timeout=1200, request_timeout=300,
        nsys='/usr/local/bin/nsys', observed_chunks=True)
    return source, request, args


def test_actual_complete_chunks_and_installed_profiler_delay(tmp_path):
    source, request, _ = source_fixture(tmp_path)
    plan = shapes.observed_shape_plan(source, request/'case.json', delay=2, limit=2)
    assert plan['chunks'] == CHUNKS and sum(plan['chunks']) == 32768
    assert plan['max_chunk'] == 14336 and plan['capture_begin_index'] == 1
    assert [x['query_tokens'] for x in plan['capture_geometry']] == [14336,14336]
    assert [x['active_seq_len'] for x in plan['capture_geometry']] == [16384,30720]
    begin, selected = shapes.capture_geometry(plan['geometry'], 1, 2)
    assert begin == 0 and [x['query_tokens'] for x in selected] == [2048,14336]
    assert shapes.capture_geometry(plan['geometry'], 0, 2) == (begin, selected)
    assert 'run/server/capture/model_step_latency.jsonl' in plan['source_evidence_sha256']


def test_older_frozen_kvmem_timer_without_terminal_field_uses_exact_input_boundary():
    rows = geometry()
    for step in rows:
        del step['batch_geometry'][0]['prefill_terminal']
    actual = shapes.complete_geometry(rows, 'cmpl-source', 32768)
    assert actual == shapes.complete_geometry(geometry(), 'cmpl-source', 32768)
    with pytest.raises(ValueError, match='complete input'):
        shapes.complete_geometry(rows[:-1], 'cmpl-source', 32768)


@pytest.mark.parametrize('mode', ['native','kvmem'])
def test_prepare_freezes_complete_same_input_actual_shapes_and_source(tmp_path, mode):
    source, request, args = source_fixture(tmp_path)
    args.mode = mode
    root = profile.prepare(args)
    plan = profile.assert_frozen(root)
    assert plan['mode'] == mode and plan['shape_helper_sha256']
    for name in ('warmup','traced'):
        case = profile.read(root/'requests'/name/'case.json')
        assert case['chunks'] == CHUNKS
        assert case['input_path'] == profile.read(request/'case.json')['input_path']
        assert case['sampling'] == profile.read(request/'case.json')['sampling']
    assert profile.read(root/'profile.json')['kvmem'] == profile.read(source/'profile.json')['kvmem']
    assert profile.read(root/'observed-shapes.json')['capture_begin_index'] == 1


def test_native_cap_only_changes_diagnostic_threshold():
    argv = ['serve','model','--long-prefill-token-threshold','2048','--max-num-batched-tokens','16424']
    actual = shapes.observed_native_argv(argv, dict(max_chunk=14336, chunks=CHUNKS))
    assert actual == argv[:3]+['14336']+argv[4:]
    assert argv[3] == '2048'


@pytest.mark.parametrize('argv', [[], ['--long-prefill-token-threshold'],
    ['--long-prefill-token-threshold','4096'],
    ['--long-prefill-token-threshold','2048','--long-prefill-token-threshold','2048']])
def test_native_cap_rejects_unknown_original_configuration(argv):
    with pytest.raises(ValueError):
        shapes.observed_native_argv(argv, dict(max_chunk=14336, chunks=CHUNKS))


@pytest.mark.parametrize('maximum,chunks', [(True,CHUNKS),(17000,CHUNKS),(14335,CHUNKS),
    (2048,CHUNKS),(14336,CHUNKS[:-1]),(14336,[True]+CHUNKS[1:])])
def test_native_cap_rejects_unbounded_or_incomplete_schedule(maximum, chunks):
    with pytest.raises(ValueError):
        shapes.observed_native_argv(['--long-prefill-token-threshold','2048'],
                                   dict(max_chunk=maximum, chunks=chunks))


@pytest.mark.parametrize('filename,field,value', [
    ('status.json','status','running'), ('status.json','protocol_passed',False),
    ('manifest.json','profile_sha256','bad'), ('manifest.json','suite_sha256','bad'),
    ('requests/32k-repeat-2/result.json','input_tokens',32767),
    ('requests/32k-repeat-2/evidence.json','refresh',[{'reason':'pressure'}]),
    ('requests/32k-repeat-2/response.json','output_ids',[1]),
    ('requests/32k-repeat-2/response.json','response_id','')])
def test_source_must_be_completed_matching_receipt(tmp_path, filename, field, value):
    source, request, _ = source_fixture(tmp_path)
    changed = profile.read(source/filename)
    changed[field] = value
    profile.save(source/filename, changed)
    with pytest.raises(ValueError):
        shapes.observed_shape_plan(source, request/'case.json', delay=2, limit=2)


def test_source_rejects_external_case_warmup_and_corrupt_input(tmp_path):
    source, request, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match='completed source request'):
        shapes.observed_shape_plan(source, tmp_path/'case.json', delay=2, limit=2)
    suite = profile.read(source/'suite.json')
    suite['cases'][0]['warmup'] = True
    profile.save(source/'suite.json', suite)
    manifest = profile.read(source/'manifest.json')
    manifest['suite_sha256'] = profile.sha(source/'suite.json')
    profile.save(source/'manifest.json', manifest)
    with pytest.raises(ValueError, match='formal'):
        shapes.observed_shape_plan(source, request/'case.json', delay=2, limit=2)
    suite['cases'][0]['warmup'] = False
    profile.save(source/'suite.json', suite)
    manifest['suite_sha256'] = profile.sha(source/'suite.json')
    profile.save(source/'manifest.json', manifest)
    (tmp_path/'tokens.i32').write_bytes(bytes(4))
    with pytest.raises(ValueError, match='input bytes'):
        shapes.observed_shape_plan(source, request/'case.json', delay=2, limit=2)


@pytest.mark.parametrize('mutation', ['gap','offset','owner','mixed','terminal','incomplete','atomic','decode'])
def test_geometry_rejects_nonidentical_public_complete_context(mutation):
    rows = geometry()
    if mutation == 'gap': rows[1]['batch_geometry'][0]['active_seq_len'] -= 1
    if mutation == 'offset': rows[1]['batch_geometry'][0]['offset'] = 1
    if mutation == 'owner': rows[1]['batch_geometry'][0]['request_id'] = 'cmpl-source-other'
    if mutation == 'mixed': rows[1]['batch_geometry'].append(deepcopy(rows[1]['batch_geometry'][0]))
    if mutation == 'terminal': rows[1]['batch_geometry'][0]['prefill_terminal'] = True
    if mutation == 'incomplete': rows.pop()
    if mutation == 'atomic': rows[1]['atomic_prefill_tokens'] = 1
    if mutation == 'decode': rows[1]['decode_requests'] = 1
    with pytest.raises(ValueError):
        shapes.complete_geometry(rows, 'cmpl-source', 32768)


@pytest.mark.parametrize('delay,limit', [(6,1),(7,1),(1,7),(-1,2),(True,2),(2,0)])
def test_capture_rejects_terminal_or_missing_or_invalid_window(delay, limit):
    with pytest.raises(ValueError):
        shapes.capture_geometry(shapes.complete_geometry(geometry(),'cmpl-source',32768),delay,limit)


@pytest.mark.parametrize('filename', ['observed-shapes.json','nsight_prefill_shapes.py'])
def test_shape_helper_and_plan_mutation_are_rejected(tmp_path, filename):
    _, _, args = source_fixture(tmp_path)
    root = profile.prepare(args)
    with (root/filename).open('a') as file:
        file.write('\n')
    with pytest.raises(ValueError, match=filename):
        profile.assert_frozen(root)


def test_source_actual_evidence_mutation_is_rejected_after_prepare(tmp_path):
    source, _, args = source_fixture(tmp_path)
    root = profile.prepare(args)
    with (source/'run/server/capture/model_step_latency.jsonl').open('a') as file:
        file.write('\n')
    with pytest.raises(ValueError, match='Observed source evidence'):
        profile.assert_frozen(root)


def test_frozen_helper_loading_ignores_import_cache(tmp_path, monkeypatch):
    _, _, args = source_fixture(tmp_path)
    root = profile.prepare(args)
    monkeypatch.setitem(sys.modules, 'nsight_prefill_shapes', ModuleType('wrong'))
    helper = profile.shape_helper(root)
    assert helper.complete_geometry(geometry(), 'cmpl-source',32768)[-1]['prefill_terminal']


def report_for(plan):
    return dict(profiled_worker_shapes=[dict(context_requests=1, generation_requests=0,
        generation_tokens=0, scheduled_tokens=row['query_tokens'], query_tokens=row['query_tokens'],
        logical_seq_len=row['logical_seq_len']) for row in plan['capture_geometry']],
        all_traced_steps_single_owner_2k_prefill=False)


@pytest.mark.parametrize('mode', ['native','kvmem'])
def test_actual_trace_checks_full_request_and_selected_nvtx(tmp_path, mode):
    _, _, args = source_fixture(tmp_path)
    args.mode = mode
    root = profile.prepare(args)
    plan = profile.read(root/'observed-shapes.json')
    profile.save(root/'requests/traced/response.json', dict(response_id='cmpl-traced'))
    profile.save(root/'requests/traced/evidence.json', dict(refresh=[]))
    path = root/('probe/model_step_latency.jsonl' if mode == 'native' else
                'run/server/capture/model_step_latency.jsonl')
    write_steps(path, geometry(response_id='cmpl-traced'))
    result = report_for(plan)
    profile.verify_shapes(root, result, mode, 2)
    assert result['actual_geometry_verified'] and result['all_traced_steps_match_observed_prefill']
    assert [row['active_seq_len'] for row in result['profiled_worker_shapes']] == [16384,30720]
    wrong = report_for(plan)
    wrong['profiled_worker_shapes'][0]['logical_seq_len'] += 2048
    with pytest.raises(ValueError, match='NVTX capture'):
        profile.verify_shapes(root, wrong, mode, 2)
    write_steps(path, geometry([2048]*15+[2016,25,7], response_id='cmpl-traced'))
    with pytest.raises(ValueError, match='complete prefill'):
        profile.verify_shapes(root, report_for(plan), mode, 2)


def test_pressure_and_observed_are_separate_modes(tmp_path):
    _, request, args = source_fixture(tmp_path)
    args.trace_kind, args.mode = 'pressure', 'kvmem'
    # Test the explicit mutual exclusion after the existing pressure guards.
    case = profile.read(request/'case.json')
    case['row']['served_input_tokens'] = 262144
    profile.save(request/'case.json', case)
    original = profile.read(args.source/'profile.json')
    original['kvmem'].update(select_budget=32768, gen_budget=16384)
    profile.save(args.source/'profile.json', original)
    with pytest.raises(ValueError, match='separate from the pressure'):
        profile.prepare(args)
