"""Recall comparison rejects mixed methods, leaks and answer-based selection."""
from copy import deepcopy
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('recall_comparison_protocol')


@pytest.fixture
def controller(protocol):
    return importlib.import_module('run_recall_comparison')


def profile():
    return dict(max_num_seqs=1, kvmem=dict(block_tokens=32, select_budget=1024,
        gen_budget=512, sink_blocks=1, recent_blocks=1, graph_alpha=0,
        guided_query_tokens=0, query_history=False, repository_gc=False))


def event(method='fps4', kind='prefill_pressure'):
    return dict(sample_id='sample-1', request_id='response-1-worker', event_type=kind,
        configured_method=method, actual_method=method, query_source='accepted_tail',
        candidate_block_count=64, selected_block_ids=[1, 5, 9], fallback_reason=None,
        query_begin=128, query_end=256, accepted_through=256, selected_tokens=96, budget=1024)


def test_three_entrance_coverage_and_common_window(protocol):
    events = [event(kind=kind) for kind in protocol.EVENT_TYPES]
    events[-1].update(query_begin=240, query_end=256, query_source='public_query',
                      query_processed_through=256, public_input_end=260)
    result = protocol.audit_events(events, 'fps4', profile(), query_span=[240, 256],
                                   required_types=protocol.EVENT_TYPES)
    assert result['event_counts'] == dict(prefill_pressure=1, decode_pressure=1, query=1)
    with pytest.raises(ValueError, match='entrances'):
        protocol.audit_events(events[:1], 'fps4', profile(), required_types=protocol.EVENT_TYPES)


def test_two_worker_dispatch_keeps_request_ownership_and_bounded_parallelism(controller):
    from threading import Barrier, Lock
    together, lock = Barrier(2), Lock()
    active = peak = 0
    seen = []
    def infer(owner):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(owner)
        together.wait(timeout=3)
        with lock:
            active -= 1
        return dict(owner=owner, response='answer-'+owner)
    results = list(controller.map_requests(['a', 'b', 'c', 'd'], infer, 2))
    assert peak == 2 and active == 0
    assert sorted(seen) == ['a', 'b', 'c', 'd']
    assert {r['owner']: r['response'] for r in results} == {k:'answer-'+k for k in seen}


def test_eight_workers_refill_a_finished_slot_while_seven_requests_are_pending(controller):
    from threading import Barrier, Event, Lock, Thread
    together, release, refill, lock = Barrier(8), Event(), Event(), Lock()
    active = peak = 0
    started, results, errors = [], [], []
    def infer(owner):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(owner)
        try:
            if owner < 8:
                together.wait(timeout=5)
            if owner in range(1, 8):
                assert release.wait(timeout=5)
            if owner == 8:
                refill.set()
            return owner
        finally:
            with lock:
                active -= 1
    def run():
        try:
            results.extend(controller.map_requests(range(10), infer, 8))
        except BaseException as error:
            errors.append(error)
    driver = Thread(target=run)
    driver.start()
    try:
        assert refill.wait(timeout=5), 'A free slot must refill before the other seven finish'
        with lock:
            assert active >= 7 and peak == 8
    finally:
        release.set()
        driver.join(timeout=6)
    assert not driver.is_alive() and not errors
    assert sorted(started) == sorted(results) == list(range(10))
    assert active == 0


def test_technical_failure_stops_refill_without_waiting_for_other_http_request(controller):
    from threading import Barrier, Event
    together, release, finished = Barrier(2), Event(), Event()
    started = []
    def infer(owner):
        started.append(owner)
        together.wait(timeout=3)
        if owner == 0:
            raise RuntimeError('technical HTTP failure')
        try:
            assert release.wait(timeout=3)
            return owner
        finally:
            finished.set()
    try:
        with pytest.raises(RuntimeError, match='technical HTTP failure'):
            list(controller.map_requests(range(20), infer, 2))
        assert sorted(started) == [0, 1]
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(timeout=3)


@pytest.mark.parametrize('owners', [2, 4, 8])
def test_parallel_profile_must_be_explicit_and_does_not_change_budgets(protocol, owners):
    configured = profile()
    configured['max_num_seqs'] = owners
    configured['recall_experiment'] = dict(concurrent_owners=owners)
    before = deepcopy(configured['kvmem'])
    assert protocol.validate_profile(configured) is configured
    assert configured['kvmem'] == before
    configured['max_num_seqs'] = 3
    with pytest.raises(ValueError, match='explicitly frozen'):
        protocol.validate_profile(configured)


def test_official_generation_limits_and_sampling_are_frozen(protocol):
    expected = {**{task:128 for task in ('niah_single_1','niah_single_2','niah_single_3',
        'niah_multikey_1','niah_multikey_2','niah_multikey_3','niah_multivalue','niah_multiquery')},
        'vt':30,'cwe':120,'fwe':50,'qa_1':32,'qa_2':32}
    assert protocol.official_task_limits() == expected
    fixture = dict(starts_in_thinking=False, generation_limit_policy='ruler_v1_official_per_task',
        task_max_tokens=expected, sampling=dict(max_tokens=128, thinking_token_budget=None,
            temperature=1.0, top_p=.95, top_k=20))
    original = deepcopy(fixture)
    for task, maximum in expected.items():
        for seed in (42, 61):
            actual = protocol.case_sampling(fixture, dict(task=task, seed=seed))
            assert actual == dict(fixture['sampling'], max_tokens=maximum, seed=seed)
    assert fixture == original


def test_official_limits_match_local_pinned_upstream_snapshot_when_available(protocol):
    import ast
    source = Path(__file__).resolve().parents[1] / 'artifacts/ruler-v1-256k-20261021-r1/upstream' / (
        'RULER-'+protocol.ruler.UPSTREAM_COMMIT) / 'scripts/data/synthetic/constants.py'
    if not source.exists():pytest.skip('Optional local frozen upstream source is not packaged with unit tests')
    tree = ast.parse(source.read_text())
    tasks = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TASKS' for t in node.targets))
    assert protocol.official_task_limits() == {
        task: tasks[family]['tokens_to_generate'] for task, family in protocol.ruler.TASK_FAMILIES.items()}


@pytest.mark.parametrize('change', ['thinking', 'missing-task', 'changed-cap', 'uniform-max', 'undeclared'])
def test_short_output_protocol_rejects_silent_or_mixed_controls(protocol, change):
    fixture = dict(starts_in_thinking=False, generation_limit_policy='ruler_v1_official_per_task',
        task_max_tokens=protocol.official_task_limits(), sampling=dict(max_tokens=128, thinking_token_budget=None))
    if change == 'thinking':fixture['starts_in_thinking'] = True
    if change == 'missing-task':fixture['task_max_tokens'].pop('qa_2')
    if change == 'changed-cap':fixture['task_max_tokens']['cwe'] = 8192
    if change == 'uniform-max':fixture['sampling']['max_tokens'] = 8192
    if change == 'undeclared':fixture.pop('generation_limit_policy')
    with pytest.raises(ValueError):protocol.case_sampling(fixture, dict(task='qa_1', seed=42))


def test_legacy_uniform_sampling_and_functional_pressure_controls_remain_valid(protocol, monkeypatch):
    fixture = dict(starts_in_thinking=False,sampling=dict(max_tokens=8192,temperature=1.0,
        top_p=.95,top_k=20,thinking_token_budget=None))
    assert protocol.case_sampling(fixture,dict(task='qa_1',seed=42)) == dict(fixture['sampling'],seed=42)
    qualifier = importlib.import_module('qualify_recall_entrypoints')
    configured = profile();configured['kvmem'].update(gen_budget=16384,rolling_active_window=True)
    short = deepcopy(fixture);short.update(task_max_tokens=protocol.official_task_limits(),
        generation_limit_policy='ruler_v1_official_per_task');short['sampling']['max_tokens']=128
    assert qualifier.functional_sampling(short,configured) == qualifier.functional_sampling(fixture,configured)
    assert qualifier.functional_sampling(short,configured,followup=True) == qualifier.functional_sampling(fixture,configured,followup=True)
    assert configured['kvmem']['select_budget']==1024 and configured['kvmem']['gen_budget']==16384


def test_public_query_processed_boundary_is_distinct_from_accepted_kv(protocol):
    query = event(kind='query')
    query.update(query_begin=250, query_end=260, accepted_through=250,
                 query_processed_through=260, public_input_end=265)
    protocol.audit_events([query], 'fps4', profile(), query_span=[250, 260])
    query['query_processed_through'] = 259
    with pytest.raises(ValueError, match='processed public-input'):
        protocol.audit_events([query], 'fps4', profile())


@pytest.mark.parametrize('change,message', [
    (dict(actual_method='mean_k'), 'Mixed-method'),
    (dict(fallback_reason='missing_query'), 'fallback'),
    (dict(query_end=257), 'unaccepted'),
    (dict(query_begin=224), '128 accepted'),
    (dict(selected_tokens=1025), 'budget'),
    (dict(selected_block_ids=[1, 1]), 'duplicate'),
    (dict(propagation_applied=True), 'enhancement'),
])
def test_event_contract_rejects_invalid_runs(protocol, change, message):
    row = event(); row.update(change)
    with pytest.raises(ValueError, match=message):
        protocol.audit_events([row], 'fps4', profile())


def test_method_profiles_only_change_method(protocol):
    baseline = profile()
    baseline['recall_experiment'] = dict(device='cpu', query_batch=4, block_batch=32)
    before = deepcopy(baseline)
    profiles = [protocol.method_profile(baseline, method) for method in protocol.METHODS]
    assert baseline == before
    for method, actual in zip(protocol.METHODS, profiles):
        assert actual['recall_experiment'].pop('method') == method
    assert all(actual == profiles[0] for actual in profiles)
    assert profiles[0]['recall_experiment']['pressure_query_window'] == 128


def comparison_rows():
    return [dict(id='a', task='niah_single_1', index=0, served_sha256='a'),
            dict(id='b', task='niah_single_1', index=1, served_sha256='b'),
            dict(id='c', task='qa_1', index=0, served_sha256='c')]


def make_results(rows, method, scores):
    return [dict(row, status='completed', method=method, input_sha256=row['served_sha256'],
                 official_case_score=score, case_score_unrounded=score, full_reference_recall=score == 100)
            for row, score in zip(rows, scores)]


def test_oracle_uses_task_weights_and_excludes_exact_mass(protocol):
    rows = comparison_rows()
    # Raw sample averaging would favor mean_k; task macro favors fixed4.
    values = dict(mean_k=[100, 100, 0], fixed4=[0, 0, 100], fps4=[0, 0, 40],
                  min_max=[20, 20, 0], exact_mass=[100, 100, 100])
    arms = {method: make_results(rows, method, scores) for method, scores in values.items()}
    result = protocol.compare_results(rows, arms)
    assert result['potential']['best_fixed_methods'] == ['mean_k', 'fixed4']
    assert result['potential']['best_fixed_score'] == 50
    assert result['potential']['per_request_oracle_score'] == 100
    assert result['potential']['gain_points'] == 50
    assert result['unique_successes']['mean_k']['sample_ids'] == ['a', 'b']
    assert result['unique_successes']['fixed4']['sample_ids'] == ['c']
    pair = result['pairwise']['mean_k__vs__fixed4']
    assert (pair['wins'], pair['losses'], pair['ties']) == (2, 1, 0)
    assert 'exact_mass' not in result['potential']['candidate_methods']


def test_incomplete_or_changed_pair_cannot_produce_oracle(protocol):
    rows = comparison_rows()
    arms = {method: make_results(rows, method, [0, 0, 0]) for method in protocol.METHODS}
    arms['fps4'].pop()
    with pytest.raises(ValueError, match='Missing'):
        protocol.compare_results(rows, arms)
    arms['fps4'] = make_results(rows, 'fps4', [0, 0, 0])
    arms['fps4'][0]['input_sha256'] = 'different'
    with pytest.raises(ValueError, match='input mismatch'):
        protocol.compare_results(rows, arms)


def test_resume_never_regenerates_complete_answer(controller, tmp_path):
    case = tmp_path / 'case'
    attempt, retained = controller.choose_attempt(case, retry_technical=False)
    assert not retained
    (attempt / 'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='retry-technical'):
        controller.choose_attempt(case, retry_technical=False)
    second, retained = controller.choose_attempt(case, retry_technical=True)
    assert second != attempt and not retained
    (second / 'response.json').write_text(json.dumps(dict(text='incorrect but complete')))
    resumed, retained = controller.choose_attempt(case, retry_technical=True)
    assert resumed == second and retained
    assert len(list((case / 'attempts').iterdir())) == 2


def test_multiple_completed_answers_fail_instead_of_choosing_best(controller, tmp_path):
    case = tmp_path / 'case'
    for name in ('0001', '0002'):
        target = case / 'attempts' / name
        target.mkdir(parents=True)
        (target / 'response.json').write_text('{}')
    with pytest.raises(ValueError, match='Multiple completed'):
        controller.choose_attempt(case, retry_technical=True)


def test_final_answer_only_and_unrounded_official_mass(protocol):
    row = dict(task='niah_multivalue', outputs=['A', 'B', 'C'])
    metric = lambda preds, refs: round(sum(word in preds[0] for word in refs[0]) / len(refs[0]) * 100, 2)
    scorer = ({'niah': dict(metric_fn=metric)}, lambda text, config: text.strip())
    result = protocol.score_case(row, dict(text='A B C</think>A'), scorer, starts_in_thinking=True)
    assert result['official_case_score'] == 33.33
    assert result['case_score_unrounded'] == pytest.approx(100 / 3)
    assert not result['full_reference_recall']


@pytest.mark.parametrize('official_caps', [False, True])
def test_client_resumes_saved_response_after_audit_failure_without_model_retry(controller, protocol,
        monkeypatch, tmp_path, official_caps):
    root, capture = tmp_path, tmp_path / 'server/capture'
    capture.mkdir(parents=True)
    token_directory = root / 'tokenizer'
    token_directory.mkdir()
    (token_directory / 'tokenizer.json').write_text('{}')
    settings = profile()
    settings.update(tokenizer=str(token_directory), port=8000, served_model_name='test', prefill_chunk_tokens=32)
    protocol.save(root / 'profile.json', settings)
    protocol.save(root / 'profiles/fps4.json', settings)
    row = dict(id='case', task='qa_1', index=0, outputs=['Paris'], input='context Question: city?',
        served_query=[100, 110], served_input_tokens=128, question='city?', served_sha256='input', seed=42)
    fixture = dict(rows=[row], sampling=dict(max_tokens=8192), qualification_ids=['case'],
        tokenizer_sha256=protocol.sha(token_directory / 'tokenizer.json'), upstream_path='upstream',
        starts_in_thinking=False, request_timeout_seconds=10)
    if official_caps:
        fixture.update(generation_limit_policy='ruler_v1_official_per_task',task_max_tokens=protocol.official_task_limits())
        fixture['sampling']['max_tokens'] = 128
    protocol.save(root / 'fixture.json', fixture)
    monkeypatch.setattr(controller, 'verify', lambda *args: (fixture, settings))
    monkeypatch.setattr(controller, 'launch_audit', lambda *args: dict(passed=True))
    monkeypatch.setattr(controller, 'load_tokens', lambda *args: [1] * 128)
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda path: object())))
    metric = lambda predictions, refs: 100. if 'Paris' in predictions[0] else 0.
    monkeypatch.setattr(controller.ruler, 'load_scorer', lambda path:
        ({'qa': dict(metric_fn=metric)}, lambda text, config: text))
    client = SimpleNamespace(stream=None, wait_retired=None, wait_owner_released=None)
    monkeypatch.setitem(sys.modules, 'qualify_graph_niah', client)
    calls = []
    response = dict(response_id='response-1', text='Paris', output_ids=[1, 2],
        seconds=1., ttft_seconds=.5, finish_reason='stop')
    pressure = event()
    pressure.update(query_begin=0, query_end=96, accepted_through=96, selected_tokens=64)
    query = event(kind='query')
    query.update(query_begin=100, query_end=110, accepted_through=100,
                 query_processed_through=110, public_input_end=128)
    log = capture / 'recall_events.jsonl'
    def stream(*args):
        assert args[4]['max_tokens'] == (32 if official_caps else 8192)
        calls.append(args)
        log.write_text(json.dumps(pressure) + '\n')
        return response
    monkeypatch.setattr(client, 'stream', stream)
    monkeypatch.setattr(client, 'wait_retired', lambda *args: dict(history_end=129))
    monkeypatch.setattr(client, 'wait_owner_released', lambda *args: dict(released=True))
    with pytest.raises(ValueError, match='entrances'):
        controller.run_client(root, 'fps4', 'qualification', capture)
    assert len(calls) == 1
    with log.open('a') as output:
        output.write(json.dumps(query) + '\n')
    result = controller.run_client(root, 'fps4', 'qualification', capture)
    assert len(calls) == 1 and result[0]['official_case_score'] == 100
    result_again = controller.run_client(root, 'fps4', 'qualification', capture)
    assert result_again == result and len(calls) == 1
    case = root / 'arms/fps4/qualification/cases/case'
    (case / result[0]['attempt'] / 'response.json').write_text('{}')
    with pytest.raises(ValueError, match='answer changed'):
        controller.saved_result(case, row, 'fps4', root)


def test_source_freeze_allows_target_build_but_binary_binding_detects_changes(protocol, tmp_path):
    source = tmp_path / 'src/vllm_kvmem'
    source.mkdir(parents=True)
    (source / 'recall.py').write_text('value = 1\n')
    binary = source / '_native.cpython-test.so'
    binary.write_bytes(b'local-build')
    sources = protocol.source_manifest(tmp_path)
    assert all(not name.endswith('.so') for name in sources)
    binary.write_bytes(b'target-gpu-build')
    protocol.verify_sources(tmp_path, sources)
    binaries = protocol.binary_manifest(tmp_path)
    protocol.verify_binaries(tmp_path, binaries)
    binary.write_bytes(b'changed-after-freeze')
    with pytest.raises(ValueError, match='binaries changed'):
        protocol.verify_binaries(tmp_path, binaries)


def test_extra_installed_binary_is_bound(protocol, tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    binary = tmp_path / 'installed-native.so'
    binary.write_bytes(b'installed')
    manifest = protocol.binary_manifest(runtime, [binary])
    assert list(manifest) == [str(binary)]
    protocol.verify_binaries(runtime, manifest)
    binary.unlink()
    with pytest.raises(ValueError, match='Missing compiled'):
        protocol.verify_binaries(runtime, manifest)


def test_functional_option_is_for_qualification_only(controller):
    args = controller.parser().parse_args(['run', '--root', '/tmp/unused', '--phase', 'formal',
                                          '--qualify-entrypoints'])
    with pytest.raises(ValueError, match='only allowed'):
        controller.run(args)


def test_functional_qualification_reuses_the_ten_case_model_process(controller, protocol, monkeypatch, tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    settings = profile()
    protocol.save(tmp_path / 'profiles/fps4.json', settings)
    fixture = dict(runtime_source=str(runtime), methods=list(protocol.METHODS))
    monkeypatch.setattr(controller, 'verify', lambda *args: (fixture, settings))
    monkeypatch.setattr(controller, 'gpu_lock', lambda *args: nullcontext())
    monkeypatch.setattr(controller, 'termination_guard', lambda: nullcontext())
    state = dict(qualified=False)
    calls = []
    process = SimpleNamespace(pid=12345, returncode=0)
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: calls.append('launch') or process)
    monkeypatch.setattr(controller, 'wait_health', lambda *args: calls.append('healthy'))
    monkeypatch.setattr(controller, 'stop_owned', lambda *args: calls.append('stop'))
    monkeypatch.setattr(controller, 'functional_qualified', lambda *args: state['qualified'])
    captures = []
    def main_client(root, method, phase, capture, **kwargs):
        calls.append('ten-cases'); captures.append(capture)
    def functional_client(root, method, capture, output):
        calls.append('functional'); captures.append(capture); state['qualified'] = True
    monkeypatch.setattr(controller, 'run_client', main_client)
    monkeypatch.setitem(sys.modules, 'qualify_recall_entrypoints', SimpleNamespace(run_client=functional_client))
    args = controller.parser().parse_args(['run', '--root', str(tmp_path), '--runtime', str(runtime),
        '--phase', 'qualification', '--method', 'fps4', '--qualify-entrypoints'])
    controller.run(args)
    assert calls == ['launch', 'healthy', 'ten-cases', 'functional', 'stop']
    assert captures[0] == captures[1]
