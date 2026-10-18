import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('timed_benchmark',
    Path(__file__).parents[1] / 'scripts/benchmark_timed_agents.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_gpu_observations_are_time_weighted_and_outages_are_excluded():
    rows = [dict(at=t, utilization=u) for t,u in [(0,0),(1,100),(3,50),(10,90),(11,90)]]
    result = benchmark.gpu_summary(rows, 0, 11)
    assert result['observed_seconds'] == 4
    assert result['time_weighted_utilization_percent'] == 72.5
    assert result['coverage_fraction'] == 4/11
    assert result['utilization_below_ten_percent_fraction'] == .25
    assert result['maximum_sample_gap_seconds'] == 7


def test_native_accepted_counts_reject_duplicate_engines_and_nonfinite_values():
    text = 'vllm:generation_tokens_total{engine="0"} 10\nvllm:num_preemptions_total{engine="0"} 0\n'
    assert benchmark.parse_native_metrics(text)['vllm:generation_tokens_total'] == 10
    with pytest.raises(ValueError, match='one native'):
        benchmark.parse_native_metrics(text + 'vllm:generation_tokens_total{engine="1"} 5\n')
    with pytest.raises(ValueError, match='Invalid'):
        benchmark.parse_native_metrics(text.replace(' 10', ' NaN'))


def native_row(begin, end, generated):
    return dict(started_at=begin, finished_at=end, values={
        'vllm:generation_tokens_total': generated, 'vllm:num_preemptions_total': 0})


def test_native_brackets_exclude_snapshots_overlapping_the_boundary():
    rows = [native_row(0,.1,3),native_row(.9,1.1,5),native_row(3.9,4.1,10),native_row(4.2,4.3,12)]
    result = benchmark.native_boundaries(rows,1,4)
    assert result['before'] == rows[0] and result['after'] == rows[-1]
    assert result['deltas']['vllm:generation_tokens_total'] == 9
    assert not benchmark.native_boundaries(rows[1:],1,4)['available']
    with pytest.raises(ValueError,match='reset'):
        benchmark.native_boundaries([native_row(0,.1,10),native_row(5,5.1,3)],1,4)


def work_row(at, submitted, completed, identity='same'):
    def counters(n):
        return dict(public_prefill_tokens=n,public_batches=n//10,public_non_prefill_tokens=n,
            prefill_request_batches=n//10,decode_request_batches=n//5)
    return dict(at=at,counter_id=identity,submitted=counters(submitted),completed=counters(completed))


def test_completed_work_bounds_do_not_pretend_submission_is_gpu_completion():
    rows = [work_row(0,10,0),work_row(2,30,20),work_row(8,90,80),work_row(12,120,110)]
    result = benchmark.public_work_summary(rows,1,10)
    assert result['completed_prefill_tokens_in_window_bounds'] == [50,120]
    assert result['submitted_inner_window']['prefill_tokens_per_second'] == 10
    assert result['submitted_inner_window']['decode_requests_per_batch'] == 2
    incomplete = benchmark.public_work_summary(rows[:-1],1,10)
    assert not incomplete['complete_brackets']
    assert 'completed_prefill_tokens_in_window_bounds' not in incomplete


def test_work_identity_or_counter_reset_invalidates_the_measurement():
    with pytest.raises(ValueError,match='identity'):
        benchmark.public_work_summary([work_row(0,10,0),work_row(2,20,10,'other')],0,2)
    with pytest.raises(ValueError,match='reset'):
        benchmark.public_work_summary([work_row(0,20,10),work_row(2,10,5)],0,2)


def measurements(pairs=1):
    return [dict(pair=i, arm=arm, results={'saturated':dict(status='completed', drained=True,
        native_client_total_output_reconciliation={'passed':True},
        output_tokens_per_second=rate)})
        for i in range(pairs) for arm,rate in [('A0',100),('P1',115)]]


def test_one_pair_is_preliminary_and_failed_streams_cannot_win():
    assert benchmark.compare(measurements())['saturated']['preliminary']
    assert not benchmark.compare(measurements())['saturated']['passes_throughput_target']
    assert benchmark.compare(measurements(3))['saturated']['passes_throughput_target']
    bad = measurements();bad[-1]['results']['saturated']['status'] = 'failed'
    with pytest.raises(ValueError,match='Failed'):
        benchmark.compare(bad)


def test_live_checkout_cannot_execute_prepared_experiment(tmp_path):
    import asyncio
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='live checkout'):
        asyncio.run(benchmark.run(SimpleNamespace(output=tmp_path)))


def test_resolved_public_profile_roundtrips_without_saving_native_only_controls(tmp_path):
    from test_serving_launcher import nvfp4_profile
    from vllm_kvmem.serve import build_launch
    profile = nvfp4_profile(tmp_path)
    profile.update(draft_method='mtp', draft_tokens=4, max_num_seqs=4,
                   max_model_len=1048576, query_replay_budget=16384)
    profile['kvmem'].update(block_tokens=32, select_budget=131072, gen_budget=131072,
        repository_blocks=32768, request_planning=True,
        session_checkpoint_policy='pm', session_capture_terminal=True)
    resolved = benchmark.resolve_profile(profile, tmp_path, tmp_path/'check')
    assert Path(resolved['model']).is_absolute()
    assert resolved.get('long_prefill_token_threshold', 1024) > 0
    again = build_launch(resolved, tmp_path/'different-config-directory', tmp_path/'run')
    assert again['profile']['long_prefill_token_threshold'] == 0
    assert again['profile']['max_num_batched_tokens'] == 16404


def test_trace_preparation_preserves_all_per_user_inputs_and_adds_only_cleanup_eos():
    import copy
    from types import SimpleNamespace
    fixture = dict(schema_version=1, agent_plans=[dict(
        initial_tokens=list(range(100, 120+i)),
        query_plan=dict(kvmem_query_span=[2, 4], kvmem_mandatory_spans=[], kvmem_reselect='auto'),
        turns=[dict(output_tokens=2+j, tool_tokens=[] if j == 0 else [800+j],
                    tool_wait_seconds=0. if j == 0 else .25) for j in range(4)])
        for i in range(2)])
    original = copy.deepcopy(fixture)
    args = SimpleNamespace(concurrency=2, warmup_turns=2, epoch_turns=4, context_tokens=141870)
    tokenizer = SimpleNamespace(eos_token_id=42)
    result = benchmark.prepare_fixture(fixture, args, {'kvmem': {'gen_budget': 100}}, tokenizer)
    assert fixture == original
    assert result == {**original, 'eos': 42}
    assert 'user_prefixes' not in result


@pytest.mark.parametrize('concurrency', [4, 8])
def test_preparation_freezes_matching_native_slots_and_all_user_plans(tmp_path, monkeypatch, concurrency):
    import json
    from test_serving_launcher import nvfp4_profile
    from vllm_kvmem.serve import build_launch

    example = Path(__file__).parents[1] / 'examples/qwen-nvfp4-offline-agent-k128-g128.json'
    profile = json.loads(example.read_text())
    paths = nvfp4_profile(tmp_path)
    profile.update({key: paths[key] for key in ('model', 'tokenizer', 'hf_config_path', 'draft_model')})
    profile['max_num_seqs'] = concurrency
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(profile))
    fixture = dict(eos=42, agent_plans=[dict(
        source_task=f'task-{i}', initial_tokens=list(range(100, 120+i)),
        query_plan=dict(kvmem_query_span=[2, 4], kvmem_mandatory_spans=[], kvmem_reselect='auto'),
        turns=[dict(output_tokens=2+j, tool_tokens=[] if j == 0 else [800+i, 900+j],
                    tool_wait_seconds=0. if j == 0 else .25) for j in range(4)])
        for i in range(concurrency)])
    fixture_path = tmp_path / 'fixture.json'
    fixture_path.write_text(json.dumps(fixture))
    checkout = tmp_path / 'checkout'
    (checkout / 'src').mkdir(parents=True)
    (checkout / 'src/frozen-marker.py').write_text('marker = 1\n')
    (checkout / 'scripts').mkdir()
    workload = Path(__file__).parents[1] / 'scripts/timed_agent_workload.py'
    (checkout / 'scripts/timed_agent_workload.py').write_bytes(workload.read_bytes())
    monkeypatch.setattr(benchmark, 'ROOT', checkout)
    output = tmp_path / 'prepared'
    args = benchmark.parser().parse_args([
        '--output', str(output), '--config', str(config), '--fixture', str(fixture_path),
        '--prepare-only', '--concurrency', str(concurrency), '--epoch-turns', '4'])

    benchmark.prepare(args)

    assert benchmark.read(output / 'fixture.json') == fixture
    manifest = benchmark.read(output / 'manifest.json')
    assert manifest['workload']['concurrency'] == concurrency
    assert manifest['fixture_sha256'] == benchmark.q.digest(fixture)
    assert manifest['source_sha256'] == benchmark.hashes(output / 'frozen')
    for arm in ('A0', 'P1'):
        frozen = benchmark.read(output / (arm + '.json'))
        assert frozen['max_num_seqs'] == concurrency
        argv = build_launch(frozen, output, output / ('run-' + arm))['vllm_argv']
        assert argv[argv.index('--max-num-seqs') + 1] == str(concurrency)
        assert argv[argv.index('--max-num-batched-tokens') + 1] == str(16384 + 5 * concurrency)


@pytest.mark.parametrize('concurrency,native_slots', [(8, 4), (4, 8), (2, 2), (16, 16)])
def test_preparation_rejects_unsupported_or_mismatched_native_concurrency(
        tmp_path, concurrency, native_slots):
    import json
    example = Path(__file__).parents[1] / 'examples/qwen-nvfp4-offline-agent-k128-g128.json'
    profile = json.loads(example.read_text())
    profile['max_num_seqs'] = native_slots
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(profile))
    args = benchmark.parser().parse_args([
        '--output', str(tmp_path / 'prepared'), '--config', str(config),
        '--fixture', str(tmp_path / 'unused.json'), '--prepare-only',
        '--concurrency', str(concurrency)])
    with pytest.raises(ValueError, match='matching native slots'):
        benchmark.prepare(args)


def test_invalid_cleanup_eos_is_rejected_before_freezing():
    from types import SimpleNamespace
    args = SimpleNamespace(concurrency=1, context_tokens=2)
    fixture = dict(prompt=[1, 2], tool=[3], tool_sizes=[1], user_prefixes=[[4]], eos=None)
    with pytest.raises(ValueError, match='cleanup EOS'):
        benchmark.prepare_fixture(fixture, args, {})


def test_runtime_variants_are_frozen_independently_of_the_common_client(tmp_path):
    import json
    (tmp_path/'frozen').mkdir()
    (tmp_path/'manifest.json').write_text('{}')
    assert benchmark.runtime_directory(tmp_path, 'A0') == tmp_path/'frozen'
    signatures = {}
    for arm in ('A0', 'P1'):
        source = tmp_path/'runtime'/arm
        (source/'src/vllm_kvmem').mkdir(parents=True)
        (source/'src/vllm_kvmem/test.py').write_text(arm)
        signatures[arm] = benchmark.hashes(source)
    (tmp_path/'manifest.json').write_text(json.dumps({'runtime_sha256': signatures}))
    assert benchmark.runtime_directory(tmp_path, 'P1') == tmp_path/'runtime/P1'
    (tmp_path/'runtime/A0/src/vllm_kvmem/test.py').write_text('mutated')
    with pytest.raises(RuntimeError, match='runtime changed'):
        benchmark.runtime_directory(tmp_path, 'A0')
