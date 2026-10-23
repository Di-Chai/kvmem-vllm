"""Full-context admission and exact reuse of the frozen RULER requests."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('benchmark_native_ruler')


def fixture():
    return dict(workers=8, sampling=dict(max_tokens=8192, temperature=1., top_p=.95,
        top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
        rows=[dict(served_input_tokens=262130)], input_directory='/frozen-inputs',
        upstream_path='/frozen-ruler', starts_in_thinking=True)


def profile():
    return dict(draft_tokens=4, max_num_seqs=8, max_num_batched_tokens=16424)


def test_full_context_reservations_fit_without_overcommitting_eight_requests(benchmark):
    log = 'Setting attention block size to 1616 tokens\nGPU KV cache size: 1,295,299 tokens'
    plan = benchmark.capacity_plan(log, fixture(), profile())
    assert plan['workers'] == 4
    per_request = plan['reserved_tokens_per_request']
    assert per_request >= 262130 + 8192 + 5 + 2 * 1616
    assert per_request % 1616 == 0
    assert 4 * per_request <= plan['admission_capacity_tokens'] < 5 * per_request
    assert plan['server_max_num_seqs'] == 8


@pytest.mark.parametrize('log', ['', 'GPU KV cache size: 1,000 tokens',
    'Setting attention block size to 1616 tokens\nGPU KV cache size: 30,000 tokens',
    'Setting attention block size to 1616 tokens\nGPU KV cache size: 1,295,299 tokens\nGPU KV cache size: 1,295,299 tokens'])
def test_missing_ambiguous_or_insufficient_capacity_never_launches_requests(benchmark, log):
    with pytest.raises(ValueError):
        benchmark.capacity_plan(log, fixture(), profile())


def test_row_order_is_same_interleaved_cohort_without_changing_seeds(benchmark):
    rows = [dict(task=task, index=index, seed=42+index)
            for task in reversed(list(benchmark.protocol.TASK_FAMILIES)) for index in (1, 0)]
    ordered = benchmark.ordered_rows(dict(rows=rows))
    assert [(r['task'], r['index']) for r in ordered] == [
        (task, index) for index in range(2) for task in benchmark.protocol.TASK_FAMILIES]
    assert all(r['seed'] == 42 + r['index'] for r in ordered)


def test_native_request_uses_full_frozen_input_and_exact_sampling_no_query_fields(benchmark, tmp_path, monkeypatch):
    frozen = fixture()
    row = dict(id='niah_single_1-0000', task='niah_single_1', index=0, seed=42,
               served_input_tokens=262130, served_sha256='expected-prompt-sha',
               served_query=[262100, 262120])
    observed = []

    def infer(target):
        case = json.loads((target / 'case.json').read_text())
        observed.append(case)
        benchmark.native.save(target / 'response.json', dict(response_id='cmpl-native', text='</think>42'))
        benchmark.native.save(target / 'result.json', dict(input_tokens=262130,
            official_case_score=100., completion_tokens=4, token_accounting_passed=True))

    monkeypatch.setattr(benchmark.native, 'client', infer)
    result = benchmark.infer(tmp_path, row, frozen, dict(port=18854))
    assert observed[0]['input_path'] == '/frozen-inputs/niah_single_1-0000.i32'
    assert observed[0]['sampling'] == dict(frozen['sampling'], seed=42)
    assert observed[0]['row'] == row
    assert 'kvmem_query_span' not in observed[0]
    assert result['text'] == '</think>42'
    assert result['task'] == row['task'] and result['index'] == 0
    assert result['input_tokens'] == 262130


def test_preemption_evidence_cannot_be_absent_or_silently_zero(benchmark):
    with pytest.raises(ValueError):
        benchmark.preemptions(['vllm:num_requests_running{engine="0"} 0'])
    assert benchmark.preemptions(['vllm:num_preemptions_total{engine="0"} 1',
                                 'vllm:num_preemptions_total{engine="1"} 2']) == 3
