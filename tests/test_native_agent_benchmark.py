import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'native_agent_benchmark_test', Path(__file__).parents[1]/'scripts/benchmark_native_agents.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def body():
    return '\n'.join([
        'vllm:generation_tokens_total{engine="0"} 123',
        'vllm:num_preemptions_total{engine="0"} 0',
        'vllm:num_requests_running{engine="0"} 8',
        'vllm:num_requests_waiting{engine="0"} 0',
        'vllm:prompt_tokens_by_source_total{model_name="model",engine="0",source="local_compute"} 200',
        'vllm:prompt_tokens_by_source_total{model_name="model",engine="0",source="local_cache_hit"} 5000'])


def test_native_counter_parser_preserves_actual_compute_and_cache_sources():
    values = benchmark.metrics(body())
    assert values['vllm:prompt_tokens_by_source_total:local_compute'] == 200
    assert values['vllm:prompt_tokens_by_source_total:local_cache_hit'] == 5000
    assert values['vllm:generation_tokens_total'] == 123


def test_multiple_engines_cannot_be_mislabeled_as_one_eight_user_engine():
    with pytest.raises(ValueError, match='exactly one native engine'):
        benchmark.metrics(body()+'\nvllm:generation_tokens_total{engine="1"} 1')


@pytest.mark.parametrize('mutated', [
    body().replace(' 123', ' nan'),
    body().replace(' 123', ' -1'),
    body().replace('vllm:num_preemptions_total', 'other_counter'),
])
def test_missing_or_invalid_native_counter_fails_accounting(mutated):
    with pytest.raises(ValueError):
        benchmark.metrics(mutated)
