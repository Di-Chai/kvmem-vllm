"""Keep cost attribution independent of context/selection policy changes."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import subprocess
import json
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
try:
    cost = importlib.import_module('qualify_graph_cost')
    analysis = importlib.import_module('analyze_graph_cost')
    execution = importlib.import_module('audit_graph_cost_run')
    benchmark = importlib.import_module('benchmark_graph_cost')
finally:
    sys.path.pop(0)


def base():
    return dict(execution_mode='compiled_graph', draft_method='mtp', draft_tokens=4,
        max_num_seqs=8, max_num_batched_tokens=16424,
        model='same-model', kvmem=dict(select_budget=36864, gen_budget=16384, block_tokens=32,
        pressure_policy='graph', action_fraction=0., query_history=False, guided_query_tokens=0,
        graph_trace=False, action_profiling=False, step_latency_interval=1,
        mean_k_backend='qw3_cuda', archive_backend='cpu', graph_alpha=.2))


def test_variants_only_change_attention_capture_and_graph_collection():
    original = base()
    before = deepcopy(original)
    values = cost.profiles(original)
    assert original == before
    stripped = []
    for name, profile in values.items():
        cost.validate_profile(profile, name)
        assert profile['kvmem']['pressure_policy'] == 'recency'
        value = deepcopy(profile)
        for key in ('action_attention', 'prefill_attention', 'graph_mode'):
            value['kvmem'].pop(key)
        stripped.append(value)
    assert stripped[0] == stripped[1] == stripped[2]


def test_coordinator_help_does_not_import_the_serving_package(tmp_path):
    script = Path(__file__).parents[1]/'scripts/benchmark_graph_cost.py'
    process = subprocess.run(['/usr/bin/python3', '-B', str(script), '--help'],
        cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert 'prepare' in process.stdout


def test_engine_reports_are_not_multiplied_by_owner_count():
    values = analysis.engine_totals([{'graph_prepare_seconds':10.,'graph_prepare_calls':100}]*8)
    assert values['graph_prepare_seconds'] == 10. and values['graph_prepare_calls'] == 100


def test_identical_input_does_not_hide_selection_or_verifier_drift():
    a = dict(input_sha256='a',query_selections={0:'q'},pressure_selections={0:'p'},
        output_sha256={0:'o'},decode_owner_steps=100)
    assert all(analysis.compare_identity(a,a).values())
    for key in ('query_selections','pressure_selections','output_sha256','decode_owner_steps'):
        b = {**a,key:'different'}
        assert not all(analysis.compare_identity(a,b).values())


def test_execution_audit_checks_boundary_overlap_not_just_interior_steps():
    rows = [dict(sequence=i, started_at=float(i), submitted_at=i+1.01,
        decode_requests=8, prefill_requests=0, prefill_tokens=0) for i in range(40)]
    assert len(execution.executed_window(rows, .5, 35.5)) == 36
    for changes in ({'prefill_tokens':1}, {'prefill_requests':1}, {'decode_requests':7}, {'sequence':99}):
        changed = deepcopy(rows)
        changed[0].update(changes)
        with pytest.raises(ValueError):execution.executed_window(changed, .5, 35.5)
    with pytest.raises(ValueError):execution.executed_window(rows[1:], .5, 35.5)


def test_changed_reference_fails_before_new_snapshot_or_model_work(tmp_path):
    reference=tmp_path/'reference';reference.mkdir()
    for name,value in [('manifest.json',dict(runtime_sha256={},profile_sha256='wrong')),
                       ('profile.json',{}),('status.json',dict(status='passed'))]:
        (reference/name).write_text(json.dumps(value))
    target=tmp_path/'new'
    with pytest.raises(ValueError,match='profile.json changed'):
        benchmark.prepare(SimpleNamespace(reference=reference,output=target,port=1234))
    assert not target.exists()


def test_time_increase_and_throughput_change_are_distinct_report_metrics():
    a=dict(cold_preparation_seconds=100,full_step_mean_ms=10,model_step_mean_ms=8,
        steady_decode_tokens_per_second=200,wall_seconds=300)
    b=dict(cold_preparation_seconds=110,full_step_mean_ms=12,model_step_mean_ms=9,
        steady_decode_tokens_per_second=220,wall_seconds=330)
    d=analysis.differences(a,b)
    assert d['full_decode_step_change_percent']==pytest.approx(20)
    assert d['decode_throughput_change_percent']==pytest.approx(10)


@pytest.mark.parametrize('key,value', [
    ('pressure_policy', 'attention'), ('graph_mode', 'multi'), ('action_profiling', True),
    ('gen_budget', 8192), ('select_budget', 32768), ('query_history', True)])
def test_rejects_policy_budget_or_diagnostic_confounders(key, value):
    profile = cost.profiles(base())['collect']
    profile['kvmem'][key] = value
    with pytest.raises(ValueError):
        cost.validate_profile(profile, 'collect')
