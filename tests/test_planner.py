"""kvmem plan: geometry, workload math, memory budget, simulator and search (CPU only)."""
import json
import math
import random

import pytest

from vllm_kvmem.planner.costs import Costs, SWAP_STACKS, STACK_ENV
from vllm_kvmem.planner.emit import overlay, environment, merge
from vllm_kvmem.planner.geometry import Geometry, from_cache_geometry, from_model_config
from vllm_kvmem.planner.memory import Machine, Placement, plan_memory
from vllm_kvmem.planner.search import candidates, evaluate, choose, requirements
from vllm_kvmem.planner.simulate import simulate
from vllm_kvmem.planner.workload import LogNormalWait, MixtureWait, EmpiricalWait, preset

# Qwen3.8-27B NVFP4 + MTP4 on the 48 GiB pool (cache-geometry.json of the 11-12 sweep).
GEOMETRY = dict(schema=2, capacity_bytes=51536592896, native_blocks=916, cpu_archive_bytes_per_token=34816,
                select_budget=36864, gen_budget=16384,
                groups=[dict(block_tokens=524288, page_bytes=56262656, payload_page_bytes=52953088,
                             layers=[f'l{i}' for i in range(16)], recurrent=True, speculative_blocks=4)
                        for _ in range(3)] +
                       [dict(block_tokens=1616, page_bytes=56262656, payload_page_bytes=56262656,
                             layers=[f'a{i}' for i in range(17)], recurrent=False, speculative_blocks=0)])
CONFIG = dict(text_config=dict(
    layer_types=(['linear_attention'] * 3 + ['full_attention']) * 16, num_hidden_layers=64,
    head_dim=256, hidden_size=5120, num_attention_heads=24, num_key_value_heads=4,
    linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=16,
    linear_num_value_heads=48, linear_value_head_dim=128, mamba_ssm_dtype='float32',
    mtp_num_hidden_layers=1, dtype='bfloat16'))
MACHINE = Machine(ram_bytes=128 << 30, ram_reserved_bytes=20 << 30, memlock_bytes=float('inf'),
                  ssd_bytes=2 << 40)


@pytest.fixture
def geometry(tmp_path):
    path = tmp_path / 'cache-geometry.json'
    path.write_text(json.dumps(GEOMETRY))
    return from_cache_geometry(path)


def test_geometry_from_capture(geometry):
    assert geometry.pool_pages == 915 and geometry.attention_page_tokens == 1616
    assert geometry.recurrent_pages_per_request == 15
    assert geometry.window_pages(36864) == 23 and geometry.running_pages(36864) == 38
    assert geometry.cpu_bytes_per_token == 34816
    # sawtooth: whole history below the budget, then budget + (growth mod gen_budget)
    assert geometry.window_tokens(20000) == 20000
    assert geometry.window_tokens(36864 + 1000) == 36864 + 1000
    assert geometry.window_tokens(36864 + 16384 + 5) == 36864 + 5


def test_geometry_from_model_config_matches_the_runtime_layout(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(CONFIG))
    g = from_model_config(path, kv_pool_bytes=51536592896, spec_tokens=4, select_budget=36864)
    assert g.cpu_bytes_per_token == 34816
    assert g.attention_page_tokens == 1616 and g.page_bytes == 56262656
    assert g.recurrent_groups == 3 and g.recurrent_pages_per_request == 15
    assert g.pool_pages == 915
    assert g.recurrent_state_bytes == 157876224          # measured restore payload


def test_tool_wait_distributions():
    d = LogNormalWait(0.9, 1.81, cap=120)
    assert d.survival(0) == 1 and d.survival(200) == 0
    assert abs(d.survival(0.9) - 0.5) < 1e-9
    # heavy tail: expected remaining wait grows with elapsed time
    assert d.expected_remaining(10) > d.expected_remaining(1) > 0
    m = MixtureWait({'q': LogNormalWait(0.1, 0.5), 'l': LogNormalWait(20, 1.0)}, {'q': 0.9, 'l': 0.1})
    rng = random.Random(3)
    names = [m.sample_class(rng)[0] for _ in range(2000)]
    assert 0.85 < names.count('q') / len(names) < 0.95
    e = EmpiricalWait([1, 2, 3, 4])
    assert e.survival(2.5) == 0.5 and e.mean() == 2.5


def test_memory_budget_and_violations(geometry):
    w = preset('toolwait-sim', 32)
    m = plan_memory(geometry, w, Placement(slots=8, residency_fraction=0.5), MACHINE)
    assert m.legal and m.gpu_pages_running == 8 * 38 and m.gpu_pages_parked_cap == 457
    assert m.parked_windows_fit == 457 // 23
    too_many = plan_memory(geometry, w, Placement(slots=30), MACHINE)
    assert not too_many.legal and 'GPU pool' in too_many.violations[0]
    long = preset('toolwait-sim', 256, max_history_tokens=1 << 20, history_tokens=200000)
    ram = plan_memory(geometry, long, Placement(slots=8), MACHINE)
    assert not ram.legal and 'CPU RAM' in ram.violations[0]
    tiered = plan_memory(geometry, long, Placement(slots=8, archive='tiered', cpu_ttl=60), MACHINE)
    assert tiered.legal and tiered.ssd_bytes_needed > 0
    pinned = plan_memory(geometry, long, Placement(slots=8, archive='tiered', pinned_archive=True),
                         Machine(**{**MACHINE.__dict__, 'memlock_bytes': 8 << 30}))
    assert any('Pinned' in v for v in pinned.violations)


def test_simulation_is_deterministic_and_responds_to_residency(geometry):
    w = preset('toolwait-sim', 32)
    costs = Costs().for_stack('direct_prepared_exempt')
    a = simulate(geometry, w, Placement(slots=8, residency_fraction=0.5), costs, duration=120, seed=7)
    b = simulate(geometry, w, Placement(slots=8, residency_fraction=0.5), costs, duration=120, seed=7)
    assert a == b and a.output_tok_s > 0 and a.turns > 0
    full = simulate(geometry, w, Placement(slots=8, residency_fraction=0.01), costs, duration=120, seed=7)
    assert full.swaps['resident'] == 0 and full.swaps['cpu'] > a.swaps['cpu']
    assert full.output_tok_s < a.output_tok_s


def test_ttl_and_ssd_tiers(geometry):
    w = preset('terminal-bench', 32)
    costs = Costs().for_stack('direct_prepared_exempt')
    ttl = simulate(geometry, w, Placement(slots=8, gpu_policy='ttl', residency_fraction=0.9, gpu_ttl=1.0),
                   costs, duration=120, seed=2)
    assert ttl.evictions['ttl'] > 0
    ssd = simulate(geometry, w, Placement(slots=8, archive='tiered', cpu_ttl=5.0), costs, duration=120, seed=2)
    assert ssd.swaps['ssd'] > 0 and ssd.ssd_sessions_peak > 0


def test_search_prefers_legal_deployable_and_marks_unbuilt(geometry):
    w = preset('toolwait-sim', 32)
    cands = list(candidates(geometry, w, residencies=(0.01, 0.5), ttls=(),
                            victims=('fifo', 'expected_wait', 'expected_wait_oracle')))
    ev = evaluate(geometry, w, MACHINE, Costs(), cands, seeds=(1,), duration=60, warmup=20)
    best = choose(ev, w)
    assert best is not None and best.deployable and best.memory.legal
    assert not any('oracle' in e.placement.victim for e in ev)     # unbuilt hints excluded by default
    assert any(e.placement.victim == 'expected_wait' and e.deployable for e in ev)
    ev_all = evaluate(geometry, w, MACHINE, Costs(), cands, seeds=(1,), duration=60, warmup=20,
                      include_unbuilt=True)
    assert any(e.needs for e in ev_all)
    assert requirements(Placement(slots=8, swap_stack='pinned'))
    strict = preset('toolwait-sim', 32, turn_latency_p90_limit=0.01)
    assert choose(ev, strict) is None


def test_emit_overlay_and_environment():
    p = Placement(slots=8, gpu_policy='ttl', residency_fraction=0.9, gpu_ttl=60.0)
    o = overlay(p)
    assert o['max_num_seqs'] == 8
    assert o['kvmem']['active_resident_ttl_ms'] == 60000 and o['kvmem']['active_resident_pressure_only'] is False
    assert o['kvmem']['session_gpu_checkpoint_bytes'] == 0
    assert environment(p) == STACK_ENV['direct_prepared_exempt']
    merged = merge({'max_num_seqs': 4, 'kvmem': {'a': 1}}, o)
    assert merged['kvmem']['a'] == 1 and merged['max_num_seqs'] == 8
    assert environment(Placement(slots=8, swap_stack='pinned')) is None
    assert set(SWAP_STACKS) >= set(STACK_ENV)


def test_report_lists_only_simulated_candidates(geometry):
    from vllm_kvmem.planner.emit import report
    w = preset('toolwait-sim', 32)
    cands = [Placement(slots=8, residency_fraction=0.5), Placement(slots=30)]   # second is illegal
    ev = evaluate(geometry, w, MACHINE, Costs(), cands, seeds=(1,), duration=30, warmup=10)
    text = report(w, geometry, {}, choose(ev, w), sorted(ev, key=lambda e: e.score, reverse=True))
    assert 'Rejected by memory' in text and 'S30' in text.split('## Rejected by memory')[1]


def test_validate_reports_errors(geometry):
    from vllm_kvmem.planner import validate as v
    rows = v.MEASURED[:2]
    old, v.MEASURED = v.MEASURED, rows
    try:
        out = v.validate(geometry, seeds=(1,))
    finally:
        v.MEASURED = old
    assert len(out['rows']) == 2 and out['current']['arms'] == 2
    assert all(abs(r['error']) < 0.25 for r in out['rows'])


def test_cli_compile_writes_plan_overlay_and_report(geometry, tmp_path):
    from vllm_kvmem.planner.__main__ import main
    gpath = tmp_path / 'g.json'
    gpath.write_text(json.dumps(GEOMETRY))
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'max_num_seqs': 4, 'kvmem': {'archive_backend': 'cpu'}}))
    out = tmp_path / 'plan'
    assert main(['compile', '--agents', '16', '--geometry', str(gpath), '--duration', '60',
                 '--base-profile', str(base), '--out', str(out)]) == 0
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['chosen'] and (out / 'report.md').exists()
    merged = json.loads((out / 'profile.json').read_text())
    slots = int(plan['chosen']['placement'].split('/')[0][1:])        # the compiler picks among 8/12/16
    assert merged['kvmem']['session_gpu_checkpoint_bytes'] == 0 and merged['max_num_seqs'] == slots
    assert slots in (8, 12, 16)


def test_learned_expected_wait_runs_in_the_simulator(geometry):
    w = preset('mixed-agents', 24)
    costs = Costs().for_stack('direct_prepared_exempt')
    r = simulate(geometry, w, Placement(slots=8, residency_fraction=0.5, victim='expected_wait'),
                 costs, duration=60, warmup=20, seed=3)
    assert r.turns > 0 and r.output_tok_s > 0


def test_batch_sizing_scales_planning_batch_and_query_capture_with_slots():
    from vllm_kvmem.planner import emit
    base = dict(max_num_seqs=8, max_num_batched_tokens=16424, request_planning_batch_tokens=16424,
                query_replay_budget=16384, draft_tokens=4,
                kvmem=dict(gen_budget=16384, historical_gc=dict(policy_params=dict(
                    thinking_only=True, query_capture_bytes=807272448, candidate_blocks=128))))
    assert emit.batch_sizing(base, 8) == {}                     # the 8-slot profile already fits
    patch = emit.batch_sizing(base, 16)
    assert patch['max_num_batched_tokens'] == patch['request_planning_batch_tokens'] == 16464
    assert patch['kvmem']['historical_gc']['policy_params']['query_capture_bytes'] == 809238528
    merged = emit.merge(base, patch)
    assert merged['kvmem']['historical_gc']['policy_params']['candidate_blocks'] == 128   # deep merge
    with pytest.raises(ValueError, match='exceeds'):
        emit.batch_sizing(base, 2000)


def test_max_context_bounds_slots_by_cpu_ram(geometry):
    from dataclasses import replace
    from vllm_kvmem.planner.memory import Machine, Placement, plan_memory
    from vllm_kvmem.planner.workload import preset
    machine = Machine(ram_bytes=126 << 30, ram_reserved_bytes=19 << 30, memlock_bytes=float('inf'))
    work = replace(preset('toolwait-sim', 32), max_context_tokens=262144)
    legal = {slots: plan_memory(geometry, work, Placement(slots=slots), machine).legal for slots in (8, 12, 16)}
    assert legal == {8: True, 12: True, 16: False}           # 256K x 34,816 B = 8.5 GiB per running request
    m = plan_memory(geometry, work, Placement(slots=16), machine)
    assert any('maximum context' in v for v in m.violations)
    assert plan_memory(geometry, replace(work, max_context_tokens=None), Placement(slots=16), machine).legal
