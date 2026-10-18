import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('weight_benchmark_test', ROOT/'scripts/benchmark_weight_precision.py')
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_only_weight_paths_and_port_change():
    base = {'model':'nv','hf_config_path':'nv','port':1,'tokenizer':'original',
            'kvmem':{'select_budget':65536},'draft_tokens':4}
    candidate = bench.expected_profile(base, Path('/weights/fp8'), 2)
    assert candidate == {**base,'model':'/weights/fp8','hf_config_path':'/weights/fp8','port':2}
    candidate['kvmem']['select_budget'] = 1
    assert base['kvmem']['select_budget'] == 65536


@pytest.mark.parametrize('name', ['src/vllm_kvmem/memory.py','src/vllm_kvmem/_native.so',
                                  'src/vllm_kvmem/runner.py','scripts/timed_agent_workload.py'])
def test_unrelated_production_or_workload_changes_rejected(name):
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        bench.runtime_delta({name:'old'}, {name:'new'})


def test_generic_loader_delta_allowed_but_file_set_fixed():
    name = 'src/vllm_kvmem/serve.py'
    assert bench.runtime_delta({name:'old'}, {name:'new'}) == {name:['old','new']}
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        bench.runtime_delta({name:'old'}, {})


def test_weight_identity_requires_complete_fp8_shards(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'fp8'}}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'one.safetensors'}}))
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        bench.model_identity(tmp_path)
    (tmp_path/'one.safetensors').write_bytes(b'weight')
    result = bench.model_identity(tmp_path)
    assert result['quant_method'] == 'fp8' and result['shards']['one.safetensors']['size'] == 6


@pytest.mark.parametrize('failure', ['not_completed','missing_history','missing_cleanup','preemption',
                                    'counter_mismatch','oracle_mismatch','changed_profile'])
def test_qualification_failure_cannot_be_used_for_performance(tmp_path, monkeypatch, failure):
    result = dict(status='passed',cleanup_completed=True,history_proofs=[{}]*6,oracle={'passed':True},
                  native_reconciliation={'passed':True,'preemptions':0,'native_tokens':10,'client_tokens':10})
    status = 'completed_pending_review'
    profile = {'port':10}
    expected_oracle = {'passed':True}
    if failure == 'not_completed': status = 'running'
    if failure == 'missing_history': result['history_proofs'] = []
    if failure == 'missing_cleanup': result['cleanup_completed'] = False
    if failure == 'preemption': result['native_reconciliation']['preemptions'] = 1
    if failure == 'counter_mismatch': result['native_reconciliation']['client_tokens'] = 9
    if failure == 'oracle_mismatch': expected_oracle = {'passed':False}
    if failure == 'changed_profile': profile['other'] = 'changed'
    for name, data in [('status.json',{'status':status}),('results.json',result),('profile.json',profile)]:
        (tmp_path/name).write_text(json.dumps(data))
    (tmp_path/'workload').mkdir()
    (tmp_path/'workload/traces.json').write_text('[]')
    gate = SimpleNamespace(verify=lambda _: {'runtime_sha256':{}},
        oracle_evidence=lambda *_:expected_oracle, qualification_profile=lambda *_:{'port':10})
    monkeypatch.setattr(bench,'module',lambda *_:gate)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        bench.qualification_proof(tmp_path, {'port':10}, {})
