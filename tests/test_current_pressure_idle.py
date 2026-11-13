"""The bounded profiler must not silently switch the compiled recall control."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

path=Path(__file__).resolve().parents[1]/'scripts/profile_current_pressure_idle.py'
spec=importlib.util.spec_from_file_location('current_idle_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def controls():
    profile=dict(port=1,execution_mode='compiled_graph',max_num_seqs=8,max_num_batched_tokens=16424,
        kvmem=dict(select_budget=36864,gen_budget=16384,block_tokens=32,sink_blocks=41,recent_blocks=0))
    source=dict(profile=profile,environment=dict(KVMEM_RUNNER_PROBE_DIR='/old/capture',
        KVMEM_LAYERWISE_QUERY_SCORE_BACKEND='tiled',KVMEM_LAYERWISE_QUERY_LAZY_READER='1',
        KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER='0'))
    actual=deepcopy(source);actual['profile']['port']=2
    actual['environment'].update(KVMEM_RUNNER_PROBE_DIR='/new/capture',PREFILL_PRESSURE_RANGE='diagnostic')
    fixture=dict(case_ids=[module.CASE],cases=1)
    return source,actual,fixture,actual['profile']


def test_diagnostic_changes_only_capture_path_and_port():
    result=module.check_effective_controls(*controls())
    assert result['passed'] and result['compiled_server_slots']==8 and result['client_workers']==1
    assert result['active']==36864 and result['gen']==16384 and not result['performance_eligible']


def test_eight_owner_diagnostic_retains_exact_declared_input_order():
    source,actual,fixture,profile=controls()
    ids=[f'original{i}' for i in range(8)]
    fixture.update(case_ids=ids[:],cases=8)
    result=module.check_effective_controls(source,actual,fixture,profile,case_ids=ids,workers=8)
    assert result['client_workers']==8 and not result['performance_eligible']
    fixture['case_ids'].reverse()
    with pytest.raises(ValueError,match='declared original'):
        module.check_effective_controls(source,actual,fixture,profile,case_ids=ids,workers=8)


@pytest.mark.parametrize('ids,workers',[(['same']*8,8),(['one'],8),(['one','two'],2)])
def test_profiler_rejects_uncontrolled_cohorts(ids,workers):
    with pytest.raises(ValueError,match='declared original cold owners'):
        module.check_effective_controls(*controls(),case_ids=ids,workers=workers)


@pytest.mark.parametrize('change',['query','pressure','eager','budget','batch','case'])
def test_changed_execution_or_workload_is_rejected(change):
    source,actual,fixture,profile=controls()
    if change=='query':actual['environment']['KVMEM_LAYERWISE_QUERY_SCORE_BACKEND']='legacy'
    if change=='pressure':actual['environment']['KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER']='1'
    if change=='eager':profile['execution_mode']='eager'
    if change=='budget':profile['kvmem']['gen_budget']=4096
    if change=='batch':profile['max_num_seqs']=1
    if change=='case':fixture['cases']=2
    with pytest.raises(ValueError):module.check_effective_controls(source,actual,fixture,profile)


def test_current_worker_installs_commit_and_publication_probes_without_vllm(tmp_path, monkeypatch):
    from prefill_commit_phase_probe import _CommitWorkerMixin
    from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
    from vllm_kvmem._prefill_state_native import State

    class BaseWorker:
        def _install_phase_probe(self):
            pass
        def execute_model(self):
            return 'execute-result'
        def sample_tokens(self):
            return 'sample-result'

    monkeypatch.setitem(sys.modules, 'layerwise_pressure_phase_probe',
                        NS(LayerwisePressurePhaseWorker=BaseWorker))
    worker_path = path.with_name('current_pressure_idle_probe.py')
    worker_spec = importlib.util.spec_from_file_location('current_idle_worker_test', worker_path)
    worker_module = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker_module)
    assert _CommitWorkerMixin in worker_module.CurrentPressureIdleWorker.__mro__
    assert 'prefill_commit_phase_probe.py' in module.HELPERS

    labels = set()
    scopes = NS(enabled=lambda: False, labels=labels, push=lambda _: None, pop=lambda: None,
                wrap=lambda owner, method, label: labels.add(label), save_manifest=lambda: None)
    worker = worker_module.CurrentPressureIdleWorker()
    worker._prefill_phase_probe = scopes
    worker._pressure_gate = NS(done=False)
    worker.model_runner = NS(_kvmem_serving=True, _kvmem_capture_runtime=None)
    monkeypatch.setenv('PREFILL_PHASE_PROBE_OUTPUT', str(tmp_path))
    # CommitTimings installs real class descriptors; restore them after the test.
    monkeypatch.setattr(BufferedPrefillHistory, 'accept', BufferedPrefillHistory.accept)
    monkeypatch.setattr(State, 'committed', State.committed)
    try:
        worker._install_phase_probe()
        installed = worker._commit_timings
        worker._install_phase_probe()
        assert worker._commit_timings is installed
        assert {'cpu_commit.buffered_accept', 'cpu_commit.native_committed',
                'buffered_history.scores', 'buffered_state.values',
                'capture._publish_logical_claim', 'logical_publication.advance',
                'publication.drain_requests'} <= labels
        assert worker.execute_model() == 'execute-result'
        assert worker.sample_tokens() == 'sample-result'
        assert not list(tmp_path.glob('cpu-commits-*.json'))
        worker._pressure_gate.done = True
        assert worker.execute_model() == 'execute-result'
        output, = tmp_path.glob('cpu-commits-*.json')
        assert json.loads(output.read_text())['rows'] == []
        assert worker.sample_tokens() == 'sample-result'
    finally:
        worker._python_gc_phase_probe.close()
