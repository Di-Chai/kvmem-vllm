"""The performance gate must distinguish overlap from incomplete publication."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('publication_benchmark',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_prefill_publication_pipeline.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def progress(**stats):
    return dict(prefill_publication_pipeline=True, pending_downloads=0, accepted_history={},
        prefill_pipeline_stats=dict(eligible_packets=2, advance_checks_with_pending=1,
            submits_with_pending=0, max_pending_per_owner=1, max_bulk_pending=1, **stats))


def test_publication_completed_during_forward_is_valid_overlap():
    # A zero submit-pending count is desirable when the background thread has
    # caught up while the new forward executes; planning observed the overlap.
    value = progress()
    assert benchmark.audit_pipeline(value, True)['prefill_pipeline_stats']['submits_with_pending'] == 0


def test_enablement_without_observed_overlap_does_not_pass():
    value = progress()
    value['prefill_pipeline_stats']['advance_checks_with_pending'] = 0
    with pytest.raises(ValueError, match='no forward planning'):
        benchmark.audit_pipeline(value, True)


@pytest.mark.parametrize('key', ['max_pending_per_owner', 'max_bulk_pending'])
def test_unbounded_backlog_does_not_pass(key):
    value = progress()
    value['prefill_pipeline_stats'][key] = 3
    with pytest.raises(ValueError, match='two-slot bound'):
        benchmark.audit_pipeline(value, True)


@pytest.mark.parametrize('field,value', [('pending_downloads', 1), ('accepted_history', {'owner': 42})])
def test_incomplete_release_does_not_pass(field, value):
    snapshot = progress()
    snapshot[field] = value
    with pytest.raises(ValueError, match='remains after release'):
        benchmark.audit_pipeline(snapshot, True)


def test_empty_baseline_pipeline_counters_are_valid():
    value = dict(prefill_publication_pipeline=False, pending_downloads=0, accepted_history={},
                 prefill_pipeline_stats={})
    benchmark.audit_pipeline(value, False)
    with pytest.raises(ValueError, match='enablement differs'):
        benchmark.audit_pipeline(value, True)


def test_mtp_uses_counter_differences_and_drafts_not_requests():
    def metrics(drafts, accepted, proposed):
        return {f'vllm:spec_decode_{name}_total{{model="test"}}': count for name, count in
                [('num_drafts', drafts), ('num_accepted_tokens', accepted), ('num_draft_tokens', proposed)]}
    value = benchmark.mtp_delta(metrics(20, 40, 80), metrics(52, 149, 208))
    assert value == dict(drafts=32, accepted_draft_tokens=109, proposed_tokens=128,
                        mean_acceptance_length=4.40625)
    assert benchmark.mtp_delta({}, {})['mean_acceptance_length'] is None


def test_pressure_control_steps_do_not_split_same_request_prefill():
    # The actual timing writer emits only model steps. Control-only refresh
    # waits appear in elapsed timestamps, while decode separates requests.
    steps = [dict(prefill_tokens=2, started_at=1, submitted_at=2, host_submit_ms=1000,
                  model_ms=800, maintenance_ms=100),
             dict(prefill_tokens=2, started_at=5, submitted_at=6, host_submit_ms=1000,
                  model_ms=800, maintenance_ms=100),
             dict(prefill_tokens=0),
             dict(prefill_tokens=3, started_at=8, submitted_at=9, host_submit_ms=1000,
                  model_ms=800, maintenance_ms=100)]
    groups = benchmark.prefill_groups(steps)
    assert [len(group) for group in groups] == [2, 1]
    value = benchmark.timing(groups[0])
    assert value['tokens'] == 4
    assert value['host_between_submissions_seconds'] == 3
    assert value['model_cuda_seconds'] == 1.6


def test_only_known_frozen_utilities_can_replace_missing_live_files(tmp_path):
    repo, reference = tmp_path / 'repo', tmp_path / 'reference'
    repo.mkdir()
    utility = reference / 'runtime/scripts/reference_service.py'
    utility.parent.mkdir(parents=True)
    utility.write_text('historical lifecycle utility\n')
    benchmark.save(reference / 'runtime-proof.json', dict(files={
        'scripts/reference_service.py': benchmark.sha(utility),
        'src/vllm_kvmem/scheduler.py': 'untrusted-placeholder'}))
    origins, inherited = benchmark.runtime_origins(repo, reference, ['scripts/reference_service.py'])
    assert origins['scripts/reference_service.py'] == utility
    assert inherited['scripts/reference_service.py']['sha256'] == benchmark.sha(utility)
    # Even an old frozen copy must never silently substitute missing live code.
    with pytest.raises(ValueError, match='dependency is missing: src/vllm_kvmem/scheduler.py'):
        benchmark.runtime_origins(repo, reference, ['src/vllm_kvmem/scheduler.py'])
    utility.write_text('modified historical utility\n')
    with pytest.raises(ValueError, match='differs from its recorded hash'):
        benchmark.runtime_origins(repo, reference, ['scripts/reference_service.py'])


def test_live_source_has_priority_over_historical_utility(tmp_path):
    repo, reference = tmp_path / 'repo', tmp_path / 'reference'
    live = repo / 'scripts/reference_service.py'
    live.parent.mkdir(parents=True)
    live.write_text('current code\n')
    reference.mkdir()
    benchmark.save(reference / 'runtime-proof.json', dict(files={}))
    origins, inherited = benchmark.runtime_origins(repo, reference, ['scripts/reference_service.py'])
    assert origins['scripts/reference_service.py'] == live
    assert inherited == {}
