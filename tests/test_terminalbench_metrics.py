import json

import pytest

from scripts.finalize_terminalbench import efficiency_metrics


def lifecycle():
    return [
        dict(event='start',trial='a',at=10),
        dict(event='agent-start',trial='a',at=12),
        dict(event='agent-start',trial='b',at=14),
        dict(event='agent-end',trial='a',at=16),
        dict(event='agent-start',trial='c',at=16),
        dict(event='agent-end',trial='b',at=18),
        dict(event='agent-end',trial='c',at=18),
        dict(event='end',trial='a',at=20),
    ]


def telemetry(at,value,*,errors=None,external=()):
    return dict(at=at,devices=[] if value is None else [dict(uuid='GPU-A',index=0,
        gpu_utilization_percent=value,memory_used_mib=70)],
        errors=errors or {},external_pids=list(external))


def save_telemetry(root,values):
    (root/'gpu-utilization.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))


def test_actual_concurrency_and_gpu_use_distinct_formal_and_agent_windows(tmp_path):
    # Outside-window startup/cleanup utilization must not alter either mean.
    save_telemetry(tmp_path,[telemetry(t,100 if t<10 or t>=20 else
                                      20 if t<12 else 80 if t<18 else 10)
                             for t in range(25)])
    result=efficiency_metrics(tmp_path,lifecycle())
    assert result['windows']['formal']['seconds']==10
    assert result['windows']['agent']['seconds']==6
    formal=result['active_agent_concurrency']['formal']
    agent=result['active_agent_concurrency']['agent']
    assert formal['peak']==agent['peak']==2  # a ending as c starts does not count three.
    assert formal['time_weighted_mean']==1
    assert agent['time_weighted_mean']==pytest.approx(10/6)
    assert formal['seconds_by_active_agents']=={'0':4,'1':2,'2':4}
    assert agent['summed_agent_seconds']==10
    for window,expected in [('formal',54),('agent',80)]:
        gpu=result['gpu'][window]
        metric=gpu['devices']['GPU-A']['metrics']['gpu_utilization_percent']
        assert gpu['status']=='sampled'
        assert metric['mean']==expected
        assert metric['coverage_fraction']==1
        assert not gpu['errors'] and not gpu['external_pids']


def test_failed_gpu_sample_does_not_become_zero_and_external_process_is_preserved(tmp_path):
    save_telemetry(tmp_path,[telemetry(t,None,errors={'devices':'failed'}) if t==15 else
                             telemetry(t,80,external=[42] if t==14 else ())
                             for t in range(10,21)])
    gpu=efficiency_metrics(tmp_path,lifecycle())['gpu']['agent']
    metric=gpu['devices']['GPU-A']['metrics']['gpu_utilization_percent']
    assert metric['mean']==80
    assert metric['coverage_fraction']==pytest.approx(5/6)
    assert gpu['errors']==[dict(at=15,errors={'devices':'failed'})]
    assert gpu['external_pids']==[42]


def test_sparse_gpu_journal_preserves_uncovered_time(tmp_path):
    save_telemetry(tmp_path,[telemetry(10,80),telemetry(20,0)])
    metric=efficiency_metrics(tmp_path,lifecycle())['gpu']['formal']['devices']['GPU-A']['metrics']['gpu_utilization_percent']
    assert metric['coverage_fraction']==.25
    assert metric['mean']==80


def test_older_run_without_dedicated_gpu_samples_reports_unavailable(tmp_path):
    # Coarse resources are deliberately not mislabeled as a full 1-second series.
    (tmp_path/'resources.jsonl').write_text(json.dumps({'gpu_memory_mib':80})+'\n')
    result=efficiency_metrics(tmp_path,lifecycle())
    assert result['gpu']['formal']['status']=='unavailable'
    assert result['gpu']['agent']['status']=='unavailable'
    assert result['active_agent_concurrency']['agent']['peak']==2


@pytest.mark.parametrize('events',[
    [dict(event='agent-start',at=12),dict(event='agent-end',at=18)],
    lifecycle()+[dict(event='agent-start',trial='a',at=13)],
    [row for row in lifecycle() if not (row.get('trial')=='b' and row['event']=='agent-end')],
])
def test_incomplete_or_ambiguous_agent_events_do_not_invent_concurrency(tmp_path,events):
    result=efficiency_metrics(tmp_path,events)
    assert result['active_agent_concurrency']['agent']['status']=='unavailable'


def test_empty_or_invalid_telemetry_has_an_explicit_reason(tmp_path):
    for content in ('','broken\n'):
        (tmp_path/'gpu-utilization.jsonl').write_text(content)
        gpu=efficiency_metrics(tmp_path,lifecycle())['gpu']['formal']
        assert gpu['status']=='unavailable' and gpu['reason']


def test_only_startup_samples_do_not_claim_a_formal_gpu_measurement(tmp_path):
    save_telemetry(tmp_path,[telemetry(1,80),telemetry(2,80)])
    gpu=efficiency_metrics(tmp_path,lifecycle())['gpu']['formal']
    assert gpu['status']=='unavailable'
    assert gpu['devices']=={}
    assert gpu['attribution_coverage_fraction']==0
