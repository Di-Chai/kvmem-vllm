import importlib
from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
monitor=importlib.import_module('monitor_value_recall')


def snapshot(at=1000,done=100,answers=0):
    return dict(at=at,queue_status='running',engine_state='running',active_arm='block_value_norm',
        phase='formal',work=dict(counter_id='engine-1',completed=dict(public_prefill_tokens=done)),
        arms={m:dict(formal=dict(status='pending',completed=answers)) for m in monitor.METHODS},
        issues=[],complete=False)


def test_long_eight_way_prefill_is_healthy_when_model_work_advances():
    first=snapshot();assert monitor.findings(first,{})==[]
    second=snapshot(at=2200,done=200)
    assert monitor.findings(second,first)==[]
    assert second['last_progress_at']==2200


def test_stalled_model_is_reported_once_until_state_changes():
    first=snapshot();monitor.findings(first,{})
    stalled=snapshot(at=2200)
    assert len(monitor.findings(stalled,first))==1
    same=snapshot(at=2800)
    assert monitor.findings(same,stalled)==[]


def test_failed_queue_is_reported_once_and_completion_notified():
    first=snapshot();monitor.findings(first,{})
    failed=deepcopy(first);failed['issues']=['进程退出'];failed['queue_status']='failed'
    assert monitor.findings(failed,first)==['进程退出']
    assert monitor.findings(deepcopy(failed),failed)==[]
    completed=snapshot(answers=260);completed['complete']=True
    for arm in completed['arms'].values():arm['formal']['status']='completed'
    assert len(monitor.findings(completed,first))==3
    assert monitor.findings(deepcopy(completed),completed)==[]


def test_transient_inspection_error_is_reported_without_stopping_monitor():
    previous=snapshot()
    failed=dict(at=2000,issues=['Monitor inspection failed: transient read'],arms={},
        complete=False,queue_status='inspection_error')
    assert monitor.findings(failed,previous)==failed['issues']
