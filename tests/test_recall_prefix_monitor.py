import importlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
m=importlib.import_module('monitor_recall_prefix_pilot')


def snap(at=0,done=0,tokens=100):
    return dict(at=at,queue_status='running',engine_state='running',active_arm='mean_k',
        arms={'mean_k':dict(completed=done)},work=dict(completed=dict(public_prefill_tokens=tokens)),
        issues=[],complete=False)


def test_prefill_without_completed_answers_is_not_a_stall():
    first=snap();assert m.findings(first,{})==[]
    advanced=snap(at=1200,tokens=1000)
    assert m.findings(advanced,first)==[]


def test_stall_and_completion_are_not_repeated():
    first=snap();m.findings(first,{})
    stalled=snap(at=1200)
    assert len(m.findings(stalled,first))==1
    again=snap(at=1800)
    assert m.findings(again,stalled)==[]
    done=snap(at=2400,done=26);done['complete']=True;done['queue_status']='completed'
    assert len(m.findings(done,again))==1
    assert m.findings(dict(done),done)==[]


def test_single_method_130_completion_does_not_require_the_old_three_26_arms(tmp_path):
    import json
    (tmp_path/'queue-status.json').write_text(json.dumps(dict(status='completed',stage='analysis')))
    arm=tmp_path/'campaign-v1/arms/block_deletion_error/qualification'
    for index in range(130):
        case=arm/'cases'/str(index);case.mkdir(parents=True);(case/'result.json').write_text('{}')
    (arm/'completed.json').write_text(json.dumps(dict(status='completed')))
    value=m.inspect(tmp_path,methods=['block_deletion_error'],expected=130,campaign_name='campaign-v1')
    assert value['complete'] and set(value['arms'])=={'block_deletion_error'}
    assert value['arms']['block_deletion_error']['completed']==130
    assert '130' in m.findings(value,{})[-1]


def test_waiting_controller_exit_is_detected(tmp_path):
    import json
    (tmp_path/'queue-status.json').write_text(json.dumps(dict(status='waiting')))
    value=m.inspect(tmp_path,is_alive=lambda pid:False,methods=['block_deletion_error'],expected=130)
    assert any('队列进程' in issue for issue in value['issues'])
