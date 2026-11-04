from pathlib import Path
from scripts.monitor_deepswe_repair import notices


def test_normal_progress_is_quiet_except_one_stable_start_event():
    first=notices(Path('/run'),{'status':'running','stage':'formal','arm':'thinking'},
        {'completed':1,'expected':8,'tasks':{}},[],[])
    later=notices(Path('/run'),{'status':'running','stage':'formal','arm':'thinking','at':200},
        {'completed':4,'expected':8,'tasks':{}},[],[])
    assert first==later and len(first)==1


def test_new_transport_error_or_dead_service_is_actionable_without_task_cancellation():
    events=notices(Path('/run'),{'status':'running','stage':'formal','arm':'lru'},
        {'tasks':{'x':{'state':'infrastructure_error','error':'HTTP500'}}},
        [{'event':'error','task':'x','error':'HTTP500'}],['tunnel'])
    kinds={x[1]['kind'] for x in events}
    assert kinds=={'arm_started','process_failed','task_infrastructure_error','transport_failure'}
    assert len({x[0] for x in events})==4


def test_preparing_has_no_false_dead_process_or_repair_event():
    assert notices(Path('/run'),{'status':'preparing'}, {},[],['model'])==[]
