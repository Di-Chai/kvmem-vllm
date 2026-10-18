import importlib.util
import json
from pathlib import Path


spec=importlib.util.spec_from_file_location('deepswe_repetition',Path(__file__).resolve().parents[1]/'benchmarks/deepswe/repetition.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def call(guard, ident, name='Bash', command='cat source.py'):
    guard.line(json.dumps({'type':'assistant','message':{'content':[{
        'type':'tool_use','id':str(ident),'name':name,'input':{'command':command}}]}}))


def result(guard, ident, text='same source', error=False):
    guard.line(json.dumps({'type':'user','message':{'content':[{
        'type':'tool_result','tool_use_id':str(ident),'content':text,'is_error':error}]}}))


def test_exact_successful_repetition_only():
    guard=module.BashRepeatGuard(3)
    for i in range(2):call(guard,i);result(guard,i)
    assert guard.first_trigger is None
    call(guard,2);result(guard,2)
    assert guard.first_trigger['consecutive_identical_successful_calls']==3


def test_result_change_error_and_intervening_tools_break_streak():
    for change in ('result','error','tool','input'):
        guard=module.BashRepeatGuard(3)
        for i in range(5):
            call(guard,i,name='Read' if i==2 and change=='tool' else 'Bash',
                 command='different' if i==2 and change=='input' else 'cat source.py')
            result(guard,i,text='changed' if i==2 and change=='result' else 'same source',
                   error=i==2 and change=='error')
        assert guard.first_trigger is None


def test_duplicate_stream_events_and_out_of_order_results():
    guard=module.BashRepeatGuard(3)
    call(guard,0);call(guard,0);call(guard,1,name='Read');call(guard,2)
    result(guard,2);result(guard,0);result(guard,1)
    assert guard.streak==1 and guard.first_trigger is None
    call(guard,3);result(guard,3)
    assert guard.first_trigger is None


def test_only_cancellations_after_owned_stop_are_expected():
    for message in ('','Cancelled by cancel scope xyz','Cancelled via cancel scope xyz by task'):
        event={'event':'error','at':20,'error':message}
        assert module.expected_stop_cancellation(event,{'at':19})
        assert not module.expected_stop_cancellation(event,{'at':21})
        assert not module.expected_stop_cancellation(event,None)
    event={'event':'cancelled','error_type':'CancelledError','at':20,'error':'new runtime format'}
    assert module.expected_stop_cancellation(event,{'at':19})
    assert not module.expected_stop_cancellation({'event':'error','at':20,'error':'HTTP 500'}, {'at':19})
