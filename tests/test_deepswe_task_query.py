import copy
import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('task_query_bridge',Path(__file__).parents[1]/'benchmarks/deepswe/claude_bridge.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)


def prepared(text, query=None):
    value=b.Prepared(([1,2,3],dict(kvmem_query_span=query or [0,len(text.encode())],
        kvmem_mandatory_spans=[dict(begin=0,end=len(text.encode()),kind='query')]),
        dict(prompt=text,query_index=0,spans=[dict(index=0,role='user',content_begin=0,content_end=len(text.encode()))]),
        [],'',False,{}))
    value.coordinate=lambda p:p
    return value


def test_large_policy_context_preserves_prompt_and_pins_and_uses_complete_task():
    instruction='Fix the complete task. 中文🙂'
    payload=dict(messages=[dict(role='user',content=[dict(type='text',text='POLICY '*1000),dict(type='text',text=instruction)])])
    p=prepared('POLICY '*1000+instruction);tokens=list(p[0]);pins=copy.deepcopy(p[1]['kvmem_mandatory_spans'])
    b.bounded_task_query(payload,instruction+'\n',4096)(p)
    assert p[0]==tokens and p[1]['kvmem_mandatory_spans']==pins
    begin,end=p[1]['kvmem_query_span'];assert p[2]['prompt'].encode()[begin:end].decode()==instruction
    assert p.query_source_audit['prompt_unchanged']


def test_supported_original_query_is_byte_for_byte_unchanged():
    p=prepared('short original');old=copy.deepcopy(p[1])
    b.bounded_task_query({'messages':[]},'irrelevant',4096)(p)
    assert p[1]==old and not hasattr(p,'query_source_audit')


def test_renderer_inserted_system_message_preserves_initial_user_source():
    text='policy '*1000+'complete task';p=prepared(text)
    p[2]['query_index']=1;p[2]['spans'][0]['index']=1
    payload=dict(messages=[dict(role='user',content=[dict(type='text',text='complete task')])])
    b.bounded_task_query(payload,'complete task\n',4096)(p)
    assert p[1]['kvmem_query_span']==[len(text)-len('complete task'),len(text)]


@pytest.mark.parametrize('kind',['missing','ambiguous','oversized_task'])
def test_source_mismatch_ambiguity_or_actual_long_task_fails_without_clipping(kind):
    instruction='task' if kind!='oversized_task' else 'task '*1000
    text='policy '*1000+instruction+(instruction if kind=='ambiguous' else '')
    payload=dict(messages=[dict(role='user',content=[dict(type='text',text=instruction if kind!='missing' else 'other')])])
    p=prepared(text);old=copy.deepcopy(p[1])
    with pytest.raises(ValueError):b.bounded_task_query(payload,instruction,4096)(p)
    assert p[1]==old
