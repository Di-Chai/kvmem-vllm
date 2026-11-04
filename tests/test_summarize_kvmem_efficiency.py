import pytest

from scripts.summarize_kvmem_efficiency import host_summary, numeric_delta, step_summary


def step(index,start,prefill=0):
    return dict(sequence=index,call_index=index,started_at=start,submitted_at=start+.5,
                prefill_tokens=prefill,decode_requests=0 if prefill else 1,
                previous_call_index=index-1,previous_end_to_model_ms=10.)


def test_sparse_model_samples_do_not_turn_unsampled_work_into_host_gaps():
    summary,_ = step_summary([step(1,1,64),step(5,5),step(6,6)],0,10)
    decode = summary['phases']['decode']
    assert summary['missing_call_indices']==3
    assert decode['consecutive_previous_end_to_model_ms']['count']==1
    assert decode['consecutive_previous_end_to_model_ms']['total']==10
    assert decode['consecutive_host_between_submissions_ms']['total']==500
    _,selected = step_summary([step(1,1),step(2,2)],1.1,2.6)
    assert [r['call_index'] for r in selected]==[2]


def span(ident,parent,stage,start,end,sequence=1,thread=1):
    return dict(span=ident,parent=parent,stage=stage,started_at=start,finished_at=end,
                wall_seconds=end-start,thread_cpu_seconds=(end-start)/2,
                sequence=sequence,thread=thread)


def test_host_phase_needs_real_step_and_background_is_not_inherited():
    values=[span(1,None,'runner.execute_model',1,1.3),
            span(2,1,'runner.prepare_attn',1.05,1.15),
            span(3,None,'action.capture',1.3,1.4),
            span(4,None,'runner.execute_model',1.6,1.8),
            span(5,None,'append.scatter',1.1,1.4,thread=2)]
    result=host_summary(values,[step(1,1.01)],0,3)
    groups={(row['phase'],row['stage']):row for row in result['stages']}
    assert ('decode','action.capture') in groups
    assert ('maintenance_only','runner.execute_model') in groups
    assert ('background_or_between_steps','append.scatter') in groups
    runner=groups['decode','runner.execute_model']
    assert runner['exclusive_wall_seconds']['total']==pytest.approx(.2)
    assert runner['inclusive_wall_seconds']['total']==pytest.approx(.3)


def test_sparse_publication_counters_start_at_zero():
    assert numeric_delta({'main':{'seconds':1}},
                         {'main':{'seconds':2},'background':{'seconds':3}})=={
                             'main':{'seconds':1},'background':{'seconds':3}}
