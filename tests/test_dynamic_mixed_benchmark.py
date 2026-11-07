import pytest

from benchmark_dynamic_mixed import analyze_timeline, validate_completion


def test_natural_eos_can_finish_before_the_cap():
    validate_completion({'prompt':[1,2], 'max_tokens':8}, [3,4],
                        {'prompt_tokens':2,'completion_tokens':2}, ['stop'], True)


@pytest.mark.parametrize('usage,reasons,done', [
    ({'prompt_tokens':2,'completion_tokens':1},['stop'],True),
    ({'prompt_tokens':1,'completion_tokens':2},['stop'],True),
    ({'prompt_tokens':2,'completion_tokens':2},['length'],True),
    ({'prompt_tokens':2,'completion_tokens':2},['stop','stop'],True),
    ({'prompt_tokens':2,'completion_tokens':2},['stop'],False),
])
def test_partial_or_misaccounted_response_is_not_success(usage,reasons,done):
    with pytest.raises(ValueError):
        validate_completion({'prompt':[1,2], 'max_tokens':8}, [3,4],usage,reasons,done)


def snapshot(at,resident,paused):
    return dict(scheduler=dict(at=at,owned_blocks=resident*2,free_blocks=40-resident*2,
        gpu_capacity={'requests':{str(i):{'owned_pages':2} for i in range(resident)}},
        request_swaps={'requests':{str(i):{'phase':'paused'} for i in range(paused)},'recent_transitions':[]}))


def test_natural_completion_is_not_counted_as_pressure_contraction():
    responses=[dict(started=100.,seconds=5. if i<8 else 20.,status='completed') for i in range(16)]
    result=analyze_timeline([snapshot(101,16,0),snapshot(106,8,0)],responses)
    assert result['changes'][-1]['completed']==8
    assert result['pressure_while_more_than_eight_pending']==[]


def test_pressure_contraction_retains_unfinished_request_count():
    responses=[dict(started=100.,seconds=20.,status='completed') for _ in range(16)]
    result=analyze_timeline([snapshot(101,16,0),snapshot(106,8,8)],responses)
    row,=result['pressure_while_more_than_eight_pending']
    assert row['resident']==8 and row['paused']==8 and row['pending_http']==16
