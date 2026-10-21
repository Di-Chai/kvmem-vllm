"""CPU eligibility oracles; no scheduler, model, score or archive mutations."""
from dataclasses import FrozenInstanceError, replace

import pytest

from vllm_kvmem.logical_prefill import (LogicalPrefillEligibility,LogicalPrefillGroup,
    LogicalPrefillSpan,plan_logical_prefill,MAX_LOGICAL_PREFILL_CHUNKS)
from vllm_kvmem.request_plan import RequestPlan


def policy(**changes):
    return replace(RequestPlan(None,reselect='off',block_tokens=32,gen_budget=16384),**changes)


def eligibility(**changes):
    return replace(LogicalPrefillEligibility(('a',),(2048,2048),4096),**changes)


def build(*,plan=None,gates=None,**changes):
    args = dict(request_id='a',owner_epoch=(101,7,3,0,('restore',2)),scheduler_step=17,
        history_begin=8192,offset=0,prompt_tokens=262144,capacity=49152)
    args.update(changes)
    return plan_logical_prefill(plan or policy(),eligibility=gates or eligibility(),**args)


def test_two_original_admissions_not_one_larger_admission(monkeypatch):
    calls = []
    original_admit,original_action = RequestPlan.admit,RequestPlan.action
    def admit(self,*args):
        calls.append(('admit',args))
        return original_admit(self,*args)
    def action(self,*args):
        calls.append(('action',args))
        return original_action(self,*args)
    monkeypatch.setattr(RequestPlan,'admit',admit)
    monkeypatch.setattr(RequestPlan,'action',action)
    group,reason = build(coalescing_authorization_id=33)
    assert reason is None and group.count == 4096
    assert calls == [('action',(8192,262144,0,49152)),('admit',(8192,2048,262144,0,49152)),
                     ('action',(10240,262144,0,49152)),('admit',(10240,2048,262144,0,49152))]
    assert group.coalescing_authorization_id == 33
    assert group.spans == (LogicalPrefillSpan(0,8192,8192,2048,0),
                           LogicalPrefillSpan(1,10240,10240,2048,2048))
    assert group.history_begin == group.local_begin == 8192 and group.history_end == 12288
    with pytest.raises(FrozenInstanceError):
        group.offset = 3
    with pytest.raises(FrozenInstanceError):
        group.spans[0].count = 4096


def test_pressure_phase_keeps_original_non_2k_aligned_history_and_dense_coordinates():
    group,reason = build(history_begin=40992,offset=8192)
    assert reason is None
    assert group.history_begin%2048 == 32
    assert [(span.history_begin,span.local_begin,span.physical_offset) for span in group.spans] == [
        (40992,32800,0),(43040,34848,2048)]


@pytest.mark.parametrize('gates,reason',[
    ({'scheduled_request_ids':()},'single_owner'),
    ({'scheduled_request_ids':('a','b')},'single_owner'),
    ({'scheduled_request_ids':('b',)},'single_owner'),
    ({'phase':'decode'},'phase'),
    ({'phase':'maintenance'},'phase'),
    ({'blocked_reasons':('restore',)},'blocked:restore'),
    ({'blocked_reasons':('publication',)},'blocked:publication'),
    ({'physical_token_budget':4095},'physical_budget'),
    ({'phase_budget':2048},'phase_budget'),
    ({'batch_budget':3072},'batch_budget'),
    ({'native_limits':(1952,2048)},'logical_admission'),
    ({'native_limits':(2048,1984)},'logical_admission'),
    ({'boundary_positions':(8192,)},'dynamic_boundary'),
    ({'boundary_positions':(9000,)},'dynamic_boundary'),
    ({'boundary_positions':(10240,)},'dynamic_boundary'),
    ({'terminal_history':8191},'terminal_prefix'),
    ({'terminal_history':10240},'terminal_prefix'),
    ({'terminal_history':12288},'terminal_prefix'),
])
def test_independent_existing_gates_are_not_bypassed_by_offline_receipt(gates,reason):
    assert build(gates=eligibility(**gates),coalescing_authorization_id=77) == (None,reason)


def test_budget_is_checked_without_charging_and_boundary_at_end_is_preserved():
    gates = eligibility(phase_budget=4096,batch_budget=4096,boundary_positions=(12288,))
    group,reason = build(gates=gates)
    assert reason is None and group.count == 4096
    assert gates.phase_budget == gates.batch_budget == 4096


@pytest.mark.parametrize('history,reason',[(47040,'action:pressure'),(48000,'logical_admission'),
                                          (49088,'action:pressure')])
def test_pressure_is_checked_at_each_original_boundary(history,reason):
    assert build(history_begin=history) == (None,reason)


def test_group_may_end_at_pressure_without_crossing_it():
    group,reason = build(history_begin=44992)
    assert reason is None and group.history_end == 49088
    assert group.request_plan.action(group.history_end,262144,0,49152) == 'pressure'


@pytest.mark.parametrize('query',[(8000,8300),(10000,11000),(10240,10272),(12000,14000)])
def test_query_replay_or_seed_anywhere_in_physical_range_declines(query):
    assert build(plan=policy(query=query,reselect='auto')) == (None,'query_replay')


def test_completed_query_before_group_does_not_disable_ordinary_suffix():
    group,reason = build(plan=policy(query=(100,200),reselect='auto'))
    assert reason is None and group.count == 4096


@pytest.mark.parametrize('point',[8192,9000,10240,12287])
def test_checkpoint_inside_or_at_start_declines_even_if_admission_would_finish(monkeypatch,point):
    monkeypatch.setattr(RequestPlan,'prefill_checkpoint',lambda self,prompt: point)
    assert build() == (None,'checkpoint')


def test_checkpoint_at_group_end_retains_exact_later_visit(monkeypatch):
    monkeypatch.setattr(RequestPlan,'prefill_checkpoint',lambda self,prompt: 12288)
    assert build()[0].history_end == 12288


@pytest.mark.parametrize('changes,reason',[
    ({'prompt_tokens':12288},'terminal_prefill'),
    ({'prompt_tokens':12000},'terminal_prefill'),
    ({'history_begin':262144},'terminal_prefill'),
    ({'history_begin':8193},'partial_logical_block'),
])
def test_tail_decode_and_partial_bridge_remain_original_path(changes,reason):
    if changes.get('history_begin',0)>49152:
        changes = {**changes,'offset':229376}
    assert build(**changes) == (None,reason)


def test_mtp_short_tail_is_not_accepted_as_two_full_logical_chunks(monkeypatch):
    # Separate MTP eligibility from the earlier generation/prefill checkpoint.
    monkeypatch.setattr(RequestPlan,'prefill_checkpoint',lambda self,prompt: None)
    monkeypatch.setattr(RequestPlan,'generation_checkpoint',lambda self,prompt: prompt)
    assert build(prompt_tokens=12291,gates=eligibility(native_prefill_lookahead=4)) == (None,'native_lookahead')
    assert build(prompt_tokens=12292,gates=eligibility(native_prefill_lookahead=4))[0].count == 4096


@pytest.mark.parametrize('retained,expected',[(0,(0,0)),(7,(7,0)),(2048,(2048,0)),
    (2049,(2048,1)),(4095,(2048,2047)),(4096,(2048,2048))])
def test_terminal_acceptance_is_per_original_child(retained,expected):
    group,_ = build()
    assert group.accepted_counts(retained) == expected


@pytest.mark.parametrize('retained',[-1,4097,True,1.5])
def test_invalid_terminal_prefix_is_not_silently_clamped(retained):
    with pytest.raises(ValueError):
        build()[0].accepted_counts(retained)


def test_owner_restore_policy_and_scheduler_identity_are_bound():
    group,_ = build()
    current = dict(request_id='a',owner_epoch=group.owner_epoch,scheduler_step=17,
        history_begin=8192,offset=0,prompt_tokens=262144,capacity=49152,request_plan=policy())
    assert group.matches(**current)
    for key,value in dict(request_id='b',owner_epoch=(102,7,3,0,('restore',2)),scheduler_step=18,
        history_begin=10240,offset=32,prompt_tokens=262145,capacity=49184,
        request_plan=policy(mandatory=((0,32,'system'),))).items():
        assert not group.matches(**{**current,key:value})
    assert not group.matches(**{**current,'owner_epoch':(101,7,3,0,('restore',3))})


@pytest.mark.parametrize('changes',[
    {'scheduled_request_ids':['a']},{'native_limits':[2048,2048]},
    {'native_limits':(2048,)},{'native_limits':(2048,False)},
    {'physical_token_budget':-1},{'phase_budget':True},{'batch_budget':3.5},
    {'native_prefill_lookahead':0},{'blocked_reasons':['restore']},
    {'boundary_positions':[0]},{'boundary_positions':(False,)},
])
def test_eligibility_snapshot_requires_bounded_immutable_values(changes):
    with pytest.raises(ValueError):
        eligibility(**changes)


@pytest.mark.parametrize('changes',[
    {'owner_epoch':[]},{'owner_epoch':('a',[])},{'owner_epoch':()},
    {'scheduler_step':True},{'coalescing_authorization_id':0},{'offset':9000},
])
def test_plan_identity_and_extents_are_validated(changes):
    with pytest.raises(ValueError):
        build(**changes)


def test_direct_group_construction_cannot_hide_bad_spans():
    group,_ = build()
    with pytest.raises(ValueError):
        replace(group,spans=list(group.spans))
    with pytest.raises(ValueError):
        replace(group,spans=(group.spans[1],group.spans[0]))
    with pytest.raises(ValueError):
        replace(group,spans=(group.spans[0],replace(group.spans[1],history_begin=10304)))
    with pytest.raises(ValueError,match='immutable RequestPlan'):
        replace(group,request_plan=policy(mandatory=[(0,32,'system')]))


def wide_gates(maximum=4,**changes):
    return eligibility(native_limits=(2048,)*maximum,physical_token_budget=2048*maximum,**changes)


@pytest.mark.parametrize('maximum',range(2,9))
def test_bounded_groups_preserve_every_original_admission_and_acceptance(maximum,monkeypatch):
    calls = []
    admit,action = RequestPlan.admit,RequestPlan.action
    def record_admit(self,*args):
        calls.append(('admit',args))
        return admit(self,*args)
    def record_action(self,*args):
        calls.append(('action',args))
        return action(self,*args)
    monkeypatch.setattr(RequestPlan,'admit',record_admit)
    monkeypatch.setattr(RequestPlan,'action',record_action)
    group,reason = build(gates=wide_gates(maximum))
    assert reason is None and group.count == maximum*2048 and len(group.spans)==maximum
    assert calls == [call for i in range(maximum) for call in (
        ('action',(8192+i*2048,262144,0,49152)),
        ('admit',(8192+i*2048,2048,262144,0,49152)))]
    assert [span.index for span in group.spans] == list(range(maximum))
    assert [span.physical_offset for span in group.spans] == list(range(0,group.count,2048))
    for accepted in [0,*range(1,group.count,2048),group.count]:
        counts = group.accepted_counts(accepted)
        assert len(counts)==maximum and sum(counts)==accepted
        assert counts == tuple(min(2048,max(0,accepted-i*2048)) for i in range(maximum))
    with pytest.raises(ValueError): group.accepted_counts(group.count+1)


@pytest.mark.parametrize('kind,expected',[
    ('physical_budget',3),('phase_budget',3),('batch_budget',3),('dynamic_boundary',3),
    ('partial_dynamic_boundary',2),('query_replay',3),('terminal_prefill',2),('terminal_prefix',2),
    ('native_lookahead',2),('checkpoint',3),('generation_checkpoint',3)])
def test_maximum_candidate_stops_at_boundary_and_retains_qualified_prefix(kind,expected,monkeypatch):
    gates,plan,args = wide_gates(8),policy(),{}
    if kind.endswith('_budget'):
        gates = replace(gates,**{'physical_token_budget' if kind=='physical_budget' else kind:6144})
    elif kind == 'dynamic_boundary': gates = replace(gates,boundary_positions=(14336,))
    elif kind == 'partial_dynamic_boundary': gates = replace(gates,boundary_positions=(14335,))
    elif kind == 'query_replay': plan = policy(query=(14336,14368),reselect='auto')
    elif kind == 'terminal_prefill': args['prompt_tokens'] = 14336
    elif kind == 'terminal_prefix': gates = replace(gates,terminal_history=14336)
    elif kind == 'native_lookahead':
        args['prompt_tokens'] = 14339
        gates = replace(gates,native_prefill_lookahead=4)
        monkeypatch.setattr(RequestPlan,'prefill_checkpoint',lambda *a:None)
        monkeypatch.setattr(RequestPlan,'generation_checkpoint',lambda self,prompt:prompt)
    elif kind == 'checkpoint': monkeypatch.setattr(RequestPlan,'prefill_checkpoint',lambda *a:14336)
    else: monkeypatch.setattr(RequestPlan,'generation_checkpoint',lambda *a:14336)
    group,reason = build(plan=plan,gates=gates,**args)
    assert reason is None and group.count==2048*expected
    assert group.history_end == 8192+2048*expected


@pytest.mark.parametrize('stop',range(2,8))
@pytest.mark.parametrize('gate',['native','action','admit'])
def test_first_failed_child_is_never_skipped_to_find_a_later_eligible_span(stop,gate,monkeypatch):
    gates = wide_gates(8)
    calls = []
    original_action,original_admit = RequestPlan.action,RequestPlan.admit
    def action(self,history,*args):
        calls.append(history)
        if gate=='action' and history==8192+stop*2048: return 'pressure'
        return original_action(self,history,*args)
    def admit(self,history,*args):
        if gate=='admit' and history==8192+stop*2048: return 1024
        return original_admit(self,history,*args)
    monkeypatch.setattr(RequestPlan,'action',action)
    monkeypatch.setattr(RequestPlan,'admit',admit)
    if gate=='native': gates=replace(gates,native_limits=(2048,)*stop+(1024,)+(2048,)*(7-stop))
    group,reason = build(gates=gates)
    assert reason is None and len(group.spans)==stop
    assert calls == [8192+i*2048 for i in range(stop+1)]


def test_g16_pressure_cycle_has_four_plus_three_complete_logical_chunks():
    first,reason = build(history_begin=32800,gates=wide_gates(4))
    assert reason is None and first.count==8192 and first.history_end==40992
    second,reason = build(history_begin=first.history_end,gates=wide_gates(4))
    assert reason is None and second.count==6144 and second.history_end==47136
    assert build(history_begin=second.history_end,gates=wide_gates(4)) == (None,'logical_admission')
    assert policy().admit(47136,2048,262144,0,49152)==1952
    seven,reason = build(history_begin=32800,gates=wide_gates(8))
    assert reason is None and len(seven.spans)==7 and seven.history_end==47136
    assert policy().action(49088,262144,0,49152)=='pressure'


@pytest.mark.parametrize('maximum', [0,1,9,64])
def test_attempt_count_and_direct_group_size_are_bounded(maximum):
    with pytest.raises(ValueError): wide_gates(maximum)
    group,_=build(gates=wide_gates(8))
    with pytest.raises(ValueError): replace(group,spans=group.spans[:maximum] if maximum<2 else group.spans+(group.spans[-1],))


def test_direct_wider_group_checks_every_middle_span_and_exact_order():
    group,_ = build(gates=wide_gates(8))
    assert MAX_LOGICAL_PREFILL_CHUNKS==8
    for index in range(8):
        with pytest.raises(ValueError):
            replace(group,spans=(*group.spans[:index],replace(group.spans[index],history_begin=40960),*group.spans[index+1:]))
    with pytest.raises(ValueError): LogicalPrefillSpan(8,16384,16384,2048,16384)
