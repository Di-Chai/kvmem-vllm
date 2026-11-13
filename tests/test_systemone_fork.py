"""Scheduler ownership and API tests; GPU qualification checks actual bytes."""
from types import SimpleNamespace as NS
import pytest

from vllm_kvmem.serving.systemone_fork import ForkPagePool
from vllm_kvmem.serving.window_manager import WorkingSetManager


class Pool:
    def __init__(self, count):
        self.blocks = [NS(block_id=i+1, ref_cnt=0, is_null=False) for i in range(count)]
    def get_num_free_blocks(self):
        return sum(b.ref_cnt == 0 for b in self.blocks)
    def get_new_blocks(self, n):
        result = [b for b in self.blocks if b.ref_cnt == 0][:n]
        assert len(result) == n
        for b in result: b.ref_cnt = 1
        return result
    def touch(self, blocks):
        for b in blocks: b.ref_cnt += 1
    def free_blocks(self, blocks):
        for b in blocks:
            assert b.ref_cnt > 0
            b.ref_cnt -= 1


def setup(count=300):
    pool = Pool(count)
    attention = object.__new__(WorkingSetManager)
    attention.native = NS(block_size=64, block_pool=pool, req_to_blocks={})
    attention.capacity = 4096
    recurrent = NS(block_size=16384, block_pool=pool, req_to_blocks={})
    manager = NS(block_pool=pool, groups=(attention,recurrent), static_blocks={1:2},
                 max_model_len=16384, _demand=lambda _:66)
    config = NS(max_concurrent_batches=1,scheduler_config=NS(async_scheduling=False,max_num_seqs=2),
                model_config=NS(enforce_eager=True),cache_config=NS(enable_prefix_caching=False,mamba_cache_mode='none'),
                speculative_config=NS(method='mtp'))
    scheduler = NS(vllm_config=config,kv_cache_manager=manager,_bounded_target=True,
                   async_refresh=False,async_restore=False,max_num_running_reqs=2)
    return scheduler,pool,attention


def acquire(ids, operation='op'):
    return dict(action='acquire',operation=operation,parent='parent',groups=[(0,ids)])


def test_reserves_cow_and_foreground_slot(monkeypatch):
    monkeypatch.delenv('KVMEM_RUNNER_PROBE_DIR',raising=False)
    scheduler,pool,attention=setup()
    fork=ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    assert fork.capacity == 4640
    assert list(map(len,fork.rows)) == [73,2]
    assert scheduler.max_num_running_reqs == scheduler.kv_cache_manager.max_authorities == 1
    assert pool.get_num_free_blocks() == 225
    ids=[b.block_id for row in fork.rows for b in row]
    assert len(ids) == len(set(ids))


def test_admission_failure_allocates_nothing():
    scheduler,pool,_=setup(140)
    with pytest.raises(ValueError,match='Insufficient'):
        ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    assert pool.get_num_free_blocks() == 140


@pytest.mark.parametrize('field,value',[('async_refresh',True),('async_restore',True),('_bounded_target',False)])
def test_rejects_unqualified_modes(field,value):
    scheduler,pool,_=setup();setattr(scheduler,field,value)
    with pytest.raises(ValueError,match='requires'):
        ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    assert pool.get_num_free_blocks() == 300


def test_lease_survives_parent_free_then_releases_exactly_once():
    scheduler,pool,attention=setup()
    fork=ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    blocks=pool.get_new_blocks(3);attention.req_to_blocks['parent']=blocks
    fork.consume([acquire([b.block_id for b in blocks])])
    assert all(b.ref_cnt == 2 for b in blocks)
    pool.free_blocks(attention.req_to_blocks.pop('parent'))
    assert all(b.ref_cnt == 1 for b in blocks)
    assert not any(b in pool.get_new_blocks(3) for b in blocks)
    fork.consume([dict(action='release',operation='op')])
    assert all(b.ref_cnt == 0 for b in blocks)
    assert not fork.leases and fork.acquired == fork.released == 1
    with pytest.raises(RuntimeError,match='Unknown'):
        fork.consume([dict(action='release',operation='op')])


@pytest.mark.parametrize('kind',['foreign','duplicate','recurrent','overlap'])
def test_invalid_lease_does_not_mutate_refs(kind):
    scheduler,pool,attention=setup()
    fork=ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    blocks=pool.get_new_blocks(2);attention.req_to_blocks['parent']=blocks
    ids=[b.block_id for b in blocks]; notice=acquire(ids)
    if kind=='foreign':notice['groups']=[(0,[99999])]
    if kind=='duplicate':notice['groups']=[(0,[ids[0],ids[0]])]
    if kind=='recurrent':notice['groups']=[(1,ids)]
    if kind=='overlap':fork.consume([notice])
    before=[b.ref_cnt for b in pool.blocks]
    with pytest.raises(RuntimeError):fork.consume([notice])
    assert before == [b.ref_cnt for b in pool.blocks]


def test_zero_shared_pages_still_has_balanced_lifecycle():
    scheduler,pool,attention=setup()
    fork=ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    fork.consume([acquire([]),dict(action='release',operation='op')])
    assert not fork.leases and fork.acquired == fork.released == 1


def test_projection_excludes_private_hidden_metadata_and_archive(monkeypatch):
    import torch
    from test_query_replay_batch import batch,ExecuteState
    from vllm_kvmem.serving.query_replay_batch import merge_query_batch
    from vllm_kvmem.adapters.capture import CapturedLayer
    from vllm_kvmem.serving.systemone_fork import ForkLane as ForkDecisions
    public=batch(('parent',),(2,),(1,),(1,))
    private=batch(('child',),(16,),(0,),None)
    combined=merge_query_batch(public,private)
    hidden=torch.arange(54).reshape(18,3)
    fork=object.__new__(ForkDecisions)
    fork.runner=NS(_kvmem_rebuild_public_metadata=lambda b: ({'public':True},{'public':True}))
    fork.frame=dict(public=public,combined=combined,end=16)
    fork.active=dict(reads=[dict(tokens=list(range(32)))],qindex=0,operation='op',mixed_steps=0,public_tokens=0,consumed=0)
    fork.emit=lambda *args,**kwargs:None
    result=fork.complete(ExecuteState(combined,hidden,[hidden+1],{'mixed':True},{'mixed':True}))
    assert result.input_batch is public
    assert result.hidden_states.shape == (2,3)
    assert torch.equal(result.hidden_states,hidden[:2])
    assert result.attn_metadata == result.slot_mappings_by_layer == {'public':True}
    assert result.num_spec_tokens_to_schedule == 1
    assert fork.frame is None and fork.active['consumed'] == 16
    c=CapturedLayer('layer',torch.arange(18),hidden,hidden,hidden)
    projected=fork.project_capture([c])[0]
    assert projected.positions.tolist() == [0,1]
    assert len(projected.query) == len(projected.key) == len(projected.value) == 2
    assert len(c.query) == 18


@pytest.mark.parametrize('terminal,refresh,expected',[(True,False,'finish'),(False,True,'detach'),(False,False,None)])
def test_terminal_and_refresh_fences_run_before_native_execution(terminal,refresh,expected):
    from vllm_kvmem.serving.systemone_fork import ForkLane as ForkDecisions
    fork=object.__new__(ForkDecisions)
    fork.active=dict(binding=NS(request_id='parent'))
    fork.expire=lambda:None
    calls=[];fork.finish=lambda **kw:calls.append('finish');fork.detach=lambda:calls.append('detach')
    output=NS(finished_req_ids={'parent'} if terminal else set(),kv_connector_metadata=NS(cancelled=set(),commands=[NS(request_id='parent')] if refresh else [],logical_prefill_groups={}))
    fork.before_step(output)
    assert calls == ([] if expected is None else [expected])


def test_refresh_injection_requires_audit_before_any_allocation(monkeypatch):
    monkeypatch.setenv('KVMEM_SYSTEMONE_VERIFY_REFRESH','1')
    monkeypatch.delenv('KVMEM_VERIFY_SESSION_RESTORE',raising=False)
    scheduler,pool,_=setup()
    with pytest.raises(ValueError,match='verification-only'):
        ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    assert pool.get_num_free_blocks()==300


def test_refresh_probe_is_once_per_live_shared_lease(monkeypatch):
    monkeypatch.setenv('KVMEM_SYSTEMONE_VERIFY_REFRESH','1')
    monkeypatch.setenv('KVMEM_VERIFY_SESSION_RESTORE','1')
    scheduler,pool,attention=setup()
    fork=ForkPagePool(scheduler,NS(select_budget=2048,gen_budget=2048))
    blocks=pool.get_new_blocks(2);attention.req_to_blocks['parent']=blocks
    fork.consume([acquire([b.block_id for b in blocks])])
    assert not fork.take_refresh_probe('other')
    assert fork.take_refresh_probe('parent')
    assert not fork.take_refresh_probe('parent')
    fork.consume([dict(action='release',operation='op')])
    assert not fork.take_refresh_probe('parent')
