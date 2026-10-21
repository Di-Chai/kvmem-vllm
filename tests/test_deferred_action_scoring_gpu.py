"""Real original scorers over frozen FP8 K/Q/table/scale inputs.

This qualifies the private fallback, not the separate publication queue's KV
lease. Real selected-layer geometry is 3 layers, 24 Q / 4 KV heads, dim 256.
"""
from dataclasses import replace
import os

import pytest
import torch

from vllm_kvmem.action_scoring import page_attention
from vllm_kvmem.deferred_action_scoring import (
    FrozenLayerInput, FrozenOwnerInput, FrozenScoreLimits, FrozenScorePool)

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit frozen fallback CUDA qualification')


def fixture(*, fused, q_dtype=torch.bfloat16):
    torch.manual_seed(731)
    device = torch.device('cuda', torch.cuda.current_device())
    layers = []
    queries = []
    for i in range(3):
        key = (torch.randn((23,4,16,512), device=device) * .7).to(torch.float8_e4m3fn)
        # Native BHN(2D) layout with a nontrivial token stride and split K view.
        key = key.transpose(1,2)[..., :256]
        angle = torch.arange(320, device=device)[:,None] * torch.linspace(.0001,.02,32,device=device)[None,:]
        trig = torch.cat((angle.cos(), angle.sin()),-1)
        layers.append(FrozenLayerInput(str(i), key, trig,
            torch.tensor([.7,1.3,.125][i],device=device),
            torch.tensor([.02,.1,.007][i],device=device), 256**-.5,q_dtype,i%2))
        queries.append(torch.randn((32,24,256),device=device,dtype=torch.bfloat16)[::2])
    owners = []
    for owner_index,length in enumerate((259,137)):
        starts = tuple(range(0,length,32))
        counts = tuple(min(32,length-x) for x in starts)
        rows = (9,2,6) if owner_index == 0 else (3,11,5)
        positions = (0,127,258) if owner_index == 0 else (3,88,136)
        tables = tuple(torch.randperm(23,device=device,dtype=torch.int32) for _ in range(2))
        masses = torch.full((len(rows),len(starts)), float('nan'),pin_memory=True)
        owners.append(FrozenOwnerInput(str(owner_index),tuple(queries),rows,positions,starts,counts,
                                      length,tables,masses))
    limits = FrozenScoreLimits(layers=3,groups=2,heads=24,dim=256,page_tokens=16,
                               block_tokens=32,max_rows=6,max_owners=2,max_segments=10,max_length=320)
    pool = FrozenScorePool(limits, device,fused=fused,max_bytes=64<<20,max_scratch_bytes=2<<30)
    return pool,tuple(layers),tuple(owners)


def oracle(layers, owners, fused):
    expected = []
    for owner in owners:
        total = None
        for index,layer in enumerate(layers):
            mass = page_attention(owner.queries[index][list(owner.query_rows)],
                torch.tensor(owner.positions,device=layer.key.device),layer.key,owner.tables[layer.group],
                layer.cos_sin,layer.key_scale,torch.tensor(owner.starts,device=layer.key.device,dtype=torch.int32),
                torch.tensor(owner.counts,device=layer.key.device,dtype=torch.int32),owner.length,
                scale=layer.scale,q_dtype=layer.q_dtype,q_scale=layer.q_scale,
                block_tokens=32,fused=fused,complete_partition=True)
            total = mass if total is None else total + mass
        expected.append((total/len(layers)).cpu())
    return expected


def seal(pool, slot):
    # Native.detach_validation records exactly this event after flags D2H.
    slot.host_flags.fill_(1)
    slot.event.record()
    pool.seal(slot)


@pytest.mark.parametrize('fused', [False,True])
@pytest.mark.parametrize('q_dtype', [torch.bfloat16,torch.float16,torch.float8_e4m3fn])
def test_original_scorer_exact_after_all_overwriteable_inputs_change(fused,q_dtype):
    pool,layers,owners = fixture(fused=fused,q_dtype=q_dtype)
    expected = oracle(layers,owners,fused)
    slot = pool.freeze(layers,owners,packet_lease=object())
    assert slot is not None
    seal(pool,slot)
    # Same producer-stream ordering: the freeze precedes the next forward's
    # source updates. Original K prefix and immutable RoPE remain leased.
    for query in owners[0].queries: query.fill_(100)
    for owner in owners:
        for table in owner.tables: table.fill_(0)
    for layer in layers:
        layer.key_scale.fill_(25)
        layer.q_scale.fill_(50)
    deltas = pool.fallback(slot)
    assert deltas == {('fused' if fused else 'reference'):6}
    for owner,want in zip(owners,expected):
        assert torch.equal(owner.masses,want)
        torch.testing.assert_close(owner.masses.sum(1),torch.ones(3),rtol=0,atol=2e-6)
    pool.release(slot)
    assert pool.snapshot()['active_slots'] == 0


def test_two_generations_reuse_stable_buffers_without_old_rows_flags_or_pool_growth():
    pool,layers,owners = fixture(fused=True)
    pointers = [(s.q.data_ptr(),s.tables.data_ptr(),s.scales.data_ptr(),s.host_flags.data_ptr()) for s in pool.storages]
    expected = oracle(layers,owners,True)
    slots = [pool.freeze(layers,owners,packet_lease=object()) for _ in range(2)]
    assert pool.freeze(layers,owners,packet_lease=object()) is None
    for slot in slots:
        assert slot.host_flags.tolist() == [-1,-1,-1]
        seal(pool,slot)
        pool.fallback(slot)
        for owner,want in zip(owners,expected): assert torch.equal(owner.masses,want)
        pool.release(slot)
    for q in owners[0].queries: q.mul_(.5)
    smaller = (replace(owners[0],query_rows=(1,),positions=(258,),masses=torch.empty((1,9),pin_memory=True)),)
    expected = oracle(layers,smaller,True)
    slot = pool.freeze(layers,smaller,packet_lease=object())
    assert slot.host_flags.tolist() == [-1,-1,-1]
    seal(pool,slot); pool.fallback(slot)
    assert torch.equal(smaller[0].masses,expected[0])
    pool.release(slot)
    assert pointers == [(s.q.data_ptr(),s.tables.data_ptr(),s.scales.data_ptr(),s.host_flags.data_ptr()) for s in pool.storages]
    assert pool.snapshot()['allocated_slots'] == 2


def test_fallback_waits_its_source_event_and_not_unrelated_producer_tail():
    pool,layers,owners = fixture(fused=True)
    expected = oracle(layers,owners,True) # Compile exact scorers before timing ordering.
    slot = pool.freeze(layers,owners,packet_lease=object())
    seal(pool,slot)
    torch.cuda._sleep(500_000_000)
    tail = torch.cuda.Event(); tail.record()
    pool.fallback(slot)
    assert not tail.query(), 'Fallback unexpectedly waited for later producer work'
    for owner,want in zip(owners,expected): assert torch.equal(owner.masses,want)
    pool.release(slot)
    tail.synchronize()


def test_real_enqueued_fallback_failure_never_releases_pending_generation():
    pool,layers,owners = fixture(fused=True)
    slot = pool.freeze(layers,owners,packet_lease=object())
    seal(pool,slot)
    original = page_attention
    calls=[]
    def failed(*args,**kwargs):
        result=original(*args,**kwargs)
        calls.append(result)
        raise RuntimeError('injected after scorer kernels')
    pool._scorer=failed
    with pytest.raises(RuntimeError,match='injected'): pool.fallback(slot)
    assert pool.snapshot()['quarantined'] and pool.snapshot()['active_slots']==1
    with pytest.raises(RuntimeError,match='quarantined'): pool.release(slot)
    assert slot.packet_lease is not None and slot.layers
    # Teardown drains real submitted work; it does not retry or release poison.
    pool.stream.synchronize()


def test_background_fallback_can_write_frontend_inference_tensors():
    from concurrent.futures import ThreadPoolExecutor
    with torch.inference_mode():
        pool,layers,owners=fixture(fused=True,q_dtype=torch.float8_e4m3fn)
        expected=oracle(layers,owners,True)
        slot=pool.freeze(layers,owners,packet_lease=object())
        assert slot.storage.q.is_inference() and owners[0].masses.is_inference()
        seal(pool,slot)
    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(pool.fallback,slot).result(timeout=30)
    for owner,want in zip(owners,expected): assert torch.equal(owner.masses,want)
    pool.release(slot)
