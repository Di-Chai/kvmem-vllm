"""Conservative per-owner coalescing keeps the global native sample budget."""
import pytest
from test_physical_prefill_scheduler import setup,reserve,request


@pytest.mark.parametrize('n,chunks',[(1,8),(2,7),(4,3),(6,2),(8,1)])
def test_multiowner_sampling_reservation(monkeypatch,tmp_path,n,chunks):
    s,r=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS','1024')
    s.prefill_physical_multi_owner=True;s._physical_native_rows=448;s.prefill_physical_max_chunks=8
    s.running=[r]+[request('p'+str(i),computed=8192,prompt=262144,output=0) for i in range(1,n)]
    s._prepare_logical_prefill()
    for owner in s.running:assert reserve(s,owner)==chunks*1024
    assert sum(len(g.spans)*32 for g in s._logical_prefill_groups.values())<=448


@pytest.mark.parametrize('kind',['query','decode','speculative'])
def test_nonhistorical_peer_keeps_original_path(monkeypatch,tmp_path,kind):
    s,r=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS','1024')
    s.prefill_physical_multi_owner=True;s._physical_native_rows=448;s.prefill_physical_max_chunks=8
    peer=request('peer',computed=8192,prompt=262144,output=0)
    if kind=='query':peer.kv_transfer_params={'kvmem_request_plan':{'kvmem_reselect':'force','kvmem_query_span':[8192,8224]}}
    if kind=='decode':peer.num_computed_tokens=peer.num_prompt_tokens
    if kind=='speculative':peer.spec_token_ids=[1]
    s.running.append(peer);s._prepare_logical_prefill()
    assert reserve(s,r)==1024 and not s._logical_prefill_groups


@pytest.mark.parametrize('kind',['publication','transaction','cadence'])
def test_native_blocked_running_peer_does_not_reserve_rows(monkeypatch,tmp_path,kind):
    from vllm.v1.request import RequestStatus
    s,r=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS','1024')
    s.prefill_physical_multi_owner=True;s._physical_native_rows=448;s.prefill_physical_max_chunks=8
    peer=request('peer',computed=8192,prompt=262144,output=0);peer.status=RequestStatus.RUNNING
    s.running.append(peer)
    if kind=='publication':s.connector.publication_blocked={'peer'}
    if kind=='transaction':s.transactions['peer']=object()
    if kind=='cadence':peer.next_decode_eligible_step=s.current_step+2
    s._prepare_logical_prefill()
    assert reserve(s,r)==8192 and set(s._logical_prefill_owners)=={'p'}
    # Stale publication feedback must never discount a waiting admission.
    s.running.remove(peer);peer.status=RequestStatus.WAITING;s.waiting=[peer]
    s._prepare_logical_prefill()
    assert reserve(s,r)==7168 and set(s._logical_prefill_owners)=={'p','peer'}


@pytest.mark.parametrize('owners,expected',[(1,[8]),(2,[7,7]),(3,[4,5,5]),
    (4,[3,3,4,4]),(5,[2,3,3,3,3]),(6,[2,2,2,2,3,3]),(8,[1,1,2,2,2,2,2,2])])
def test_remainder_reservations_fill_only_existing_native_rows(monkeypatch,tmp_path,owners,expected):
    s,r=setup(monkeypatch,tmp_path)
    monkeypatch.setenv('KVMEM_PREFILL_CHUNK_TOKENS','1024')
    s.prefill_physical_multi_owner=True;s.prefill_physical_fill_rows=True
    s._physical_native_rows=448;s.prefill_physical_max_chunks=8
    s.running=[r]+[request('p'+str(i),computed=8192,prompt=262144,output=0) for i in range(1,owners)]
    s._prepare_logical_prefill()
    tokens=[reserve(s,owner) for owner in s.running]
    assert sorted(n//1024 for n in tokens)==expected
    assert sum(tokens)//32 <= 448
    assert all(g.count==len(g.spans)*1024 for g in s._logical_prefill_groups.values())
    # A later iteration with a nonhistorical peer must discard old reservations.
    s.running[-1].num_computed_tokens=s.running[-1].num_prompt_tokens
    s._prepare_logical_prefill()
    assert not s._logical_owner_chunks and not s._logical_prefill_owners


def test_row_remainder_is_bounded_and_rotates_without_starvation():
    from vllm_kvmem.serving.multiowner_prefill import owner_chunk_allocations
    totals=[0]*6
    for step in range(6):
        allocation=owner_chunk_allocations(6,448,1024,8,rotation=step)
        assert sum(allocation)==14 and min(allocation)==2 and max(allocation)==3
        totals=[a+b for a,b in zip(totals,allocation)]
    assert len(set(totals))==1
    assert owner_chunk_allocations(16,448,1024,8)==(0,)*16
    assert sum(owner_chunk_allocations(3,448,2048,8))==7
