"""Multi-reader leases, atomic groups and independent native batch coordinates."""
from types import SimpleNamespace as NS
import pytest
from test_systemone_fork import setup, acquire
from vllm_kvmem.systemone_fork import ForkPagePool, ForkDecisions


@pytest.mark.parametrize('branches', [2, 4])
def test_shared_page_references_survive_independent_retirement(monkeypatch, branches):
    monkeypatch.setenv('KVMEM_SYSTEMONE_BRANCHES', str(branches))
    scheduler, pool, attention = setup(1000)
    scheduler.vllm_config.scheduler_config.max_num_seqs = branches + 1
    fork = ForkPagePool(scheduler, NS(select_budget=2048, gen_budget=2048))
    ids = [b.block_id for rows in fork.lane_rows for row in rows for b in row]
    assert len(ids) == len(set(ids)) == branches * 75
    blocks = pool.get_new_blocks(2)
    attention.req_to_blocks['parent'] = blocks
    for i in range(branches):
        fork.consume([acquire([b.block_id for b in blocks], 'op'+str(i))])
    assert all(b.ref_cnt == branches + 1 for b in blocks)
    fork.consume([dict(action='release', operation='op0')])
    assert all(b.ref_cnt == branches for b in blocks)
    pool.free_blocks(attention.req_to_blocks.pop('parent'))
    assert all(b.ref_cnt == branches-1 for b in blocks)
    for i in reversed(range(1, branches)):
        fork.consume([dict(action='release', operation='op'+str(i))])
    assert all(b.ref_cnt == 0 for b in blocks)
    assert fork.acquired == fork.released == branches and not fork.lease_parents


def test_multi_reservation_is_checked_before_any_allocation(monkeypatch):
    monkeypatch.setenv('KVMEM_SYSTEMONE_BRANCHES', '4')
    scheduler, pool, _ = setup(365)
    scheduler.vllm_config.scheduler_config.max_num_seqs = 5
    with pytest.raises(ValueError, match='Insufficient'):
        ForkPagePool(scheduler, NS(select_budget=2048, gen_budget=2048))
    assert pool.get_num_free_blocks() == 365


def coordinator(monkeypatch, size=4):
    import torch
    from test_systemone_active import queue
    q, binding = queue()
    q.runner.model_config = NS(max_model_len=16384)
    q.runner.device = 'cpu'
    root = object.__new__(ForkDecisions)
    root.runner, root.pending = q.runner, {}
    root.emit = root.memory = lambda *a, **kw: None
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    root.lanes = []
    for _ in range(size):
        lane = NS(active=None, pending={})
        def start(job, public, index, lane=lane):
            lane.active = job
            job['history'] = public.num_computed_tokens_np[index]
            return True
        lane.start = start
        lane.expire = lambda: None
        def finish(error=None, lane=lane):
            lane.active['result'] = {'error': error}
            lane.active = None
        lane.finish = finish
        root.lanes.append(lane)
    return root, binding


def submit(root, operation, size=4, **extra):
    reference = dict(namespace=['tenant', 'parent'], fork_group=dict(id='group', size=size), **extra)
    return root.rpc('submit', operation, reference, [dict(tokens=[1, 2], labels=[3])])


@pytest.mark.parametrize('size', [2, 4])
def test_group_admission_is_atomic_at_one_history(monkeypatch, size):
    root, _ = coordinator(monkeypatch, size)
    public = NS(req_ids=['native-parent'], num_computed_tokens_np=[50])
    for i in range(size-1):
        assert submit(root, str(i), size)['status'] == 'pending'
    root.admit(public)
    assert all(lane.active is None for lane in root.lanes)
    submit(root, str(size-1), size)
    root.admit(public)
    assert all(lane.active['history'] == 50 for lane in root.lanes)
    assert 'error' in submit(root, 'late', size)


def test_one_stale_group_member_rejects_all_before_ownership(monkeypatch):
    root, _ = coordinator(monkeypatch, 2)
    submit(root, 'a', 2)
    submit(root, 'b', 2, history_end=49)
    root.admit(NS(req_ids=['native-parent'], num_computed_tokens_np=[50]))
    assert all(lane.active is None for lane in root.lanes)
    assert all('error' in job['result'] for job in root.pending.values())


def test_cancelling_waiting_group_rejects_peer_without_allocations(monkeypatch):
    root, _ = coordinator(monkeypatch, 4)
    submit(root, 'a'); submit(root, 'b')
    root.rpc('cancel', 'a')
    assert 'error' in root.rpc('poll', 'b')
    assert all(lane.active is None for lane in root.lanes)


def test_finishing_or_cancelling_one_admitted_member_preserves_peers(monkeypatch):
    root, _ = coordinator(monkeypatch, 4)
    for i in range(4): submit(root, str(i))
    root.admit(NS(req_ids=['native-parent'], num_computed_tokens_np=[50]))
    root.lanes[0].active['result'] = {'status': 'done'}
    root.lanes[0].active = None
    root.rpc('cancel', '0')  # Ordinary HTTP finally cleanup must not cancel peers.
    assert all(lane.active['result']['status'] == 'pending' for lane in root.lanes[1:])
    root.rpc('cancel', '1')
    assert root.lanes[1].active is None
    assert all(lane.active['result']['status'] == 'pending' for lane in root.lanes[2:])


def test_four_private_batches_have_independent_logits_positions_and_slots():
    from test_query_replay_batch import batch
    from vllm_kvmem.query_replay_batch import merge_query_batch
    public = batch(('parent',), (2,), (4,), (1,))
    merged = public
    for i, count in enumerate((16, 32, 24, 8)):
        merged = merge_query_batch(merged, batch((str(i),), (count,), (i,), None))
    assert merged.req_ids == ['parent', '0', '1', '2', '3']
    assert merged.idx_mapping_np.tolist() == [4, 0, 1, 2, 3]
    assert merged.query_start_loc_np.tolist() == [0, 2, 18, 50, 74, 82]
    assert merged.logits_indices.tolist() == [0, 1, 17, 49, 73, 81]
    assert merged.num_draft_tokens_per_req.tolist() == [1, 0, 0, 0, 0]
    assert public.num_tokens == 2 and public.num_reqs == 1
