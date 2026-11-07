from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.request_swap import FencedPageSwap, SwapPhase


class Pool:
    def __init__(self, count):
        self.blocks = [NS(block_id=i + 1, ref_cnt=0, is_null=False) for i in range(count)]

    def get_num_free_blocks(self):
        return sum(b.ref_cnt == 0 for b in self.blocks)

    def get_new_blocks(self, count):
        if count > self.get_num_free_blocks():
            raise BufferError('pool full')
        rows = [b for b in self.blocks if b.ref_cnt == 0][:count]
        for b in rows:
            b.ref_cnt = 1
        return rows

    def free_blocks(self, blocks):
        for b in blocks:
            assert b.ref_cnt == 1
            b.ref_cnt = 0


class Event:
    done = False

    def query(self):
        return self.done


def fixture(count=12):
    pool = Pool(count)
    groups = [NS(block_pool=pool, req_to_blocks={}, num_cached_block={'a': 0}),
              NS(block_pool=pool, req_to_blocks={}, num_cached_block={'a': 0}, mamba_cache_mode='none')]
    groups[0].offsets = {'a': 128}
    groups[0].pending = {}
    groups[0].imports = {}
    groups[0].req_to_blocks['a'] = pool.get_new_blocks(3)
    groups[1].req_to_blocks['a'] = pool.get_new_blocks(2)
    groups[0].req_to_blocks['peer'] = pool.get_new_blocks(1)
    request = NS(request_id='a', num_computed_tokens=160, num_in_flight_tokens=0,
                 next_decode_eligible_step=17, is_finished=lambda: False)
    manager = NS(block_pool=pool, coordinator=NS(single_type_managers=groups))
    return FencedPageSwap(manager), request, groups, pool


def test_pause_waits_for_checkpoint_then_rebinds_exact_groups_without_changing_window():
    swap, request, groups, pool = fixture()
    peer = groups[0].req_to_blocks['peer'][0]
    ticket = swap.prepare(request)
    assert request.next_decode_eligible_step == 2**62
    event = Event()
    assert not swap.checkpoint_complete(ticket, event)
    assert pool.get_num_free_blocks() == 6
    event.done = True
    assert swap.checkpoint_complete(ticket, event)
    assert pool.get_num_free_blocks() == 11 and ticket.phase is SwapPhase.PAUSED
    reused = pool.get_new_blocks(3)
    assert [b.block_id for b in reused] == list(ticket.original_blocks[0])
    new = swap.prepare_restore(ticket)
    assert list(map(len, new)) == [3, 2]
    assert new != ticket.original_blocks
    swap.restore_dispatched(ticket)
    event.done = False
    assert not swap.restore_complete(ticket, event)
    assert request.next_decode_eligible_step == 2**62
    event.done = True
    assert swap.restore_complete(ticket, event)
    assert request.next_decode_eligible_step == 17
    assert request.num_computed_tokens == 160 and groups[0].offsets == {'a': 128}
    assert groups[0].req_to_blocks['peer'] == [peer] and peer.ref_cnt == 1
    assert all(group.num_cached_block == {'a': 0} for group in groups)


def test_restore_capacity_failure_and_unpublished_rollback_are_atomic():
    swap, request, groups, pool = fixture(7)
    ticket = swap.prepare(request)
    event = Event(); event.done = True
    swap.checkpoint_complete(ticket, event)
    held = pool.get_new_blocks(2)
    assert swap.prepare_restore(ticket) is None
    assert all('a' not in group.req_to_blocks for group in groups)
    pool.free_blocks(held)
    free = pool.get_num_free_blocks()
    assert swap.prepare_restore(ticket)
    swap.abort_restore(ticket)
    assert pool.get_num_free_blocks() == free and ticket.phase is SwapPhase.PAUSED
    swap.prepare_restore(ticket); swap.restore_dispatched(ticket)
    with pytest.raises(ValueError, match='must drain'):
        swap.abort_restore(ticket)


@pytest.mark.parametrize('corruption', ['history', 'inflight', 'offset', 'pages', 'references'])
def test_stale_capture_never_releases_pages(corruption):
    swap, request, groups, pool = fixture()
    ticket = swap.prepare(request)
    if corruption == 'history': request.num_computed_tokens += 1
    if corruption == 'inflight': request.num_in_flight_tokens = 1
    if corruption == 'offset': groups[0].offsets['a'] += 1
    if corruption == 'pages': groups[0].req_to_blocks['a'].reverse()
    if corruption == 'references': groups[0].req_to_blocks['a'][0].ref_cnt = 2
    event = Event(); event.done = True
    free = pool.get_num_free_blocks()
    with pytest.raises((RuntimeError, ValueError)):
        swap.checkpoint_complete(ticket, event)
    assert pool.get_num_free_blocks() == free


def test_duplicate_and_foreign_ack_cannot_release_reused_pages():
    swap, request, groups, pool = fixture()
    ticket = swap.prepare(request)
    event = Event(); event.done = True
    swap.checkpoint_complete(ticket, event)
    held = pool.get_new_blocks(5)
    with pytest.raises(ValueError, match='already completed'):
        swap.checkpoint_complete(ticket, event)
    assert all(b.ref_cnt == 1 for b in held)
    pool.free_blocks(held)
    swap.prepare_restore(ticket); swap.restore_dispatched(ticket); swap.restore_complete(ticket, event)
    second = swap.prepare(request)
    with pytest.raises(ValueError):
        swap.restore_complete(ticket, event)
    assert swap.entries['a'] is second


def test_cancellation_only_discards_after_checkpoint_fence():
    swap, request, groups, pool = fixture()
    ticket = swap.prepare(request)
    request.is_finished = lambda: True
    with pytest.raises(ValueError, match='fully paused'):
        swap.discard_paused(ticket)
    event = Event(); event.done = True
    swap.checkpoint_complete(ticket, event)
    swap.discard_paused(ticket)
    assert not swap.entries and pool.get_num_free_blocks() == 11


def test_failed_capture_can_unfreeze_only_after_its_reads_drain():
    swap, request, groups, pool = fixture()
    ticket = swap.prepare(request)
    event = Event()
    assert not swap.abort_capture(ticket, event)
    assert request.next_decode_eligible_step == 2**62
    event.done = True
    assert swap.abort_capture(ticket, event)
    assert request.next_decode_eligible_step == 17
    assert swap._ids(swap._rows('a')) == ticket.original_blocks
    assert pool.get_num_free_blocks() == 6 and not swap.entries


@pytest.mark.parametrize('kind', ['inflight', 'finished', 'pending', 'import', 'shared', 'null', 'cross_group'])
def test_prepare_refuses_unsafe_boundaries_without_freezing_request(kind):
    swap, request, groups, pool = fixture()
    if kind == 'inflight': request.num_in_flight_tokens = 1
    if kind == 'finished': request.is_finished = lambda: True
    if kind == 'pending': groups[0].pending['a'] = object()
    if kind == 'import': groups[0].imports['a'] = object()
    if kind == 'shared': groups[0].req_to_blocks['a'][0].ref_cnt = 2
    if kind == 'null': groups[0].req_to_blocks['a'][0].is_null = True
    if kind == 'cross_group': groups[1].req_to_blocks['a'][0] = groups[0].req_to_blocks['a'][0]
    with pytest.raises(ValueError): swap.prepare(request)
    assert request.next_decode_eligible_step == 17 and not swap.entries
