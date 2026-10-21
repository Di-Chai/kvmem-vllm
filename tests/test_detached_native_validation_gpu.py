"""Real DMA/event proof for detached native flags with one reused GPU bank.

These tests isolate ticket lifetime from attention math (covered by the native
FA2/XQA and deferred Action tests). No second GPU validation bank is allocated.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from threading import get_ident

import pytest
import torch

from vllm_kvmem.native_attention import NativeAttentionStats


pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit detached native CUDA qualification')


def bank_fixture():
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.mode, bank.disabled, bank.batch_reason = 'reuse', None, None
    bank.names = ('a', 'b', 'c')
    bank.rows = torch.empty(1, device='cuda', dtype=torch.int32)
    bank.control = torch.empty(1, device='cuda', dtype=torch.int32)
    bank.valid_flags = torch.empty(3, device='cuda', dtype=torch.int32)
    bank.host = {name: torch.empty(3, dtype=torch.int32, pin_memory=True)
                 for name in ('valid_flags', 'ranges', 'rows')}
    bank.max_rows, bank.upload_event = 256, None
    bank.counts, bank.validation_failures = Counter(), 0
    bank._detached_owner_thread = get_ident()
    bank.disarm()
    return bank


def frame_metadata(bank, *, owners=1):
    bank.plans = {f'owner-{i}': {'rows': [0, 31]} for i in range(owners)}
    bank.slots = {rid: (0, 2, 16) for rid in bank.plans}
    bank.reduced = set(bank.names)
    bank.validation_dirty = True
    bank.frame_success = Counter({'reuse:prefill': len(bank.names) * owners})
    bank.counts.update(bank.frame_success)


def host_flags():
    return torch.full((3,), -1, dtype=torch.int32, pin_memory=True)


def put_flags(bank, values):
    bank.valid_flags.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))


@pytest.mark.parametrize('nondefault_stream', [False, True])
def test_rearm_overwrites_one_gpu_bank_before_old_worker_resolution(nondefault_stream):
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        bank = bank_fixture()
        frame_metadata(bank, owners=2)
        put_flags(bank, (1, 0, 1))
        first = bank.detach_validation(host_flags(), torch.cuda.Event())
        # arm really clears the same GPU flags. The old D2H must precede it on
        # the producer stream, even though its CPU result has not been read.
        bank.arm({})
        frame_metadata(bank)
        put_flags(bank, (1, 1, 1))
        second = bank.detach_validation(host_flags(), torch.cuda.Event())
        bank.arm({})
        frame_metadata(bank)
        put_flags(bank, (0, 0, 0))
        live_plans, live_success = bank.plans, bank.frame_success.copy()
    with ThreadPoolExecutor(max_workers=1) as worker:
        newer = worker.submit(second.resolve).result()
        older = worker.submit(first.resolve).result()
    assert newer.valid and older.invalid_names == ('b',)
    assert bank.apply_detached_validation(second, newer)
    assert not bank.apply_detached_validation(first, older)
    assert bank.plans is live_plans and bank.frame_success == live_success
    assert bank.validation_dirty and not bank.frame_invalid and not bank.invalid
    assert bank.counts == {'reuse:prefill': 6, 'fallback:prefill:invalid_native_mass': 6}
    assert bank.validation_failures == 1 and not bank._detached_pending


def test_cuda_graph_replays_share_gpu_flags_but_not_detached_results():
    bank = bank_fixture()
    source = torch.ones(3, dtype=torch.int32, device='cuda')
    # Only the GPU producer is captured. Its dynamic input changes on replay;
    # independent pinned ticket copies follow each replay on the same stream.
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        bank.valid_flags.copy_(source)
    tickets = []
    expected = [('a',), ('c',)]
    for values in ((0, 1, 1), (1, 1, 0)):
        bank.arm({})
        frame_metadata(bank)
        source.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        graph.replay()
        tickets.append(bank.detach_validation(host_flags(), torch.cuda.Event()))
    bank.arm({})
    source.fill_(1)
    graph.replay()
    with ThreadPoolExecutor(max_workers=1) as worker:
        for ticket, invalid in reversed(list(zip(tickets, expected))):
            result = worker.submit(ticket.resolve).result()
            assert result.invalid_names == invalid
            assert not bank.apply_detached_validation(ticket, result)
    assert bank.validation_failures == 2
    assert bank.counts['reuse:prefill'] == 0
    assert bank.counts['fallback:prefill:invalid_native_mass'] == 6
    assert bank.valid_flags.cpu().tolist() == [1, 1, 1]


def test_resolved_unapplied_host_and_event_are_reusable_on_real_cuda():
    bank, host, event = bank_fixture(), host_flags(), torch.cuda.Event()
    frame_metadata(bank)
    put_flags(bank, (0, 1, 1))
    old = bank.detach_validation(host, event)
    with ThreadPoolExecutor(max_workers=1) as worker:
        old_result = worker.submit(old.resolve).result()
        # Models the pool-release -> frame-ready interval. Counter collection
        # has not happened, but it no longer depends on either DMA resource.
        bank.arm({})
        frame_metadata(bank, owners=2)
        put_flags(bank, (1, 1, 0))
        new = bank.detach_validation(host, event)
        new_result = worker.submit(new.resolve).result()
    assert old_result.invalid_names == ('a',) and new_result.invalid_names == ('c',)
    assert host.tolist() == [1, 1, 0]
    assert not bank.apply_detached_validation(old, old_result)
    assert not bank.apply_detached_validation(new, new_result)
    assert bank.counts['fallback:prefill:invalid_native_mass'] == 9
    assert bank.counts['reuse:prefill'] == 0 and not bank._detached_pending


def test_unresolved_real_dma_buffers_cannot_be_reused_even_after_gpu_completion():
    bank, host, event = bank_fixture(), host_flags(), torch.cuda.Event()
    frame_metadata(bank)
    put_flags(bank, (0, 1, 1))
    old = bank.detach_validation(host, event)
    event.synchronize()
    bank.arm({})
    frame_metadata(bank)
    with pytest.raises(ValueError, match='still leased'):
        bank.detach_validation(host, event)
    assert host.tolist() == [0, 1, 1]
    result = old.resolve()
    assert result.invalid_names == ('a',)
    assert not bank.apply_detached_validation(old, result)


def test_real_event_wait_error_is_permanent_and_retains_dma_lease():
    class FailingEvent:
        def __init__(self):
            self.event = torch.cuda.Event()
            self.waits = 0

        def record(self):
            self.event.record()

        def synchronize(self):
            self.waits += 1
            self.event.synchronize()
            raise OSError('injected completion error after real DMA')

    bank, host, event = bank_fixture(), host_flags(), FailingEvent()
    frame_metadata(bank)
    put_flags(bank, (1, 0, 1))
    ticket = bank.detach_validation(host, event)
    bank.arm({})
    frame_metadata(bank)
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(OSError, match='completion error'):
            worker.submit(ticket.resolve).result()
    with pytest.raises(RuntimeError, match='quarantined'):
        ticket.resolve()
    with pytest.raises(ValueError, match='still leased'):
        bank.detach_validation(host, event)
    assert ticket.error is not None and not ticket.resolved
    assert bank._detached_pending[ticket.sequence] is ticket
    assert ticket.host_flags is host and ticket.event is event and event.waits == 1
    assert bank.validation_failures == 0
