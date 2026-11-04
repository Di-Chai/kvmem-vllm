"""Accepted Q retains its FIFO publication fence across two bounded DMA slots."""
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.pressure_query_capture import PressureQueryWindow, validate_capture_options
from test_prefill_publication_pipeline import enabled_fixture, mark_pipeline
from test_request_publication import packet, close
from test_pressure_query_capture import batch


def qrows(begin, count):
    return torch.stack([torch.arange(begin, begin+count)[:, None, None].expand(count, 1, 2)
                        + layer*1000 for layer in range(2)]).to(torch.bfloat16)


def enable_q(runtime):
    runtime.pressure_query_enabled = True
    runtime.pressure_query_layers = (0, 1)
    runtime.pressure_query_heads = 1
    runtime.pressure_query_window = 128
    runtime.pressure_query_stats = dict(packets=0, bytes=0, accepted_rows=0)
    for owner in runtime.authorities.values():
        owner.pressure_query_window = PressureQueryWindow()
        owner.pressure_query_window.accept(0, 4, qrows(0, 4), prefill=True)


def with_q(runtime, item, begin, count):
    source = [NS(query=row) for row in qrows(begin, count)]
    payload, descriptors = runtime._capture_pressure_queries(item.download.payload, source,
        batch(list(item.request_ids), [begin], [count], [True]), (0,))
    item.download.payload = payload
    item.pressure_queries = descriptors
    # Simulate the next public model forward overwriting captured Q storage.
    for layer in source:
        layer.query.fill_(-999)
    return item


def test_pipeline_allowed_while_early_release_and_coalescing_remain_rejected():
    validate_capture_options(NS(prefill_publication_pipeline=True))
    for key in ('offline_prefill_coalescing', 'prefill_physical_coalescing',
                'early_capture_slot_release', 'prefill_capture_ahead'):
        with pytest.raises(ValueError, match=key):
            validate_capture_options(NS(prefill_publication_pipeline=True, **{key: True}))


def test_two_slots_commit_only_confirmed_q_and_clone_before_recycling(tmp_path, monkeypatch):
    runtime, owner, other, archive = enabled_fixture(tmp_path)
    enable_q(runtime)
    first = with_q(runtime, mark_pipeline(packet(runtime, [('a', 4, 4)])), 4, 4)
    entered, release = Event(), Event()
    original = owner.memory.begin_append

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(owner.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert runtime.prefill_pipeline_can_advance('a')
        second = with_q(runtime, mark_pipeline(packet(runtime, [('a', 8, 4)], confirmed=False)), 8, 4)
        assert not runtime.prefill_pipeline_can_advance('a')
        assert torch.equal(owner.pressure_query_window.read(4)[0], qrows(0, 4))
        release.set()
        runtime.join_cpu_scatter()
        assert owner.history_end == 8
        assert torch.equal(owner.pressure_query_window.read(8)[0], qrows(0, 8))
        assert second.download.owners and not second.published
        with pytest.raises(RuntimeError, match='accepted boundary'):
            owner.pressure_query_window.read(12)
        first.download.payload.zero_()
        assert torch.equal(owner.pressure_query_window.read(8)[0], qrows(0, 8))
        runtime.begin_step()
        assert runtime.drain_requests({'a'}, wait=True)
        second.download.payload.zero_()
        assert torch.equal(owner.pressure_query_window.read(12)[0], qrows(0, 12))
        assert runtime.pressure_query_stats['accepted_rows'] == 8
    finally:
        release.set()
        close(runtime, (owner, other), archive)
