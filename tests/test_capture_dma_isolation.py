"""Actual packet construction with independently delayed CPU-only DMA fences."""
from dataclasses import replace
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.terminal import TerminalPrefix
from test_authority import setup, drain
from test_request_publication import fixture, close


class FakeEvent:
    def __init__(self, ready):
        self.ready = Event()
        self.entered = Event()
        self.waits = 0
        if ready:
            self.ready.set()

    def query(self):
        return self.ready.is_set()

    def synchronize(self):
        self.waits += 1
        self.entered.set()
        assert self.ready.wait(10), 'Test DMA fence was never released'


class FakePool:
    def __init__(self, ready):
        self.buffers = [object(), object()]
        self.pending = {}
        self.ready = ready
        self.released = []

    def download(self, payload):
        slot = next((i for i in range(2) if i not in self.pending), None)
        if slot is None:
            return None
        event = FakeEvent(self.ready)
        download = NS(payload=payload, slot=slot, event=event, query=event.query, owners=(payload,))
        self.pending[slot] = download
        return download

    def release(self, download):
        assert self.pending[download.slot] is download and download.query()
        del self.pending[download.slot]
        self.released.append(download)
        download.owners = ()
        return True


def configure(runtime, split=True):
    runtime.split_capture_dma = split
    runtime.decode_capture_limit = 5
    runtime.names = ('layer0', 'layer1')
    runtime.scales = torch.ones(2)
    runtime.downloads = FakePool(False)
    runtime.decode_downloads = FakePool(True) if split else None


def sample(runtime, specs, actual=None, rejected=None):
    # specs: id, logical begin, CPU upper bound, is_prefill.
    actual = actual or [spec[2] for spec in specs]
    cpu_starts = np.cumsum([0, *[spec[2] for spec in specs]])
    gpu_starts = torch.tensor(np.cumsum([0, *actual]), dtype=torch.int64)
    positions = torch.cat([torch.arange(begin, begin+count) for (_, begin, _, _), count in zip(specs, actual)])
    tokens = positions.numel()
    captures = [NS(name=name, positions=positions,
        key=(torch.arange(tokens*2).reshape(tokens, 1, 2)+layer*13).bfloat16(),
        value=torch.ones(tokens, 1, 2, dtype=torch.bfloat16)) for layer, name in enumerate(runtime.names)]
    batch = NS(req_ids=tuple(s[0] for s in specs), num_reqs=len(specs), num_tokens=tokens,
        query_start_loc=gpu_starts, query_start_loc_np=cpu_starts,
        positions=positions, is_prefilling_np=np.array([s[3] for s in specs]),
        has_prefill=any(s[3] for s in specs), num_computed_tokens_np=np.array([s[1] for s in specs]),
        num_scheduled_tokens=np.array([s[2] for s in specs]))
    rejection = torch.tensor(rejected or [0]*len(specs))
    runtime.submit(captures, batch, rejection)
    return captures, batch


def make_ready(runtime):
    for packet in runtime.queue:
        packet.download.event.ready.set()


@pytest.mark.parametrize('configured,env,enabled', [(True, None, True), (False, None, False),
    (True, '0', False), (False, '1', True)])
def test_pool_capacity_is_fixed_and_flag_precedence_is_explicit(tmp_path, monkeypatch, configured, env, enabled):
    import vllm_kvmem.capture_runtime as capture
    monkeypatch.delenv('KVMEM_SPLIT_CAPTURE_DMA', raising=False)
    if env is not None:
        monkeypatch.setenv('KVMEM_SPLIT_CAPTURE_DMA', env)
    options = RuntimeOptions(split_capture_dma=configured)
    monkeypatch.setattr(capture.RuntimeOptions, 'from_vllm', lambda _: options)
    allocations = []
    monkeypatch.setattr(capture, 'PackedDownloads', lambda size, slots, device:
        allocations.append((size, slots)) or NS(pending={}, capacity_bytes=size*slots))
    layer = NS(num_kv_heads=1, head_dim=2,
        attn=NS(kv_cache_dtype='fp8_e4m3', _v_scale=torch.tensor(1.)))
    runner = NS(_kvmem_layers=('layer',), _kvmem_prepared_archive=object(),
        model=NS(named_modules=lambda: [('layer', layer)]), device=torch.device('cpu'), decode_query_len=5,
        vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=1024, max_num_seqs=8)))
    runtime = CaptureRuntime(runner, tmp_path)
    assert runtime.split_capture_dma is enabled
    expected = [(1024*4+264, 2)]
    if enabled:
        expected += [(8*5*4+264, 2)]
    assert allocations == expected
    monkeypatch.setenv('KVMEM_SPLIT_CAPTURE_DMA', 'true')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        CaptureRuntime(runner, tmp_path)


@pytest.mark.parametrize('adaptive', [False, True])
def test_split_payload_uses_actual_gpu_rows_not_cpu_mtp_upper_bounds(tmp_path, adaptive):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    specs = [('b', 4, 8, True), ('a', 4, 4, False), ('c', 100, 6, True), ('d', 10, 2, False)]
    actual = [8, 2, 6, 1] if adaptive else [8, 4, 6, 2]
    captures, batch = sample(runtime, specs, actual, [0, 1, 0, 0])
    assert [p.lane for p in runtime.queue] == ['decode', 'bulk']
    assert all(not p.confirmed for p in runtime.queue)
    for packet, indices in zip(runtime.queue, [(1, 3), (0, 2)]):
        count = len(indices)
        data = packet.download.payload
        header_bytes = (4*count+1)*8
        header = data[:header_bytes].view(torch.int64).tolist()
        wanted_counts = [actual[i] for i in indices]
        assert header[:count+1] == np.cumsum([0, *wanted_counts]).tolist()
        assert header[count+1:2*count+1] == [specs[i][1] for i in indices]
        assert header[3*count+1:] == [[0, 1, 0, 0][i] for i in indices]
        raw_bytes = runtime.row_elements*packet.tokens*runtime.raw_dtype.itemsize
        raw = data[header_bytes:header_bytes+raw_bytes].view(runtime.raw_dtype).reshape(2, packet.tokens, 1, 2)
        source_rows = [row for i in indices for row in range(int(batch.query_start_loc[i]), int(batch.query_start_loc[i+1]))]
        expected = torch.stack([capture.key[source_rows] for capture in captures])
        assert torch.equal(raw[:, :sum(wanted_counts)], expected)
    # This packet-format test deliberately has unregistered c/d. Drop only its
    # test-owned storage after checking every emitted byte range.
    runtime.queue.clear()
    runtime.downloads.pending.clear(); runtime.decode_downloads.pending.clear()
    close(runtime, (a, b), archive)


def test_two_delayed_bulk_slots_do_not_block_three_more_decode_steps(tmp_path):
    runtime, a, b, archive = fixture(tmp_path, threshold=8)
    configure(runtime)
    c, _ = setup(tmp_path/'c', 'c', archive)
    runtime.authorities['c'] = c
    runtime.accepted['c'] = runtime.rejected['c'] = 0
    runtime.last_steps['c'] = []
    sample(runtime, [('b', 4, 8, True), ('a', 4, 1, False)])
    runtime.begin_step(set(), {})
    assert runtime.start_cpu_scatter()
    bulk = next(p for p in runtime.queue if p.lane == 'bulk')
    assert bulk.download.event.entered.wait(5)
    try:
        sample(runtime, [('c', 0, 8, True), ('a', 5, 1, False)])
        runtime.begin_step(set(), {})
        assert len(runtime.downloads.pending) == 2
        for begin in (6, 7, 8):
            sample(runtime, [('a', begin, 1, False)])
            runtime.begin_step(set(), {})
            assert runtime.drain_requests({'a'}, wait=True)
            assert a.history_end == begin+1
            assert len(runtime.downloads.pending) == 2
            assert not runtime.decode_downloads.pending
        assert b.history_end == 4 and c.history_end == 0
        assert not runtime._cpu_scatter_future.done()
        assert all(d.owners for d in runtime.downloads.pending.values())
        metrics = runtime._publication_metrics_snapshot()
        assert metrics['main/decode'].get('dma_wait_calls', 0) == 0
        assert metrics['main/decode'].get('slot_wait_calls', 0) == 0
    finally:
        make_ready(runtime)
        runtime.join_cpu_scatter()
        runtime.drain(wait=True)
        close(runtime, (a, b, c), archive)
    assert not runtime.downloads.pending and not runtime.decode_downloads.pending
    assert runtime._publication_metrics_snapshot()['background/bulk']['dma_wait_calls'] >= 1


@pytest.mark.parametrize('terminal_end', [4, 5])
def test_split_unconfirmed_mtp_and_cancel_clip_before_own_publication(tmp_path, terminal_end):
    runtime, a, b, archive = fixture(tmp_path, threshold=8)
    configure(runtime)
    accepted_actions = []
    actions = {rid: object() for rid in ('a', 'b')}
    runtime.action = NS(capture=lambda *_: dict(actions), accept=lambda packet, rid, retained:
        accepted_actions.append((rid, retained, packet.action_inputs.pop(rid))))
    sample(runtime, [('b', 4, 8, True), ('a', 4, 4, False)], actual=[8, 3], rejected=[0, 1])
    assert not runtime.drain_requests({'a'}, wait=True)
    assert a.history_end == 4
    prefix = TerminalPrefix(terminal_end, 7, 1, 'FINISHED_ABORTED')
    runtime.begin_step({'a'}, {'a': prefix})
    assert runtime.drain_requests({'a'}, wait=True)
    assert a.history_end == terminal_end
    assert runtime.last_steps['a'][-1] == dict(history_begin=4, scheduled=3,
        accepted=terminal_end-4, rejected=1, terminal_discarded=6-terminal_end)
    assert b.history_end == 4 and len(runtime.downloads.pending) == 1
    assert accepted_actions == [('a', terminal_end-4, actions['a'])]
    assert runtime.queue[0].action_inputs == {'b': actions['b']}
    assert next(iter(runtime.downloads.pending.values())).event.waits == 0
    assert not runtime.decode_downloads.pending
    make_ready(runtime); runtime.drain(wait=True)
    assert accepted_actions[-1] == ('b', 8, actions['b'])
    close(runtime, (a, b), archive)


def test_disabled_split_retains_original_whole_batch_fence(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime, split=False)
    sample(runtime, [('b', 4, 8, True), ('a', 4, 1, False)])
    assert len(runtime.queue) == 1 and runtime.queue[0].request_ids == ('b', 'a')
    runtime.begin_step(set(), {})
    assert not runtime.drain_requests({'a'}, wait=False)
    assert a.history_end == 4 and not runtime.queue[0].published
    make_ready(runtime); runtime.drain(wait=True)
    close(runtime, (a, b), archive)


@pytest.mark.parametrize('failure', ['enqueue', 'bulk_backpressure'])
def test_split_enqueue_failure_preserves_previously_submitted_owner(tmp_path, monkeypatch, failure):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    def fail(*args, **kwargs):
        # Even a mixed batch has already submitted its independent small DMA
        # before touching a full bulk pool. The engine still has to wait for
        # bulk capacity before returning this mixed submit; no claim otherwise.
        assert len(runtime.decode_downloads.pending) == 1
        assert runtime.queue[0].lane == 'decode'
        raise OSError('enqueue failed')
    if failure == 'enqueue':
        monkeypatch.setattr(runtime.downloads, 'download', fail)
    else:
        runtime.downloads.pending = {0: object(), 1: object()}
        monkeypatch.setattr(runtime, 'ensure_download_slot', fail)
    with pytest.raises(OSError, match='enqueue failed'):
        sample(runtime, [('b', 4, 8, True), ('a', 4, 1, False)])
    assert len(runtime.queue) == 1 and runtime.queue[0].request_ids == ('a',)
    assert not runtime.queue[0].confirmed and runtime.queue[0].download.owners
    assert not runtime.drain_requests({'a'}, wait=True)
    runtime.begin_step({'a'}, {'a': TerminalPrefix(4, 5, 0, 'FINISHED_ABORTED')})
    runtime.drain(wait=True)
    assert a.history_end == 4 and not runtime.decode_downloads.pending
    runtime.downloads.pending.clear()
    close(runtime, (a, b), archive)
