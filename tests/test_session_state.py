from dataclasses import replace
from types import SimpleNamespace
import os

import pytest
import torch

from vllm_kvmem.native_state import NativeRecurrentState
from vllm_kvmem.session_state import PromptStateAdapter


def fixture(device='cpu', draft=True):
    conv = torch.arange(8*5*4, dtype=torch.float32, device=device).reshape(8, 5, 4)
    ssm = torch.arange(8*2*3*3, dtype=torch.float32, device=device).reshape(8, 2, 3, 3)
    # A head-major strided native cache, with distinct pages for each request.
    backing = torch.arange(10*4*2*6, dtype=torch.float32, device=device).reshape(10, 4, 2, 6)
    cache = backing.bfloat16().transpose(1, 2)
    gdn_table = torch.tensor([[1, 2], [3, 4], [5, 6]], device=device)
    # Null entries outside the visible sliding prefix must never be copied.
    draft_table = torch.tensor([[0, 1, 5], [0, 2, 6], [0, 8, 9]], device=device)
    layer = SimpleNamespace(layer_name='draft.attn', kv_cache=cache, num_kv_heads=2, head_size=3)
    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={'a': 0, 'b': 1},
            num_computed_tokens_np=torch.tensor([8, 8, 8]).numpy(),
            prefill_len=SimpleNamespace(np=torch.tensor([11, 11, 11]).numpy())),
        _kvmem_request_generations={'a': 1, 'b': 2},
        model_state=SimpleNamespace(num_accepted_tokens_gpu=torch.tensor([1, 1, 7], device=device)),
        block_tables=SimpleNamespace(num_blocks=SimpleNamespace(np=torch.tensor([[2, 2, 2], [3, 3, 3]]).numpy()),
            block_tables=[SimpleNamespace(gpu=gdn_table), SimpleNamespace(gpu=draft_table)],
            kernel_block_sizes=[1, 4]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None, SimpleNamespace(layer_names=['draft.attn'])]),
        speculator=SimpleNamespace(model=SimpleNamespace(modules=lambda: [layer])))
    recurrent = NativeRecurrentState.__new__(NativeRecurrentState)
    recurrent.runner, recurrent.groups = runner, {0: {'gdn': (conv, ssm)}}
    adapter = PromptStateAdapter.__new__(PromptStateAdapter)
    adapter.runner, adapter.recurrent, adapter._identity = runner, recurrent, object()
    adapter.contract = SimpleNamespace(sliding_window=5, method='dflash', validate_native_groups=lambda _: (1,)) if draft else None
    return adapter, conv, ssm, cache, draft_table


@pytest.mark.parametrize('draft', [False, True])
def test_cpu_owned_checkpoint_restores_new_request_pages_and_preserves_neighbours(draft):
    adapter, conv, ssm, cache, _ = fixture(draft=draft)
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    assert snapshot.bytes == (20 + 18) * 4 + (4*2*6*2 if draft else 0)
    assert all(t.device.type == 'cpu' for t in (*snapshot.recurrent.values(), *snapshot.draft.values()))
    before = conv.clone(), ssm.clone(), cache.clone()
    del adapter.runner.req_states.req_id_to_index['a']
    del adapter.runner._kvmem_request_generations['a']
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    conv[1:3].fill_(-17); ssm[1:3].fill_(-17); cache[1].fill_(-17)
    assert adapter.restore(snapshot, 'new') is None
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(conv[3:5], before[0][3:5]) and torch.equal(ssm[3:5], before[1][3:5])
    assert torch.equal(conv[6], before[0][6]) and torch.equal(ssm[6], before[1][6])
    assert torch.equal(cache[2], before[2][2])
    assert torch.equal(cache[8], before[2][1] if draft else before[2][8])
    assert adapter.runner.model_state.num_accepted_tokens_gpu.tolist() == [1, 1, 1]


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1' or not torch.cuda.is_available(),reason='Opt-in CUDA checkpoint gate')
def test_pinned_snapshot_fences_gpu_copy_before_source_reuse():
    adapter,conv,ssm,_,_=fixture('cuda',draft=False)
    before=conv.clone(),ssm.clone()
    snapshot=adapter.capture('a',8,max_bytes=4096)
    assert all(t.is_pinned() for t in snapshot.recurrent.values())
    conv[1].fill_(-11);ssm[1].fill_(-11)
    # Serving polls this event through pool retirement; this GPU gate waits
    # explicitly so it can inspect CPU bytes and submit the next restoration.
    snapshot.completion.synchronize()
    assert snapshot.query() and not snapshot.source_owners
    adapter.runner.req_states.req_id_to_index['new']=2
    adapter.runner._kvmem_request_generations['new']=3
    completed=adapter.restore(snapshot,'new');completed.synchronize()
    assert torch.equal(conv[5],before[0][1]) and torch.equal(ssm[5],before[1][1])
    assert torch.equal(conv[3:5],before[0][3:5]) and torch.equal(ssm[3:5],before[1][3:5])


def test_invalid_checkpoint_or_layout_fails_before_any_destination_write():
    adapter, conv, ssm, cache, table = fixture()
    with pytest.raises(BufferError, match='byte reservation'):
        adapter.capture('a', 8, max_bytes=1)
    with pytest.raises(ValueError, match='prefill boundary'):
        adapter.capture('a', 7, max_bytes=4096)
    adapter.runner.model_state.num_accepted_tokens_gpu[0] = 2
    with pytest.raises(ValueError, match='canonicalization'):
        adapter.capture('a', 8, max_bytes=4096)
    adapter.runner.model_state.num_accepted_tokens_gpu[0] = 1
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    before = conv.clone(), ssm.clone(), cache.clone()
    broken = dict(snapshot.draft)
    broken['draft.attn'] = broken['draft.attn'].float()
    with pytest.raises(ValueError, match='Draft checkpoint schema'):
        adapter.restore(replace(snapshot, draft=broken), 'b')

    with pytest.raises(ValueError, match='another worker'):
        adapter.restore(replace(snapshot, owner=object()), 'b')
    with pytest.raises(ValueError, match='Incomplete'):
        adapter.restore(replace(snapshot, recurrent={}), 'b')
    table[1, 1] = 0
    with pytest.raises(ValueError, match='null or aliased'):
        adapter.restore(snapshot, 'b')
    assert torch.equal(conv, before[0]) and torch.equal(ssm, before[1]) and torch.equal(cache, before[2])


def test_decode_snapshot_uses_accepted_gdn_row_and_keeps_neighbour_state():
    adapter, conv, ssm, cache, _ = fixture()
    adapter.capture_decode = True
    adapter.runner.req_states.prefill_len.np[0] = 5
    adapter.runner.model_state.num_accepted_tokens_gpu[0] = 2
    adapter.runner._kvmem_capture_runtime = SimpleNamespace(last_steps={
        'a': [{'history_begin': 6, 'accepted': 2, 'scheduled': 2}]})
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    before_conv, before_ssm, before_cache = conv.clone(), ssm.clone(), cache.clone()
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    assert torch.equal(snapshot.recurrent['gdn', 'ssm'], before_ssm[2])
    assert torch.equal(snapshot.recurrent['gdn', 'conv'][:3], before_conv[1, 1:4])
    assert torch.equal(conv[3:], before_conv[3:]) and torch.equal(ssm[3:], before_ssm[3:])
    assert torch.equal(cache, before_cache)
    assert adapter.runner.model_state.num_accepted_tokens_gpu.tolist() == [1, 1, 7]


@pytest.mark.parametrize('history',[0,7])
def test_terminal_padded_prefill_tail_preserves_first_accepted_state(history):
    adapter,conv,ssm,cache,_=fixture()
    adapter.contract=SimpleNamespace(method='mtp',validate_native_groups=lambda _:())
    adapter.runner.req_states.prefill_len.np[0]=history+1
    adapter.runner._kvmem_capture_runtime=SimpleNamespace(
        last_steps={'a':[{'history_begin':history,'accepted':1,'scheduled':2}]},
        authorities={'a':SimpleNamespace(history_end=history+1)})
    adapter.recurrent.conv_dim_first=False;adapter.recurrent.conv_widths={'gdn':4}
    before_conv,before_ssm,before_cache=conv.clone(),ssm.clone(),cache.clone()
    snapshot=adapter.capture('a',history+1,max_bytes=4096,terminal=True)
    assert torch.equal(snapshot.recurrent['gdn','ssm'],before_ssm[1])
    assert torch.equal(snapshot.recurrent['gdn','conv'],before_conv[1])
    assert torch.equal(conv,before_conv) and torch.equal(ssm,before_ssm)
    assert torch.equal(cache,before_cache)


def test_clipped_continuation_keeps_accepted_state_and_repeated_metadata_is_idempotent(monkeypatch):
    from vllm_kvmem.native_state import canonicalize_before_non_speculative_batch
    adapter, conv, ssm, _, _ = fixture()
    runner = adapter.runner
    runner.speculative_config = object()
    runner.model_state.num_accepted_tokens_gpu[0] = 2
    runner._kvmem_capture_runtime = SimpleNamespace(drain=lambda **_: None,
        drain_requests=lambda *a, **kw: True,
        last_steps={'a': [{'history_begin': 6, 'accepted': 2, 'scheduled': 2}]})
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    monkeypatch.setattr('vllm_kvmem.native_state.NativeRecurrentState', lambda _: adapter.recurrent)
    batch = SimpleNamespace(num_reqs=2, req_ids=['a', 'b'], num_draft_tokens_per_req=[0, 7],
        is_prefilling_np=[False, False], num_computed_tokens_np=[8, 8])
    old_conv, old_ssm = conv.clone(), ssm.clone()
    canonicalize_before_non_speculative_batch(runner, batch)
    assert torch.equal(conv[1, :3], old_conv[1, 1:4])
    assert torch.equal(ssm[1], old_ssm[2])
    assert torch.equal(conv[3:], old_conv[3:]) and torch.equal(ssm[3:], old_ssm[3:])
    accepted_conv, accepted_ssm = conv.clone(), ssm.clone()
    canonicalize_before_non_speculative_batch(runner, batch)
    assert torch.equal(conv, accepted_conv) and torch.equal(ssm, accepted_ssm)


def test_snapshot_uses_exact_visible_logical_tail_without_guessing_physical_page_order():
    adapter, _, _, cache, table = fixture()
    adapter.runner.req_states.num_computed_tokens_np[0] = 9
    adapter.runner.req_states.prefill_len.np[0] = 12
    snapshot = adapter.capture('a', 9, max_bytes=4096)
    assert snapshot.draft_begin == 5
    expected = torch.cat((cache[1, :, 1:].transpose(0, 1), cache[5, :, :1].transpose(0, 1)))
    assert torch.equal(snapshot.draft['draft.attn'], expected)
    table[0, 2] = table[0, 1]
    with pytest.raises(ValueError, match='aliased'):
        adapter.capture('a', 9, max_bytes=4096)


def test_prepared_restore_reuses_owned_packed_payload_without_another_host_gather(monkeypatch):
    import vllm_kvmem.session_state as state_module
    adapter, conv, ssm, cache, _ = fixture()
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    expected = conv.clone(), ssm.clone(), cache.clone()
    offset = 0
    for tensor in (*snapshot.recurrent.values(), *snapshot.draft.values()):
        assert tensor.data_ptr() == snapshot.payload.data_ptr() + offset
        offset += tensor.numel() * tensor.element_size()
    assert offset == snapshot.bytes == snapshot.payload.numel()
    prepared = adapter.prepare_restore(snapshot, 'b')
    adapter.runner.model_state.num_accepted_tokens_gpu[1] = 7

    # Replace only this module's torch reference: unrelated PyTorch operations
    # and CPU tests remain untouched, and any redundant gather fails the test.
    class PackedOnlyTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def cat(self, *args, **kwargs):
            pytest.fail('An owned packed checkpoint must not be gathered again')

    monkeypatch.setattr(state_module, 'torch', PackedOnlyTorch())
    assert adapter.submit_restore(prepared) is None
    adapter.validate_restore(prepared)
    assert torch.equal(conv[3], expected[0][1])
    assert torch.equal(ssm[3], expected[1][1])
    assert torch.equal(cache[2], expected[2][1])
    assert torch.equal(conv[5:], expected[0][5:])
    assert adapter.runner.model_state.num_accepted_tokens_gpu.tolist() == [1, 1, 7]


@pytest.mark.parametrize('mutation', ['generation', 'index', 'removed'])
def test_prepared_restore_rejects_recycled_native_destination_before_commit(mutation):
    adapter, conv, ssm, cache, _ = fixture()
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    prepared = adapter.prepare_restore(snapshot, 'b')
    before = conv.clone(), ssm.clone(), cache.clone()
    if mutation == 'generation':
        adapter.runner._kvmem_request_generations['b'] += 1
    elif mutation == 'index':
        adapter.runner.req_states.req_id_to_index['b'] = 2
    else:
        del adapter.runner.req_states.req_id_to_index['b']
        del adapter.runner._kvmem_request_generations['b']
    with pytest.raises(ValueError, match='recycled'):
        adapter.validate_restore(prepared)
    assert all(torch.equal(actual, old) for actual, old in zip((conv, ssm, cache), before))


def test_legacy_unpacked_snapshot_still_restores_exact_model_state():
    adapter, conv, ssm, cache, _ = fixture()
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    # Older snapshots have no retained packed owner and can contain individual
    # CPU allocations; the fallback must preserve the same typed layout.
    snapshot = replace(snapshot, payload=None,
        recurrent={key: tensor.clone() for key, tensor in snapshot.recurrent.items()},
        draft={key: tensor.clone() for key, tensor in snapshot.draft.items()})
    before = conv.clone(), ssm.clone(), cache.clone()
    assert adapter.restore(snapshot, 'b') is None
    assert torch.equal(conv[3], before[0][1])
    assert torch.equal(ssm[3], before[1][1])
    assert torch.equal(cache[2], before[2][1])


def test_pending_snapshot_keeps_source_and_rejects_cpu_consumers_until_its_fence():
    from test_core import Event
    adapter, conv, ssm, cache, _ = fixture()
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    prepared = adapter.prepare_restore(snapshot, 'b')
    fence, owner = Event(), object()
    pending = replace(snapshot, completion=fence, source_owners=(owner,))
    before = conv.clone(), ssm.clone(), cache.clone()
    assert not pending.query() and pending.source_owners == (owner,)
    with pytest.raises(RuntimeError, match='payload is not ready'):
        adapter.prepare_restore(pending, 'b')
    prepared.snapshot = pending
    with pytest.raises(RuntimeError, match='payload is not ready'):
        adapter.submit_restore(prepared)
    assert all(torch.equal(actual, old) for actual, old in zip((conv, ssm, cache), before))
    fence.done = True
    assert pending.query() and not pending.source_owners and pending.completion is None
    fence.query = lambda: pytest.fail('Ready snapshots must not query CUDA on every engine step')
    assert pending.query()
    assert adapter.submit_restore(prepared) is None
    assert torch.equal(conv[3], before[0][1]) and torch.equal(ssm[3], before[1][1])


@pytest.mark.parametrize('record_fails', [False, True])
def test_capture_submission_returns_without_host_wait_and_fences_failed_submission(monkeypatch, record_fails):
    """Exercise the CUDA submission branch with CPU-backed tensor storage."""
    from test_core import Event
    import vllm_kvmem.session_state as state_module
    adapter, _, _, _, _ = fixture(draft=False)
    copied, synchronized = [], []

    class DeviceSnapshot(torch.Tensor):
        @property
        def device(self):
            # Storage and copy dispatch remain CPU-only; only capture's device
            # choice is simulated. No CUDA allocator/driver is initialized.
            return torch.device('cuda:0')

    class Completion(Event):
        def record(self, stream):
            copied.append(stream)
            if record_fails:
                raise OSError('injected event recording failure')

        def synchronize(self):
            pytest.fail('Successful capture must not synchronously wait on its event')

    completion = Completion()
    stream = SimpleNamespace(synchronize=lambda: synchronized.append(True))

    class SubmissionTorch:
        cuda = SimpleNamespace(Event=lambda: completion, current_stream=lambda _: stream)

        def __getattr__(self, name):
            return getattr(torch, name)

        def cat(self, values):
            return torch.Tensor._make_subclass(DeviceSnapshot, torch.cat(values), False)

        def empty(self, *args, **kwargs):
            assert kwargs.pop('pin_memory') is True
            return torch.empty(*args, **kwargs)

    monkeypatch.setattr(state_module, 'torch', SubmissionTorch())
    if record_fails:
        with pytest.raises(OSError, match='event recording failure'):
            adapter.capture('a', 8, max_bytes=4096)
        assert synchronized == [True]
    else:
        snapshot = adapter.capture('a', 8, max_bytes=4096)
        assert copied == [stream] and not synchronized
        assert not snapshot.query() and len(snapshot.source_owners) == 1
        assert snapshot.payload.numel() == snapshot.bytes
        completion.done = True
        assert snapshot.query() and not snapshot.source_owners and snapshot.completion is None


@pytest.mark.parametrize('optimized', [False, True])
def test_checkpoint_canonicalization_host_wait_ablation_preserves_exact_bytes(monkeypatch, tmp_path, optimized):
    import json
    adapter, conv, ssm, _, _ = fixture(draft=False)
    adapter.runner._kvmem_probe_output = tmp_path
    adapter.runner._kvmem_capture_runtime = SimpleNamespace(
        last_steps={'a': [{'history_begin': 6, 'accepted': 2, 'scheduled': 2}]},
        authorities={'a': SimpleNamespace(history_end=8)})
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    adapter.runner.req_states.prefill_len.np[0] = 5
    adapter.runner.model_state.num_accepted_tokens_gpu[0] = 2
    before_conv, before_ssm = conv.clone(), ssm.clone()
    waits = []
    canonicalize = adapter.recurrent.canonicalize_accepted

    def with_event(*args, **kwargs):
        canonicalize(*args, **kwargs)
        return SimpleNamespace(synchronize=lambda: waits.append(True))

    monkeypatch.setattr(adapter.recurrent, 'canonicalize_accepted', with_event)
    monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', str(int(optimized)))
    snapshot = adapter.capture('a', 8, max_bytes=4096, terminal=True)
    assert waits == ([] if optimized else [True])
    assert torch.equal(snapshot.recurrent['gdn', 'ssm'], before_ssm[2])
    assert torch.equal(snapshot.recurrent['gdn', 'conv'][:3], before_conv[1, 1:4])
    assert torch.equal(conv[3:], before_conv[3:]) and torch.equal(ssm[3:], before_ssm[3:])
    trace = json.loads((tmp_path/'checkpoint_timing.jsonl').read_text())
    assert trace['optimized'] is optimized and trace['status'] == 'submitted'
    assert trace['snapshot_bytes'] == snapshot.bytes
    assert ('canonicalize_wait' in trace['host_stage_seconds']) is (not optimized)
    assert 'accepted_counter_read' not in trace['host_stage_seconds']
    assert trace['host_sync_wait_seconds'] <= trace['host_total_seconds']
