from dataclasses import replace
from types import SimpleNamespace

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
    adapter.contract = SimpleNamespace(sliding_window=5, validate_native_groups=lambda _: (1,)) if draft else None
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


def test_clipped_continuation_keeps_accepted_state_and_repeated_metadata_is_idempotent(monkeypatch):
    from vllm_kvmem.native_state import canonicalize_before_non_speculative_batch
    adapter, conv, ssm, _, _ = fixture()
    runner = adapter.runner
    runner.speculative_config = object()
    runner.model_state.num_accepted_tokens_gpu[0] = 2
    runner._kvmem_capture_runtime = SimpleNamespace(drain=lambda **_: None,
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
