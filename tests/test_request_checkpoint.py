from dataclasses import replace
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.request_checkpoint import (
    CacheRows, NativeRequestCheckpointAdapter, capture_exact, checkpoint_size, restore_exact,
)


def fixture(device='cpu'):
    # Real native attention packing has a strided page/head/token view and
    # interleaved K/V. Preserve byte patterns, including NaN payloads.
    backing = torch.arange(9 * 5 * 3 * 16, dtype=torch.int64, device=device).remainder(256).to(torch.uint8)
    cache = backing.reshape(9, 5, 3, 16).transpose(1, 2)
    key, value = cache.split(8, dim=-1)
    conv = torch.arange(9 * 7 * 4, dtype=torch.float32, device=device).reshape(9, 7, 4)
    ssm = torch.arange(9 * 2 * 3 * 3, dtype=torch.float32, device=device).reshape(9, 2, 3, 3)
    counter = torch.tensor([3, 7, 1], device=device)
    def bindings(rows, slot=0):
        return (CacheRows('key', key, rows), CacheRows('value', value, rows),
                CacheRows('conv', conv, rows), CacheRows('ssm', ssm, rows),
                CacheRows('num_accepted', counter, (slot,)))
    return bindings, (cache, conv, ssm, counter)


def test_exact_roundtrip_preserves_partial_page_bytes_speculative_rows_and_neighbours():
    bindings, tensors = fixture()
    expected = [tensor.clone() for tensor in tensors]
    original = bindings((1, 5))
    size = checkpoint_size(original)
    saved = capture_exact(('a', 2, 3), original, max_bytes=size)
    assert saved.query() and saved.bytes == size
    for tensor in tensors[:-1]: tensor[1].fill_(0); tensor[5].fill_(0)
    tensors[-1][0] = 0
    done = restore_exact(saved, ('a', 2, 3), bindings((6, 2), 1))
    assert done.query()
    for actual, old in zip(tensors[:-1], expected[:-1]):
        assert torch.equal(actual[6], old[1]) and torch.equal(actual[2], old[5])
        assert torch.equal(actual[3], old[3]) and torch.equal(actual[4], old[4])
    assert tensors[-1].tolist() == [0, 3, 1]


def test_fp8_payload_roundtrips_without_float_conversion():
    raw = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
    data = raw.view(torch.float8_e4m3fn)
    saved = capture_exact(('a', 1, 1), [CacheRows('fp8', data, (1, 3))], max_bytes=128)
    old = raw.clone()
    raw[1].zero_(); raw[3].zero_()
    restore_exact(saved, ('a', 1, 1), [CacheRows('fp8', data, (0, 2))])
    assert torch.equal(raw[0], old[1]) and torch.equal(raw[2], old[3])


@pytest.mark.parametrize('change', ['missing', 'name', 'dtype', 'shape', 'rows', 'identity', 'payload'])
def test_restore_checks_entire_schema_before_first_write(change):
    bindings, tensors = fixture()
    saved = capture_exact(('a', 2, 3), bindings((1, 5)), max_bytes=8192)
    target = list(bindings((6, 2), 1))
    identity = saved.identity
    if change == 'missing': target.pop()
    if change == 'name': target[-1] = replace(target[-1], name='wrong')
    if change == 'dtype': target[-1] = replace(target[-1], tensor=tensors[-1].float())
    if change == 'shape': target[-1] = replace(target[-1], tensor=tensors[-1][:, None])
    if change == 'rows': target[-1] = replace(target[-1], rows=(1, 2))
    if change == 'identity': identity = ('a', 3, 3)
    if change == 'payload': saved.payload = saved.payload[:-1]
    before = [t.clone() for t in tensors]
    with pytest.raises(ValueError): restore_exact(saved, identity, target)
    assert all(torch.equal(a, b) for a, b in zip(tensors, before))


def test_budget_failure_before_payload_allocation(monkeypatch):
    bindings, _ = fixture()
    def forbidden(*a, **kw): pytest.fail('over-budget capture allocated a payload')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(BufferError):
        capture_exact(('a', 1, 1), bindings((1, 5)), max_bytes=1)


@pytest.mark.parametrize('rows', [(), (-1,), (99,), (1, 1), (True,)])
def test_invalid_row_descriptors_rejected(rows):
    with pytest.raises(ValueError): checkpoint_size([CacheRows('bad', torch.ones(3, 4), rows)])


def test_capture_refuses_duplicate_names_and_duplicate_physical_rows():
    tensor = torch.ones(4, 8)
    with pytest.raises(ValueError):
        checkpoint_size([CacheRows('a', tensor, (1,)), CacheRows('a', tensor, (2,))])
    with pytest.raises(ValueError, match='Aliased'):
        checkpoint_size([CacheRows('a', tensor, (1,)), CacheRows('b', tensor, (1,))])


def native_fixture(device='cpu'):
    from vllm_kvmem.serving.native_state import NativeRecurrentState
    target = torch.arange(12 * 4 * 2 * 8, device=device).remainder(256).to(torch.uint8).reshape(12, 4, 2, 8).transpose(1, 2)
    mtp = target.clone() + 1
    conv = torch.arange(12 * 5 * 4, dtype=torch.float32, device=device).reshape(12, 5, 4)
    ssm = torch.arange(12 * 2 * 3 * 3, dtype=torch.float32, device=device).reshape(12, 2, 3, 3)
    def layer(name, cache):
        return NS(num_kv_heads=2, head_dim=4,
                  attn=NS(layer_name=name, kv_cache=cache, kv_cache_dtype='fp8_e4m3'))
    modules = {'target': layer('target.attn', target)}
    draft = layer('mtp.attn', mtp)
    pages = ((1, 3), (5,), (6, 7))
    tables = [NS(gpu=torch.tensor([row], device=device)) for row in pages]
    runner = NS(req_states=NS(req_id_to_index={'a': 0}), _kvmem_request_generations={'a': 9},
                block_tables=NS(block_tables=tables, blocks_per_kv_block=[1, 1, 1],
                    kernel_block_sizes=[4, 4, 1], num_blocks=NS(np=torch.tensor([[2], [1], [2]]).numpy())),
                model=NS(named_modules=lambda: modules.items()), _kvmem_layers=['target'],
                _kvmem_archive_layers=['target', '__mtp__.draft'], _kvmem_mtp_modules={'__mtp__.draft': draft},
                model_state=NS(num_accepted_tokens_gpu=torch.tensor([3], device=device)),
                kv_cache_config=NS(kv_cache_groups=[NS(layer_names=['target.attn']),
                                                  NS(layer_names=['mtp.attn']), NS(layer_names=['gdn'])]))
    recurrent = object.__new__(NativeRecurrentState)
    recurrent.runner = runner
    recurrent.groups = {2: {'gdn': (conv, ssm)}}
    adapter = object.__new__(NativeRequestCheckpointAdapter)
    adapter.runner, adapter.recurrent, adapter._slots = runner, recurrent, {}
    return adapter, pages, (target, mtp, conv, ssm)


def test_native_adapter_restores_target_mtp_and_every_gdn_row_in_original_slot():
    adapter, pages, tensors = native_fixture()
    old = [t.clone() for t in tensors]
    saved = adapter.capture('a', 4, pages, max_bytes=16384)
    assert len(saved.schemas) == 7
    assert [s.rows for s in saved.schemas] == [2, 2, 1, 1, 2, 2, 1]
    replacement = ((2, 4), (8,), (1, 3))
    for table, row in zip(adapter.runner.block_tables.block_tables, replacement):
        table.gpu[0] = torch.tensor(row)
    for tensor in tensors: tensor.zero_()
    adapter.runner.model_state.num_accepted_tokens_gpu.zero_()
    adapter.restore(saved, 'a', 4, replacement).synchronize()
    for tensor, expected, source, destination in zip(tensors, old,
            (pages[0], pages[1], pages[2], pages[2]),
            (replacement[0], replacement[1], replacement[2], replacement[2])):
        for src, dst in zip(source, destination): assert torch.equal(tensor[dst], expected[src])
    # Capturing all rows preserves pending acceptance; it never normalizes to 1.
    assert adapter.runner.model_state.num_accepted_tokens_gpu.item() == 3
    adapter.release('a')
    assert adapter._slots == {}


@pytest.mark.parametrize('kind', ['generation', 'slot', 'attention_table', 'state_table'])
def test_native_adapter_refuses_recycled_identity_and_stale_allocator_witness(kind):
    adapter, pages, tensors = native_fixture()
    saved = adapter.capture('a', 4, pages, max_bytes=16384)
    if kind == 'generation': adapter.runner._kvmem_request_generations['a'] += 1
    if kind == 'slot': adapter.runner.req_states.req_id_to_index['a'] = 1
    if kind == 'attention_table': adapter.runner.block_tables.block_tables[0].gpu[0, 0] = 2
    if kind == 'state_table': adapter.runner.block_tables.block_tables[2].gpu[0, 0] = 8
    before = [t.clone() for t in tensors]
    with pytest.raises(ValueError): adapter.restore(saved, 'a', 4, pages)
    assert all(torch.equal(t, old) for t, old in zip(tensors, before))


def test_native_adapter_failed_capture_releases_only_its_unpublished_owner():
    adapter, pages, tensors = native_fixture()
    with pytest.raises(BufferError): adapter.capture('a', 4, pages, max_bytes=1)
    assert adapter._slots == {}
    saved = adapter.capture('a', 5, pages, max_bytes=16384)
    with pytest.raises(ValueError, match='Duplicate'): adapter.capture('a', 6, pages, max_bytes=16384)
    assert saved.query() and adapter._slots == {'a': (0, 9)}


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='Opt-in exact swap CUDA byte qualification')
def test_cuda_exact_swap_payload_owns_data_after_original_pages_reused():
    bindings, tensors = fixture('cuda')
    expected = [tensor.clone() for tensor in tensors]
    saved = capture_exact(('gpu', 5, 11), bindings((1, 5)), max_bytes=8192)
    assert saved.payload.is_pinned()
    saved.completion.synchronize()
    for tensor in tensors[:-1]: tensor[1].zero_(); tensor[5].zero_()
    tensors[-1][0] = 0
    done = restore_exact(saved, saved.identity, bindings((6, 2), 1))
    done.synchronize()
    for tensor, old in zip(tensors[:-1], expected[:-1]):
        assert torch.equal(tensor[6], old[1]) and torch.equal(tensor[2], old[5])
        assert torch.equal(tensor[3], old[3])
    assert tensors[-1].tolist() == [0, 3, 1]
