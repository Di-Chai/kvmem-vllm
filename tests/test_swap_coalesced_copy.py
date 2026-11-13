"""KVMEM_SWAP_COALESCED_COPY: whole-row attention bindings, adjacent rows copied as one region."""
import pytest
import torch

from vllm_kvmem.serving import request_checkpoint as rc
from vllm_kvmem.serving.request_checkpoint import CacheRows, capture_exact, checkpoint_size, restore_exact


def caches(device):
    # The real attention cache: a block-major physical buffer with a strided
    # logical (block, head, token, K|V) view, as native_vllm_fp8_views expects.
    backing = torch.arange(12 * 5 * 3 * 16, dtype=torch.int64, device=device).remainder(251).to(torch.uint8)
    cache = backing.reshape(12, 5, 3, 16).transpose(1, 2)
    conv = torch.arange(12 * 7 * 4, dtype=torch.float32, device=device).reshape(12, 7, 4)
    return backing.reshape(12, -1), cache, conv


def test_physical_rows_are_the_dense_block_bytes():
    backing, cache, _ = caches('cpu')
    rows = rc.physical_rows(cache)
    assert rows.shape == (12, 240) and rows.data_ptr() == backing.data_ptr()
    assert torch.equal(rows, backing)
    with pytest.raises(ValueError, match='not dense'):
        rc.physical_rows(cache[:, :, :, :8])              # a K-only split view


def test_runs_merge_only_contiguous_ascending_neighbours():
    _, cache, conv = caches('cpu')
    rows = rc.physical_rows(cache)
    assert rc._runs(rows, (5, 6, 7, 2, 3, 9)) == [(0, 5, 3), (3, 2, 2), (5, 9, 1)]
    key = cache.split(8, dim=-1)[0]                        # strided rows: never merged
    assert rc._runs(key, (5, 6, 7)) == [(0, 5, 1), (1, 6, 1), (2, 7, 1)]


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_coalesced_roundtrip_is_byte_identical(monkeypatch, device):
    backing, cache, conv = caches(device)
    rows = (5, 6, 7, 2, 3, 9)
    def bindings(selected):
        return (CacheRows('kv', rc.physical_rows(cache), selected), CacheRows('conv', conv, selected))
    monkeypatch.setenv('KVMEM_SWAP_COALESCED_COPY', '0')
    plain = capture_exact(('r', 1, 1), bindings(rows), max_bytes=checkpoint_size(bindings(rows)))
    monkeypatch.setenv('KVMEM_SWAP_COALESCED_COPY', '1')
    saved = capture_exact(('r', 1, 1), bindings(rows), max_bytes=checkpoint_size(bindings(rows)))
    if device == 'cuda':
        torch.cuda.synchronize()
    assert torch.equal(saved.payload, plain.payload)       # same payload layout and bytes
    expected = backing.clone(), conv.clone()
    for row in rows:
        backing[row].fill_(0); conv[row].fill_(0)
    target = (8, 9, 10, 0, 1, 4)                             # restored into other pages
    restore_exact(saved, ('r', 1, 1), bindings(target))
    if device == 'cuda':
        torch.cuda.synchronize()
    for old, new in zip(rows, target):
        assert torch.equal(backing[new], expected[0][old]) and torch.equal(conv[new], expected[1][old])
    assert torch.equal(backing[11], expected[0][11])        # untouched neighbour


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_SWAP_COALESCED_COPY', 'yes')
    with pytest.raises(ValueError, match='KVMEM_SWAP_COALESCED_COPY'):
        rc.coalesced_copies()
