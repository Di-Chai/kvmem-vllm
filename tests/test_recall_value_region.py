"""Target-only K/V gathers preserve partial tails, owned lifetimes and scores."""
import pytest
import torch

from test_query_key_fps_performance import setup
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.recall_value_reference import prepare_reference, score_value_records


def test_target_kv_regions_skip_draft_layers_and_survive_archive_release(tmp_path, monkeypatch):
    a, archive, records, raw = setup(tmp_path)
    calls = []
    original = archive.read_region
    def read(*args, **kwargs):
        calls.append((args[2], args[3], len(args[1])))
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, 'read_region', read)
    order = (2, 0, 1)
    shape = (3, 2, 32, 2, 4)
    target = (torch.empty(shape, dtype=a.raw_dtype), torch.empty(shape, dtype=torch.uint8))
    (keys, values), stats = a.read_score_kv([records[i].handle for i in order], destination=target)
    assert keys is target[0] and values is target[1]
    assert calls == [(0, 2*32*2*4, 2), (3*32*2*4, 2*32*2*4, 2)]
    for out, old in enumerate(order):
        count = records[old].count
        assert torch.equal(keys[out, :, :count].view(torch.uint8), raw[:2, old*32:old*32+count].view(torch.uint8))
        assert bool((values[out, :, :count] == 255).all())
        assert bool((keys[out, :, count:].float() == 0).all())
        assert bool((values[out, :, count:] == 0).all())
    assert stats['key_read_bytes'] == stats['value_read_bytes'] == 2*2*32*2*4
    assert stats['draft_key_read_bytes'] == stats['draft_value_read_bytes'] == 0
    assert a.close() and archive.close()
    assert torch.equal(keys[1].view(torch.uint8), raw[:2, :32].view(torch.uint8))


@pytest.mark.parametrize('failure', ['key', 'value'])
def test_region_queue_failure_releases_every_read_lease(tmp_path, monkeypatch, failure):
    a, archive, records, _ = setup(tmp_path)
    original = archive.read_region
    def read(*args, **kwargs):
        return None if bool(args[2]) == (failure == 'value') else original(*args, **kwargs)
    monkeypatch.setattr(archive, 'read_region', read)
    with pytest.raises(BlockingIOError, match='queue is full'):
        a.read_score_kv([r.handle for r in records])
    assert a.close() and archive.close()


@pytest.mark.parametrize('method', ['block_value_norm', 'block_deletion_error'])
def test_region_packed_attention_scores_and_every_operand_match_legacy(tmp_path, monkeypatch, method):
    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
                        repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 3*32*2*4*2, 16, gather_workers=2)
    a = RequestAuthority('value-regions', config, archive, 3, 2, 4,
                         scoring_layers=2, raw_dtype=torch.float8_e4m3fn, resident_records=4)
    gen = torch.Generator().manual_seed(632)
    keys = torch.randn(3, 67, 2, 4, generator=gen).to(torch.float8_e4m3fn)
    values = torch.randn(3, 67, 2, 4, generator=gen).to(torch.float8_e4m3fn).view(torch.uint8)
    assert a.append(keys, values, history_begin=0, local_rope_begin=0) and a.flush()
    for job, _ in a.writes:
        job.result()
    a.poll()
    records = sorted(a.records.values(), key=lambda r: r.history_begin)
    a.recall_value_scales = torch.tensor([.371, 1.293])
    q = torch.randn(2, 7, 4, 4, generator=torch.Generator().manual_seed(227))
    mask = [True, False, True]
    monkeypatch.setenv('KVMEM_RECALL_DEVICE', 'cpu')
    monkeypatch.setenv('KVMEM_RECALL_QUERY_BATCH', '128')
    monkeypatch.setenv('KVMEM_RECALL_BLOCK_BATCH', '2')
    legacy, _ = prepare_reference(a, records, q, mask, packed=True, cache_tiles=True)
    region, stats = prepare_reference(a, records, q, mask, packed=True, region=True, cache_tiles=True)
    for layer in range(2):
        for head in range(2):
            for begin, end in ((0, 2), (2, 3)):
                assert all(torch.equal(x, y) for x, y in zip(legacy.read_blocks(begin, end, layer, head),
                                                             region.read_blocks(begin, end, layer, head)))
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND', 'streaming_packed')
    old, _ = score_value_records(a, records, q, mask, method)
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND', 'streaming_region')
    new, stats = score_value_records(a, records, q, mask, method)
    assert old == new
    assert max(new) > 0 and new[0] != new[2]
    assert stats['operand_backend'] == 'region_packed' and stats['operand_pack_seconds'] == 0
    assert stats['archive_region_stats']['draft_value_read_bytes'] == 0
    assert a.close() and archive.close()


def test_region_rejects_duplicate_and_noncontiguous_destinations(tmp_path):
    a, archive, records, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        a.read_score_kv([records[0].handle]*2)
    shape = (1, 2, 32, 2, 4)
    invalid = (torch.empty(shape, dtype=a.raw_dtype).transpose(-1, -2), torch.empty(shape, dtype=torch.uint8))
    with pytest.raises(ValueError, match='contiguous CPU'):
        a.read_score_kv([records[0].handle], destination=invalid)
    assert a.close() and archive.close()
