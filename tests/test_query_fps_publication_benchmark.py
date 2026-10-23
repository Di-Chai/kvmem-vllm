"""CPU-only checks for the real cache benchmark's ownership and reference audit."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import benchmark_query_fps_publication as bench


def fixture():
    generator=torch.Generator().manual_seed(176)
    counts=[32,7,32]
    keys=torch.randn((2,sum(counts),2,8),generator=generator).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    meta=dict(history=128,starts=[0,32,96],counts=counts)
    return keys,meta


def test_owned_slab_preserves_original_bytes_partial_lengths_and_generations():
    keys,meta=fixture();authority=bench.OwnedMmapAuthority(keys,meta)
    slab,stats=authority.read_score_keys([authority.records[1].handle,authority.records[0].handle])
    raw=slab.view(torch.uint8).numpy()
    np.testing.assert_array_equal(raw[0,:,:7],keys[:,32:39])
    np.testing.assert_array_equal(raw[1],keys[:,:32])
    assert np.count_nonzero(raw[0,:,7:])==0
    assert stats['key_read_bytes']==39*2*2*8
    assert stats['value_read_bytes']==stats['draft_key_read_bytes']==0
    original=keys.copy();raw[:]=0
    np.testing.assert_array_equal(keys,original)
    with pytest.raises(ValueError,match='stale or foreign'):
        authority.read_score_keys([NS(slot=0,generation=2)])


def test_actual_frozen_and_current_cache_cold_snapshots_have_identical_bytes_and_scores():
    torch.set_num_threads(1)
    keys,meta=fixture();authority=bench.OwnedMmapAuthority(keys,meta)
    versions=bench.load_versions(bench.OLD_MODULE)
    queries=torch.randn((2,2,4,8),generator=torch.Generator().manual_seed(91))
    score_mask=torch.tensor([True,False,True]);reference=None
    for arm,module in versions.items():
        cache=module.RawKeyFPSCache(authority.archive,2,2,8,dtype=torch.float16,max_bytes=1024**2)
        with cache.use():
            values,offsets,scores,stats,timing=bench.measured_run(module,cache,authority,queries,score_mask,device='cpu')
            raw_offsets=offsets.clone()
            actual_hash=bench.tensor_digest(values)
            expected_hash=bench.expected_cache_digest(keys,meta['counts'],raw_offsets.numpy())
            assert actual_hash==expected_hash
            assert stats['cache_rebuilt_blocks']==3 and stats['cache_hit_blocks']==0
            assert timing['total_wall_seconds']>=timing['snapshot_wall_seconds']>=0
            assert scores[1].item()==0
            if reference is None:reference=(raw_offsets,scores.clone(),actual_hash)
            else:
                assert torch.equal(offsets,reference[0])
                assert torch.equal(scores,reference[1])
                assert actual_hash==reference[2]
            del values,offsets,scores
        cache.clear()
        assert cache.active==cache.bytes==cache.capacity==authority.archive._query_window_cache_bytes==0


def test_interleaving_and_exclusive_failure_preservation(tmp_path):
    assert bench.interleaved_schedule(3)==[(0,'old'),(0,'new'),(1,'new'),(1,'old'),(2,'old'),(2,'new')]
    assert bench.interleaved_schedule(1)==[(0,'old'),(0,'new')]
    with pytest.raises(ValueError):bench.interleaved_schedule(2)
    output=tmp_path/'failed'
    with pytest.raises(ValueError,match='threads'):
        bench.run(NS(output=output,threads=0))
    before=(output/'failure.json').read_bytes()
    assert bench.read(output/'status.json')['status']=='failed'
    with pytest.raises(FileExistsError):bench.run(NS(output=output))
    assert (output/'failure.json').read_bytes()==before


def test_score_vector_digest_handles_scalar_chunks_and_preserves_signed_zero():
    import hashlib
    import json

    scores = torch.tensor([0., -0., 0.25, 0.5], dtype=torch.float32)
    expected = hashlib.sha256()
    expected.update(json.dumps(dict(shape=[4], dtype='torch.float32'), sort_keys=True).encode()+b'\n')
    expected.update(scores.view(torch.uint8).numpy().tobytes())
    assert bench.tensor_digest(scores) == expected.hexdigest()
    changed = scores.clone(); changed[1] = 0.
    assert bench.tensor_digest(changed) != bench.tensor_digest(scores)
