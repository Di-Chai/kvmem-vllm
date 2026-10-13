import pytest

from vllm_kvmem.gpu_raw_cache import RawCacheIndex


def test_raw_cache_is_generation_and_namespace_qualified():
    cache=RawCacheIndex(4,2)
    first=cache.reserve(('a',0,1))
    other=cache.reserve(('b',0,1))
    assert first!=other
    assert cache.get(('a',0,2)) is None
    assert cache.get(('a',0,1))==first
    assert cache.get(('b',0,1))==other
    cache.drop_namespace('a')
    assert cache.get(('a',0,1)) is None and cache.get(('b',0,1))==other
    assert len(cache.free)+len(cache.entries)==cache.slots


def test_per_user_quota_does_not_evict_another_active_user():
    cache=RawCacheIndex(4,2)
    for namespace in ('a','b'):
        for block in (0,1):cache.reserve((namespace,block,1))
    retained=[cache.get(('b',i,1)) for i in (0,1)]
    cache.reserve(('a',2,1))
    assert cache.get(('a',0,1)) is None and cache.counts=={'a':2,'b':2}
    assert [cache.get(('b',i,1)) for i in (0,1)]==retained
    assert cache.evictions==1


def test_one_gpu_write_batch_never_assigns_a_slot_twice():
    cache=RawCacheIndex(2,1)
    key=('a',1,1)
    slot=cache.reserve(key)
    assert cache.reserve(('a',2,1),{key}) is None
    assert cache.get(key)==slot and len(cache.entries)==1
    assert cache.reserve(('a',2,1))==slot
    assert cache.get(key) is None


@pytest.mark.parametrize('key',[('a',0),('',0,1),('a',True,1),('a',0,-1)])
def test_raw_cache_rejects_unqualified_keys(key):
    with pytest.raises(ValueError):RawCacheIndex(2,1).get(key)
