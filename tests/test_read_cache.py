import pytest
from vllm_kvmem.read_cache import RecordReadCache


def test_cache_bounds_generations_and_read_ownership_survive_eviction():
    cache = RecordReadCache(12, 8)
    a, b = ('a', 0, 1), ('b', 0, 1)
    assert cache.put(a, b'abcd') and cache.put(b, b'efgh')
    held = cache.get(a)
    assert cache.put(('a', 1, 1), b'ijkl')
    assert cache.put(('a', 2, 1), b'mnop')
    assert cache.used_bytes == 12 and cache.by_namespace == {'a': 8, 'b': 4}
    assert cache.get(a) is None and held == b'abcd'
    assert cache.get(b) == b'efgh'
    assert cache.put(('c', 0, 1), b'qrst')
    assert cache.used_bytes == 12
    cache.remove(b)
    assert cache.get(b) is None
    assert cache.put(('b', 0, 2), b'uvwx')
    assert cache.get(('b', 0, 1)) is None
    cache.clear()
    assert cache.used_bytes == 0 and not cache.by_namespace
    assert held == b'abcd'


def test_cache_cannot_mutate_authority_or_hide_backing_buffer_memory():
    cache = RecordReadCache(4, 4)
    key = ('a', 0, 1)
    assert not cache.put(key, b'12345')
    with pytest.raises(ValueError, match='own immutable'):
        cache.put(key, memoryview(b'large backing buffer')[:1])
    assert cache.put(key, b'1234')
    with pytest.raises(ValueError, match='immutable data'):
        cache.put(key, b'5678')
    assert cache.get(key) == b'1234' and cache.used_bytes == 4
    assert not RecordReadCache(0, 0).put(key, b'1')
