import asyncio
import pytest

from vllm_kvmem.elastic import ActiveBudget, CacheGeometry, HistoryStore, select_context


def test_history_isolation_version_and_transactional_reservations():
    store = HistoryStore(48, 3, 10)
    key, session, prompt = store.begin('a', 'same', 0, [1, 2], 3)
    assert prompt == [1, 2] and store.bytes == 0
    with pytest.raises(RuntimeError):
        store.begin('a', 'same', 0, [4], 1)
    other, _, _ = store.begin('b', 'same', 0, [7], 1)
    store.finish(key, [1, 2], [3], waiting_for_tool=True)
    assert session.waiting_for_tool and list(session.tokens) == [1, 2, 3]
    assert store.cache_salt('a', 'same') != store.cache_salt('b', 'same')
    store.abort(other)
    with pytest.raises(RuntimeError):
        store.begin('a', 'same', 0, [4], 1)
    again, _, prompt = store.begin('a', 'same', 1, [4], 1)
    assert prompt == [1, 2, 3, 4]
    store.abort(again)
    assert store.bytes == 12 and list(session.tokens) == [1, 2, 3]


def test_expiry_never_reclaims_active_owner_and_budget_includes_pending_growth():
    now = [0]
    store = HistoryStore(24, 2, 5, clock=lambda: now[0])
    key, _, _ = store.begin('a', 's', 0, [1], 3)
    with pytest.raises(BufferError):
        store.begin('b', 's', 0, [2], 2)
    now[0] = 10
    store.expire()
    assert key in store.entries
    store.finish(key, [1], [2])
    now[0] = 16
    store.expire()
    assert not store.entries and store.bytes == 0
    with pytest.raises(LookupError):
        store.begin('a', 's', 1, [3], 1)


def test_output_overrun_does_not_publish_history():
    store = HistoryStore(32, 2, 5)
    key, session, _ = store.begin('a', 's', 0, [1], 1)
    with pytest.raises(ValueError):
        store.finish(key, [1], [2, 3])
    with pytest.raises(ValueError):
        store.finish(key, [9], [2])
    assert session.busy and not session.tokens and store.bytes == 0
    store.abort(key)


def test_user_history_quota_cannot_starve_another_user():
    store = HistoryStore(100, 8, 10, per_user_bytes=16, per_user_sessions=1)
    a, _, _ = store.begin('a', 's', 0, [1], 2)
    with pytest.raises(BufferError):
        store.begin('a', 'other', 0, [2], 1)
    store.finish(a, [1], [2, 3])
    with pytest.raises(BufferError):
        store.begin('a', 's', 1, [4], 1)
    b, _, _ = store.begin('b', 's', 0, [9], 1)
    store.finish(b, [9], [10])
    assert store.bytes == 20 and store.user_bytes == {'a': 12, 'b': 8}
    store.delete('a', 's')
    assert store.bytes == 8 and store.user_bytes['a'] == 0


def test_native_geometry_accounts_for_pages_and_recurrent_speculation():
    geometry = CacheGeometry((
        {'block_tokens': 100, 'page_bytes': 1000, 'recurrent': False, 'speculative_blocks': 0},
        {'block_tokens': 100, 'page_bytes': 2000, 'recurrent': True, 'speculative_blocks': 3}), 20000)
    assert geometry.active_bytes(101) == 12000


def test_context_selection_is_explicit_bounded_and_does_not_destroy_history():
    tokens = list(range(20))
    assert select_context(tokens, [[0, 2], [16, 20]], 6) == [0, 1, 16, 17, 18, 19]
    assert tokens == list(range(20))
    with pytest.raises(BufferError):
        select_context(tokens, [[0, 2], [16, 20]], 5)
    for spans in ([[1, 3], [2, 4]], [[0, 21]], [[False, 1]], []):
        with pytest.raises(ValueError):
            select_context(tokens, spans, 20)


def test_deadlines_fairness_cancellation_and_byte_pressure():
    async def scenario():
        budget = ActiveBudget(100, 3, 5)
        a = budget.enter('a', 60, 10)
        await a.ready
        b = budget.enter('b', 60, 10)
        duplicate = budget.enter('a', 30, 10)
        c = budget.enter('c', 30, .01)
        await c.ready
        assert budget.bytes == 90 and not b.ready.done() and not duplicate.ready.done()
        budget.release(c)
        expired = budget.enter('d', 80, .001)
        await asyncio.sleep(.005)
        budget.release(a)
        await b.ready
        await duplicate.ready
        with pytest.raises(TimeoutError):
            await expired.ready
        assert budget.bytes == 90
        budget.release(b)
        budget.release(duplicate)
        assert budget.bytes == 0 and not budget.active and not budget.waiting
    asyncio.run(scenario())
