"""KVMEM_LIFECYCLE_SNAPSHOT_CACHE: session views spliced from cached per-entry JSON
are byte-identical to json.dumps(SessionLedger.snapshot())."""
import json
import types

from vllm_kvmem.serving.session_control import SessionLedger, _Entry


def ledger(entries):
    fake = types.SimpleNamespace(entries=entries)
    fake.snapshot = lambda: SessionLedger.snapshot(fake)
    fake.snapshot_json = lambda: SessionLedger.snapshot_json(fake)
    return fake


def entry(name, **kw):
    e = _Entry(namespace=(name, name), generation=1, request_id=None, phase='idle', history=4096, window=2048,
               tokens=b'abc' + name.encode(), query=(4000, 4032), mandatory=((0, 64, 'root'),),
               active_records=tuple((i * 32, 32) for i in range(64)), allocated_records=7,
               protected_records=5, ready_records=3)
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def same(fake):
    assert fake.snapshot_json() == json.dumps(fake.snapshot())


def test_identical_bytes_through_every_kind_of_change():
    entries = {('a', 'a'): entry('a'), ('b', 'b'): entry('b'), ('c', 'c'): entry('c', phase='active')}
    fake = ledger(entries)
    same(fake)
    same(fake)                                          # cached
    a = entries[('a', 'a')]
    a.active_records = a.active_records[:10] + ((9999, 7),)
    same(fake)
    a.history, a.window = 5000, 2500
    same(fake)
    a.query, a.mandatory = None, ()
    same(fake)
    a.tokens = b'new prefix'
    same(fake)
    a.generation += 1; a.ready_records = 9
    same(fake)
    entries[('c', 'c')].phase = 'idle'
    same(fake)
    entries[('b', 'b')].phase = 'evicting'
    same(fake)
    del entries[('a', 'a')]
    same(fake)
    assert set(fake._view_json) == {('c', 'c')}         # stale views are dropped
    entries.clear()
    same(fake)
    assert fake.snapshot_json() == '[]'


def test_equal_but_new_active_records_tuple_reencodes_safely():
    entries = {('a', 'a'): entry('a')}
    fake = ledger(entries)
    same(fake)
    entries[('a', 'a')].active_records = tuple(list(entries[('a', 'a')].active_records))
    same(fake)


def test_scheduler_file_bytes_match_with_and_without_the_cache(tmp_path, monkeypatch):
    import re
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).parent))
    from test_capacity import make_gate, request
    from vllm_kvmem.serving.scheduler import KVMemScheduler
    gate, _, _ = make_gate(blocks=5)
    req = request('a')
    assert gate.allocate_slots(req, 64) is not None
    entries = {('s1', 's1'): entry('s1'), ('s2', 's2'): entry('s2'), ('s3', 's3'): entry('s3', phase='active')}
    sessions = ledger(entries)
    sessions.hits = sessions.misses = sessions.waits = 0
    sessions.owner_snapshot = lambda: []
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler._bounded_target = True
    scheduler.kv_cache_manager = gate
    scheduler.requests = {'a': req}
    scheduler.transactions = {}
    scheduler.connector = SimpleNamespace(pending={}, sessions=sessions)
    monkeypatch.setenv('KVMEM_LIFECYCLE_TRACE', '1')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    path = tmp_path / 'scheduler_resources.json'
    text = lambda: re.sub(r'"at": [0-9.e+-]+', '"at": 0', path.read_text())
    for change in (None, lambda: setattr(entries[('s1', 's1')], 'history', 9000),
                   lambda: setattr(entries[('s3', 's3')], 'phase', 'idle')):
        if change:
            change()
        monkeypatch.delenv('KVMEM_LIFECYCLE_SNAPSHOT_CACHE', raising=False)
        assert scheduler._write_lifecycle_snapshot(force=True)
        plain = text()
        monkeypatch.setenv('KVMEM_LIFECYCLE_SNAPSHOT_CACHE', '1')
        assert scheduler._write_lifecycle_snapshot(force=True)
        assert text() == plain and '\\u0000' not in plain
        assert len(json.loads(plain)['session_views']) == sum(e.phase == 'idle' for e in entries.values())
