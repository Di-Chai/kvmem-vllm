import pytest

torch = pytest.importorskip("torch")

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive


def setup(tmp_path, name="a", archive=None, resident=4):
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
                        repository_blocks=8, sink_blocks=1, recent_blocks=0)
    archive = archive or AsyncArchive(str(tmp_path), 2 * 4 * 1 * 2 * 3, 16)
    return RequestAuthority(name, config, archive, 2, 1, 2, resident_records=resident), archive


def payload(begin, count):
    raw = torch.arange(begin * 4, (begin + count) * 4, dtype=torch.float32)
    raw = raw.reshape(count, 2, 1, 2).transpose(0, 1).to(torch.bfloat16)
    return raw, raw.to(torch.uint8)


def drain(authority):
    assert authority.flush()
    for job, _ in authority.writes:
        job.result(timeout=5)
    authority.poll()


def test_decode_extends_one_tail_and_only_complete_records_are_archived(tmp_path):
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 3), history_begin=0, local_rope_begin=0, accepted=2)
    assert authority.history_end == 2
    assert archive.tier.used_slots == 0
    first = next(iter(authority.records.values())).handle
    raw, values = authority.read_record(first)
    assert torch.equal(raw, payload(0, 2)[0])
    assert torch.equal(values, payload(0, 2)[1])
    assert authority.append(*payload(2, 5), history_begin=2, local_rope_begin=2, accepted=3)
    drain(authority)
    assert authority.history_end == 5
    assert archive.tier.used_slots == 1
    records = list(authority.records.values())
    assert records[0].handle == first and records[0].count == 4
    assert records[0].raw is None and records[0].archived
    assert records[1].count == 1 and not records[1].submitted
    assert torch.equal(authority.read_record(first)[0], payload(0, 4)[0])
    handles, means = authority.mean_keys()
    assert handles[0] == first
    assert torch.equal(means[:, 0], payload(0, 4)[0].float().mean(dim=1))
    assert torch.equal(means[:, 1], payload(4, 1)[0].float().mean(dim=1))
    assert authority.close()
    assert archive.tier.used_slots == 0
    assert archive.close()


def test_backpressure_does_not_publish_or_consume_history(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path, resident=1)
    original_write = archive.write
    monkeypatch.setattr(archive, "write", lambda *a, **kw: None)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    assert not authority.append(*payload(4, 1), history_begin=4, local_rope_begin=4)
    assert authority.history_end == 4 and len(authority.records) == 1
    monkeypatch.setattr(archive, "write", original_write)
    drain(authority)
    assert authority.append(*payload(4, 1), history_begin=4, local_rope_begin=4)
    assert authority.history_end == 5
    assert authority.close()
    assert archive.close()


def test_shared_archive_namespaces_and_live_read_lease_survive_other_cancel(tmp_path):
    a, archive = setup(tmp_path)
    b, _ = setup(tmp_path, "b", archive)
    assert a.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    assert b.append(*payload(20, 4), history_begin=0, local_rope_begin=0)
    drain(a)
    drain(b)
    ha = next(iter(a.records.values())).handle
    hb = next(iter(b.records.values())).handle
    assert ha == hb  # Same local slot/generation, different request namespaces.
    lease = a.memory.repository.pin(ha)
    assert not a.close()
    assert torch.equal(b.read_record(hb)[0], payload(20, 4)[0])
    lease.reset()
    assert a.close()
    assert archive.tier.used_slots == 1
    assert b.close()
    assert archive.close()


def test_zero_acceptance_and_bad_history_never_enter_retrieval_index(tmp_path):
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0, accepted=0)
    assert not authority.records
    assert authority.history_end == 0
    with pytest.raises(ValueError, match="history"):
        authority.append(*payload(1, 1), history_begin=1, local_rope_begin=0)
    assert authority.history_end == 0
    assert authority.close()
    assert archive.close()


def test_query_selection_uses_archived_accepted_keys_and_append_keeps_logical_history(tmp_path):
    authority, archive = setup(tmp_path)
    raw = torch.cat([torch.full((2, 4, 1, 2), value, dtype=torch.bfloat16)
                     for value in (0, 1, 10, -10)], dim=1)
    values = torch.zeros_like(raw, dtype=torch.uint8)
    assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
    drain(authority)
    handles, _ = authority.mean_keys()
    queries = torch.ones(2, 1, 2, 2)
    plan = authority.select_for_queries(queries)
    assert plan.total_window_tokens == 8
    assert [r.block_id for r in plan.remaps] == [handles[0].slot, handles[2].slot]
    assert torch.equal(authority.read_record(handles[2])[0], raw[:, 8:12])
    from vllm_kvmem.authority import _Ready
    authority.memory.attach_completion(_Ready())
    assert authority.memory.commit_selection()
    del plan
    assert authority.append(*payload(16, 1), history_begin=16, local_rope_begin=8)
    assert authority.history_end == 17
    assert len(authority.records) == 5
    assert authority.close()
    assert archive.close()


def test_shared_queue_progress_can_come_from_another_request(tmp_path):
    from threading import Event
    from vllm_kvmem.capture_runtime import CaptureRuntime
    archive = AsyncArchive(str(tmp_path), 48, 16, workers=1, max_pending=1)
    a, _ = setup(tmp_path, "a", archive, resident=1)
    b, _ = setup(tmp_path, "b", archive, resident=1)
    real = archive.tier
    started, finish = Event(), Event()

    class DelayedTier:
        def __getattr__(self, key):
            return getattr(real, key)

        def io(self, *args):
            started.set()
            assert finish.wait(5)
            return real.io(*args)

    archive.tier = DelayedTier()
    try:
        assert a.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        assert started.wait(5)
        assert b.append(*payload(20, 4), history_begin=0, local_rope_begin=0)
        assert not b.writes
        assert not b.append(*payload(24, 1), history_begin=4, local_rope_begin=4)
    finally:
        finish.set()
    for job, _ in a.writes:
        job.result(timeout=5)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.authorities = {"a": a, "b": b}
    runtime._wait_archive_progress()  # Already completed elsewhere; retry, don't fail.
    drain(b)
    assert b.append(*payload(24, 1), history_begin=4, local_rope_begin=4)
    assert a.close() and b.close()
    assert archive.close()
