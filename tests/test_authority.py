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


def test_submitted_snapshot_consumes_resident_budget_until_completion_is_observed(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path, resident=1)
    original_write = archive.write
    admitted = []
    observed = False

    class DeferredObservation:
        def __init__(self, job):
            self.job = job

        def done(self):
            return observed and self.job.done()

        def result(self, timeout=None):
            return self.job.result(timeout)

    def write(*args):
        job = original_write(*args)
        assert job is not None
        admitted.append(job)
        return DeferredObservation(job)

    monkeypatch.setattr(archive, "write", write)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    record = next(iter(authority.records.values()))
    assert record.submitted and record.raw is None and not record.archived
    admitted[0].result(timeout=5)
    # I/O-owned snapshots still count even after staging tensors were released;
    # durable publication, rather than submission, makes the budget reusable.
    assert not authority.append(*payload(4, 1), history_begin=4, local_rope_begin=4)
    assert authority.history_end == 4 and not record.archived
    observed = True
    assert authority.append(*payload(4, 1), history_begin=4, local_rope_begin=4)
    assert record.archived and authority.history_end == 5
    assert authority.close() and archive.close()


def test_sealed_batch_retry_preserves_record_order_bytes_checksums_and_accepted_prefix(tmp_path, monkeypatch):
    import hashlib

    authority, archive = setup(tmp_path)
    original_write = archive.write
    packets = []
    admit = False

    def write(namespace, leases, data):
        assert type(data) is bytes
        packets.append(([(lease.handle.slot, lease.handle.generation) for lease in leases], data))
        return original_write(namespace, leases, data) if admit else None

    monkeypatch.setattr(archive, "write", write)
    raw, values = payload(0, 12)
    expected_raw, expected_values = raw.clone(), values.clone()
    assert authority.append(raw, values, history_begin=0, local_rope_begin=0, accepted=9)
    raw.zero_()
    values.zero_()
    assert not authority.flush()
    records = list(authority.records.values())
    expected_packets = [expected_raw[:, begin:begin+4].contiguous().view(torch.uint8).numpy().tobytes()
                        + expected_values[:, begin:begin+4].contiguous().numpy().tobytes()
                        for begin in (0, 4)]
    keys = [(r.handle.slot, r.handle.generation) for r in records[:2]]
    assert packets == [(keys, b"".join(expected_packets))] * 2
    assert [r.digest for r in records[:2]] == [hashlib.sha256(data).hexdigest() for data in expected_packets]
    assert records[2].count == 1 and records[2].digest is None
    for i, record in enumerate(records):
        begin = i * 4
        assert torch.equal(record.key_sum, expected_raw[:, begin:begin+record.count].float().sum(dim=1))
    admit = True
    drain(authority)
    assert packets[-1] == packets[0]
    for i, record in enumerate(records):
        actual_raw, actual_values = authority.read_record(record.handle)
        begin = i * 4
        assert torch.equal(actual_raw, expected_raw[:, begin:begin+record.count])
        assert torch.equal(actual_values, expected_values[:, begin:begin+record.count])
    assert authority.history_end == 9
    assert authority.close() and archive.close()


def test_cancelled_unsubmitted_record_drops_staging_only_after_lease_release(tmp_path, monkeypatch):
    import weakref

    authority, archive = setup(tmp_path, resident=1)
    monkeypatch.setattr(archive, "write", lambda *args: None)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    record = next(iter(authority.records.values()))
    raw = weakref.ref(record.raw)
    lease = authority.memory.repository.pin(record.handle)
    del record
    assert not authority.close()
    assert raw() is not None and archive.tier.used_slots == 0
    lease.reset()
    assert authority.close()
    assert raw() is None and not authority.records
    assert authority.close() and archive.close()
