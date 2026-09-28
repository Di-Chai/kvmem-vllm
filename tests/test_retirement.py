import json
import time
from threading import Event
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.options import RuntimeOptions
from test_authority import setup, payload


@pytest.mark.parametrize('verify', [False, True])
def test_retirement_drains_real_archive_and_labels_optional_readback(tmp_path, monkeypatch, verify):
    authority, archive = setup(tmp_path / 'archive')
    assert authority.append(*payload(0, 5), history_begin=0, local_rope_begin=0)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(memory=authority.memory.config, verify_archive_on_finish=verify)
    runtime.config = runtime.options.memory
    runtime.runner = NS(vllm_config=NS(max_concurrent_batches=1), decode_query_len=1)
    runtime.output = tmp_path
    runtime.archive = archive
    runtime.downloads = NS(pending={})
    runtime.queue = []
    runtime.authorities = {'a': authority}
    runtime.accepted, runtime.rejected, runtime.last_steps = {'a': 5}, {'a': 0}, {'a': []}
    runtime.completed, runtime.batches = [], 1
    original = authority.read_record
    reads = []

    def read(handle):
        reads.append(handle)
        return original(handle)

    monkeypatch.setattr(authority, 'read_record', read)
    runtime.finish({'a'})
    report = json.loads((tmp_path / 'archive.json').read_text())
    assert report['archive_verification'] == ('full_checksum' if verify else 'publication_only')
    assert report['completed'][0]['sealed_records_archived'] == 1
    assert report['completed'][0]['sealed_records_verified'] == int(verify)
    assert len(reads) == int(verify)
    assert not runtime.authorities and archive.tier.used_slots == 0
    assert not runtime.accepted and not runtime.rejected
    assert archive.close()


def test_async_retirement_is_nonblocking_and_preserves_other_archive_namespaces(tmp_path, monkeypatch):
    from vllm_kvmem.retirement import RetirementQueue
    a, archive = setup(tmp_path / 'archive')
    b, _ = setup(tmp_path / 'unused', name='b', archive=archive)
    assert a.append(*payload(0, 5), history_begin=0, local_rope_begin=0)
    started, release = Event(), Event()
    original = a.read_record
    def held_read(handle):
        started.set()
        assert release.wait(5)
        return original(handle)
    monkeypatch.setattr(a, 'read_record', held_read)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(memory=a.memory.config, async_archive_retirement=True)
    runtime.config = runtime.options.memory
    runtime.runner = NS(vllm_config=NS(max_concurrent_batches=1), decode_query_len=1)
    runtime.output, runtime.archive = tmp_path, archive
    runtime.downloads, runtime.queue = NS(pending={}), []
    runtime.authorities = {'a': a, 'b': b}
    runtime.accepted, runtime.rejected, runtime.last_steps = {'a': 5}, {'a': 0}, {'a': []}
    runtime.completed, runtime.batches, runtime.retiring = [], 1, {}
    runtime.retirement = RetirementQueue(2, True)
    try:
        runtime.finish({'a'})
        assert started.wait(5)
        assert not runtime.retirement_done('a')
        assert runtime.authorities == {'b': b} and set(runtime.retiring) == {'a'}
        # A surviving request can publish while a's integrity read is blocked.
        assert b.append(*payload(8, 5), history_begin=0, local_rope_begin=0)
        for job, _ in b.writes:
            job.result(timeout=5)
        b.poll()
        release.set()
        deadline = time.monotonic() + 5
        while not runtime.retirement_done('a'):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert archive.tier.used_slots == 1
        report = json.loads((tmp_path / 'archive.json').read_text())
        assert report['active_requests'] == 1 and report['retiring_requests'] == 0
        assert report['completed'][0]['sealed_records_verified'] == 1
        assert b.read_record(next(iter(b.records.values())).handle)[0].shape[1] == 4
        assert runtime.retirement.close()
    finally:
        release.set()
        runtime.retirement.executor.shutdown(wait=True)
    assert b.close() and archive.close()


def test_retirement_queue_bound_and_checksum_failure_cannot_acknowledge_cleanup(tmp_path):
    from vllm_kvmem.retirement import RetirementQueue
    from test_authority import drain
    authority, archive = setup(tmp_path)
    authority.append(*payload(0, 5), history_begin=0, local_rope_begin=0)
    drain(authority)
    next(iter(authority.records.values())).digest = 'corrupted'
    queue = RetirementQueue(1, True)
    queue.submit('a', authority)
    with pytest.raises(BufferError, match='capacity'):
        queue.submit('b', authority)
    with pytest.raises(RuntimeError, match='checksum'):
        queue.jobs['a'].result(timeout=5)
    with pytest.raises(RuntimeError, match='checksum'):
        queue.poll('a')
    assert 'a' in queue.jobs and not queue.close()
    assert archive.tier.used_slots == 1
    queue.executor.shutdown(wait=True)
    assert authority.close() and archive.close()


def test_background_reclamation_waits_for_external_generation_lease(tmp_path):
    from vllm_kvmem.retirement import RetirementQueue
    from test_authority import drain
    authority, archive = setup(tmp_path)
    authority.append(*payload(0, 5), history_begin=0, local_rope_begin=0)
    drain(authority)
    handle = next(iter(authority.records.values())).handle
    lease = authority.memory.repository.pin(handle)
    queue = RetirementQueue(1, False)
    try:
        queue.submit('a', authority)
        deadline = time.monotonic() + 5
        while not authority.closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert queue.poll('a') is None and archive.tier.used_slots == 1
        lease.reset()
        result = queue.jobs['a'].result(timeout=5)
        assert result['sealed_records_verified'] == 0
        assert queue.poll('a')['history_end'] == 5
        assert archive.tier.used_slots == 0 and queue.close()
    finally:
        lease.reset()
        queue.executor.shutdown(wait=True)
    assert archive.close()
