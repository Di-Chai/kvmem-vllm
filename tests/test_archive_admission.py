import pytest

from vllm_kvmem.archive_admission import ArchiveAdmission, ArchiveCapacity


def test_pending_dma_and_speculation_debit_ready_space_without_changing_query_size():
    gate = ArchiveAdmission(128, 4096)
    gate.update(1, {'a': ArchiveCapacity(1024, 1), 'b': ArchiveCapacity(2048, 4)}, {'a', 'b'})
    # Previous accepted 128-token query has not reached the archive yet.
    assert not gate.allows('a', 1152, 8)
    assert gate.allows('b', 2048, 128)  # A blocked request does not block its neighbour.
    # Native sampling rejected seven of the eight scheduled candidate rows.
    assert gate.allows('a', 1025, 8)
    # New worker acknowledgement moves the debit into actual repository usage.
    gate.update(2, {'a': ArchiveCapacity(1025, 0)}, {'a', 'b'})
    assert gate.allows('a', 1025, 8)  # Still fits in the existing mutable tail.
    assert not gate.allows('a', 1145, 8)
    gate.update(3, {'a': ArchiveCapacity(1025, 1)}, {'a', 'b'})
    assert gate.allows('a', 1145, 8)


def test_capacity_bound_holds_for_every_tail_and_unpublished_accepted_suffix():
    gate = ArchiveAdmission(16, 64)
    sequence = 0
    for tail in range(16):
        for ready in range(4):
            sequence += 1
            gate.update(sequence, {'a': ArchiveCapacity(64 + tail, ready)}, {'a'})
            for pending in range(24):
                for query in (1, 8, 16):
                    capacity = ready * 16 + (16 - tail if tail else 0)
                    assert gate.allows('a', 64 + tail + pending, query) == (pending + query <= capacity)
        gate.forget('a')  # Separate independent request history for the next tail.


def test_sequence_history_and_finished_request_isolation():
    gate = ArchiveAdmission(128, 10)
    gate.update(1, {'a': ArchiveCapacity(128, 1)}, {'a'})
    with pytest.raises(ValueError, match='Stale'):
        gate.update(1, {}, {'a'})
    with pytest.raises(ValueError, match='backwards'):
        gate.update(2, {'a': ArchiveCapacity(127, 1)}, {'a'})
    with pytest.raises(ValueError, match='capacity'):
        gate.update(2, {'a': ArchiveCapacity(128, 11)}, {'a'})
    assert gate.sequence == 1
    gate.update(2, {'a': ArchiveCapacity(128, 1)}, set())
    assert not gate.snapshots
    assert gate.allows('new', 0, 1280) and not gate.allows('new', 0, 1281)


def test_native_connector_returns_capacity_even_without_model_or_refresh_work():
    pytest.importorskip('vllm')
    from test_connector import pair, poll
    scheduler, worker = pair()
    gate = ArchiveAdmission(128, 10)
    scheduler.scheduler_archive_update = lambda sequence, capacity: gate.update(sequence, capacity, {'a', 'b'})
    worker.archive_snapshot = lambda: {'a': ArchiveCapacity(128, 0), 'b': ArchiveCapacity(256, 3)}
    first = poll(scheduler, worker)
    assert not first.is_empty() and gate.sequence == 1
    assert not gate.allows('a', 128, 8) and gate.allows('b', 256, 8)
    worker.archive_snapshot = lambda: {'a': ArchiveCapacity(128, 1)}
    poll(scheduler, worker)
    assert gate.sequence == 2 and gate.allows('a', 128, 8)


def test_feedback_cannot_publish_candidate_rows_before_native_eos_clipping():
    pytest.importorskip('vllm')
    from types import SimpleNamespace
    from vllm_kvmem.capture_runtime import CaptureRuntime
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    def premature_drain(*args, **kwargs):
        raise AssertionError('EOS clipping has not arrived; do not drain capture')
    runtime.drain = premature_drain
    runtime.authorities = {'a': SimpleNamespace(history_end=128, poll=lambda: 0,
        memory=SimpleNamespace(repository=SimpleNamespace(ready_slots=3)))}
    assert runtime.archive_capacity() == {'a': ArchiveCapacity(128, 3)}


def test_scheduler_preserves_native_query_and_only_defers_request_without_credits(monkeypatch):
    pytest.importorskip('vllm')
    from types import SimpleNamespace
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.scheduler import KVMemScheduler
    from vllm_kvmem.admission import QueryPolicy
    monkeypatch.setattr(Scheduler, '_reserve_prefill_lookahead', lambda self, request, history, tokens: tokens)
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler._bounded_target = True
    scheduler.session_capture_decode = False
    scheduler.window_managers = {0: SimpleNamespace(offsets={})}
    scheduler.query_policy = QueryPolicy(16, 64, 64)
    scheduler.archive_admission = ArchiveAdmission(16, 24)
    scheduler.archive_admission.update(1, {'a': ArchiveCapacity(64, 0), 'b': ArchiveCapacity(64, 2)}, {'a', 'b'})
    a, b = [SimpleNamespace(request_id=k, num_prompt_tokens=64) for k in ('a', 'b')]
    assert scheduler._reserve_prefill_lookahead(a, 64, 8) == 0
    assert scheduler._reserve_prefill_lookahead(b, 64, 8) == 8
    scheduler.archive_admission.update(2, {'a': ArchiveCapacity(64, 1)}, {'a', 'b'})
    assert scheduler._reserve_prefill_lookahead(a, 64, 8) == 8
