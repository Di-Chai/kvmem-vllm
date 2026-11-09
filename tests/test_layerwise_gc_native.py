"""KVMEM_LAYERWISE_HISTORICAL_GC on the real native archive, payload arena and GC worker."""
import pytest

from vllm_kvmem import _native as n
from test_historical_manager import actual_archive, append, eventually, steps


def test_current_window_hold_keeps_held_records_out_of_retirement(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    oldest = [lease.handle for lease in leases[:4]]
    try:
        hold = scope.protect_current(oldest)
        assert hold is not None
        steps(archive, worker, scope)
        # Unheld, LRU retires exactly these four oldest (see the archive LRU
        # test); held, it meets the same pressure with the four newer ones.
        assert all(repo.pin(h) for h in oldest)
        assert scope.stats['retired'] == 4
        assert not any(repo.pin(lease.handle) for lease in leases[4:])
        del hold
    finally:
        leases.clear()
        eventually(archive.close)


def test_a_paused_scope_commits_no_retirement_until_resumed(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    try:
        scope.pause_policy(True)
        steps(archive, worker, scope)
        assert scope.policy_paused and scope.stats['retired'] == 0
        scope.pause_policy(False)
        steps(archive, worker, scope)
        assert scope.stats['retired'] == 4
    finally:
        leases.clear()
        eventually(archive.close)


def test_prepared_reader_sources_are_gc_read_rights_with_exact_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, values = append(archive, repo, scope)
    try:
        pins = [repo.pin(lease.handle) for lease in leases]
        reader = archive.prepare_region_reader('a', pins)
        assert all(isinstance(source, n.HistoricalPayloadRead) for source in reader.sources)
        assert [bytes(memoryview(source)) for source in reader.sources] == values
        assert reader.close()
        monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '0')
        with pytest.raises(ValueError, match='no-GC'):
            archive.prepare_region_reader('a', pins)
    finally:
        pins = None
        leases.clear()
        eventually(archive.close)


def test_checkpoint_policy_hold_keeps_its_records_until_dropped(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from vllm_kvmem.layerwise_gc_holds import checkpoint_hold
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    oldest = [lease.handle for lease in leases[:4]]
    try:
        rows = [[(h.slot, h.generation, 32*i, 32, 32*i) for i, h in enumerate(oldest)]]
        window = scope.protect_current(oldest)     # the capturing request's live window hold
        # This foreground worker acknowledges a pause only when advanced.
        import threading
        stop = threading.Event()
        def advance():
            while not stop.is_set():
                worker.advance()
                stop.wait(.001)
        pump = threading.Thread(target=advance)
        pump.start()
        try:
            hold = checkpoint_hold(NS(memory=NS(_history_scope=scope)), rows)
        finally:
            stop.set()
            pump.join()
        assert hold is not None
        del window                                 # the request finishes; only the checkpoint holds
        steps(archive, worker, scope)
        assert all(repo.pin(h) for h in oldest) and scope.stats['retired'] == 4
        del hold
    finally:
        leases.clear()
        eventually(archive.close)


@pytest.mark.parametrize('route', ['python', 'native', 'paired'])
def test_prepared_reads_through_gc_read_rights_copy_exact_bytes(tmp_path, monkeypatch, route):
    monkeypatch.setenv('KVMEM_LAYERWISE_HISTORICAL_GC', '1')
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, values = append(archive, repo, scope)
    try:
        pins = [repo.pin(lease.handle) for lease in leases]
        reader = archive.prepare_region_reader('a', pins)
        if route == 'native':
            reader.native = n.RecordSources(list(reader.sources), archive.record_bytes)
        indices, regions = (5, 0, 7), ((1, 3), (4, 2))
        target = bytearray(len(indices)*5)
        assert reader.read(indices, regions, destination=target, paired_native=route == 'paired') is not None
        expected = b''.join(values[i][1:4] for i in indices) + b''.join(values[i][4:6] for i in indices)
        assert bytes(target) == expected
        assert reader.close()
    finally:
        pins = reader = None
        leases.clear()
        eventually(archive.close)
