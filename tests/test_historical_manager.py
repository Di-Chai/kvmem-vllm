"""Native manager bindings and real archive-backed asynchronous LRU decisions."""
import time

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.memory_storage import MemoryArchive


def eventually(predicate):
    end = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < end
        time.sleep(.001)


def configuration(mode=n.HistoricalGcMode.DELETE, background=False, max_scopes=1):
    config = n.HistoricalGcConfig()
    config.mode = mode
    config.background = background
    config.max_scopes = max_scopes
    config.poll_ms = 1
    config.scan_blocks = 8
    config.delete_blocks = 2
    config.ready_low = 2
    config.ready_high = 4
    return config


def actual_archive(tmp_path, mode=n.HistoricalGcMode.DELETE, background=False):
    archive = MemoryArchive(str(tmp_path), 8, 16, native_payloads=True)
    worker = archive.enable_history_gc(configuration(mode, background))
    repo = n.Repository(8, 4)
    scope = archive.register_history_scope('a', repo)
    return archive, worker, repo, scope


def append(archive, repo, scope, count=8):
    leases = [repo.reserve(i*4, 4).commit() for i in range(count)]
    values = [bytes([i+1])*8 for i in range(count)]
    archive.write('a', leases, b''.join(values)).result(5)
    for lease in leases:
        assert scope.update(lease, n.HISTORY_SEALED | n.HISTORY_GPU)
    return leases, values


def drain(archive, scope):
    observed = []
    for completion in scope.completions(8):
        if completion.cpu and not archive.retire_cpu_record('a', completion.handle):
            continue
        assert scope.acknowledge(completion)
        observed.append(completion)
    return observed


def steps(archive, worker, scope, count=16, ack=True):
    for _ in range(count):
        worker.advance()
        if ack:
            drain(archive, scope)


@pytest.mark.parametrize('mode', [n.HistoricalGcMode.OFF, n.HistoricalGcMode.SHADOW])
def test_off_and_shadow_keep_all_real_archive_data(tmp_path, mode):
    archive, worker, repo, scope = actual_archive(tmp_path, mode)
    leases, values = append(archive, repo, scope)
    try:
        steps(archive, worker, scope)
        assert all(repo.pin(lease.handle) for lease in leases)
        assert bytes(archive.read('a', leases).result(5)[0]) == b''.join(values)
        assert scope.stats['retired'] == scope.stats['cpu_released'] == 0
        assert archive.io_usage()['cpu_authority_bytes'] == 64
        if mode == n.HistoricalGcMode.SHADOW:
            assert scope.stats['shadow_decisions'] == 4
    finally:
        leases.clear()
        eventually(archive.close)


def test_actual_archive_lru_cpu_releases_do_not_wait_for_gpu_leases(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    try:
        steps(archive, worker, scope)
        assert scope.stats['retired'] == scope.stats['cpu_released'] == 4
        assert scope.stats['slots_released'] == 0
        assert repo.ready_slots == 0
        assert archive.io_usage()['cpu_authority_bytes'] == 32
        assert len(archive._entries) == 4
        assert leases[0].record.token_begin == 0
        assert not repo.pin(leases[0].handle)
        with pytest.raises(KeyError):
            archive.read('a', [leases[0]])
        leases.clear()
        steps(archive, worker, scope)
        assert repo.ready_slots == 4
        assert scope.stats['pending_slots'] == 0
    finally:
        leases.clear()
        eventually(archive.close)


def test_observing_cpu_completion_does_not_acknowledge_descriptor_cleanup(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    leases, _ = append(archive, repo, scope)
    try:
        steps(archive, worker, scope, ack=False)
        pending = scope.completions(8)
        assert len(pending) == 4 and all(event.cpu for event in pending)
        leases.clear()
        steps(archive, worker, scope, ack=False)
        assert repo.ready_slots == 0
        assert len(scope.completions(8)) == 4
        drain(archive, scope)
        steps(archive, worker, scope)
        assert repo.ready_slots == 4
        assert len(archive._entries) == 4
    finally:
        leases.clear()
        eventually(archive.close)


def test_native_background_manager_progresses_without_foreground_policy_calls(tmp_path):
    archive, _, repo, scope = actual_archive(tmp_path, background=True)
    # Pause while establishing this test's fixed use ordering.
    scope.pause_policy(True)
    leases, _ = append(archive, repo, scope)
    try:
        scope.used(100)
        assert scope.update(leases[6], n.HISTORY_SEALED)
        scope.used(200)
        scope.pause_policy(False)
        eventually(lambda: scope.stats['cpu_released'] == 4)
        assert not repo.pin(leases[6].handle)  # Cold even though it was created recently.
        assert repo.pin(leases[7].handle)
        assert repo.ready_slots == 0
        leases.clear()
        eventually(lambda: (drain(archive, scope), scope.stats['slots_released'] == 4)[1])
    finally:
        leases.clear()
        eventually(archive.close)


def test_native_scope_registry_is_bounded_and_foreign_owners_are_rejected(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path, n.HistoricalGcMode.OFF)
    lease = repo.reserve(0, 4).commit()
    archive.write('a', [lease], b'original').result(5)
    with pytest.raises(BufferError, match='scope capacity'):
        archive.register_history_scope('b', n.Repository(8, 4))
    foreign = n.Repository(8, 4).reserve(0, 4).commit()
    allocation = next(iter(archive._entries.values())).allocation
    source = archive.tier.payloads.acquire(archive.tier.payload_handle(allocation))
    try:
        with pytest.raises(ValueError, match='foreign repository'):
            scope.publish(foreign, source, n.HISTORY_SEALED)
    finally:
        del source
        lease.reset()
        foreign.reset()
        eventually(archive.close)


def test_history_gc_requires_native_storage_and_setup_before_archive_use(tmp_path):
    legacy = MemoryArchive(str(tmp_path / 'legacy'), 8, 2)
    try:
        with pytest.raises(ValueError, match='native CPU'):
            legacy.enable_history_gc(configuration())
    finally:
        assert legacy.close()
    archive = MemoryArchive(str(tmp_path / 'native'), 8, 8, native_payloads=True)
    repo = n.Repository(8, 4)
    lease = repo.reserve(0, 4).commit()
    archive.write('a', [lease], b'original').result(5)
    try:
        with pytest.raises(RuntimeError, match='before archive use'):
            archive.enable_history_gc(configuration())
    finally:
        lease.reset()
        eventually(archive.close)


@pytest.mark.parametrize('keep', [True, False])
def test_configured_factory_parameters_control_real_archive_reclamation(tmp_path, keep):
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.historical_gc import HistoricalGcSettings

    memory = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=12,
                         repository_blocks=8, sink_blocks=0, recent_blocks=0)
    config = HistoricalGcSettings(mode='delete', policy='lru',
        policy_params={'keep_mandatory': keep}, ready_low=2, ready_high=4).native(memory, 1)
    config.background = False
    archive = MemoryArchive(str(tmp_path), 8, 16, native_payloads=True)
    worker = archive.enable_history_gc(config)
    repo = n.Repository(8, 4)
    scope = archive.register_history_scope('a', repo)
    leases, _ = append(archive, repo, scope)
    assert all(scope.update(lease, n.HISTORY_SEALED | n.HISTORY_GPU | n.HISTORY_MANDATORY)
               for lease in leases)
    try:
        steps(archive, worker, scope)
        stats = archive.history_stats()
        assert stats['policy'] == 'lru' and stats['policy_params'] == {'keep_mandatory': keep}
        assert scope.stats['retired'] == scope.stats['cpu_released'] == (0 if keep else 4)
        assert archive.io_usage()['cpu_authority_bytes'] == (64 if keep else 32)
        assert all(bool(repo.pin(lease.handle)) == (keep or i >= 4) for i, lease in enumerate(leases))
        assert all(lease.record.token_count == 4 for lease in leases)
    finally:
        leases.clear()
        eventually(archive.close)


@pytest.mark.parametrize('policy,params', [
    ('unknown', {}), ('lru', {'typo': True}),
    *[('lru', {'keep_mandatory': v}) for v in (0, 1, 1.0, 'true')],
])
def test_native_worker_validates_factory_before_scope_registration_even_when_off(tmp_path, policy, params):
    archive = MemoryArchive(str(tmp_path), 8, 16, native_payloads=True)
    config = configuration(mode=n.HistoricalGcMode.OFF)
    config.policy, config.policy_params = policy, params
    try:
        with pytest.raises(ValueError, match='historical_gc.policy'):
            archive.enable_history_gc(config)
        assert archive._history_worker is None
    finally:
        eventually(archive.close)
