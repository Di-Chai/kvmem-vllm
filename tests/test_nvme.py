from vllm_kvmem import _native


def test_strict_reservation_owns_physical_space_but_leaves_no_named_file(tmp_path):
    import os
    from pathlib import Path
    total = 4 * 1024 * 1024
    arena = _native.NvmeTier(str(tmp_path), total, 4096, preallocate=True)
    descriptors = []
    for path in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(path)
        except OSError:
            continue
        if target.startswith(str(tmp_path / 'kvmem-reserve-')):
            assert target.endswith(' (deleted)')
            state = path.stat()
            assert state.st_size == total and state.st_blocks * 512 >= total
            descriptors.append(path)
    assert len(descriptors) == 1 and not list(tmp_path.iterdir())
    allocation = arena.allocate(0)
    arena.io([_native.IoSpan(allocation.slot, 0, 4096)], b'x' * 4096, True)
    result = bytearray(4096)
    arena.io([_native.IoSpan(allocation.slot, 0, 4096)], result, False)
    assert result == b'x' * 4096
    del arena
    assert not descriptors[0].exists()


def test_reservation_failure_does_not_fall_back_to_sparse_success_or_leave_file(tmp_path):
    import subprocess
    import sys
    script = '''
import resource, signal, sys
from pathlib import Path
from vllm_kvmem import _native
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
try:
    _native.NvmeTier(sys.argv[1], 8192, 4096, True)
except RuntimeError:
    assert not list(Path(sys.argv[1]).iterdir())
    print('reservation failed safely')
else:
    raise AssertionError('A failed reservation must never become sparse success')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'reservation failed safely'


def test_packed_nvme_io_and_generation_qualified_release(tmp_path):
    arena = _native.NvmeTier(str(tmp_path), 4096 * 4, 4096)
    allocations = [arena.allocate(i) for i in range(4)]
    assert [a.slot for a in allocations] == [0, 1, 2, 3]
    source = bytes(i % 251 for i in range(4096 * 4))
    spans = [_native.IoSpan(a.slot, i * 4096, 4096) for i, a in enumerate(allocations)]
    stats = arena.io(spans, source, True)
    assert stats["bytes"] == len(source)
    assert stats["syscalls"] == 1
    target = bytearray(len(source))
    assert arena.io(spans, target, False)["syscalls"] == 1
    assert target == source
    assert arena.allocate(99).slot == -1
    assert arena.release(allocations[0])
    replacement = arena.allocate(99)
    assert replacement.slot == 0
    assert arena.release(allocations[0])  # Stale callback cannot free replacement.
    assert arena.used_slots == 4
    assert list(tmp_path.iterdir()) == []  # Ephemeral backing unlinked at open.
