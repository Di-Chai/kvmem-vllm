from vllm_kvmem import _native


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
