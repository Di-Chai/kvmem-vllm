"""KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB: identical bytes, parallel route for smaller regions."""
import os
import numpy as np
import pytest

from vllm_kvmem.core import memory_storage
from vllm_kvmem.core.memory_storage import MemoryArchive


def gather(monkeypatch, mib):
    if mib is None:
        monkeypatch.delenv('KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB', raising=False)
    else:
        monkeypatch.setenv('KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB', str(mib))
    record = 2 * 1024**2
    archive = MemoryArchive('', record, 64, gather_workers=4, cpu_direct_record_seal=True)
    rng = np.random.default_rng(3)
    sources = [rng.integers(0, 256, record, dtype=np.uint8).tobytes() for _ in range(40)]
    out = np.zeros(40 * 256 * 1024, dtype=np.uint8)
    tasks = archive._gather(out, sources, 512 * 1024, 256 * 1024)        # 10 MiB region read
    expected = b''.join(s[512*1024:768*1024] for s in sources)
    archive.close()
    return tasks, out.tobytes() == expected, archive.archive_usage() if hasattr(archive, 'archive_usage') else None


def test_default_keeps_inline_copy(monkeypatch):
    tasks, same, _ = gather(monkeypatch, None)
    assert same and tasks == 1


def test_lower_threshold_splits_across_workers_with_identical_bytes(monkeypatch):
    tasks, same, _ = gather(monkeypatch, 4)
    assert same and tasks == 4


def test_invalid_threshold_is_rejected(monkeypatch):
    monkeypatch.setenv('KVMEM_CPU_REGION_GATHER_PARALLEL_MIN_MIB', '0')
    with pytest.raises(ValueError, match='PARALLEL_MIN_MIB'):
        memory_storage.region_gather_parallel_min_bytes()
