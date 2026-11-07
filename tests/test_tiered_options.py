"""Factory and scheduler agree on SSD backing, independent of CPU residency."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.archive_limits import tiered_capacity_records
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.coordination import HistoryAdmission
from vllm_kvmem.options import RuntimeOptions


def options(**changes):
    values = dict(memory=KVMemConfig(block_tokens=16, select_budget=32, gen_budget=64,
                                   repository_blocks=128, sink_blocks=1, recent_blocks=0),
        archive_backend='tiered', cpu_read_cache_bytes=8192, nvme_max_bytes=65536,
        archive_io_max_bytes=73728, tiered_segment_bytes=8192,
        session_capacity=2, session_checkpoint_bytes=4096, async_archive_retirement=True,
        request_planning=True, cpu_history_coordination=True)
    values.update(changes)
    return RuntimeOptions(**values)


@pytest.mark.parametrize('changes', [
    {'cpu_read_cache_bytes': 0}, {'nvme_max_bytes': 0}, {'archive_io_max_bytes': 0},
    {'nvme_preallocate': True}, {'cpu_history_coordination': False},
    {'tiered_cpu_low_watermark': .9}, {'tiered_cpu_high_watermark': float('nan')},
    {'tiered_cpu_low_watermark': True}, {'tiered_segment_bytes': False},
    {'cpu_direct_record_seal': True}, {'assembly_read_prefetch': True},
    {'repository_gc': True},
])
def test_unsupported_tiered_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        options(**changes)


def test_roundtrip_does_not_change_memory_policy():
    original = options()
    from dataclasses import asdict
    payload = {k: getattr(original, k) for k in (
        'archive_backend', 'cpu_read_cache_bytes', 'nvme_max_bytes', 'archive_io_max_bytes',
        'tiered_segment_bytes', 'tiered_cpu_low_watermark', 'tiered_cpu_high_watermark',
        'session_capacity', 'session_checkpoint_bytes', 'async_archive_retirement',
        'request_planning', 'cpu_history_coordination')}
    payload.update(asdict(original.memory))
    result = RuntimeOptions.from_vllm(NS(
        kv_transfer_config=NS(kv_connector_extra_config={'kvmem': payload}),
        cache_config=NS(cache_dtype='fp8_e4m3')))
    assert result.memory == original.memory
    assert result.tiered_segment_bytes == 8192
    assert result.archive_backend == 'tiered'
    assert result.rolling_active_window == original.rolling_active_window
    assert result.pressure_policy == original.pressure_policy


def test_factory_and_history_quota_use_disk_not_per_request_cpu(tmp_path):
    from vllm_kvmem import _native
    from vllm_kvmem.storage import make_shared_archive
    cfg = options()
    archive = make_shared_archive(cfg, tmp_path, 4096, 4)
    try:
        assert archive.capacity_bytes == 65536
        assert tiered_capacity_records(cfg, 4096, 4) == 16
        gate = HistoryAdmission(archive.capacity_bytes, 16, 256, 64, reserve_progress=False)
        requests = [NS(request_id=str(i), session_id=None, num_prompt_tokens=32,
                       sampling_params=NS(max_tokens=0)) for i in range(4)]
        for request in requests:
            gate.allocated(request)
        # Four turns fit their guaranteed SSD backing, though CPU has two records.
        assert gate.snapshot()['active'] == 4
        repo = _native.Repository(8, 16)
        leases = [repo.reserve(i*16, 16).commit() for i in range(8)]
        for i, lease in enumerate(leases):
            archive.write(str(i//2), [lease], bytes([i])*4096).result(5)
        gate.observe(archive.history_used_bytes())
        for i, lease in enumerate(leases):
            assert archive.read(str(i//2), [lease]).result(5)[0] == bytes([i])*4096
        assert archive.history_used_bytes() == 8*4096
    finally:
        assert archive.close()


def test_ssd_rounding_and_impossible_record_fail_before_open(tmp_path):
    from vllm_kvmem.storage import make_shared_archive
    cfg = options(nvme_max_bytes=65535)
    assert tiered_capacity_records(cfg, 4096, 4) == 15
    with pytest.raises(ValueError, match='one complete'):
        make_shared_archive(replace(cfg, nvme_max_bytes=4095), tmp_path/'missing', 4096, 4)
    assert not (tmp_path/'missing').exists()


def test_factory_rejects_region_scratch_overflow_and_accepts_exact_serving_bound(tmp_path):
    from vllm_kvmem import _native
    from vllm_kvmem.storage import make_shared_archive
    # Six complete K/G records; paired K/V outputs plus fallback scratch/cache
    # copies need three extents even when all source records start on CPU.
    size = 6 * 4096
    with pytest.raises(ValueError, match='full serving batch'):
        make_shared_archive(options(archive_io_max_bytes=3*size-1),
                            tmp_path/'rejected', 4096, 4)
    assert not (tmp_path/'rejected').exists()
    archive = make_shared_archive(options(archive_io_max_bytes=3*size),
                                  tmp_path/'accepted', 4096, 4)
    try:
        repo = _native.Repository(6, 16)
        leases = [repo.reserve(i*16, 16).commit() for i in range(6)]
        payload = b''.join(bytes([i])*4096 for i in range(6))
        archive.write('a', leases, payload).result(5)
        result, _ = archive.read_regions('a', leases, ((0, 2048), (2048, 2048))).result(5)
        expected = b''.join(bytes([i])*2048 for i in range(6)) * 2
        assert result == expected
        assert archive.io_usage()['archive_io_peak_bytes'] == 3*size
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
    finally:
        assert archive.close()
