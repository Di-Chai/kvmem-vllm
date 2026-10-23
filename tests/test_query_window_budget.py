"""The shared overlap cache cannot consume admitted CPU history capacity."""
from types import SimpleNamespace
import time

import pytest

from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.coordination import HistoryAdmission
from vllm_kvmem.historical_admission import HistoricalHistoryAdmission
from vllm_kvmem.historical_gc import HistoricalGcSettings, historical_record_charge
from vllm_kvmem.query_window_budget import split_cpu_history_budget
from vllm_kvmem.storage import make_shared_archive


def options(*, historical=False, **updates):
    values = dict(memory=KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0),
        archive_backend='cpu', cpu_read_cache_bytes=10000,
        query_score_mode='overlap_16_8', query_window_cache_bytes=1000,
        session_capacity=1, archive_io_max_bytes=0, cpu_gather_workers=1,
        async_seal=True, historical_gc=(HistoricalGcSettings(mode='off',
            ready_low=1, ready_high=2) if historical else None),
        nvme_preallocate=False, nvme_max_bytes=0,
        cpu_history_cache_bytes_per_request=0)
    return SimpleNamespace(**(values | updates))


def close(archive):
    deadline = time.monotonic() + 5
    while not archive.close():
        assert time.monotonic() < deadline
        time.sleep(.001)


@pytest.mark.parametrize('historical', [False, True])
def test_worker_and_admission_share_one_global_reservation(tmp_path, historical):
    settings = options(historical=historical)
    history_bytes, cache_bytes = split_cpu_history_budget(settings)
    record_bytes, block_tokens = 192, settings.memory.block_tokens
    archive = make_shared_archive(settings, tmp_path, record_bytes, max_requests=1)
    try:
        assert (history_bytes, cache_bytes) == (9000, 1000)
        if historical:
            charge = historical_record_charge(record_bytes, block_tokens)
            admission = HistoricalHistoryAdmission(history_bytes, charge,
                settings.memory.repository_blocks, block_tokens)
            admitted_records = admission.ledger.capacity_records
        else:
            charge = record_bytes
            admission = HistoryAdmission(history_bytes, block_tokens,
                record_bytes // block_tokens, max_tokens=64, reserve_progress=False)
            admitted_records = admission.capacity_bytes // record_bytes
        assert archive.tier.capacity == admitted_records == history_bytes // charge
        assert archive.tier.capacity * charge + cache_bytes <= settings.cpu_read_cache_bytes
        # The cache is independently reserved; charging its actual allocation
        # as accepted history would make observe(0) fail before any append.
        archive._query_window_cache_bytes = 700
        assert archive.history_used_bytes() == 0
        admission.observe(archive.history_used_bytes())
        usage = archive.io_usage()
        assert usage['query_window_cache_reserved_bytes'] == 1000
        assert usage['query_window_cache_bytes'] == 700
    finally:
        archive._query_window_cache_bytes = 0
        close(archive)


def test_reservation_is_once_per_archive_and_mode_off_preserves_capacity(tmp_path):
    settings = options(query_score_mode='mean_k')
    archive = make_shared_archive(settings, tmp_path, 192, max_requests=8)
    try:
        assert split_cpu_history_budget(settings) == (10000, 0)
        assert archive.tier.capacity == 10000 // 192
        assert archive.io_usage()['query_window_cache_reserved_bytes'] == 0
    finally:
        close(archive)


def test_nvme_cpu_read_cache_and_window_cache_share_the_configured_cpu_limit(tmp_path):
    settings = options(archive_backend='nvme')
    archive = make_shared_archive(settings, tmp_path, 192, max_requests=1)
    try:
        assert archive.read_cache.capacity_bytes == 9000
        assert archive.io_usage()['query_window_cache_reserved_bytes'] == 1000
        assert archive.capacity_bytes == 64 * 192
    finally:
        close(archive)


@pytest.mark.parametrize('total', [0, 999, 1000])
def test_insufficient_cache_reservation_fails_before_archive_creation(tmp_path, total):
    with pytest.raises(ValueError, match='CPU memory budget'):
        make_shared_archive(options(cpu_read_cache_bytes=total), tmp_path/'unused', 192, 1)
    assert not (tmp_path/'unused').exists()


def test_cache_reservation_must_leave_capacity_for_one_history_record(tmp_path):
    with pytest.raises(ValueError, match='one record'):
        make_shared_archive(options(cpu_read_cache_bytes=1100), tmp_path/'unused', 192, 1)
    assert not (tmp_path/'unused').exists()
