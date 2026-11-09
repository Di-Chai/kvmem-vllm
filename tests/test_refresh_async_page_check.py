"""KVMEM_REFRESH_ASYNC_PAGE_CHECK: deferred page-reservation reads verify like the blocking read."""
import os
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import pressure_promise


def test_flag_is_opt_in_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_REFRESH_ASYNC_PAGE_CHECK', raising=False)
    assert pressure_promise.refresh_async_page_check() is False
    monkeypatch.setenv('KVMEM_REFRESH_ASYNC_PAGE_CHECK', '1')
    assert pressure_promise.refresh_async_page_check() is True
    monkeypatch.setenv('KVMEM_REFRESH_ASYNC_PAGE_CHECK', 'yes')
    with pytest.raises(ValueError):
        pressure_promise.refresh_async_page_check()


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='Explicit GPU qualification only')
def test_deferred_read_sees_the_row_at_enqueue_time_and_verifies_like_the_blocking_read():
    import torch
    table = torch.arange(40, dtype=torch.int32, device='cuda').reshape(4, 10)
    row = table[2, :6]
    expected = row.cpu().tolist()
    ok = pressure_promise.defer_page_check(row, expected)
    table[2].fill_(-1)                     # a later in-stream update must not leak into the check
    pressure_promise.verify_page_checks(NS(promise_page_checks=(ok,)))
    bad = pressure_promise.defer_page_check(table[1, :6], expected)
    with pytest.raises(ValueError, match='page reservation mismatch'):
        pressure_promise.verify_page_checks(NS(promise_page_checks=(bad,)))
