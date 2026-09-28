"""Real native GC + shared archive using serving policy and pressure handling."""
from threading import Event
from types import SimpleNamespace
import time

import pytest

pytest.importorskip('torch')
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.repository_gc import serving_gc_config


def test_serving_reserve_matches_original_policy_and_rejects_unreclaimable_capacity():
    memory = KVMemConfig()
    gc = serving_gc_config(memory, 4224)
    assert (gc.ready_low, gc.ready_high) == (204, 409)
    assert gc.candidate_ingest_tokens == 409 * 128
    assert gc.recent_tokens == memory.recent_blocks * 128 and gc.background
    with pytest.raises(ValueError, match=r'K\+G'):
        serving_gc_config(KVMemConfig(repository_blocks=1536), 4224)


def test_shared_archive_recycles_both_request_namespaces_with_serving_pressure_path(tmp_path):
    pytest.importorskip('vllm')
    import torch
    from vllm_kvmem.authority import RequestAuthority, _Ready
    from vllm_kvmem.capture_runtime import CaptureRuntime
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.storage import AsyncArchive
    from test_authority import payload, drain

    memory = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
                        repository_blocks=24, sink_blocks=1, recent_blocks=0)
    policy = serving_gc_config(memory, 8)
    archive = AsyncArchive(str(tmp_path), 48, 48)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(memory=memory, repository_gc=True)
    runtime.gc_progress = Event()
    runtime.gc_wait_seconds = 0.0
    runtime.gc_wait_count = 0
    runtime._archive_wait_deadline = None
    runtime.authorities = {name: RequestAuthority(name, memory, archive, 2, 1, 2,
        resident_records=8, gc_config=policy, gc_progress=runtime.gc_progress) for name in ('a', 'b')}
    sinks = {}
    try:
        for name, authority in runtime.authorities.items():
            assert authority.append(*payload(0, 16), history_begin=0, local_rope_begin=0)
            drain(authority)
            sinks[name] = next(iter(authority.records.values())).handle
        for _ in range(40):
            for name, authority in runtime.authorities.items():
                plan = authority.select_for_queries(torch.ones(2, 1, 2, 2))
                authority.memory.attach_completion(_Ready())
                assert authority.memory.commit_selection()
                del plan
                deadline = time.monotonic() + 5
                while not authority.append(*payload(authority.history_end, 8),
                        history_begin=authority.history_end, local_rope_begin=8):
                    assert time.monotonic() < deadline
                    runtime._wait_archive_progress()
                runtime._archive_wait_deadline = None
                drain(authority)
                assert len(authority.records) <= 24
                assert archive.tier.used_slots <= 48
        for name, authority in runtime.authorities.items():
            assert authority.history_end == 336 and authority.gc_reclaimed_records > 50
            assert torch.equal(authority.read_record(sinks[name])[0], payload(0, 4)[0])
        # A sufficiently fast background worker can replenish before append
        # stalls. Reclamation and bounded slots, not an artificial pause, are
        # the required result.
    finally:
        for authority in runtime.authorities.values():
            assert authority.close()
        assert archive.tier.used_slots == 0 and archive.close()


def test_pressure_timeout_is_bounded_and_never_claims_release(tmp_path):
    pytest.importorskip('vllm')
    from vllm_kvmem.capture_runtime import CaptureRuntime
    from vllm_kvmem.options import RuntimeOptions
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(repository_gc=True)
    runtime.authorities = {'held': SimpleNamespace(poll=lambda: 0, writes=[])}
    runtime._archive_wait_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match='no append progress'):
        runtime._wait_archive_progress()
    assert set(runtime.authorities) == {'held'}


def test_capacity_fallback_waits_for_background_progress_without_running_policy():
    pytest.importorskip('vllm')
    from threading import Timer
    from vllm_kvmem.capture_runtime import CaptureRuntime
    from vllm_kvmem.options import RuntimeOptions
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(repository_gc=True)
    runtime.authorities = {'held': SimpleNamespace(poll=lambda: 0, writes=[])}
    runtime._archive_wait_deadline = None
    runtime.gc_progress = Event()
    runtime.gc_wait_count, runtime.gc_wait_seconds = 0, 0.0
    notifier = Timer(.01, runtime.gc_progress.set)
    notifier.start()
    try:
        runtime._wait_archive_progress()
        assert runtime.gc_wait_count == 1 and runtime.gc_wait_seconds > 0
        assert not runtime.gc_progress.is_set() and 'held' in runtime.authorities
    finally:
        notifier.join()
