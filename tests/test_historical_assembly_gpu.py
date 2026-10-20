"""Real GPU assembly after historical retirement; separate from model gates."""
import os
import time

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='explicit GPU opt-in required')


@pytest.mark.parametrize('packed,direct', [(False, False), (True, False), (True, True)])
def test_retired_admitted_window_rebuilds_correct_bytes_and_frees_cpu_before_gpu_finishes(tmp_path, packed, direct):
    import torch
    from vllm import _custom_ops as ops
    from vllm_kvmem import _native as n
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.codec import native_vllm_fp8_views, neox_rope
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler

    cfg = KVMemConfig(block_tokens=16, select_budget=32, gen_budget=32,
                      repository_blocks=8, sink_blocks=0, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 16*64*3, 16, native_payloads=True)
    gc = n.HistoricalGcConfig()
    gc.mode, gc.background = n.HistoricalGcMode.DELETE, False
    gc.ready_low, gc.ready_high, gc.scan_blocks = 5, 6, 8
    worker = archive.enable_history_gc(gc)
    authority = RequestAuthority('a', cfg, archive, 1, 1, 64)
    torch.manual_seed(17)
    raw = torch.randn(1, 64, 1, 64, dtype=torch.bfloat16)
    value = torch.randn_like(raw)
    scale = torch.tensor(.03, device='cuda')
    fp8 = (value.float()/scale.cpu()).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert authority.append(raw, fp8, history_begin=0, local_rope_begin=0)
    for job, _ in authority.writes:
        job.result(5)
    handles = tuple(r.handle for r in authority.records.values())
    for _ in range(16):
        worker.advance()
        authority.poll()
    assert authority.gc_reclaimed_records == 2
    plan = authority.memory.select({h.slot: 1. for h in handles[2:]})
    assert plan.total_window_tokens == 32
    authority.memory.repository.retire_permanently(handles[2])
    for _ in range(8):
        worker.advance()
    assert archive.io_usage()['cpu_authority_bytes'] == 2*archive.record_bytes

    angles = torch.randn(96, 16, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    table = torch.tensor([3, 5, 1, 7, 2, 6, 4, 8], dtype=torch.int32, device='cuda')
    cache = torch.zeros(10, 1, 8, 128, dtype=torch.uint8, device='cuda')
    key, values = native_vllm_fp8_views(cache, 1, 64)
    layer = LayerWindow(key, values, trig, scale, table)
    assembler = NativeWindowAssembler((layer,), 16, chunk_blocks=1,
                                      packed=packed, direct_pinned_reads=direct)
    plan_result = assembler.assemble(authority, plan)
    continuation = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(continuation):
        continuation.wait_event(plan_result.event)
        torch.cuda._sleep(1_000_000_000)
        observed = cache.clone()
        complete.record(continuation)
    try:
        deadline = time.monotonic()+5
        while archive.io_usage()['cpu_authority_bytes'] != archive.record_bytes:
            assert time.monotonic() < deadline
            worker.advance()
            authority.poll()
            time.sleep(.001)
        assert not complete.query()
        assert authority.gc.scope.stats['slots_released'] == 0
        complete.synchronize()
        reference = torch.zeros_like(cache)
        rk, rv = native_vllm_fp8_views(reference, 1, 64)
        positions = torch.arange(32, device='cuda')
        rotated = neox_rope(raw[0, 32:].cuda(), trig[positions, :16], trig[positions, 16:])
        slots = table[positions//8].long()*8 + positions % 8
        ops.reshape_and_cache_flash(rotated, value[0, 32:].cuda(), rk, rv, slots,
                                    'fp8_e4m3', scale, scale)
        assert torch.equal(observed, reference)
        authority.memory.attach_completion(plan_result)
        assert authority.memory.commit_selection()
    finally:
        complete.synchronize()
        assert assembler.close()
        del plan_result, plan
        deadline = time.monotonic()+5
        while not authority.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
        while not archive.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
