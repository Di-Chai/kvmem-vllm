import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1",
                                reason="Explicit GPU qualification only")


@pytest.mark.parametrize("packed,direct", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("retain_sink,fail_second_read", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize('optimized',[False,True])
def test_selected_archive_blocks_rebuild_native_pages_in_bounded_chunks(tmp_path, monkeypatch, retain_sink, fail_second_read, packed, direct,optimized):
    if optimized and not direct:
        pytest.skip('Split assembly requires direct packed reads')
    from vllm import _custom_ops as ops
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.codec import native_vllm_fp8_views, neox_rope
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler
    torch.manual_seed(71)
    cfg = KVMemConfig(block_tokens=16, select_budget=32, gen_budget=32,
                      repository_blocks=8, sink_blocks=int(retain_sink), recent_blocks=0)
    layers, heads, dim = 2, 2, 64
    archive = AsyncArchive(str(tmp_path), layers * 16 * heads * dim * 3, 8)
    authority = RequestAuthority("a", cfg, archive, layers, heads, dim)
    raw = torch.randn(layers, 64, heads, dim, dtype=torch.bfloat16)
    raw[:, 16:32].add_(2)
    raw[:, 32:48].add_(4)  # Positive query prefers the middle block, not newest tail.
    values = torch.randn_like(raw)
    scale = torch.tensor(0.03, device="cuda")
    fp8 = (values.float() / scale.cpu()).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert authority.append(raw, fp8, history_begin=0, local_rope_begin=0)
    for job, _ in authority.writes:
        job.result(timeout=5)
    authority.poll()
    plan = authority.select_for_queries(torch.ones(layers, 1, heads, dim))
    assert plan.total_window_tokens == 32
    native_layers, expected = [], []
    angles = torch.randn(128, 16, device="cuda")
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    table = torch.tensor([3, 5, 1, 7, 2, 6, 4, 8], dtype=torch.int32, device="cuda")
    original_positions = torch.arange(64, dtype=torch.int64, device="cuda")
    original_slots = table[original_positions // 8].long() * 8 + original_positions % 8
    for i in range(layers):
        actual = torch.zeros(10, heads, 8, dim * 2, dtype=torch.uint8, device="cuda")
        k, v = native_vllm_fp8_views(actual, heads, dim)
        original_raw = raw[i].cuda()
        original_rotated = neox_rope(original_raw, trig[original_positions, :16], trig[original_positions, 16:])
        ops.reshape_and_cache_flash(original_rotated, values[i].cuda(), k, v, original_slots,
                                    "fp8_e4m3", scale, scale)
        reference = actual.clone()
        rk, rv = native_vllm_fp8_views(reference, heads, dim)
        native_layers.append(LayerWindow(k, v, trig, scale, table))
        expected.append((rk, rv))
    from vllm_kvmem.gpu_raw_cache import GpuRawCache
    raw_cache=GpuRawCache(2*layers*16*heads*dim*2,(layers,16,heads,dim),torch.device('cuda:0'),1,2) if optimized else None
    extra=dict(raw_cache=raw_cache,reuse_resident_v=optimized)
    assembler = NativeWindowAssembler(tuple(native_layers), 16, chunk_blocks=1, packed=packed,
                                      direct_pinned_reads=direct,**extra)
    if fail_second_read:
        from vllm_kvmem.memory import Phase
        if packed:
            from vllm_kvmem import packed_assembly
            original_read = packed_assembly.read_packed_records
            read = lambda handles, **kwargs: original_read(authority, handles, **kwargs)
        else:
            read = authority.read_records
        calls = 0

        def fail_after_first_chunk(handles, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Injected archive read failure after GPU work")
            return read(handles, **kwargs)

        if packed:
            monkeypatch.setattr(packed_assembly, "read_packed_records",
                                lambda current, handles, **kwargs: fail_after_first_chunk(handles, **kwargs))
        else:
            monkeypatch.setattr(authority, "read_records", fail_after_first_chunk)
        with pytest.raises(OSError, match="Injected"):
            assembler.assemble(authority, plan)
        assert calls == 2 and authority.memory.phase is Phase.FAILED
        with pytest.raises(RuntimeError, match="ASSEMBLING"):
            authority.memory.commit_selection()
        authority.memory._fence.event.synchronize()
        assert assembler.close()
        del plan
        assert authority.close()
        assert archive.close()
        return
    result = assembler.assemble(authority, plan)
    result.event.synchronize()
    assert result.rebuilt_tokens + result.skipped_tokens == 32
    assert result.rebuilt_tokens >= 16
    if not retain_sink:
        assert result.chunks == 2 and result.rebuilt_tokens == 32
    chosen = torch.tensor(list(range(0 if retain_sink else 16, 16 if retain_sink else 32)) + list(range(32, 48)))
    positions = torch.arange(32, dtype=torch.int64, device="cuda")
    slots = table[positions // 8].long() * 8 + positions % 8
    for i, (rk, rv) in enumerate(expected):
        selected_raw = raw[i, chosen].cuda()
        rotated = neox_rope(selected_raw, trig[positions, :16], trig[positions, 16:])
        ops.reshape_and_cache_flash(rotated, values[i, chosen].cuda(), rk, rv, slots,
                                    "fp8_e4m3", scale, scale)
        assert torch.equal(native_layers[i].key.view(torch.uint8), rk.view(torch.uint8))
        assert torch.equal(native_layers[i].value.view(torch.uint8), rv.view(torch.uint8))
    authority.memory.attach_completion(result)
    assert authority.memory.commit_selection()
    assert assembler.close()
    del plan, result
    if retain_sink:
        checkpoint = authority.memory.checkpoint_window()
        # Alternately overwrite position 16, then restore a cold block whose
        # previous baked position is also 16. It must never use the skip path.
        for cold in (1, 2, 1):
            plan = authority.memory.select({i: float(i == cold) for i in range(4)})
            assert not next(r for r in plan.remaps if r.block_id == cold).skip
            assembler = NativeWindowAssembler(tuple(native_layers), 16, chunk_blocks=1,
                packed=packed, direct_pinned_reads=direct,**extra)
            result = assembler.assemble(authority, plan)
            result.event.synchronize()
            assert result.rebuilt_tokens == 16 and result.skipped_tokens == 16
            for i, (rk, rv) in enumerate(expected):
                selected_raw = raw[i, cold*16:(cold+1)*16].cuda()
                rotated = neox_rope(selected_raw, trig[positions[16:], :16], trig[positions[16:], 16:])
                ops.reshape_and_cache_flash(rotated, values[i, cold*16:(cold+1)*16].cuda(),
                    rk, rv, slots[16:], 'fp8_e4m3', scale, scale)
                assert torch.equal(native_layers[i].key.view(torch.uint8), rk.view(torch.uint8))
                assert torch.equal(native_layers[i].value.view(torch.uint8), rv.view(torch.uint8))
            authority.memory.attach_completion(result)
            assert authority.memory.commit_selection() and assembler.close()
            del plan, result
        # A session import gets different native pages. Restore the checkpoint's
        # original noncontiguous frame and reload V as well as re-RoPE K, even
        # where a record's dense position did not change.
        fresh_table = table.flip(0).contiguous()
        for layer, (rk, rv) in zip(native_layers, expected):
            layer.kernel_page_table = fresh_table
            layer.key.view(torch.uint8).fill_(19)
            layer.value.view(torch.uint8).fill_(19)
            rk.view(torch.uint8).fill_(19)
            rv.view(torch.uint8).fill_(19)
        plan = authority.memory.restore_window(checkpoint)
        assembler = NativeWindowAssembler(tuple(native_layers), 16, chunk_blocks=1,
            packed=packed, direct_pinned_reads=direct,**extra)
        result = assembler.assemble(authority, plan)
        result.event.synchronize()
        assert result.rebuilt_tokens == 32 and result.skipped_tokens == 0
        fresh_slots = fresh_table[positions // 8].long() * 8 + positions % 8
        for i, (rk, rv) in enumerate(expected):
            selected_raw = raw[i, chosen].cuda()
            rotated = neox_rope(selected_raw, trig[positions, :16], trig[positions, 16:])
            ops.reshape_and_cache_flash(rotated, values[i, chosen].cuda(),
                rk, rv, fresh_slots, 'fp8_e4m3', scale, scale)
            assert torch.equal(native_layers[i].key.view(torch.uint8), rk.view(torch.uint8))
            assert torch.equal(native_layers[i].value.view(torch.uint8), rv.view(torch.uint8))
        authority.memory.attach_completion(result)
        assert authority.memory.commit_selection() and assembler.close()
        del plan, result, checkpoint
    assert authority.close()
    assert archive.close()


@pytest.mark.parametrize('packed,direct', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('chunk_blocks', [1, 2])
def test_partial_terminal_tail_rebuilds_exact_native_fp8_bytes(tmp_path, packed, direct, chunk_blocks):
    from vllm import _custom_ops as ops
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.codec import native_vllm_fp8_views, neox_rope
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler
    cfg = KVMemConfig(block_tokens=16, select_budget=32, gen_budget=32, repository_blocks=8,
                      sink_blocks=1, recent_blocks=1)
    layers, heads, dim, count = 2, 2, 64, 35
    archive = AsyncArchive(str(tmp_path), layers * 16 * heads * dim * 3, 8)
    authority = RequestAuthority('partial', cfg, archive, layers, heads, dim)
    torch.manual_seed(881)
    raw = torch.randn(layers, count, heads, dim, dtype=torch.bfloat16)
    values = torch.randn_like(raw)
    scale = torch.tensor(.03, device='cuda')
    encoded = (values.float() / .03).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert authority.append(raw, encoded, history_begin=0, local_rope_begin=0)
    for job, _ in authority.writes:
        job.result(timeout=5)
    authority.poll()
    checkpoint = authority.memory.checkpoint_window(allow_partial=True)
    plan = authority.memory.restore_window(checkpoint)
    table = torch.tensor([3, 5, 1, 7, 2, 6, 4, 8], dtype=torch.int32, device='cuda')
    angles = torch.randn(128, 16, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    positions = torch.arange(count, device='cuda')
    slots = table[positions // 8].long() * 8 + positions % 8
    native, expected = [], []
    for i in range(layers):
        backing = torch.zeros(10, heads, 8, dim * 2, dtype=torch.uint8, device='cuda')
        k, v = native_vllm_fp8_views(backing, heads, dim)
        ref = backing.clone(); rk, rv = native_vllm_fp8_views(ref, heads, dim)
        rotated = neox_rope(raw[i].cuda(), trig[positions, :16], trig[positions, 16:])
        ops.reshape_and_cache_flash(rotated, values[i].cuda(), rk, rv, slots, 'fp8_e4m3', scale, scale)
        native.append(LayerWindow(k, v, trig, scale, table)); expected.append((rk, rv))
    assembler = NativeWindowAssembler(tuple(native), 16, chunk_blocks=chunk_blocks, packed=packed,
                                      direct_pinned_reads=direct)
    result = assembler.assemble(authority, plan); result.event.synchronize()
    assert result.rebuilt_tokens == 35 and result.skipped_tokens == 0
    for actual, (key, value) in zip(native, expected):
        assert torch.equal(actual.key.view(torch.uint8), key.view(torch.uint8))
        assert torch.equal(actual.value.view(torch.uint8), value.view(torch.uint8))
    authority.memory.attach_completion(result); assert authority.memory.commit_selection()
    assert assembler.close()
    del plan, checkpoint, result
    assert authority.close() and archive.close()


def test_resident_value_moves_crossing_chunks_and_raw_hits_are_byte_exact(tmp_path):
    from vllm import _custom_ops as ops
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.codec import native_vllm_fp8_views, neox_rope
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler
    from vllm_kvmem.gpu_raw_cache import GpuRawCache
    cfg=KVMemConfig(block_tokens=16,select_budget=48,gen_budget=48,
                    repository_blocks=12,sink_blocks=0,recent_blocks=0)
    layers,heads,dim=2,2,64
    archive=AsyncArchive(str(tmp_path),layers*16*heads*dim*3,12)
    authority=RequestAuthority('cycle',cfg,archive,layers,heads,dim)
    torch.manual_seed(9983)
    raw=torch.randn(layers,96,heads,dim,dtype=torch.bfloat16)
    values=torch.randn_like(raw)
    scale=torch.tensor(.03,device='cuda')
    encoded=(values.float()/.03).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert authority.append(raw,encoded,history_begin=0,local_rope_begin=0)
    authority.publish_sealed(lambda:None)
    table=torch.tensor([7,4,1,8,2,6,3,5],dtype=torch.int32,device='cuda')
    angles=torch.randn(128,16,device='cuda')
    trig=torch.cat((angles.cos(),angles.sin()),dim=-1).bfloat16()
    old_positions=torch.arange(96,device='cuda')
    old_slots=table[old_positions//16].long()*16+old_positions%16
    native,expected=[],[]
    for i in range(layers):
        backing=torch.zeros((10,heads,16,dim*2),dtype=torch.uint8,device='cuda')
        k,v=native_vllm_fp8_views(backing,heads,dim)
        rotated=neox_rope(raw[i].cuda(),trig[old_positions,:16],trig[old_positions,16:])
        ops.reshape_and_cache_flash(rotated,values[i].cuda(),k,v,old_slots,'fp8_e4m3',scale,scale)
        reference=backing.clone()
        native.append(LayerWindow(k,v,trig,scale,table))
        expected.append(native_vllm_fp8_views(reference,heads,dim))
    raw_cache=GpuRawCache(3*layers*16*heads*dim*2,(layers,16,heads,dim),torch.device('cuda:0'),1,3)
    positions=torch.arange(48,device='cuda')
    slots=table[positions//16].long()*16+positions%16
    for chosen in ((1,3,5),(0,1,3)):
        plan=authority.memory.select({i:float(i in chosen) for i in range(6)})
        assert [r.block_id for r in plan.remaps]==list(chosen)
        assembler=NativeWindowAssembler(tuple(native),16,chunk_blocks=1,packed=True,
            direct_pinned_reads=True,raw_cache=raw_cache,reuse_resident_v=True)
        result=assembler.assemble(authority,plan)
        result.event.synchronize()
        selected=torch.tensor([t for i in chosen for t in range(i*16,(i+1)*16)])
        for i,(rk,rv) in enumerate(expected):
            rotated=neox_rope(raw[i,selected].cuda(),trig[positions,:16],trig[positions,16:])
            ops.reshape_and_cache_flash(rotated,values[i,selected].cuda(),rk,rv,slots,'fp8_e4m3',scale,scale)
            assert torch.equal(native[i].key.view(torch.uint8),rk.view(torch.uint8))
            assert torch.equal(native[i].value.view(torch.uint8),rv.view(torch.uint8))
        if chosen==(0,1,3):
            assert result.resident_v_reused_tokens==32 and result.raw_cache_hit_tokens==32
            assert result.uploaded_bytes==authority.record_bytes
        authority.memory.attach_completion(result)
        assert authority.memory.commit_selection() and assembler.close()
        del plan,result
    assert authority.close() and archive.close()


def test_two_staging_slots_preserve_delayed_gpu_consumers_across_refreshes():
    from vllm_kvmem.window_assembly import _AssemblyStaging

    device = torch.device('cuda', torch.cuda.current_device())
    staging = _AssemblyStaging(4096, device)
    initial_buffers = None
    observed = []
    for refresh in range(2):
        owner = object()
        staging.claim(owner)
        for chunk in range(5):
            expected = refresh * 5 + chunk + 1
            source = torch.full((4096,), expected, dtype=torch.uint8)
            destination, ready, slot = staging.destination(source.numel())
            transfer = staging.transfers.upload(source, destination, ready=ready)
            if transfer is None:
                next(iter(staging.transfers.pending.values())).event.synchronize()
                transfer = staging.transfers.upload(source, destination, ready=ready)
            assert transfer is not None
            torch.cuda.current_stream(device).wait_event(transfer.event)
            # H2D can finish long before the consumer reads the slot. A pool
            # protected only by DMA completion corrupts earlier chunks here.
            torch.cuda._sleep(20_000_000)
            observed.append((destination.clone(), expected))
            staging.consumed(slot)
        buffers = tuple(tensor.data_ptr() for tensor in staging.buffers)
        if initial_buffers is None:
            initial_buffers = buffers
        else:
            assert buffers == initial_buffers
        staging.release(owner)
    torch.cuda.current_stream(device).synchronize()
    for actual, expected in observed:
        assert torch.equal(actual.cpu(), torch.full((4096,), expected, dtype=torch.uint8))
    assert staging.close()


@pytest.mark.parametrize('trig_dtype',[torch.float32,torch.bfloat16])
def test_real_qwen_geometry_cached_assembly_matches_legacy_across_1600_token_pages(tmp_path,trig_dtype):
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.window_assembly import LayerWindow,NativeWindowAssembler
    from vllm_kvmem.gpu_raw_cache import GpuRawCache
    layers,heads,dim,block,page,count=17,4,256,128,1600,4096
    cfg=KVMemConfig(block_tokens=block,select_budget=2048,gen_budget=2048,
                    repository_blocks=64,sink_blocks=0,recent_blocks=0)
    archive=AsyncArchive(str(tmp_path),layers*block*heads*dim*3,64)
    owner=RequestAuthority('real-geometry',cfg,archive,layers,heads,dim,resident_records=64,scoring_layers=16)
    torch.manual_seed(7093)
    raw=torch.randn(layers,count,heads,dim,dtype=torch.bfloat16)
    encoded=torch.randint(0,127,raw.shape,dtype=torch.uint8)
    assert owner.append(raw,encoded,history_begin=0,local_rope_begin=0)
    owner.publish_sealed(lambda:None)
    table=torch.tensor([4,2,6,1],dtype=torch.int32,device='cuda')
    angles=torch.randn(6400,32,device='cuda')
    trig=torch.cat((angles.cos(),angles.sin()),dim=-1).to(trig_dtype)
    positions=torch.arange(count,device='cuda')
    slots=table[positions//page].long()*page+positions%page
    legacy,cached=[],[]
    for i in range(layers):
        backing=torch.zeros((8,heads,page,dim*2),dtype=torch.uint8,device='cuda')
        key,value=native_vllm_fp8_views(backing,heads,dim)
        scale=torch.tensor(.025+i*.001,device='cuda')
        assemble_fp8(raw[i].cuda(),encoded[i].view(torch.float8_e4m3fn).cuda(),positions,slots,
                     trig,key,value,scale)
        other=backing.clone()
        k2,v2=native_vllm_fp8_views(other,heads,dim)
        legacy.append(LayerWindow(key,value,trig,scale,table))
        cached.append(LayerWindow(k2,v2,trig,scale,table))
    cache=GpuRawCache(layers*2048*heads*dim*2,(layers,block,heads,dim),torch.device('cuda:0'),1,16)
    for chosen in (range(8,24),range(0,16),range(8,24)):
        plan=owner.memory.select({i:float(i in chosen) for i in range(32)})
        before=NativeWindowAssembler(tuple(legacy),block,chunk_blocks=3,packed=True,direct_pinned_reads=True)
        after=NativeWindowAssembler(tuple(cached),block,chunk_blocks=3,packed=True,direct_pinned_reads=True,
                                     raw_cache=cache,reuse_resident_v=True)
        baseline=before.assemble(owner,plan)
        optimized=after.assemble(owner,plan)
        baseline.event.synchronize();optimized.event.synchronize()
        for left,right in zip(legacy,cached):
            assert torch.equal(left.key.view(torch.uint8),right.key.view(torch.uint8))
            assert torch.equal(left.value.view(torch.uint8),right.value.view(torch.uint8))
        owner.memory.attach_completion(optimized)
        assert owner.memory.commit_selection() and before.close() and after.close()
        del plan,baseline,optimized
    assert owner.close() and archive.close()
