"""Explicit real-CUDA qualification of bounded immutable read lookahead."""
import math
import os
from threading import Event

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


@pytest.mark.parametrize('delay_read', [False, True])
def test_prefetch_matches_bytes_for_three_chunks_partial_tail_and_cold_restore(tmp_path, monkeypatch, delay_read):
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem import split_assembly
    from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler

    block, layers, heads, dim, page = 32, 2, 2, 64, 16
    count = 600*block+7
    config = KVMemConfig(block_tokens=block, select_budget=301*block, gen_budget=302*block,
                         repository_blocks=1024, sink_blocks=0, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), layers*block*heads*dim*2, 1024,
                            cpu_direct_record_seal=True)
    owner = RequestAuthority('prefetch-gpu', config, archive, layers, heads, dim,
                             raw_dtype=torch.float8_e4m3fn, resident_records=640)
    torch.manual_seed(3010971)
    raw = torch.randn(layers,count,heads,dim).to(torch.float8_e4m3fn)
    encoded = torch.randint(0,127,raw.shape,dtype=torch.uint8)
    assert owner.append(raw, encoded, history_begin=0, local_rope_begin=0)
    owner.publish_sealed(lambda: archive.wait_for_capacity(.05))
    page_count = math.ceil(count/page)
    table = (torch.randperm(page_count,device='cuda')+1).to(torch.int32)
    angles = torch.randn((page_count+1)*page,16,device='cuda')
    trig = torch.cat((angles.cos(),angles.sin()),dim=-1)
    positions = torch.arange(count,device='cuda')
    slots = table[positions//page].long()*page+positions%page
    frames = [[], []]
    for layer_index in range(layers):
        backing = torch.zeros((page_count+1,heads,page,2*dim),dtype=torch.uint8,device='cuda')
        key,value = native_vllm_fp8_views(backing,heads,dim)
        scale = torch.tensor(.04052734375+layer_index*.00390625,device='cuda')
        assemble_fp8(raw[layer_index].cuda(),encoded[layer_index].view(torch.float8_e4m3fn).cuda(),
                     positions,slots,trig,key,value,scale)
        other_key,other_value = native_vllm_fp8_views(backing.clone(),heads,dim)
        frames[0].append(LayerWindow(key,value,trig,scale,table))
        frames[1].append(LayerWindow(other_key,other_value,trig,scale,table))
    chosen = set(range(1,600,2))|{600}
    scores = {r.handle.slot: float(r.history_begin//block in chosen) for r in owner.records.values()}
    plan = owner.memory.select(scores,action=False)
    selected = tuple((r.block_id,r.n_tokens,r.to_base) for r in plan.remaps)
    assert len(selected) == 301 and selected[-1][1] == 7
    native_gather = archive._gather_region
    native_launch = split_assembly.assemble_split_layer
    for restoring in (False,True):
        if restoring:
            checkpoint = owner.memory.checkpoint_window(allow_partial=True)
            plan = owner.memory.restore_window(checkpoint)
            for layer in frames[0]+frames[1]:
                layer.kernel_page_table = table.flip(0).contiguous()
                layer.key.view(torch.uint8).fill_(199)
                layer.value.view(torch.uint8).fill_(199)
        assemblers = [NativeWindowAssembler(tuple(frame),block,chunk_blocks=128,packed=True,
            direct_pinned_reads=True,reuse_resident_v=True,raw_dtype=torch.float8_e4m3fn,
            read_prefetch=flag) for frame,flag in zip(frames,(False,True))]
        baseline = assemblers[0].assemble(owner,plan)
        baseline.event.synchronize()
        reading, launched = Event(), Event()
        gather_calls = [0]
        def delayed_gather(*args):
            gather_calls[0] += 1
            if delay_read and not restoring and gather_calls[0] == 2:
                reading.set()
                assert launched.wait(5), 'CPU prefetch blocked current layer submission'
            return native_gather(*args)
        first_layer = [True]
        def launch(*args,**kwargs):
            if first_layer[0]:
                first_layer[0] = False
                if delay_read and not restoring:
                    assert reading.wait(5), 'Lookahead did not start before current layers'
                launched.set()
                # Give GPU scratch consumers a real tail while host work starts
                # the other slot. Consumers still retain their own CUDA fence.
                torch.cuda._sleep(5_000_000)
            return native_launch(*args,**kwargs)
        monkeypatch.setattr(archive,'_gather_region',delayed_gather)
        monkeypatch.setattr(split_assembly,'assemble_split_layer',launch)
        optimized = assemblers[1].assemble(owner,plan)
        optimized.event.synchronize()
        monkeypatch.setattr(archive,'_gather_region',native_gather)
        monkeypatch.setattr(split_assembly,'assemble_split_layer',native_launch)
        assert tuple((r.block_id,r.n_tokens,r.to_base) for r in plan.remaps) == selected
        for left,right in zip(*frames):
            assert torch.equal(left.key.view(torch.uint8),right.key.view(torch.uint8))
            assert torch.equal(left.value.view(torch.uint8),right.value.view(torch.uint8))
        assert optimized.rebuilt_tokens == baseline.rebuilt_tokens == 300*block+7
        assert optimized.partial_tokens == baseline.partial_tokens == 7
        assert optimized.chunks == baseline.chunks == 4
        assert optimized.uploaded_bytes == baseline.uploaded_bytes
        stats = optimized.read_stats
        assert stats['assembly_prefetch_submitted'] == stats['assembly_prefetch_completed'] == 2
        assert stats['assembly_prefetch_abandoned'] == 0
        assert optimized.resident_v_reused_tokens == (0 if restoring else 300*block)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        for assembler in assemblers:
            assert len(assembler.transfers.buffers) == len(assembler.staging.buffers) == 2
            assert len({b.data_ptr() for b in assembler.transfers.buffers}) == 2
            assert len({b.data_ptr() for b in assembler.staging.buffers}) == 2
            assert not assembler.transfers.preparing and not assembler.transfers._prepared
        owner.memory.attach_completion(optimized)
        assert owner.memory.commit_selection()
        assert all(assembler.close() for assembler in assemblers)
        assert all(assembler.staging.close() for assembler in assemblers)
        del plan,baseline,optimized
        if restoring:
            del checkpoint
    assert owner.close() and archive.close()
