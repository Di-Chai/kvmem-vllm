"""Full ready-archive assembly component; excludes publication and selection waits.

The real traced selection has 1,024 selected pages, 93 unmoved pages, and 931
reconstructed pages. Both arms use the same MemoryArchive, native selection,
FP8 source bytes, noncontiguous physical page table and reconstruction kernels.
Only CPU read lookahead changes. GPU cache reset and byte checks are untimed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time

import torch

from vllm_kvmem import _native
from vllm_kvmem.assembly import assemble_fp8
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.codec import native_vllm_fp8_views
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler


def hashes(root):
    manifest = json.loads((root/'frozen.json').read_text())
    actual = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in manifest['files']}
    assert actual == manifest['files'], 'Frozen execution files changed'
    return actual


def run(args):
    assert 2 <= args.rounds <= 20 and 1 <= args.warmup <= 4
    proof = hashes(args.root)
    assert Path(_native.__file__).resolve().is_relative_to((args.root/'runtime').resolve())
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    shape = layers,block,heads,dim = 17,32,4,256
    page,chunk = 16,128
    trace = json.loads((args.root/'pressure-selection.json').read_text())
    history = trace['history_begin']
    selected = {begin//block for begin,count in trace['selected_history_ranges'] if count == block}
    assert history == 49088 and len(selected) == 1024
    records = history//block
    raw_bytes = layers*block*heads*dim
    archive = MemoryArchive(str(args.root/'cpu-only'),raw_bytes*2,2048,
                            cpu_direct_record_seal=True,cpu_seal_hash_workers=1)
    config = KVMemConfig(block_tokens=block,select_budget=32768,gen_budget=16384,
                         repository_blocks=2048,sink_blocks=0,recent_blocks=0)
    owner = RequestAuthority('assembly-component',config,archive,layers,heads,dim,
                             raw_dtype=torch.float8_e4m3fn,resident_records=256)
    setup_begin = time.perf_counter()
    for begin in range(0,records,chunk):
        ids = torch.arange(begin,min(begin+chunk,records),dtype=torch.int32)
        raw = ((ids%110)+8).to(torch.uint8)[None,:,None,None,None]
        values = ((ids*7+3)%127).to(torch.uint8)[None,:,None,None,None]
        raw = raw.expand(layers,len(ids),block,heads,dim).contiguous().reshape(layers,-1,heads,dim)
        values = values.expand(layers,len(ids),block,heads,dim).contiguous().reshape_as(raw)
        assert owner.append(raw.view(torch.float8_e4m3fn),values,
                            history_begin=begin*block,local_rope_begin=begin*block)
        owner.publish_sealed(lambda: archive.wait_for_capacity(.05))
    raw = values = None
    scores = {r.handle.slot: float(r.history_begin//block in selected) for r in owner.records.values()}
    plan = owner.memory.select(scores,action=False)
    remaps = tuple((r.block_id,r.n_tokens,r.from_base,r.to_base,bool(r.skip)) for r in plan.remaps)
    assert len(remaps) == 1024 and sum(not r[-1] for r in remaps) == 931
    assert all(r.n_tokens == block and r.working_k_resident for r in plan.remaps)
    page_count = history//page
    table = (torch.randperm(page_count,device='cuda')+1).to(torch.int32)
    angles = torch.randn(history+page,16,device='cuda')
    trig = torch.cat((angles.cos(),angles.sin()),dim=-1)
    positions = torch.arange(history,device='cuda')
    slots = table[positions//page].long()*page+positions%page
    ids = positions//block
    raw_gpu = ((ids%110)+8).to(torch.uint8)[:,None,None].expand(-1,heads,dim).contiguous()
    values_gpu = ((ids*7+3)%127).to(torch.uint8)[:,None,None].expand(-1,heads,dim).contiguous()
    seeds,frames = [], {False:[],True:[]}
    for index in range(layers):
        backing = torch.zeros((page_count+1,heads,page,2*dim),dtype=torch.uint8,device='cuda')
        key,value = native_vllm_fp8_views(backing,heads,dim)
        scale = torch.tensor(.04052734375+index*.0009765625,device='cuda')
        assemble_fp8(raw_gpu.view(torch.float8_e4m3fn),values_gpu.view(torch.float8_e4m3fn),
                     positions,slots,trig,key,value,scale)
        seeds.append(backing)
        for flag in frames:
            k,v = native_vllm_fp8_views(backing.clone(),heads,dim)
            frames[flag].append(LayerWindow(k,v,trig,scale,table))
    raw_gpu = values_gpu = None
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_begin
    staging = {False:None,True:None}
    first_pointers = {}
    evidence = dict(scope=__doc__,geometry=dict(layers=layers,block_tokens=block,heads=heads,dim=dim,
        history_tokens=history,selected_records=1024,skipped_records=93,rebuilt_records=931,
        chunk_blocks=chunk,chunks=8,upload_bytes=931*raw_bytes),
        configuration=dict(raw_dtype='float8_e4m3fn',archive='MemoryArchive',gather_workers=8,
            cpu_mean_index_simd=False,raw_cache_bytes=0,reuse_resident_v=True,
            pinned_slots_per_arm=2,gpu_staging_slots_per_arm=2),
        setup_seconds=setup_seconds,source_hashes=proof,samples=[],byte_checks=[],warmup=[],
        device=torch.cuda.get_device_name(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        source_order_sha256=hashlib.sha256(json.dumps(remaps).encode()).hexdigest())
    last_result = None

    def sample(flag):
        nonlocal last_result
        # Recreate the same resident source window outside the timed interval.
        for seed,layer in zip(seeds,frames[flag]):
            key,value = native_vllm_fp8_views(seed,heads,dim)
            layer.key.view(torch.uint8).copy_(key.view(torch.uint8))
            layer.value.view(torch.uint8).copy_(value.view(torch.uint8))
        torch.cuda.synchronize()
        begin_event,end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        cpu = time.process_time()
        start = time.perf_counter()
        begin_event.record()
        assembler = NativeWindowAssembler(tuple(frames[flag]),block,chunk_blocks=chunk,
            packed=True,direct_pinned_reads=True,reuse_resident_v=True,raw_dtype=torch.float8_e4m3fn,
            read_prefetch=flag,staging=staging[flag])
        if staging[flag] is None:
            staging[flag] = assembler.staging
        assembler._shared_staging = True
        result = assembler.assemble(owner,plan)
        end_event.record()
        submitted = time.perf_counter()
        result.event.synchronize()
        end_event.synchronize()
        assert assembler.close()
        finished = time.perf_counter()
        pointers = tuple(b.data_ptr() for b in assembler.transfers.buffers+assembler.staging.buffers)
        if flag in first_pointers:
            assert pointers == first_pointers[flag], 'Fixed staging addresses changed'
        first_pointers[flag] = pointers
        assert len(set(pointers)) == 4
        assert result.rebuilt_tokens == 931*block and result.skipped_tokens == 93*block
        assert result.chunks == 8 and result.uploaded_bytes == 931*raw_bytes
        assert result.resident_v_reused_tokens == 931*block
        assert not assembler.transfers.preparing and not assembler.transfers._prepared
        if flag:
            assert result.read_stats['assembly_prefetch_submitted'] == result.read_stats['assembly_prefetch_completed']
            assert result.read_stats['assembly_prefetch_abandoned'] == 0
            assert result.read_stats['assembly_prefetch_submitted'] > 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        last_result = result
        return dict(enabled=flag,wall_ms=(finished-start)*1000,
            host_submission_ms=(submitted-start)*1000,completion_and_release_ms=(finished-submitted)*1000,
            process_cpu_ms=(time.process_time()-cpu)*1000,
            current_stream_elapsed_ms=begin_event.elapsed_time(end_event),
            host_prepare_ms=result.host_prepare_seconds*1000,read_stats=result.read_stats,
            uploaded_bytes=result.uploaded_bytes)

    try:
        for trial in range(args.warmup+args.rounds):
            order = (False,True) if trial%2 == 0 else (True,False)
            destination = evidence['warmup'] if trial < args.warmup else evidence['samples']
            for flag in order:
                destination.append(dict(trial=trial,**sample(flag)))
            equal = all(torch.equal(a.key.view(torch.uint8),b.key.view(torch.uint8)) and
                        torch.equal(a.value.view(torch.uint8),b.value.view(torch.uint8))
                        for a,b in zip(frames[False],frames[True]))
            assert equal, 'Off/on native K or V bytes differ'
            evidence['byte_checks'].append(dict(trial=trial,equal=equal,layers=layers,
                checked='Entire native physical K/V arena including skipped and unused pages'))
        summary = {}
        for flag in (False,True):
            rows = [row for row in evidence['samples'] if row['enabled'] == flag]
            summary['on' if flag else 'off'] = dict(samples=len(rows),
                median_wall_ms=statistics.median(r['wall_ms'] for r in rows),
                mean_wall_ms=statistics.mean(r['wall_ms'] for r in rows),
                total_wall_ms=sum(r['wall_ms'] for r in rows),
                median_host_submission_ms=statistics.median(r['host_submission_ms'] for r in rows),
                median_completion_and_release_ms=statistics.median(r['completion_and_release_ms'] for r in rows),
                median_read_queue_ms=statistics.median(r['read_stats'].get('read_queue_wait_seconds',0)*1000 for r in rows),
                median_cache_gather_ms=statistics.median(r['read_stats'].get('cache_copy_seconds',0)*1000 for r in rows))
        evidence['summary'] = summary
        evidence['median_wall_reduction_percent'] = (1-summary['on']['median_wall_ms']/summary['off']['median_wall_ms'])*100
        evidence['mean_wall_reduction_percent'] = (1-summary['on']['mean_wall_ms']/summary['off']['mean_wall_ms'])*100
        evidence['memory'] = dict(archive=archive.io_usage(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            pinned_staging_bytes_per_arm=2*chunk*raw_bytes*2,
            gpu_staging_bytes_per_arm=2*chunk*raw_bytes*2,resident_v_bytes_per_arm=931*raw_bytes)
        assert hashes(args.root) == proof
        owner.memory.attach_completion(last_result)
        assert owner.memory.commit_selection()
        for pool in staging.values():
            assert pool.close()
        del plan,last_result
        assert owner.close() and archive.close()
        evidence['cleanup_complete'] = True
        (args.root/'component.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps({key:evidence[key] for key in ('summary','median_wall_reduction_percent','mean_wall_reduction_percent','cleanup_complete')},indent=2))
    except BaseException as error:
        evidence['error'] = repr(error)
        (args.root/'component-failed.json').write_text(json.dumps(evidence,indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--warmup',type=int,default=2)
    args = parser.parse_args()
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
