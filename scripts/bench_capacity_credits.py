"""Isolate KVMem capacity accounting with real native full-attention managers.

No GPU, model or I/O. A small allocator facade preserves real physical block
ownership; identical requests repeatedly ask for already-admitted zero growth.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_kvmem.capacity import ReservedKVCacheManager
from vllm_kvmem.window_manager import WorkingSetManager


def gate(cls):
    pool = BlockPool(4096, False, 1600)
    spec = FullAttentionSpec(block_size=1600, num_kv_heads=4, head_size=256, dtype=torch.uint8)
    managers = [WorkingSetManager(FullAttentionManager(spec,pool,False,0,1600),32768,16384)
                for _ in range(4)]
    class Native:
        block_pool = pool
        max_model_len = 262144
        coordinator = NS(single_type_managers=tuple(managers))
        def allocate_slots(self, request, count, reserved_blocks=0):
            total = request.history+count
            need = sum(m.get_num_blocks_to_allocate(request.request_id,total,(),request.history,
                                                   request.history,total) for m in managers)
            if need > pool.get_num_free_blocks()-reserved_blocks:
                return None
            return tuple(m.allocate_new_blocks(request.request_id,total,total) for m in managers)
        def free(self,request):
            for m in managers: m.free(request.request_id)
    cfg = NS(speculative_config=None,max_concurrent_batches=1,
             cache_config=NS(enable_prefix_caching=False,mamba_cache_mode='none'))
    g = cls(Native(),cfg)
    requests = [NS(request_id=str(i),num_prompt_tokens=32768,max_tokens=4096,history=0) for i in range(8)]
    for request in requests:
        assert g.allocate_slots(request,32768) is not None
        request.history = 32768
    return g,requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('vllm_kvmem._capacity_before',a.before)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pairs={'before':gate(module.ReservedKVCacheManager),'after':gate(ReservedKVCacheManager)}
    samples={key:[] for key in pairs}
    def cycle(key):
        g,requests=pairs[key]
        if key=='after':g.begin_step()
        try:
            for request in requests: assert g.allocate_slots(request,0) is not None
        finally:
            if key=='after':g.end_step()
    for key in pairs:
        for _ in range(5):cycle(key)
    for repeat in range(7):
        for key in (('before','after') if repeat%2 else ('after','before')):
            start=time.perf_counter()
            for _ in range(50):cycle(key)
            samples[key].append((time.perf_counter()-start)*1000/50)
    scans={}
    for key,(g,requests) in pairs.items():
        original=g._owned;calls=[]
        def counted(request_id):calls.append(1);return original(request_id)
        g._owned=counted;cycle(key);g._owned=original
        scans[key]=len(calls)
        for request in requests:g.free(request)
    result=dict(scope='CPU capacity accounting; 8 requests; 4 full groups; 1600-token pages; no GDN/GPU/I/O',
        before_sha256=hashlib.sha256(a.before.read_bytes()).hexdigest(),
        results={key:dict(median_cycle_ms=statistics.median(rows),samples_cycle_ms=rows,
                         owned_scans_per_cycle=scans[key]) for key,rows in samples.items()})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
