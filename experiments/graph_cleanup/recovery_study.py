"""Copy/read recovery components using archived real native FP8 attention KV.

No global cache drops. Disk tests use the actual filesystem and disclose its
type; a Ceph mount is not labeled a local SSD. This does not time the complete
KVMem admission/page-table/recurrent-state restoration path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from protocol import quantile


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def run(args):
    import torch
    args.output.mkdir(parents=True,exist_ok=False)
    source=torch.load(args.archive,map_location='cpu',weights_only=True)
    kv=source['kv']
    save(args.output/'environment.json',dict(gpu=torch.cuda.get_device_name(),
        filesystem=subprocess.check_output(['df','-T',str(args.output)],text=True),
        scope=source['scope'],repeats=20,cache_policy='posix_fadvise DONTNEED for this file only; remote/server caches not controlled'))
    results=[]
    for length in (1024,4096,8192,16384):
        cpu=kv[:,:length].contiguous()
        pinned=torch.empty_like(cpu,pin_memory=True);pinned.copy_(cpu)
        device=torch.empty(cpu.shape,dtype=torch.uint8,device='cuda')
        expected=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        path=args.output/f'kv-{length}.bin'
        with path.open('xb',buffering=0) as f:
            view=memoryview(cpu.numpy()).cast('B')
            offset=0
            while offset<len(view):offset+=f.write(view[offset:])
            os.fsync(f.fileno())
        rows=[]
        for method in ('pageable_cpu','pinned_cpu','file_warm','file_client_cache_advised_cold'):
            for repeat in range(-2,20):
                torch.cuda.synchronize()
                t=time.perf_counter();read_s=0.
                if method.startswith('file_'):
                    with path.open('rb',buffering=0) as f:
                        if method.endswith('advised_cold'):
                            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                        target=memoryview(pinned.numpy()).cast('B');offset=0
                        while offset<len(target):
                            n=f.readinto(target[offset:])
                            if not n:raise RuntimeError('Incomplete archive read')
                            offset+=n
                    read_s=time.perf_counter()-t
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                device.copy_(cpu if method=='pageable_cpu' else pinned,non_blocking=method!='pageable_cpu')
                end.record();end.synchronize()
                elapsed=time.perf_counter()-t
                if repeat>=0:rows.append(dict(method=method,repeat=repeat,wall_seconds=elapsed,
                    file_read_seconds=read_s,gpu_copy_ms=start.elapsed_time(end)))
            if hashlib.sha256(device.cpu().numpy().tobytes()).hexdigest()!=expected:
                raise RuntimeError('Restored raw native KV differs')
        entry=dict(tokens=length,bytes=cpu.numel(),sha256=expected,raw_copy_verified=True,
            recurrent_state_included=False,physical_page_installation_included=False,measurements=rows,
            summaries={method:dict(p50_ms=quantile([r['wall_seconds']*1000 for r in rows if r['method']==method],.5),
                                   p95_ms=quantile([r['wall_seconds']*1000 for r in rows if r['method']==method],.95))
                       for method in sorted({r['method'] for r in rows})})
        results.append(entry);save(args.output/'results.json',results)
        print(json.dumps(dict(tokens=length,bytes=cpu.numel(),summaries=entry['summaries'])),flush=True)
        del device,pinned,cpu


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
