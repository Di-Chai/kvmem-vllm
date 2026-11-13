"""CPU-only replay of one qualified per-layer pressure's immutable KV reads.

Reuse the existing MemoryArchive, repository leases and native gather. This
isolates thread dispatch from byte copies; it is not model throughput evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import time
from types import SimpleNamespace as NS

import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import record_key
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.serving.packed_assembly import read_packed_records


def run(args):
    if args.output.exists():
        raise ValueError('Preserve prior component evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence = json.loads(args.evidence.read_text())
    event = next(e for e in evidence['events'] if e['event']=='pressure_selected')
    refresh = next(e for e in evidence['refresh'] if e['selection']=='pressure')
    if (event['accepted_history'] != refresh['history_begin'] or
            event['actual_method'] != 'accumulated_attention_per_layer'):
        raise ValueError('Expected the same qualified pressure event')
    if [e['layer'] for e in event['layers']] != list(range(16)):
        raise ValueError('Incomplete target-layer selections')
    layer_bytes=32*4*256
    raw_bytes=17*layer_bytes
    record_bytes=2*raw_bytes
    extent=event['accepted_history']//32
    skipped=refresh['assembly_skipped_tokens']//32
    orders=[list(e['selected_starts']) for e in event['layers']]
    if orders[0][:skipped] != list(range(0,skipped*32,32)):
        raise ValueError('Component only supports this first pressure stable prefix')
    orders.append(list(orders[0]))  # MTP follows the original layer-zero choice.
    for i in (0,16):orders[i]=orders[i][skipped:]
    if any(p%32 or not 0<=p//32<extent for layer in orders for p in layer):
        raise ValueError('Partial or out-of-history read')
    total=sum(map(len,orders))*2*layer_bytes
    if total != refresh['assembly_uploaded_bytes']:
        raise ValueError('Component reads differ from the observed byte count')
    torch.set_num_threads(1)
    archive=MemoryArchive(str(args.output.parent/'unused-cpu-only'),record_bytes,extent+2,
        gather_workers=8,cpu_direct_record_seal=True,cpu_seal_hash_workers=1)
    repo=_native.Repository(extent+2,32)
    leases=[repo.reserve(i*32,32).commit() for i in range(extent)]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(_native.__file__),args.evidence,
         *(Path(__file__).resolve().parents[1]/'src/vllm_kvmem'/name for name in
           ['memory_storage.py','storage.py','packed_assembly.py'])]}
    try:
        for begin in range(0,extent,32):
            group=leases[begin:begin+32]
            views=[]
            for i in range(begin,begin+len(group)):
                raw=b''.join(struct.pack('<Q',i*256+layer)*(layer_bytes//8) for layer in range(17))
                value=b''.join(struct.pack('<Q',(1<<64)-1-i*256-layer)*(layer_bytes//8) for layer in range(17))
                views.extend((raw,value))
            job=archive.write_from_views('diagnostic',group,views)
            if job is None:raise RuntimeError('Unexpected setup admission failure')
            job.result()
        views=None
        sources=archive._region_sources([archive.key('diagnostic',lease) for lease in leases])
        authority=NS(archive=archive,shape=(17,32,4,256),record_bytes=record_bytes,
            poll=lambda:None,records={record_key(lease.handle):NS(archived=True,count=32) for lease in leases},
            memory=NS(request_id='diagnostic',admitted_read=lambda h:(repo.pin(h),None)))
        buffers=[torch.empty(2*256*layer_bytes,dtype=torch.uint8) for _ in range(2)]
        for buffer in buffers:buffer.fill_(90)
        operations=[];pairs=[]
        for layer,order in enumerate(orders):
            for begin in range(0,len(order),256):
                ids=[p//32 for p in order[begin:begin+256]]
                values=[sources[i] for i in ids]
                handles=[leases[i].handle for i in ids]
                pair=[]
                for offset in (layer*layer_bytes,raw_bytes+layer*layer_bytes):
                    pair.append((offset,layer_bytes))
                    operations.append((handles,values,((offset,layer_bytes),)))
                pairs.append((handles,values,tuple(pair)))

        def measure(kind,verify=False):
            tasks=0;started=time.perf_counter()
            for index,(handles,values,regions) in enumerate(pairs if kind=='paired' else operations):
                offset=regions[0][0]
                target=buffers[index%2][:len(values)*layer_bytes*len(regions)]
                if kind=='paired':
                    _,stats=read_packed_records(authority,handles,destination=target,regions=regions)
                    tasks+=stats['cache_gather_tasks']
                elif kind=='read':
                    _,stats=read_packed_records(authority,handles,destination=target,
                                               region=(offset,layer_bytes))
                    tasks+=stats['cache_gather_tasks']
                else:
                    tasks+=archive._gather(memoryview(target.numpy()),values,offset,layer_bytes)
                if verify:
                    actual=memoryview(target.numpy())
                    for field,(offset,length) in enumerate(regions):
                        for i,source in enumerate(values):
                            begin=(field*len(values)+i)*layer_bytes
                            assert actual[begin:begin+layer_bytes].tobytes()==source[offset:offset+length]
            seconds=time.perf_counter()-started
            assert archive.io_usage()['archive_io_pending_bytes']==0
            return dict(seconds=seconds,tasks=tasks,bytes=total)

        validations=[];samples=[]
        for workers in (1,2,4,8):
            archive.gather_workers=workers
            for kind in ('read','paired','gather'):
                measure(kind,verify=True)
            validations.append(dict(workers=workers,all_kv_bytes_equal=True))
        for repeat in range(args.repeats):
            order=[1,2,4,8]
            order=order[repeat%4:]+order[:repeat%4]
            if repeat%2:order.reverse()
            for workers in order:
                archive.gather_workers=workers
                kinds=['read','paired','gather'];kinds=kinds[repeat%3:]+kinds[:repeat%3]
                if repeat%2:kinds.reverse()
                for kind in kinds:
                    samples.append(dict(repeat=repeat,workers=workers,kind=kind,**measure(kind)))
        summary=[]
        for workers in (1,2,4,8):
            for kind in ('read','paired','gather'):
                seconds=[s['seconds'] for s in samples if s['workers']==workers and s['kind']==kind]
                median=statistics.median(seconds)
                summary.append(dict(workers=workers,kind=kind,median_seconds=median,
                    min_seconds=min(seconds),max_seconds=max(seconds),GB_per_second=total/median/1e9))
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in source_hashes.items()):
            raise ValueError('Component implementation or evidence changed during measurement')
        report=dict(source_hashes=source_hashes,pressure_history=event['accepted_history'],
            original_pressure_copied_bytes=total,region_operations=len(operations),
            affinity=sorted(os.sched_getaffinity(0)),source_record_bytes=record_bytes,
            validations=validations,samples=samples,summary=summary,io_usage=archive.io_usage(),
            limitations=['CPU-only synthetic immutable archive bytes with actual selected record order and K/V extents.',
                'Pageable pre-touched buffers; no GPU DMA or model/GIL competitor.',
                'Uses public repository pins; native assembly-plan admission scanning is not represented.',
                'Every configuration uses the same already allocated source and destination buffers; no affinity changes.'])
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    finally:
        assert archive.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    run(parser.parse_args())
