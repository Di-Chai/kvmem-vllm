#!/usr/bin/env python3
"""Interleaved full cold RawKeyFPSCache snapshot/scoring A/B on fixed real Q/K.

This is a component benchmark. An owned mmap-Key slab adapter replaces the live
native archive reader, identically in both arms. Cold means empty derived FPS
cache, not flushed OS page caches; input hashing deliberately precedes timing.
No model generation, target labels or answer-dependent selection are involved.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import sys
from threading import Event, RLock, Thread
import time
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import torch

from benchmark_ruler import save
from analyze_query_key_fps import native_selections

REPO=Path(__file__).resolve().parents[1]
CASE='niah_multikey_2-0011'
DIAGNOSTIC=REPO/'artifacts/ruler-retrieval-diagnostic-20261023-r4'
CAPTURE=DIAGNOSTIC/'compressed/cases'/CASE/'diagnostic'
OFFSETS=REPO/'artifacts/query-key-fps-fixed-operands-20261023-r1'/CASE/'representative-token-offsets.npy'
OLD_MODULE=REPO/'artifacts/ruler-key-fps-halo-20261023-r1/runtime/src/vllm_kvmem/query_key_fps.py'
POOL_BYTES=2*1024**3
SCOPE=('Real fixed 256K diagnostic Q/K; actual cache.snapshot and scoring APIs. Owned mmap FP8 '
    'Key-slab adapter, not live native archive reading. Derived-cache cold, OS file cache uncontrolled '
    'and warmed by input hashing. No prefill, group-policy change, model generation or end-to-end throughput claim.')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):result.update(block)
    return result.hexdigest()


def identity(path):
    stat=Path(path).stat()
    return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)


def tensor_digest(tensor):
    """Hash active logical values only, without cloning an entire cache slab."""
    require(tensor.device.type=='cpu','Hash validation requires CPU data outside timing')
    result=hashlib.sha256()
    result.update(json.dumps(dict(shape=list(tensor.shape),dtype=str(tensor.dtype)),sort_keys=True).encode()+b'\n')
    for chunk in tensor:
        raw=chunk.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        result.update(memoryview(raw).cast('B'))
    return result.hexdigest()


def load_versions(old_path):
    """Keep the frozen relative base-cache dependency separate from current src."""
    old_path=Path(old_path).resolve()
    package_name='_fps_publication_frozen_'+hashlib.sha256(str(old_path).encode()).hexdigest()[:12]
    package=ModuleType(package_name);package.__path__=[str(old_path.parent)];package.__package__=package_name
    sys.modules[package_name]=package
    spec=importlib.util.spec_from_file_location(package_name+'.query_key_fps',old_path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    new=importlib.import_module('vllm_kvmem.query_key_fps')
    require(old.RawKeyFPSCache is not new.RawKeyFPSCache,'Old/new cache implementations are not isolated')
    return dict(old=old,new=new)


class OwnedMmapAuthority:
    """Original-block/generation adapter returning bounded owned CPU Key slabs.

    NumPy copies mmap bytes directly into the caller-owned CPU/pinned slab. No
    query, gold target or representative offsets enter this reader. It never
    returns borrowed mmap tensor views and never reads V or draft-layer data.
    """
    def __init__(self,keys,meta):
        require(keys.dtype==np.uint8 and keys.ndim==4,'Expected original FP8 byte array')
        self.keys,self.history_end=keys,meta['history']
        self.starts,self.counts=list(meta['starts']),list(meta['counts'])
        require(len(self.starts)==len(self.counts) and sum(self.counts)==keys.shape[1] and
            all(type(n) is int and 0<n<=32 for n in self.counts),'Invalid accepted Key partition')
        self.begins=np.r_[0,np.cumsum(self.counts)]
        self.records=[NS(history_begin=start,count=count,handle=NS(slot=i,generation=1))
            for i,(start,count) in enumerate(zip(self.starts,self.counts))]
        self.archive=NS(_lock=RLock())
        self.read_batches=0
        self.read_bytes=0

    def read_score_keys(self,handles,*,pin_memory=False):
        layers,_,heads,dim=self.keys.shape
        slots=[]
        for handle in handles:
            require(type(handle.slot) is int and 0<=handle.slot<len(self.records) and handle.generation==1,
                'Mmap adapter received a stale or foreign generation')
            slots.append(handle.slot)
        require(len(set(slots))==len(slots),'Mmap adapter received duplicate handles')
        require(len(slots)<=32,'FPS adapter read exceeds bounded 32-block tile')
        packed=torch.zeros((len(slots),layers,32,heads,dim),dtype=torch.float8_e4m3fn,pin_memory=pin_memory)
        byte_view=packed.view(torch.uint8).numpy()
        byte_count=0;started=time.perf_counter()
        for row,slot in enumerate(slots):
            begin,end=int(self.begins[slot]),int(self.begins[slot+1])
            np.copyto(byte_view[row,:,:end-begin],self.keys[:,begin:end],casting='no')
            byte_count+=(end-begin)*layers*heads*dim
        self.read_batches+=1;self.read_bytes+=byte_count
        return packed,dict(key_read_bytes=byte_count,value_read_bytes=0,draft_key_read_bytes=0,
            key_read_batches=1,key_resident_bytes=0,key_region_seconds=time.perf_counter()-started)

    def read_records(self,handles):
        raise AssertionError('Publication A/B must use the same optimized Key-only read path')


def expected_cache_digest(keys,counts,offsets,*,dtype=torch.float16):
    """Independent raw-Key gather from the previously saved CPU FPS offsets."""
    layers,_,heads,dim=keys.shape;blocks=len(counts)
    require(offsets.shape==(layers,blocks,4,heads),'Saved FPS offsets have another geometry')
    limits=np.asarray(counts)[None,:,None,None]
    require(np.all((offsets>=-1)&(offsets<limits)),'Saved FPS offsets exceed original valid tokens')
    begins=np.r_[0,np.cumsum(counts)]
    result=hashlib.sha256()
    result.update(json.dumps(dict(shape=[layers,blocks,4,heads,dim],dtype=str(dtype)),sort_keys=True).encode()+b'\n')
    head_indices=np.arange(heads)[None,None,:]
    for layer in range(layers):
        for begin in range(0,blocks,128):
            end=min(begin+128,blocks);indices=np.asarray(offsets[layer,begin:end],dtype=np.int64)
            positions=begins[begin:end,None,None]+np.maximum(indices,0)
            raw=np.array(keys[layer,positions,head_indices,:],copy=True)
            values=torch.from_numpy(raw).view(torch.float8_e4m3fn).to(dtype)
            values.masked_fill_(torch.from_numpy(indices<0)[...,None],0)
            result.update(memoryview(values.view(torch.uint8).numpy()).cast('B'))
    return result.hexdigest()


def interleaved_schedule(repeats):
    require(type(repeats) is int and repeats in (1,3),'Use one qualification pair or three interleaved pairs')
    return [(round_,arm) for round_ in range(repeats) for arm in (('old','new') if round_%2==0 else ('new','old'))]


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


class RSSSampler:
    def __init__(self,interval=.01):
        self.interval=interval;self.stop=Event();self.peak=0
    def _sample(self):
        while not self.stop.wait(self.interval):self.peak=max(self.peak,rss_bytes())
    def __enter__(self):
        self.before=self.peak=rss_bytes();self.worker=Thread(target=self._sample,daemon=True);self.worker.start();return self
    def __exit__(self,*error):
        self.peak=max(self.peak,rss_bytes());self.stop.set();self.worker.join()


def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def measured_run(module,cache,authority,queries,score_mask,*,device):
    """Only real snapshot + scoring; no hashes, equality checks or native picker."""
    require(cache.active==1,'Keep the cache owner pinned throughout timing and consumption')
    require(cache.bytes==0 and cache.capacity==0 and authority.archive._query_window_cache_bytes==0,
        'A cold trial started with a live derived cache')
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with RSSSampler() as memory:
        synchronize(device);began=time.perf_counter()
        representatives,indices,stats=cache.snapshot(authority,authority.records,optimized=True,device=device)
        synchronize(device);built=time.perf_counter()
        scores=module.key_fps_scores(queries,representatives,indices,score_mask,query_half=True,optimized=True)
        synchronize(device);finished=time.perf_counter()
    require(stats['cache_hit_blocks']==0 and stats['cache_rebuilt_blocks']==len(authority.records),
        'Cold run silently reused representative records')
    timing=dict(snapshot_wall_seconds=built-began,scoring_wall_seconds=finished-built,total_wall_seconds=finished-began,
        rss_before_bytes=memory.before,rss_peak_sampled_bytes=memory.peak,rss_sample_interval_seconds=memory.interval,
        process_cumulative_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else 0)
    return representatives,indices,scores,stats,timing


def validate_result(representatives,indices,scores,meta,offsets,expected_hash,profile):
    cpu_scores=scores.detach().cpu().float().numpy()
    cpu_indices=indices.detach().cpu().to(torch.int8)
    require(np.array_equal(cpu_indices.numpy(),offsets),'FPS offsets differ from saved CPU reference')
    cache_hash=tensor_digest(representatives)
    require(cache_hash==expected_hash,'Published cache bytes differ from original selected raw Keys')
    require(np.isfinite(cpu_scores).all() and np.all(cpu_scores>=0) and
        np.all(cpu_scores[~np.asarray(meta['score_mask'],dtype=bool)]==0),'Invalid native score-mask probabilities')
    selections=native_selections(profile,meta,dict(captured_mean_k=np.asarray(meta['mean_scores']),candidate=cpu_scores))
    return dict(indices_sha256=tensor_digest(cpu_indices),active_cache_bytes_sha256=cache_hash,
        scores_sha256=tensor_digest(torch.from_numpy(cpu_scores)),selected_starts=selections['candidate'],
        selected_tokens=sum(n for start,n in zip(meta['starts'],meta['counts']) if start in set(selections['candidate'])),
        saved_offsets_exact=True,original_key_cache_bytes_exact=True,native_picker_verified=True),cpu_indices,cpu_scores


@contextmanager
def exclusive_gpu(device):
    if torch.device(device).type!='cuda':yield;return
    import benchmark_block_graph as lifecycle
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not lifecycle.gpu_processes(),'Foreign GPU owner; component benchmark will not stop it')
        yield


def summarize(runs):
    result={}
    for arm in ('old','new'):
        rows=[row for row in runs if row['arm']==arm]
        result[arm]={key:dict(values=[r['timing'][key] for r in rows],median=statistics.median(r['timing'][key] for r in rows))
            for key in ('snapshot_wall_seconds','scoring_wall_seconds','total_wall_seconds')}
        result[arm]['publication_seconds']=[row['stats'].get('fps_publish_seconds') for row in rows]
    old,new=result['old']['total_wall_seconds']['median'],result['new']['total_wall_seconds']['median']
    result['total_speedup_ratio']=old/new if new else None
    result['total_time_reduction_fraction']=1-new/old if old else None
    result['limit']='Component full cold cache+score time only; sampled CPU peak is approximate and maxrss is cumulative. Not end-to-end throughput.'
    return result


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    runs=[];cache=None
    try:
        save(output/'status.json',dict(status='validating',at=time.time()))
        require(1<=args.threads<=8,'Use between one and eight CPU threads')
        torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=False
        schedule=interleaved_schedule(args.repeats)
        capture=args.capture.resolve();old_path=args.old_module.resolve();offset_path=args.offsets.resolve()
        files=dict(queries=capture/'queries.pt',keys=capture/'raw_keys.npy',metadata=capture/'archive.json',
            offsets=offset_path,profile=args.profile.resolve(),old_module=old_path,
            old_base=old_path.with_name('query_window_scoring.py'),new_module=REPO/'src/vllm_kvmem/query_key_fps.py',
            new_base=REPO/'src/vllm_kvmem/query_window_scoring.py',benchmark=Path(__file__).resolve())
        old_proof=old_path.parents[3]/'runtime-proof.json'
        require(sha(old_path)==read(old_proof)['files']['src/vllm_kvmem/query_key_fps.py'] and
            sha(files['old_base'])==read(old_proof)['files']['src/vllm_kvmem/query_window_scoring.py'],
            'Frozen old cache source differs from its runtime proof')
        files['old_runtime_proof']=old_proof
        proof={name:dict(path=str(path),sha256=sha(path),stat=identity(path)) for name,path in files.items()}
        save(output/'source-proof.json',proof)
        for name in ('old_module','old_base','new_module','new_base','benchmark'):
            folder=output/'source'/name;folder.mkdir(parents=True)
            shutil.copy2(files[name],folder/files[name].name)
        meta=read(files['metadata']);profile=read(files['profile'])
        require(meta['history']==261952 and len(meta['starts'])==8186 and meta['dtype']=='float8_e4m3fn' and
            meta['raw_k_scale']==1.,'Expected the fixed 256K 0011 unit-scale FP8 capture')
        keys=np.load(files['keys'],mmap_mode='r');offsets=np.load(files['offsets'],allow_pickle=False)
        require(list(keys.shape)==meta['key_shape'],'Original Key shape changed')
        queries_cpu=torch.load(files['queries'],map_location='cpu',weights_only=True)
        require(queries_cpu.ndim==4 and queries_cpu.shape[0]==keys.shape[0] and queries_cpu.shape[-1]==keys.shape[-1],
            'Original Query/Key geometry differs')
        expected=expected_cache_digest(keys,meta['counts'],offsets)
        authority=OwnedMmapAuthority(keys,meta);versions=load_versions(old_path)
        layers,_,heads,dim=keys.shape
        caches={arm:module.RawKeyFPSCache(authority.archive,layers,heads,dim,dtype=torch.float16,max_bytes=POOL_BYTES)
            for arm,module in versions.items()}
        environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
            cpu_threads=torch.get_num_threads(),device=args.device,omp_num_threads=os.getenv('OMP_NUM_THREADS'),
            openblas_num_threads=os.getenv('OPENBLAS_NUM_THREADS'),schedule=schedule,repeats_per_arm=args.repeats,
            query_shape=list(queries_cpu.shape),key_shape=list(keys.shape),pool_capacity_bytes=POOL_BYTES,
            optimized=True,scope=SCOPE,expected_active_cache_bytes_sha256=expected,
            timing_validation='Large equality/hash/native picker validation occurs after synchronize and timer stop.')
        with exclusive_gpu(args.device):
            queries=queries_cpu.to(args.device);score_mask=torch.tensor(meta['score_mask'],dtype=torch.bool)
            if torch.device(args.device).type=='cuda':environment['gpu_name']=torch.cuda.get_device_name(args.device)
            save(output/'environment.json',environment)
            first=None
            for trial,(round_,arm) in enumerate(schedule):
                directory=output/f'{trial:02d}-{arm}-round-{round_+1}';directory.mkdir()
                for item in caches.values():
                    require(item.active==0,'Prior owner is still consuming a cache view');item.clear()
                gc.collect();cache=caches[arm]
                save(output/'status.json',dict(status='running',trial=trial,arm=arm,completed=len(runs),at=time.time()))
                save(directory/'attempt.json',dict(arm=arm,round=round_,at=time.time(),cold_derived_cache=True))
                representatives=indices=scores=cpu_indices=cpu_scores=None
                try:
                    with cache.use():
                        representatives,indices,scores,stats,timing=measured_run(versions[arm],cache,authority,queries,score_mask,device=args.device)
                        save(directory/'measurement.json',dict(arm=arm,round=round_,stats=stats,timing=timing))
                        audit,cpu_indices,cpu_scores=validate_result(representatives,indices,scores,meta,offsets,expected,profile)
                        np.save(directory/'indices.npy',cpu_indices.numpy(),allow_pickle=False)
                        np.save(directory/'scores.npy',cpu_scores,allow_pickle=False)
                        if first is None:first=dict(audit=audit,scores=cpu_scores.copy())
                        difference=np.abs(cpu_scores-first['scores'])
                        audit.update(score_max_abs_difference=float(difference.max(initial=0)),
                            scores_exact_to_first=np.array_equal(cpu_scores,first['scores']),
                            native_picker_exact_to_first=audit['selected_starts']==first['audit']['selected_starts'])
                        require(audit['scores_exact_to_first'] and audit['native_picker_exact_to_first'],
                            'Old/new full page scores or native picker changed')
                        save(directory/'audit.json',audit)
                        result=dict(trial=trial,round=round_,arm=arm,stats=stats,timing=timing,audit=audit,
                            files={name:sha(directory/name) for name in ('measurement.json','indices.npy','scores.npy','audit.json')})
                        save(directory/'result.json',result);runs.append(result)
                finally:
                    # Drop all borrowed cache views before budget release, also
                    # when independent validation rejects a completed trial.
                    representatives=indices=scores=cpu_indices=cpu_scores=None
                    cache.clear()
                    require(cache.active==0 and authority.archive._query_window_cache_bytes==0,
                        'Derived cache owner/budget not released')
                    save(directory/'cleanup.json',dict(cache_bytes=cache.bytes,shared_pool_bytes=authority.archive._query_window_cache_bytes,
                        active_uses=cache.active))
                save(output/'progress.json',dict(completed=len(runs),expected=len(schedule),runs=runs))
                print(json.dumps(dict(trial=trial,arm=arm,timing=timing,publish_seconds=stats.get('fps_publish_seconds')),allow_nan=False),flush=True)
            del queries,score_mask
            synchronize(args.device)
        for name,path in files.items():
            require(identity(path)==proof[name]['stat'] and sha(path)==proof[name]['sha256'],'Fixed source changed: '+name)
        save(output/'summary.json',dict(status='completed',scope=SCOPE,runs=runs,summary=summarize(runs),
            exact_saved_offsets=True,exact_original_key_cache_bytes=True,exact_scores_and_native_picker=True,
            no_model_generation=True,source_files_unchanged=True))
        save(output/'cleanup.json',dict(shared_pool_bytes=authority.archive._query_window_cache_bytes,
            active_uses=sum(item.active for item in caches.values()),all_cache_bytes=sum(item.bytes for item in caches.values())))
        save(output/'status.json',dict(status='completed',completed=len(runs),at=time.time()))
    except BaseException as error:
        save(output/'failure.json',dict(error=repr(error),completed=len(runs),at=time.time()))
        save(output/'status.json',dict(status='failed',completed=len(runs),error=repr(error),at=time.time()))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capture',type=Path,default=CAPTURE)
    parser.add_argument('--offsets',type=Path,default=OFFSETS)
    parser.add_argument('--profile',type=Path,default=DIAGNOSTIC/'compressed/profile.json')
    parser.add_argument('--old-module',type=Path,default=OLD_MODULE)
    parser.add_argument('--repeats',type=int,choices=(1,3),default=3)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=1)
    run(parser.parse_args(argv))


if __name__=='__main__':main()
