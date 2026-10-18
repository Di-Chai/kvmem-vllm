#!/usr/bin/env python3
"""CPU-only cost bound for a redundant pre-seal MeanK reduction.

This does not implement copy-only scatter, change production code, or predict
serving throughput. It times the separable current Torch reduction, then calls
the real append/scatter/finalize/archive path and verifies its accepted history.
Run without --run to inspect the bounded allocation plan without running work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive


@dataclass(frozen=True)
class Geometry:
    target_layers: int = 16
    mtp_layers: int = 1
    heads: int = 4
    dim: int = 256
    block: int = 32
    tokens: int = 2048
    owners: int = 8
    rejected: int = 3

    @property
    def layers(self):
        return self.target_layers + self.mtp_layers

    @property
    def records(self):
        return self.tokens // self.block

    @property
    def record_bytes(self):
        return self.layers * self.block * self.heads * self.dim * 2

    def allocation_plan(self):
        sizes = dict(
            source_raw_and_values_bytes=self.owners*self.layers*(self.tokens+self.rejected)*self.heads*self.dim*2,
            destination_raw_and_values_bytes=self.owners*self.records*self.record_bytes,
            immutable_archive_bytes=self.owners*self.records*self.record_bytes,
            one_archive_pack_bytes=self.records*self.record_bytes,
            sum_and_mean_index_bytes=self.owners*self.records*self.target_layers*self.heads*self.dim*6,
            retained_delta_bytes=self.owners*self.records*self.target_layers*self.heads*self.dim*4,
            one_widened_block_bytes=self.target_layers*self.block*self.heads*self.dim*4,
        )
        sizes['conservative_tensor_payload_peak_bytes'] = sum(sizes.values())
        return sizes

    def validate(self):
        if any(type(x) is not int or x <= 0 for x in (
                self.target_layers,self.heads,self.dim,self.block,self.tokens,self.owners)):
            raise ValueError('Geometry must contain positive integers')
        if self.mtp_layers not in (0,1) or not 0 <= self.rejected <= 8:
            raise ValueError('Only zero/one MTP layer and up to eight rejected rows are supported')
        if self.block < 8 or self.tokens % self.block or self.owners > 8:
            raise ValueError('Require B>=8, complete prefill blocks, and at most eight owners')
        if self.allocation_plan()['conservative_tensor_payload_peak_bytes'] > 2*1024**3:
            raise ValueError('Benchmark payload estimate exceeds its 2 GiB bound')


def sources(g):
    """Finite mixed-sign FP8 values and distinct owner/layer/token identities."""
    g.validate()
    result=[]
    # Exclude the two E4M3 NaN codes. Tiny and large signs exercise cancellation.
    codes=torch.tensor([0,1,3,7,8,24,56,72,112,126,128,129,131,135,136,152,184,200,240,254],dtype=torch.uint8)
    lane=torch.arange(g.heads*g.dim).reshape(g.heads,g.dim)
    for owner in range(g.owners):
        raw=torch.empty(g.layers,g.tokens+g.rejected,g.heads,g.dim,dtype=torch.uint8)
        values=torch.empty_like(raw)
        for layer in range(g.layers):
            for begin in range(0,g.tokens+g.rejected,g.block):
                count=min(g.block,g.tokens+g.rejected-begin)
                positions=torch.arange(begin,begin+count)[:,None,None]
                offsets=positions+lane[None]+owner*7+layer*3
                raw[layer,begin:begin+count]=codes[offsets % len(codes)]
                values[layer,begin:begin+count]=((offsets*13+owner*17)%256).to(torch.uint8)
        result.append((raw.view(torch.float8_e4m3fn),values))
    return result


def preseal_deltas(g,inputs):
    """The exact n>B-small-decode expression currently in append_authority_batch."""
    return [raw[:g.target_layers,start:start+g.block].float().sum(dim=1)
            for raw,_ in inputs for start in range(0,g.tokens,g.block)]


@contextmanager
def native_timings(enabled):
    """Local subprocess-only wrappers, retaining the real native functions."""
    data={name:dict(calls=0,seconds=0.) for name in ('scatter_history_batch','finalize_mean_fp8')}
    old={name:getattr(_native,name) for name in data}
    if enabled:
        for name,original in old.items():
            def wrapped(*args,_name=name,_original=original,**kwargs):
                started=time.perf_counter()
                try:
                    return _original(*args,**kwargs)
                finally:
                    data[_name]['calls']+=1
                    data[_name]['seconds']+=time.perf_counter()-started
            setattr(_native,name,wrapped)
    try:
        yield data
    finally:
        for name,original in old.items():
            setattr(_native,name,original)


def make_owners(g):
    cfg=KVMemConfig(block_tokens=g.block,select_budget=g.tokens,gen_budget=2*g.block,
        repository_blocks=g.records+2,sink_blocks=1,recent_blocks=0)
    archive=MemoryArchive('',g.record_bytes,g.owners*(g.records+2),workers=1,
        max_pending=g.owners,gather_workers=1)
    archive.async_seal=True
    owners=[RequestAuthority(f'micro-{i}',cfg,archive,g.layers,g.heads,g.dim,
        raw_dtype=torch.float8_e4m3fn,resident_records=g.records+1,
        scoring_layers=g.target_layers,mean_k_backend='qw3_cuda') for i in range(g.owners)]
    return owners,archive


def verify_history(g,owners,inputs):
    hashes={name:hashlib.sha256() for name in ('raw','values','mean_index')}
    checked=0
    for owner,(source_raw,source_v) in zip(owners,inputs):
        if owner.history_end!=g.tokens:
            raise AssertionError('Rejected rows entered accepted history')
        records=sorted(owner.records.values(),key=lambda r:r.history_begin)
        if len(records)!=g.records:
            raise AssertionError('Incomplete logical history')
        for record in records:
            begin=record.history_begin
            raw,values=owner.read_record(record.handle)
            if (record.count!=g.block or
                not torch.equal(raw.view(torch.uint8),source_raw[:,begin:begin+g.block].view(torch.uint8)) or
                not torch.equal(values,source_v[:,begin:begin+g.block])):
                raise AssertionError('Archived accepted raw K/V differs from source')
            # Existing real finalizer, deliberately poisoned initial sums:
            # proves full sealing overwrites prior deltas rather than adds them.
            reference=torch.full_like(record.key_sum,12345.)
            _native.finalize_mean_fp8(raw.view(torch.uint8).numpy(),reference.numpy(),g.block)
            expected=(reference/g.block).to(torch.float16)
            if not torch.equal(reference,record.key_sum) or not torch.equal(expected,record.mean_key):
                raise AssertionError('Full-seal index depends on overwritten incremental sum')
            for name,tensor in (('raw',raw.view(torch.uint8)),('values',values),
                                ('mean_index',record.mean_key.view(torch.uint8))):
                hashes[name].update(memoryview(tensor.contiguous().numpy()).cast('B'))
            checked+=1
    return dict(history_tokens_per_owner=g.tokens,owners=len(owners),records_checked=checked,
        rejected_rows_excluded=True,archived_raw_value_byte_equal=True,
        mean_index_bit_equal=True,poisoned_prior_sum_overwritten=True,
        sha256={k:v.hexdigest() for k,v in hashes.items()})


def run_append(g,inputs,*,measure=True):
    owners,archive=make_owners(g)
    try:
        entries=[(a,raw,values,dict(history_begin=0,local_rope_begin=0,accepted=g.tokens))
                 for a,(raw,values) in zip(owners,inputs)]
        with native_timings(measure) as native:
            start=time.perf_counter() if measure else None
            consumed=append_authority_batch(entries)
            append_seconds=time.perf_counter()-start if measure else None
        if consumed!=g.owners:
            raise AssertionError('Unexpected append capacity deferral')
        start=time.perf_counter() if measure else None
        for a in owners:
            a.publish_sealed(lambda:archive.wait_for_capacity(10))
        drain_seconds=time.perf_counter()-start if measure else None
        proof=verify_history(g,owners,inputs)
        return dict(append_host_seconds=append_seconds,publication_drain_seconds=drain_seconds,
                    native=native,proof=proof)
    finally:
        for a in owners:
            for job,_ in tuple(a.writes):job.result(timeout=30)
            if not a.close():raise AssertionError('Benchmark retained an authority lease')
        if archive.tier.used_slots or not archive.close():
            raise AssertionError('Benchmark retained CPU archive storage')


def partial_contract():
    """Real two-append history: partial sum stays live, full seal overwrites it."""
    g=Geometry(target_layers=2,mtp_layers=1,heads=1,dim=64,block=8,tokens=8,owners=1)
    inputs=sources(g);raw,values=inputs[0];owners,archive=make_owners(g);a=owners[0]
    try:
        assert append_authority_batch([(a,raw[:,:9],values[:,:9],
            dict(history_begin=0,local_rope_begin=0,accepted=7))])==1
        record=next(iter(a.records.values()))
        expected=raw[:2,:7].float().sum(dim=1)
        assert a.history_end==7 and record.count==7 and record.mean_key is None
        assert torch.equal(record.key_sum,expected) and bool(torch.count_nonzero(expected))
        _,index=a.mean_keys()
        assert torch.equal(index[:,0],(expected/7).half())
        # Completing one token uses the real qualified fused/scatter path.
        assert append_authority_batch([(a,raw[:,7:10],values[:,7:10],
            dict(history_begin=7,local_rope_begin=7,accepted=1))])==1
        a.publish_sealed(lambda:archive.wait_for_capacity(10))
        sequential=torch.zeros_like(expected)
        for token in range(8):sequential+=raw[:2,token].float()
        assert torch.equal(record.key_sum,sequential)
        proof=verify_history(g,owners,inputs)
        proof.update(partial_sum_required=True,partial_index_bit_equal=True,
            completed_seal_matches_token_order=True,mtp_excluded_from_mean=True)
        return proof
    finally:
        for job,_ in tuple(a.writes):job.result(timeout=30)
        assert a.close() and archive.close()


def benchmark(g,*,repeats=3,warmups=1):
    g.validate()
    if not 1<=repeats<=5 or not 0<=warmups<=2:
        raise ValueError('Use 1..5 repeats and 0..2 warmups')
    inputs=sources(g)
    samples=[]
    for repeat in range(-warmups,repeats):
        start=time.perf_counter()
        deltas=preseal_deltas(g,inputs)
        delta_seconds=time.perf_counter()-start
        delta_bytes=sum(d.numel()*d.element_size() for d in deltas)
        del deltas
        row=run_append(g,inputs)
        if repeat>=0:
            row.update(preseal_delta_seconds=delta_seconds,retained_delta_bytes=delta_bytes)
            samples.append(row)
        gc.collect()
    del inputs
    return dict(samples=samples,partial_contract=partial_contract(),
        median_preseal_delta_seconds=statistics.median(r['preseal_delta_seconds'] for r in samples),
        median_current_append_host_seconds=statistics.median(r['append_host_seconds'] for r in samples),
        conservative_scope='Pre-seal delta is measured separately, then repeated inside unchanged append. '
            'Its time is an isolated CPU opportunity bound, not a measured optimization or GPU throughput gain. '
            'Append includes allocation, native scatter/finalize, seal submission and Python bookkeeping. '
            'Archive workers can overlap append. Native stage times are nested in append, never additive.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Execute CPU timing; omit for allocation plan only')
    p.add_argument('--output',type=Path)
    p.add_argument('--owners',type=int,default=8)
    p.add_argument('--prefill-tokens',type=int,default=2048)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if not 1<=args.threads<=4:raise ValueError('Use 1..4 CPU Torch threads')
    g=Geometry(tokens=args.prefill_tokens,owners=args.owners);g.validate()
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/'src/vllm_kvmem/authority.py',root/'csrc/bindings.cpp',Path(_native.__file__)]
    report=dict(status='prepared',geometry=asdict(g),allocation_plan=g.allocation_plan(),
        scope='CPU-only existing FP8 accepted-history append, not model inference; no production edits.',
        environment=dict(python=platform.python_version(),torch=torch.__version__,cpu=platform.processor(),
            affinity_cpus=len(os.sched_getaffinity(0)),torch_threads=args.threads),
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    if args.run:
        torch.set_num_threads(args.threads)
        report['rss_before_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['results']=benchmark(g,repeats=args.repeats,warmups=args.warmups)
        report['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['status']='completed'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
