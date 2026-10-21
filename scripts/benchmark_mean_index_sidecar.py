"""Diagnostic CPU-finalize vs GPU ordered-sum component; no model TPS claim.

Two scopes are reported. index_only starts with the common raw archive capture
already available on both devices. capture_to_archive includes the full existing
raw/V payload D2H, extra sidecar kernel/D2H, original logical-2K append, CPU mean
conversion, immutable CPU seal/hash/publication, and separate owner retirement.
The fixture has no model, graph construction, pressure picker or concurrent GPU
work; fresh authority staging allocation is included, so it is not a warm serving
throughput estimate. Sidecar allocation is fixed and reported outside timing.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch

from mean_index_sidecar import Geometry, OrderedMeanSidecar, SumConsumer, cpu_jobs
from vllm_kvmem import _native


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def fixture(tokens, *, geometry=None, seed=738):
    """Use the real fused capture packer; preserve its header/raw/V layout."""
    from vllm_kvmem.capture_runtime import CaptureRuntime
    from vllm_kvmem.capture_pack import PackLimits
    geometry = geometry or Geometry()
    geometry.sizes(tokens)
    names = tuple(f'target.{i}.attn' for i in range(geometry.scoring_layers))
    names += tuple(f'mtp.{i}.attn' for i in range(geometry.layers-geometry.scoring_layers))
    torch.manual_seed(seed)
    positions = torch.arange(tokens,device='cuda',dtype=torch.int64)
    captures = tuple(NS(name=name, positions=positions,
        key=torch.randn(tokens,geometry.heads,geometry.dim,device='cuda',dtype=torch.bfloat16)*3,
        value=torch.randn(tokens,geometry.heads,geometry.dim,device='cuda',dtype=torch.bfloat16)*4)
        for name in names)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = NS(capture_fused_pack=True)
    runtime.raw_dtype = torch.float8_e4m3fn
    runtime.scales = torch.linspace(.3,1.7,geometry.layers,device='cuda')
    runtime.names = names
    runtime._capture_pack_limits = PackLimits(geometry.layers,geometry.heads,geometry.dim,
                                              geometry.capacity_tokens,8)
    batch = NS(num_reqs=1,num_tokens=tokens,positions=positions,
               query_start_loc=torch.tensor([0,tokens],device='cuda',dtype=torch.int32))
    payload,count = runtime._capture_payload(captures,batch,
        torch.zeros(1,device='cuda',dtype=torch.int32),(0,))
    assert count == tokens and runtime.capture_pack_stats.get('fused_batches') == 1
    n = geometry.layers*tokens*geometry.heads*geometry.dim
    shape = (geometry.layers,tokens,geometry.heads,geometry.dim)
    raw = payload[40:40+n].view(torch.float8_e4m3fn).reshape(shape)
    host = torch.empty(payload.shape,pin_memory=True,dtype=torch.uint8)
    host.copy_(payload,non_blocking=True)
    completed = torch.cuda.Event()
    completed.record(); completed.synchronize()
    host_raw = host[40:40+n].view(torch.float8_e4m3fn).reshape(shape)
    host_values = host[40+n:].reshape(shape)
    assert host[:40].view(torch.int64).tolist() == [0,tokens,0,0,0]
    expected = torch.empty(tokens//32,geometry.scoring_layers,geometry.heads,geometry.dim)
    _native.finalize_mean_batch(cpu_jobs(host_raw,expected))
    mean = torch.stack([(row/32).half() for row in expected])
    digests = [hashlib.sha256(host_raw[:,i:i+32].view(torch.uint8).numpy().tobytes()+
                            host_values[:,i:i+32].numpy().tobytes()).hexdigest()
               for i in range(0,tokens,32)]
    sidecar = OrderedMeanSidecar(geometry,archive_names=runtime.names,
                                target_names=names[:geometry.scoring_layers])
    # Pack inputs are now fenced and can be freed. Both arms use this exact
    # immutable quantized output; input preparation never enters timed scopes.
    del runtime,captures
    return NS(geometry=geometry,tokens=tokens,payload=payload,raw=raw,host=host,
              host_raw=host_raw,host_values=host_values,expected=expected,mean=mean,
              digests=digests,sidecar=sidecar,download_done=completed,names=names)


@torch.inference_mode()
def index_once(env, mode):
    output = torch.empty_like(env.expected)
    started = time.perf_counter()
    jobs = cpu_jobs(env.host_raw,output)
    timing = {}
    ticket = None
    if mode == 'cpu':
        _native.finalize_mean_batch(jobs)
    else:
        ticket = env.sidecar.submit(env.raw)
        sums,invalid = ticket.resolve()
        consumer = SumConsumer(sums,invalid,_native.finalize_mean_batch)
        consumer(jobs)
        timing.update(env.sidecar.timings(),fallback_records=consumer.fallback_records)
    finalized = time.perf_counter()
    means = [(row/32).half() for row in output]
    divided = time.perf_counter()
    if ticket is not None:
        ticket.release()
    ended = time.perf_counter()
    timing.update(wall_ms=(ended-started)*1000,
                  finalize_ms=(finalized-started)*1000,cpu_divide_half_ms=(divided-finalized)*1000,
                  sidecar_release_ms=(ended-divided)*1000)
    return timing,output,torch.stack(means)


@torch.inference_mode()
def append_once(env, mode):
    from vllm_kvmem.authority import RequestAuthority,append_authority_batch
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    g = env.geometry
    cfg = KVMemConfig(block_tokens=32,select_budget=32768,gen_budget=16384,
                     repository_blocks=2048,sink_blocks=32,recent_blocks=64)
    record_bytes = g.layers*32*g.heads*g.dim*2
    archive = MemoryArchive('/tmp/mean-index-sidecar-no-files',record_bytes,2048,
        max_pending=8,io_max_bytes=record_bytes*(env.tokens//32+1)*2,
        cpu_direct_record_seal=True,cpu_seal_hash_workers=8)
    archive.async_seal = True  # make_shared_archive sets this from serving options
    owner = RequestAuthority('component',cfg,archive,g.layers,g.heads,g.dim,
        raw_dtype=torch.float8_e4m3fn,resident_records=env.tokens//32+1,
        scoring_layers=g.scoring_layers,mean_k_backend='qw3_cuda')
    ticket = None
    started = time.perf_counter()
    env.host.copy_(env.payload,non_blocking=True)
    env.download_done.record()
    if mode == 'gpu':
        ticket = env.sidecar.submit(env.raw)
        sums,invalid = ticket.resolve()  # includes the preceding common payload D2H
        consumer = SumConsumer(sums,invalid,_native.finalize_mean_batch)
    else:
        env.download_done.synchronize()
        consumer = None
    downloaded = time.perf_counter()
    with consumer.installed() if consumer is not None else nullcontext():
        for begin in range(0,env.tokens,2048):
            count = min(2048,env.tokens-begin)
            assert append_authority_batch([(owner,env.host_raw[:,begin:begin+count],
                env.host_values[:,begin:begin+count],dict(history_begin=begin,
                    local_rope_begin=begin,accepted=count))]) == 1
    appended = time.perf_counter()
    owner.publish_sealed(lambda: archive.wait_for_capacity(.01))
    sealed = time.perf_counter()
    timing = {}
    if ticket is not None:
        timing.update(env.sidecar.timings(),fallback_records=consumer.fallback_records)
        ticket.release()
    published = time.perf_counter()
    timing.update(download_and_sidecar_ms=(downloaded-started)*1000,
        append_ms=(appended-downloaded)*1000,publication_tail_ms=(sealed-appended)*1000,
        sidecar_release_ms=(published-sealed)*1000,publication_wall_ms=(published-started)*1000)
    # Numerical/checksum validation is outside both timed phases. Retirement
    # remains measured, since deferring cleanup must not masquerade as a win.
    records = sorted(owner.records.values(),key=lambda r:r.history_begin)
    assert owner.history_end == env.tokens and len(records) == env.tokens//32
    assert [r.digest for r in records] == env.digests
    assert all(torch.equal(r.key_sum.view(torch.int32),env.expected[i].view(torch.int32)) and
               torch.equal(r.mean_key.view(torch.int16),env.mean[i].view(torch.int16))
               for i,r in enumerate(records))
    usage = archive.io_usage()
    assert usage['seal_records'] == env.tokens//32
    assert usage['seal_direct_bytes'] == env.tokens*g.layers*g.heads*g.dim*2
    records.clear()
    retired = time.perf_counter()
    assert owner.close() and archive.close()
    timing['retirement_ms'] = (time.perf_counter()-retired)*1000
    timing['complete_lifecycle_ms'] = timing['publication_wall_ms']+timing['retirement_ms']
    return timing,usage


def summary(samples):
    return {key:dict(median=statistics.median([row[key] for row in samples]),
                     min=min(row[key] for row in samples),max=max(row[key] for row in samples))
            for key in samples[0] if key.endswith('_ms')}


def run(args):
    if not 2 <= args.rounds <= 30 or not 1 <= args.warmup <= 8:
        raise ValueError('Bounded interleaved round counts required')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__),Path(__file__).with_name('mean_index_sidecar.py'),
               root/'src/vllm_kvmem/authority.py',root/'src/vllm_kvmem/capture_pack.py',
               root/'src/vllm_kvmem/memory_storage.py',root/'csrc/bindings.cpp',Path(_native.__file__)]
    fingerprints = {str(path):sha(path) for path in sources}
    cases = []
    for tokens in args.tokens:
        env = fixture(tokens)
        # Exclude lazy compilation/cold buffers from formal samples, retain its
        # elapsed cost independently. Every formal pair revalidates exact bits.
        cold = time.perf_counter()
        index_once(env,'gpu')
        cold_seconds = time.perf_counter()-cold
        samples = {scope:{mode:[] for mode in ('cpu','gpu')}
                   for scope in ('index_only','capture_to_archive')}
        archive_usage = {}
        for scope in samples:
            for trial in range(args.warmup+args.rounds):
                for mode in (('cpu','gpu') if trial%2 == 0 else ('gpu','cpu')):
                    if scope == 'index_only':
                        timing,sums,means = index_once(env,mode)
                        assert torch.equal(sums.view(torch.int32),env.expected.view(torch.int32))
                        assert torch.equal(means.view(torch.int16),env.mean.view(torch.int16))
                    else:
                        timing,usage = append_once(env,mode)
                        archive_usage[mode] = usage
                    if trial >= args.warmup:
                        samples[scope][mode].append(timing)
        sizes = env.geometry.sizes(tokens)
        capacity = env.geometry.sizes(env.geometry.pages*32)
        cases.append(dict(tokens=tokens,geometry=asdict(env.geometry),
            target_names=list(env.names[:env.geometry.scoring_layers]),mtp_excluded=list(env.names[env.geometry.scoring_layers:]),
            payload_bytes=env.payload.numel(),active_sidecar=sizes,one_slot_capacity=capacity,
            hypothetical_two_slot_gpu_plus_pinned_bytes=2*(capacity['gpu_bytes']+capacity['pinned_bytes']),
            cold_first_submit_seconds=cold_seconds,exact_sums_means_digests=True,
            actual_archive_usage=archive_usage,samples=samples,
            summary={scope:{mode:summary(rows) for mode,rows in arms.items()}
                     for scope,arms in samples.items()}))
        del env
        gc.collect(); torch.cuda.empty_cache()
    assert fingerprints == {str(path):sha(path) for path in sources}, 'Sources changed during component'
    output = dict(diagnostic_only=True,source_sha256=fingerprints,torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),torch_cpu_threads=torch.get_num_threads(),
        warmup_rounds=args.warmup,measured_rounds=args.rounds,
        scope=__doc__,production_changed=False,cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(output,stream,indent=2); stream.write('\n')
    print(json.dumps(dict(output=str(args.output),cases=[dict(tokens=c['tokens'],summary=c['summary'])
                                                       for c in cases]),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,nargs='+',default=[32,1984,2048,14336])
    parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
