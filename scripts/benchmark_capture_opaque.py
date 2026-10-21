"""Compiled Q/K/V capture component, including Python and opaque dispatch.

Seventeen independent projection owners model 16 target attention layers and
one gated MTP prefill layer. This is not a full-model benchmark: GEMM, attention,
MTP decode, scoring, archival transfer and publication are absent. Compiled
projection segments are called separately, as attention separates them in the
model; no cross-layer kernel fusion is introduced by this fixture.

Prepare freezes source without importing torch. Run only with externally owned
GPU and --run-gpu. CUDA profiler samples are separate from ABBA wall timings.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import FunctionType, SimpleNamespace as NS

CAPACITY = 16424
LAYERS = 17
HEADS, KV_HEADS, DIM, ROTARY = 24, 4, 256, 64
MROPE = (11, 11, 10)
NORM_KERNEL = '_fused_qk_rmsnorm_rope_gate_kernel'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def abba_pairs(pairs):
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 12 or pairs % 2:
        raise ValueError('Use an even number of at least 12 matched AB/BA pairs')
    return tuple(('reference', 'opaque') if i % 2 == 0 else ('opaque', 'reference')
                 for i in range(pairs))


def validate_tokens(tokens):
    values = tuple(tokens)
    if values != (2048, 14336):
        raise ValueError('Qualified component shapes are exactly Q2048 and Q14336')
    return values


def prepare(directory, source_root):
    if directory.exists():
        raise ValueError('Prepare requires a new experiment directory')
    source_root = source_root.resolve()
    package = source_root/'src/vllm_kvmem'
    directory.mkdir(parents=True)
    shutil.copytree(package, directory/'runtime/vllm_kvmem',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), directory/'benchmark_capture_opaque.py')
    files = sorted(p for p in (directory/'runtime').rglob('*') if p.is_file())
    files.append(directory/'benchmark_capture_opaque.py')
    manifest = dict(schema=1, source_root=str(source_root),
        files={str(p.relative_to(directory)):sha256(p) for p in files},
        scope=__doc__, cases=[2048,14336],capacity=CAPACITY,layers=LAYERS,
        preparation_initializes_cuda=False)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_sources(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = directory/relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Frozen source changed or missing: '+relative)
    if Path(__file__).resolve() != (directory/'benchmark_capture_opaque.py').resolve():
        raise ValueError('Execute the frozen benchmark script from the prepared directory')
    return manifest


class Owner:
    pass


def compiled_projection(layer, label):
    import torch
    from vllm_kvmem.capture import _project_with_capture
    def forward(qkv, positions):
        return _project_with_capture(layer, qkv, positions)
    # Distinct code objects keep one compiled segment per Attention owner. Do
    # not raise Dynamo's global recompile limit or let a hot path fall eager.
    original = forward
    forward = FunctionType(original.__code__.replace(co_name=label, co_qualname=label),
        original.__globals__, label, original.__defaults__, original.__closure__)
    return torch.compile(forward, fullgraph=True, dynamic=True, options={
        'enable_auto_functionalized_v2':False, 'combo_kernels':True,
        'benchmark_combo_kernel':True})


def fixture():
    import torch
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm_kvmem.capture import CapturedLayer
    from vllm_kvmem.capture_compile import capture_compile_fingerprint
    from vllm_kvmem.capture_opaque import install_opaque_capture
    generator = torch.Generator(device='cuda').manual_seed(7729)
    def random(*shape):
        return torch.randn(*shape,device='cuda',dtype=torch.bfloat16,generator=generator)
    # Same MRoPE frequency convention and partial rotation as the deployed model.
    inv = 1.0 / (10000000.0 ** (torch.arange(0, ROTARY, 2, device='cuda').float()/ROTARY))
    angles = torch.arange(65536,device='cuda').float()[:,None]*inv[None,:]
    cache = torch.cat((angles.cos(),angles.sin()),-1)
    positions_base = torch.stack((torch.arange(14336,device='cuda')+8192,
        torch.arange(14336,device='cuda')+8199,torch.arange(14336,device='cuda')+8211))
    width = (2*HEADS+2*KV_HEADS)*DIM
    qkv = [random(14336,width) for _ in range(LAYERS)]
    weights = [(random(DIM),random(DIM)) for _ in range(LAYERS)]
    class Layer(torch.nn.Module):
        pass
    arms={}
    for mode in ('reference','opaque'):
        layers=[]
        for index in range(LAYERS):
            layer=Layer();layer.attn=Layer()
            layer.attn.layer_name=f'capture_component.{mode}.{index}.attn'
            layer.num_heads,layer.num_kv_heads,layer.head_dim=HEADS,KV_HEADS,DIM
            layer.q_size,layer.kv_size=HEADS*DIM,KV_HEADS*DIM
            layer.q_norm=NS(weight=weights[index][0],variance_epsilon=1e-6)
            layer.k_norm=NS(weight=weights[index][1])
            layer.rotary_emb=NS(rotary_dim=ROTARY,cos_sin_cache=cache,mrope_section=MROPE)
            layer._kvmem_capture_name=layer.attn.layer_name
            layer._kvmem_capture_buffers=CapturedLayer(layer.attn.layer_name,positions_base,
                torch.full((CAPACITY,HEADS,DIM),-7,device='cuda',dtype=torch.bfloat16),
                torch.full((CAPACITY,KV_HEADS,DIM),-7,device='cuda',dtype=torch.bfloat16),
                torch.full((CAPACITY,KV_HEADS,DIM),-7,device='cuda',dtype=torch.bfloat16))
            if index==LAYERS-1:
                layer._kvmem_capture_gate=torch.ones((),device='cuda',dtype=torch.int32)
                layer._kvmem_mtp_prefix_only=True
            layers.append(layer)
        static={layer.attn.layer_name:layer.attn for layer in layers}
        runner=NS(model=NS(named_modules=lambda layers=layers:
            [(str(i),layer) for i,layer in enumerate(layers[:-1])]),
            _kvmem_layers=[str(i) for i in range(LAYERS-1)],
            _kvmem_mtp_modules={'mtp':layers[-1]},
            vllm_config=NS(compilation_config=NS(static_forward_context=static)))
        stats=install_opaque_capture(runner,capture_compile_fingerprint()) if mode=='opaque' else None
        context=ForwardContext(no_compile_layers=static,attn_metadata={},slot_mapping={})
        calls=[compiled_projection(layer,f'{mode}_projection_{i}') for i,layer in enumerate(layers)]
        arms[mode]=NS(layers=layers,runner=runner,stats=stats,context=context,calls=calls)
    def invoke(mode,tokens):
        arm=arms[mode]
        with override_forward_context(arm.context):
            result=None
            for i,call in enumerate(arm.calls):
                result=call(qkv[i][:tokens],positions_base[:,:tokens])
            return result
    return arms,qkv,positions_base,invoke


def raw(layer):
    buffers=layer._kvmem_capture_buffers
    return buffers.query,buffers.key,buffers.value


def bit_equal(left,right):
    import torch
    return torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))


def correctness(arms,qkv,positions,tokens):
    import torch
    from vllm.forward_context import override_forward_context
    checked=[]
    for i in range(LAYERS):
        outputs={}
        for mode in ('reference','opaque'):
            arm=arms[mode]
            for value in raw(arm.layers[i]): value.fill_(-7)
            with override_forward_context(arm.context):
                outputs[mode]=arm.calls[i](qkv[i][:tokens],positions[:,:tokens])
        for label,a,b in zip(('q','k','v','gate'),outputs['reference'],outputs['opaque'],strict=True):
            if not bit_equal(a,b): raise RuntimeError(f'{i}/{label}: compiled output mismatch')
        for label,a,b in zip(('raw_q','raw_k','raw_v'),raw(arms['reference'].layers[i]),
                             raw(arms['opaque'].layers[i]),strict=True):
            if not bit_equal(a,b): raise RuntimeError(f'{i}/{label}: captured buffer mismatch')
            if not bool((b[tokens:] == -7).all().item()):
                raise RuntimeError(f'{i}/{label}: unwritten tail changed')
        if not bit_equal(raw(arms['opaque'].layers[i])[2][:tokens],
                         qkv[i][:tokens,-KV_HEADS*DIM:].reshape(tokens,KV_HEADS,DIM)):
            raise RuntimeError('Raw V differs from the noncontiguous QKV slice')
        checked.append(i)
        del outputs
    # MTP gate changes are device values, not a new compile/owner installation.
    i=LAYERS-1
    for mode in ('reference','opaque'):
        arm=arms[mode];layer=arm.layers[i]
        snapshots=[value.clone() for value in raw(layer)]
        layer._kvmem_capture_gate.zero_()
        with override_forward_context(arm.context):
            outputs=arm.calls[i](qkv[i][:tokens]*.5,positions[:,:tokens]+1)
        if not all(bit_equal(a,b) for a,b in zip(snapshots,raw(layer),strict=True)):
            raise RuntimeError('Disabled MTP gate overwrote capture buffers')
        del outputs,snapshots
        layer._kvmem_capture_gate.fill_(1)
    torch.cuda.synchronize()
    return dict(bitwise_outputs_and_full_raw_buffers=True,checked_owners=checked,
                disabled_mtp_gate_preserved_buffers=True,untouched_capacity_tail=True)


def compiled_copy_proofs(cache):
    """Prove all three buffer copies, allowing Inductor combo partition changes."""
    proofs=[]
    run=re.compile(r'\s*(triton_\w+)\.run\((.*)\)')
    tensor=re.compile(r'(?:arg\d+_\d+|buf\d+)')
    def call(line):
        match=run.fullmatch(line)
        if not match:return None
        args=[]
        for value in match[2].split(','):
            value=value.strip()
            if not tensor.fullmatch(value):break
            args.append(value)
        if len(args)<2 or len(args)%2:return None
        n=len(args)//2
        return dict(kernel=match[1],pairs=list(zip(args[:n],args[n:],strict=True)))
    for path in sorted(cache.rglob('*.py')):
        lines=path.read_text().splitlines()
        for i,line in enumerate(lines):
            if not line.lstrip().startswith(NORM_KERNEL+'_0.run('):continue
            before=[];after=[]
            for j in range(max(0,i-50),i):
                info=call(lines[j])
                if info and 'aten.clone' in '\n'.join(lines[max(0,j-10):j]):
                    before.append(dict(line=j+1,**info))
            for j in range(i+1,min(len(lines),i+35)):
                if lines[j].lstrip().startswith('return '):break
                info=call(lines[j])
                if info:after.append(dict(line=j+1,**info))
            clones=[pair for item in before for pair in item['pairs']]
            backs=[pair for item in after for pair in item['pairs']]
            if (len(clones)!=3 or len(set(clones))!=3 or
                    sorted(backs)!=sorted((b,a) for a,b in clones) or
                    not all(a.startswith('arg') and b.startswith('buf') and b in line
                            for a,b in clones)):continue
            proofs.append(dict(path=str(path),sha256=sha256(path),norm_line=i+1,
                clone_calls=before,copyback_calls=after,
                kernel_order=[x['kernel'] for x in before]+[NORM_KERNEL]+[x['kernel'] for x in after]))
    return proofs


def audit_kernel_sequence(names, proofs, owners):
    """Match entire per-owner launch sequences, rather than guessing copy names."""
    sequences=sorted({tuple(p['kernel_order']) for p in proofs},key=len,reverse=True)
    candidates={(0,0):[]}
    for owner in range(owners):
        following={}
        for (offset,count),selected in candidates.items():
            for sequence in sequences:
                stop=offset+len(sequence)
                if tuple(names[offset:stop])==sequence:
                    following[(stop,count+1)]=selected+[list(sequence)]
        candidates=following
    matched=candidates.get((len(names),owners))
    if matched is None:
        raise RuntimeError('Reference clone/copyback source does not cover actual owner kernel sequences')
    return matched


def profile_copies(invoke,mode,tokens,path):
    import torch
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        output=invoke(mode,tokens)
        torch.cuda.synchronize()
    del output
    profiler.export_chrome_trace(str(path))
    events=sorted((event for event in profiler.events() if str(event.device_type).endswith('CUDA')),
                  key=lambda event:event.time_range.start)
    norms=[event for event in events if NORM_KERNEL in event.name]
    if len(norms)!=LAYERS:
        raise RuntimeError(f'{mode}: expected17 actual projection kernels, got{len(norms)}')
    non_norm=[event for event in events if NORM_KERNEL not in event.name]
    if mode=='opaque' and non_norm:
        raise RuntimeError('Opaque compiled projection still launches other GPU operations: '+
                           str(Counter(event.name for event in non_norm)))
    copy_proofs=compiled_copy_proofs(path.parent/'inductor_cache') if mode=='reference' else []
    matched_sequences=audit_kernel_sequence([event.name for event in events],copy_proofs,LAYERS) if mode=='reference' else []
    # Keep names/durations and the complete trace: names alone are insufficient
    # to prove whether a fused generic kernel is a clone or its copyback.
    return dict(trace=str(path),trace_sha256=sha256(path),compiled_copy_proofs=copy_proofs,
        matched_owner_kernel_sequences=matched_sequences,
        actual_norm_kernels=len(norms),non_norm_kernels=len(non_norm),
        norm_kernel_sum_ms=sum(e.device_time_total for e in norms)/1000,
        non_norm_kernel_sum_ms=sum(e.device_time_total for e in non_norm)/1000,
        kernels=[dict(name=e.name,duration_us=e.device_time_total) for e in events])


def measure(invoke,mode,tokens,begin,end):
    import torch
    torch.cuda.synchronize()
    base=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();begin.record()
    output=invoke(mode,tokens)
    submitted=time.perf_counter();end.record();end.synchronize()
    finished=time.perf_counter()
    result=dict(wall_ms=(finished-started)*1000,host_submit_ms=(submitted-started)*1000,
        cuda_interval_ms=begin.elapsed_time(end),base_allocated_bytes=base,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()-base)
    del output
    return result


def run(directory, *, pairs, warmup, profile_repeats):
    manifest=verify_sources(directory)
    abba=abba_pairs(pairs)
    if warmup<3 or profile_repeats<1:
        raise ValueError('Use at least3 warmups and one separate profiler sample')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(directory/'inductor_cache')
    os.environ['TRITON_CACHE_DIR']=str(directory/'triton_cache')
    sys.path.insert(0,str(directory/'runtime'))
    import torch
    from vllm_kvmem.capture_compile import capture_compile_fingerprint
    from vllm_kvmem import capture_opaque
    if not str(Path(capture_opaque.__file__).resolve()).startswith(str((directory/'runtime').resolve())+'/'):
        raise ValueError('Imported live capture code instead of frozen source')
    torch.set_grad_enabled(False)
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    result=dict(scope=__doc__,manifest_sha256=sha256(directory/'manifest.json'),
        source_sha256=manifest['files'],capture_compile_fingerprint=capture_compile_fingerprint(),
        torch_version=torch.__version__,triton_version=importlib.metadata.version('triton'),
        vllm_version=importlib.metadata.version('vllm'),gpu=torch.cuda.get_device_name(),
        compile_options=dict(fullgraph=True,dynamic=True,enable_auto_functionalized_v2=False,
                             combo_kernels=True,benchmark_combo_kernel=True),
        capacities=dict(tokens=CAPACITY,owners=LAYERS,q_heads=HEADS,kv_heads=KV_HEADS,
            head_dim=DIM,rotary_dim=ROTARY,mrope_section=MROPE,cos_sin_rows=65536),
        independent_qkv_per_owner=True,independent_capture_buffers_per_arm=True,
        arm_inputs_and_weights_identical=True,compiled_segments_per_arm=LAYERS,
        MTP_prefill_layers=1,target_attention_layers=16,cases=[])
    with torch.inference_mode():
        arms,qkv,positions,invoke=fixture()
        for tokens in validate_tokens((2048,14336)):
            started=time.perf_counter()
            correct=correctness(arms,qkv,positions,tokens)
            cold=time.perf_counter()-started
            for _ in range(warmup):
                for mode in ('reference','opaque'): output=invoke(mode,tokens);del output
            torch.cuda.synchronize()
            samples=[];begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            for pair,order in enumerate(abba):
                for mode in order:
                    samples.append(dict(pair=pair,mode=mode,**measure(invoke,mode,tokens,begin,end)))
            summary={mode:{metric:dict(mean=statistics.mean(values),median=statistics.median(values),
                        minimum=min(values),maximum=max(values))
                    for metric in ('wall_ms','host_submit_ms','cuda_interval_ms','peak_incremental_allocated_bytes')
                    for values in [[sample[metric] for sample in samples if sample['mode']==mode]]}
                for mode in ('reference','opaque')}
            case=dict(tokens=tokens,correctness=correct,
                cold_compile_and_correctness_seconds=cold,warmup_per_arm=warmup,
                matched_pairs=pairs,samples=samples,summary=summary,complete=False)
            result['cases'].append(case)
            (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            profiles={mode:[profile_copies(invoke,mode,tokens,
                       directory/f'profile-{tokens}-{mode}-{i}.json') for i in range(profile_repeats)]
                for mode in ('reference','opaque')}
            case.update(separate_kernel_profiles=profiles,
                actual_opaque_stats=arms['opaque'].stats.snapshot(),complete=True)
            (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(tokens=tokens,summary=summary)),flush=True)
    wrappers=sorted(p for p in (directory/'inductor_cache').rglob('*.py') if p.is_file())
    result['compiled_wrapper_sha256']={str(p.relative_to(directory)):sha256(p) for p in wrappers}
    verify_sources(directory)
    result['complete']=True
    (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    p=sub.add_parser('run');p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--pairs',type=int,default=12)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--profile-repeats',type=int,default=1)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.directory,args.source_root)
    else:
        if not args.run_gpu:parser.error('Run requires --run-gpu and external exclusive GPU ownership')
        if (args.directory/'results.json').exists():raise ValueError('Use a fresh prepared experiment')
        run(args.directory,pairs=args.pairs,warmup=args.warmup,profile_repeats=args.profile_repeats)


if __name__=='__main__':
    main()
