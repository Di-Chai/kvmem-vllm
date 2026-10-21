"""Read-only recovery of r3 copy auditing, with optional separate new-arm profile.

The original results.json and failed exit/log are never rewritten. --profile-new
uses that experiment's frozen fixture and source, never current production code.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import benchmark_capture_opaque as audit


def frozen_module(root):
    path=root/'benchmark_capture_opaque.py'
    spec=importlib.util.spec_from_file_location('frozen_opaque_component',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_sources(root)
    return module


def profile_new(root):
    frozen=frozen_module(root)
    target=root/'post-profile-14336-opaque.json'
    if target.exists():raise ValueError('Separate profile already exists')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor_cache')
    os.environ['TRITON_CACHE_DIR']=str(root/'triton_cache')
    sys.path.insert(0,str(root/'runtime'))
    import torch
    from vllm_kvmem import capture_opaque
    if not Path(capture_opaque.__file__).resolve().is_relative_to((root/'runtime').resolve()):
        raise ValueError('Frozen production runtime is not active')
    torch.set_grad_enabled(False)
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    with torch.inference_mode():
        arms,qkv,positions,invoke=frozen.fixture()
        equality=frozen.correctness(arms,qkv,positions,14336)
        for _ in range(3):output=invoke('opaque',14336);del output
        torch.cuda.synchronize()
        profile=frozen.profile_copies(invoke,'opaque',14336,target)
        # Post-run hashes are evidence for reconstructed inputs only. The
        # original run did not store byte hashes, so identity to that run rests
        # on the exact frozen deterministic fixture, seed and environment.
        hashes=[]
        for value in qkv:
            host=value.view(torch.uint8).cpu().numpy()
            hashes.append(hashlib.sha256(memoryview(host)).hexdigest())
            del host
        pos=positions.contiguous().view(torch.uint8).cpu().numpy()
        detail=dict(correctness=equality,profile=profile,
            reconstructed_qkv_sha256=hashes,
            reconstructed_positions_sha256=hashlib.sha256(memoryview(pos)).hexdigest(),
            original_input_byte_hashes_available=False,
            reconstruction_note='Same frozen fixture with CUDA generator seed7729, same torch version; original input identity is not retrospectively byte-proven.',
            stats=arms['opaque'].stats.snapshot(),
            frozen_script_sha256=audit.sha256(root/'benchmark_capture_opaque.py'),
            controller_sha256=audit.sha256(__file__))
        (root/'post-profile-details.json').write_text(json.dumps(detail,indent=2)+'\n')
    frozen.verify_sources(root)


def trace_kernels(path):
    events=json.loads(path.read_text())['traceEvents']
    return sorted((e for e in events if e.get('cat')=='kernel'),key=lambda e:e['ts'])


def analyze(root):
    frozen=frozen_module(root)
    before=audit.sha256(root/'results.json')
    result=json.loads((root/'results.json').read_text())
    proofs=audit.compiled_copy_proofs(root/'inductor_cache')
    cases=[]
    for case in result['cases']:
        tokens=case['tokens'];profiles={}
        for mode in ('reference','opaque'):
            path=root/f'profile-{tokens}-{mode}-0.json'
            if not path.exists() and mode=='opaque' and tokens==14336:
                path=root/'post-profile-14336-opaque.json'
            if not path.exists():
                profiles[mode]=None
                continue
            kernels=trace_kernels(path);names=[e['name'] for e in kernels]
            norms=[e for e in kernels if audit.NORM_KERNEL in e['name']]
            if len(norms)!=17:raise ValueError('Incorrect real norm kernel count')
            if mode=='reference':
                sequences=audit.audit_kernel_sequence(names,proofs,17)
            else:
                if len(kernels)!=17:raise ValueError('Opaque profile has extra GPU kernels')
                sequences=[[name] for name in names]
            profiles[mode]=dict(path=str(path),sha256=audit.sha256(path),kernel_count=len(kernels),
                norm_kernels=len(norms),copy_kernels=len(kernels)-len(norms),
                kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
                norm_sum_ms=sum(e['dur'] for e in norms)/1000,
                copy_sum_ms=sum(e['dur'] for e in kernels if audit.NORM_KERNEL not in e['name'])/1000,
                audited_owner_sequences=sequences)
        samples=case['samples']
        if case['matched_pairs']!=12 or len(samples)!=24:
            raise ValueError('Incomplete paired timing')
        for pair,order in enumerate(audit.abba_pairs(12)):
            actual=tuple(r['mode'] for r in samples if r['pair']==pair)
            if actual!=order:raise ValueError('ABBA order changed')
        required=['bitwise_outputs_and_full_raw_buffers','disabled_mtp_gate_preserved_buffers','untouched_capacity_tail']
        if not all(case['correctness'].get(k) is True for k in required):
            raise ValueError('Original numeric qualification incomplete')
        summary=case['summary']
        a,b=summary['reference'],summary['opaque']
        cases.append(dict(tokens=tokens,original_case_complete_flag=case['complete'],
            original_timing_and_equality_complete=True,copy_audit_complete=all(profiles.values()),
            cold_compile_and_correctness_seconds=case['cold_compile_and_correctness_seconds'],
            summary=summary,profiles=profiles,
            wall_delta_ms=b['wall_ms']['mean']-a['wall_ms']['mean'],
            wall_change_percent=(b['wall_ms']['mean']/a['wall_ms']['mean']-1)*100,
            cuda_interval_change_percent=(b['cuda_interval_ms']['mean']/a['cuda_interval_ms']['mean']-1)*100))
    if before!=audit.sha256(root/'results.json'):raise ValueError('Original result was modified')
    inputs=dict(generator_device='cuda',generator_seed=7729,qkv_owners=17,
        qkv_shape_per_owner=[14336,14336],qkv_dtype='bfloat16',
        weights_generated_after_all_qkv=True,mrope_offsets=[8192,8199,8211],
        frozen_script_sha256=audit.sha256(root/'benchmark_capture_opaque.py'))
    return dict(original_result_sha256=before,original_completed_successfully=False,
        original_failure='Second-shape profiler audit assumed fixed combo partition; numerical qualification and all12pairs had already completed and were saved.',
        cases=cases,generated_wrapper_proofs=proofs,
        input_recipe=inputs,input_recipe_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        separate_new14k_profile=(json.loads((root/'post-profile-details.json').read_text())
            if (root/'post-profile-details.json').exists() else None),
        all_required_evidence_complete=all(c['copy_audit_complete'] for c in cases),
        source_sha256={str(p):audit.sha256(p) for p in [Path(__file__),Path(audit.__file__),
            root/'manifest.json',root/'benchmark_capture_opaque.py',root/'results.json']},
        limits=['17 projection segments, not full model; no attention/GEMM/archive/scoring included.',
            'CUDA event interval includes GPU idle due to host dispatch; kernel sums are separate profiled samples.',
            'Component clone combo partitions vary:2K has51 total kernels;14K has83. Full-model trace had6 per target projection; do not equate kernel counts.',
            'MTP prefilling only; tiny decode/graph behavior is covered by separate qualification.',
            'Full-model intervening kernels may hide opaque dispatch time;2K component regression is not a full-model speed prediction.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-new',action='store_true');p.add_argument('--run-gpu',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new post-audit output path')
    if a.profile_new:
        if not a.run_gpu:p.error('--profile-new requires --run-gpu and external GPU ownership')
        profile_new(a.root)
    a.output.write_text(json.dumps(analyze(a.root),indent=2)+'\n')
