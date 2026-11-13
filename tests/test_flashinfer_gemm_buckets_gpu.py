"""Real FP8/NVFP4 GEMMs across the 4096 -> 8192 runtime bucket boundary.

No forced output tokens: compare actual matrix values against independently
decoded FP32 operands, retain the old 4136 tactic, and prove 8192 was profiled.
Only explicit temporary autotune files are loaded/saved. Run after the model
benchmark has stopped, with KVMEM_RUN_GPU_TESTS=1.
"""
import ast
import json
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')


def _config_rows(key):
    # FlashInfer file keys are repr((custom_op, runner, nearest_profile, ...)).
    # Extras contain torch.dtype attribute expressions, not Python literals;
    # parse only the literal operator/profile nodes, never evaluate extras.
    expression = ast.parse(key,mode='eval').body
    operator = ast.literal_eval(expression.elts[0])
    shape = ast.literal_eval(expression.elts[2])[0]
    return shape[-2] if operator=='fp8_gemm' else shape[0]


def _tactics(path):
    return {key:value for key,value in json.loads(path.read_text()).items()
            if not key.startswith('_')}


def _matrices(kind):
    import torch
    from flashinfer.gemm import gemm_base
    m,k,n = 4136,512,1024
    generator = torch.Generator(device='cuda').manual_seed(79241)
    if kind.startswith('fp8'):
        # Exactly representable binary fractions avoid conflating input
        # quantization error with incorrect GEMM/tactic shape application.
        a = (torch.randint(-16,17,(1,m,k),generator=generator,device='cuda').float()/8).to(torch.float8_e4m3fn)
        b_nt = (torch.randint(-16,17,(1,n,k),generator=generator,device='cuda').float()/8).to(torch.float8_e4m3fn)
        a_scale = torch.tensor(.5,device='cuda')
        b_scale = torch.tensor(.25,device='cuda')
        reference = (a.float() @ b_nt.float().transpose(-1,-2) * .125).bfloat16()
        backend = kind.split('_',1)[1]
        def invoke(rows):
            return gemm_base.bmm_fp8(a[:,:rows],b_nt.transpose(-1,-2),a_scale,b_scale,
                                    torch.bfloat16,backend=backend)
        return invoke,lambda rows:reference[:,:rows],(m,k,n)

    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    # Independent IEEE E2M1 value table. Generate packed nibbles and varying
    # per-16-element E4M3 scales directly instead of reusing a GEMM dequantizer.
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
    scale_values = torch.tensor([.25,.5,.75,1,1.5,2],device='cuda')
    def packed(rows):
        codes = torch.randint(0,16,(rows,k),generator=generator,device='cuda',dtype=torch.uint8)
        values = lut[codes.long()]
        sf = scale_values[torch.randint(0,len(scale_values),(rows,k//16),generator=generator,device='cuda')]
        scale = sf.to(torch.float8_e4m3fn)
        byte = codes[:,0::2] | (codes[:,1::2] << 4)
        # This is a layout transform only; expected values use the original
        # unswizzled scale matrix and independently unpacked nibble values.
        swizzled = block_scale_interleave(scale.view(torch.uint8)).reshape((rows+127)//128*128,k//16)
        return byte,swizzled,values*scale.float().repeat_interleave(16,dim=-1)
    a,a_sf,a_float = packed(m)
    b,b_sf,b_float = packed(n)
    alpha = torch.tensor(.125,device='cuda')
    reference = (a_float @ b_float.T * .125).bfloat16()
    def invoke(rows):
        return gemm_base.mm_fp4(a[:rows],b.T,a_sf,b_sf.T,alpha,
                               out_dtype=torch.bfloat16,backend='cutlass')
    return invoke,lambda rows:reference[:rows],(m,k,n)


@pytest.mark.parametrize('kind',['fp8_cutlass','fp8_cublas','nvfp4_cutlass','fp8_cudnn'])
def test_profiled_boundary_tactic_matches_independent_values_and_runtime_cache(kind,tmp_path,monkeypatch,record_property):
    import torch
    from flashinfer.autotuner import AutoTuner,autotune
    from flashinfer.gemm import gemm_base
    from vllm_kvmem.serving.flashinfer_gemm_buckets import _CONFIG_NAMES,_align_gemm_overrides

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('Qualified deployment is Blackwell SM12x')
    if kind=='fp8_cudnn' and not gemm_base._is_cudnn_override_shape_available():
        pytest.skip('cuDNN FP8 auto is excluded on SM12x without backend >=9.23.1 override-shape support')
    monkeypatch.setenv('FLASHINFER_AUTOTUNER_LOAD_FROM_FILE','0')
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        invoke,reference,geometry = _matrices(kind)
        baseline = AutoTuner()
        monkeypatch.setattr(AutoTuner,'_instance',baseline)
        with autotune(tune_mode=True,tuning_buckets=(4136,)):
            invoke(4118)
        torch.cuda.synchronize()
        seed_path = tmp_path/'baseline-4136.json'
        baseline.save_configs(str(seed_path))
        old_tactics = _tactics(seed_path)
        assert old_tactics and {_config_rows(key) for key in old_tactics}=={4136}
        baseline_hits = []
        baseline_search = baseline.search_cache
        def lookup_baseline(*args,**kwargs):
            result = baseline_search(*args,**kwargs)
            baseline_hits.append(bool(result[0]))
            return result
        monkeypatch.setattr(baseline,'search_cache',lookup_baseline)
        before = {rows:invoke(rows).clone() for rows in (4113,4118,4136)}
        assert len(baseline_hits)==3 and not any(baseline_hits)

        candidate = AutoTuner()
        monkeypatch.setattr(AutoTuner,'_instance',candidate)
        assert candidate.load_configs(str(seed_path))
        measured_shapes = []
        original_profile = candidate._profile_single_kernel
        def profile(runner,inputs,tactic,config,*args,**kwargs):
            measured_shapes.append(tuple(inputs[0].shape))
            return original_profile(runner,inputs,tactic,config,*args,**kwargs)
        monkeypatch.setattr(candidate,'_profile_single_kernel',profile)
        state = SimpleNamespace(alignments=[])
        configs = {name:getattr(gemm_base,name) for name in _CONFIG_NAMES}
        with autotune(tune_mode=True,tuning_buckets=(4136,)):
            with _align_gemm_overrides(candidate,configs,autotune,state):
                invoke(4118)
        torch.cuda.synchronize()
        assert measured_shapes and {shape[-2] if kind.startswith('fp8') else shape[0]
                                    for shape in measured_shapes}=={8192}
        assert state.alignments and all(row['aligned_buckets']==[8192] for row in state.alignments)
        assert '_apply_tuning_overrides' not in vars(candidate)

        tuned_path = tmp_path/'aligned-8192.json'
        candidate.save_configs(str(tuned_path))
        new_tactics = _tactics(tuned_path)
        assert all(new_tactics.get(key)==value for key,value in old_tactics.items())
        added = set(new_tactics)-set(old_tactics)
        assert added and {_config_rows(key) for key in added}=={8192}

        # A fresh singleton proves the saved file is usable, independent of
        # the live profiling cache and temporary warmup mapper hook.
        restored = AutoTuner()
        monkeypatch.setattr(AutoTuner,'_instance',restored)
        assert restored.load_configs(str(tuned_path))
        hits = []
        search = restored.search_cache
        def lookup(*args,**kwargs):
            result = search(*args,**kwargs)
            hits.append(bool(result[0]))
            return result
        monkeypatch.setattr(restored,'search_cache',lookup)
        errors = {}
        for rows in (4113,4118,4136):
            after = invoke(rows)
            expected = reference(rows)
            # BF16 relative tolerance is about one output ULP; a small absolute
            # tolerance covers cancellation near zero. Neither permits a
            # wrong-row/scale/tactic-shape result to pass.
            torch.testing.assert_close(before[rows],expected,rtol=.008,atol=.015625)
            torch.testing.assert_close(after,expected,rtol=.008,atol=.015625)
            torch.testing.assert_close(after,before[rows],rtol=.008,atol=.015625)
            errors[rows] = float((after.float()-expected.float()).abs().max())
        torch.cuda.synchronize()
        assert len(hits)==3 and all(hits)
        evidence = dict(kind=kind,geometry=geometry,real_rows=[4113,4118,4136],
            profiled_rows=8192,profile_measurements=len(measured_shapes),
            seed_tactics=len(old_tactics),new_tactics=len(added),old_tactics_unchanged=True,
            baseline_runtime_cache_hits=baseline_hits,runtime_file_cache_hits=hits,maximum_absolute_errors=errors,
            independent_reference='FP32 matmul of independently decoded quantized values; BF16 output')
        (tmp_path/'qualification.json').write_text(json.dumps(evidence,indent=2)+'\n')
        record_property('qualification',json.dumps(evidence))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
