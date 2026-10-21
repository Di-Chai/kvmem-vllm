"""Pure host checks for freezing and unbiased opaque-capture measurement order."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/benchmark_capture_opaque.py'
spec = importlib.util.spec_from_file_location('capture_opaque_benchmark',PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_abba_order_pairs_have_balanced_counts_and_counterbalance():
    pairs=bench.abba_pairs(12)
    assert len(pairs)==12
    assert pairs[:2]==(('reference','opaque'),('opaque','reference'))
    assert tuple(mode for pair in pairs for mode in pair)==('reference','opaque','opaque','reference')*6
    assert bench.validate_tokens([2048,14336])==(2048,14336)


@pytest.mark.parametrize('invalid',[0,10,13,True])
def test_insufficient_or_unbalanced_pairs_are_rejected(invalid):
    with pytest.raises(ValueError,match='12 matched'):
        bench.abba_pairs(invalid)


def test_freeze_isolated_from_live_edits_and_rejects_tampering(tmp_path,monkeypatch):
    source=tmp_path/'source'
    package=source/'src/vllm_kvmem'
    package.mkdir(parents=True)
    (package/'capture.py').write_text('original source')
    (package/'__pycache__').mkdir()
    (package/'__pycache__/stale.pyc').write_bytes(b'stale')
    directory=tmp_path/'run'
    manifest=bench.prepare(directory,source)
    assert manifest['preparation_initializes_cuda'] is False
    assert not (directory/'runtime/vllm_kvmem/__pycache__').exists()
    with pytest.raises(ValueError,match='Execute the frozen'):
        bench.verify_sources(directory)
    monkeypatch.setattr(bench,'__file__',str(directory/'benchmark_capture_opaque.py'))
    (package/'capture.py').write_text('new live source')
    assert bench.verify_sources(directory)==manifest
    frozen=directory/'runtime/vllm_kvmem/capture.py'
    frozen.write_text('changed frozen source')
    with pytest.raises(ValueError,match='changed or missing'):
        bench.verify_sources(directory)
    with pytest.raises(ValueError,match='new experiment'):
        bench.prepare(directory,source)


def test_import_and_prepare_do_not_import_torch_or_initialize_cuda(tmp_path):
    source=tmp_path/'source';(source/'src/vllm_kvmem').mkdir(parents=True)
    (source/'src/vllm_kvmem/__init__.py').write_text('')
    code='''import runpy,sys
from pathlib import Path
module=runpy.run_path(sys.argv[1],run_name='bench_no_main')
assert 'torch' not in sys.modules
module['prepare'](Path(sys.argv[2]),Path(sys.argv[3]))
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(PATH),str(tmp_path/'frozen'),str(source)],
                          capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_gpu_run_requires_explicit_opt_in_before_import(tmp_path):
    result=subprocess.run([sys.executable,'-B',str(PATH),'run','--directory',str(tmp_path)],
                          capture_output=True,text=True)
    assert result.returncode==2
    assert 'external exclusive GPU ownership' in result.stderr


def test_combo_copy_audit_follows_actual_reverse_arguments_not_kernel_names(tmp_path):
    path=tmp_path/'wrapper.py'
    path.write_text('''    # Original ATen: [aten.clone]
    triton_poi_fused_0.run(arg9_1, arg10_1, arg7_1, buf3, buf4, buf5, count, stream=stream)
    _fused_qk_rmsnorm_rope_gate_kernel_0.run(qkv, buf3, buf4, buf5)
    triton_poi_fused_1.run(buf3, buf4, buf5, arg9_1, arg10_1, arg7_1, count, stream=stream)
''')
    proof=bench.compiled_copy_proofs(tmp_path)
    assert len(proof)==1
    assert proof[0]['clone_calls'][0]['kernel']=='triton_poi_fused_0'
    assert proof[0]['copyback_calls'][0]['kernel']=='triton_poi_fused_1'
    path.write_text(path.read_text().replace('buf3, buf4, buf5, arg9_1','buf3, buf4, buf5, wrong'))
    assert bench.compiled_copy_proofs(tmp_path)==[]


def test_sequence_audit_allows_three_or_five_kernels_but_requires_every_owner():
    norm=bench.NORM_KERNEL
    proofs=[dict(kernel_order=['clone_all',norm,'back_all']),
            dict(kernel_order=['clone_q','clone_kv',norm,'back_q','back_kv'])]
    actual=['clone_q','clone_kv',norm,'back_q','back_kv','clone_all',norm,'back_all']
    assert len(bench.audit_kernel_sequence(actual,proofs,2))==2
    with pytest.raises(RuntimeError,match='cover actual'):
        bench.audit_kernel_sequence(actual,proofs,3)
    with pytest.raises(RuntimeError,match='cover actual'):
        bench.audit_kernel_sequence(actual+['unknown'],proofs,2)
