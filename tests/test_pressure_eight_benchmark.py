from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from test_query_stage_efficiency_driver import prepared_inputs, prefill_state_inputs


def module():
    p = Path(__file__).resolve().parents[1]/'scripts/benchmark_pressure_eight.py'
    s = importlib.util.spec_from_file_location('pressure_eight_test', p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


@pytest.fixture
def cohort(prepared_inputs, tmp_path):
    _, args = prepared_inputs; d = module(); b = d.base
    source = args['source_arm']; f = b.read(args['fixture_path'])
    original_row, original_binding = f['rows'][0], f['bindings'][0]
    f['rows'], f['bindings'] = [], []
    for i in range(8):
        cid = f'group{i}_q0000'
        f['rows'].append(dict(original_row, id=cid))
        f['bindings'].append(dict(original_binding, id=cid))
    f['case_ids'] = [r['id'] for r in f['rows']]
    b.save(args['fixture_path'], f, replace=True)
    profile = b.read(source/'profile.json')
    profile['kv_cache_memory_bytes'] = 48*1024**3
    profile['kvmem']['cpu_read_cache_bytes'] = 160_000_000_000
    b.save(source/'profile.json', profile, replace=True)
    runtime = source/'runtime-v1'
    (runtime/'scripts/audit_layerwise_gilopt.py').write_text(
        'def read(path): return {}\ndef save_new(path, value): pass\n')
    binary = runtime/'src/vllm_kvmem/_native.test.so'; binary.write_bytes(b'frozen-native')
    b.save(source/'runtime-proof.json', b.tree(runtime), replace=True)
    variants = {'baseline': {'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER':'1',
                            'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':'1'},
                'candidate': {'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER':'1',
                              'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS':'2'}}
    (tmp_path/'pyproject.toml').write_text('[project]\nname="fixture"\n')
    return d, dict(source_arm=source, fixture_path=args['fixture_path'],
                   autotune_source=args['autotune_source'], variants=variants)


def test_freeze_eight_workers_same_static_controls_and_completion_gate(cohort, tmp_path):
    d, args = cohort; root = tmp_path/'cohort'
    p = d.prepare(root, **args)
    assert p['workers'] == 8 and p['gpu_confirmed_gate'] and p['first_answers_only']
    assert not any(p[k] for k in ('formal_resume','scheduled_monitor','external_grading','automatic_retry'))
    assert p['native_extension']['sha256']
    assert p['analysis_modules'] == list(d.ANALYSIS_MODULES)
    for trial in p['trials']:
        arm = root/trial['name']; runtime = arm/'runtime-v1'
        assert (runtime/'src/vllm_kvmem/layerwise_query.py').read_text().endswith(d.GATE)
        assert d.base.tree(runtime) == p['source_runtime_sha256']
        assert d.base.read(arm/'profile.json')['kvmem']['gpu_memory_admission'] == 'static'
        assert len(d.base.read(arm/'fixture.json')['rows']) == 8
    result = subprocess.run([sys.executable, '-B', str(root/'launch.py'), 'verify', '--root', str(root)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    # The source runtime predates the analyzers. They must still be importable
    # from the frozen bundle without the workspace scripts on PYTHONPATH.
    code = ('import sys; sys.path[:0] = '+repr([str(root/'cohort-analysis'),
            str(root/'source-runtime/scripts')])+'; from analyze_layerwise_dynamic8 import measure')
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match='Preserve'):
        d.prepare(root, **args)
    path = root/d.HELPER; path.write_text(path.read_text()+'\nchanged=True\n')
    with pytest.raises(ValueError, match='Frozen input/control'):
        d.verify(root)


def test_eight_way_acceptance_binary_and_source_are_identical_across_abba(cohort,prepared_inputs,tmp_path):
    d,args=cohort
    binary,_,_=prefill_state_inputs(prepared_inputs,tmp_path)
    args.update(variants={name:{'KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT':str(value)}
                          for name,value in [('baseline',0),('candidate',1)]},
                overlays=('csrc/buffered_prefill_state.cpp',),
                overlay_source=prepared_inputs[1]['overlay_source'],
                prefill_state_binary=binary,
                order=('baseline','candidate','candidate','baseline'))
    root=tmp_path/'batch-eight';plan=d.prepare(root,**args)
    report=plan['prefill_state_extension']
    assert report['sha256']==d.base.sha(binary)
    assert plan['workers']==8 and plan['gpu_confirmed_gate']
    for trial in plan['trials']:
        runtime=root/trial['name']/'runtime-v1'
        assert d.base.sha(runtime/report['relative_path'])==d.base.sha(binary)
        assert (runtime/'src/vllm_kvmem/layerwise_query.py').read_text().endswith(d.GATE)
    result=subprocess.run([sys.executable,'-B',str(root/'launch.py'),'verify','--root',str(root)],
                          text=True,capture_output=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('fault', ['duplicate_group','seven','cpu','gpu','dynamic','budget','batch'])
def test_cohort_rejects_changed_controls_or_inadmissible_gate(cohort, fault):
    d, args = cohort
    f = deepcopy(d.base.read(args['fixture_path']))
    p = deepcopy(d.base.read(args['source_arm']/'profile.json'))
    ids = f['case_ids'][:]
    if fault == 'duplicate_group': ids[1] = ids[0].replace('q0000','q0001')
    elif fault == 'seven': ids.pop()
    elif fault == 'cpu': p['kvmem']['cpu_read_cache_bytes'] = 20*1024**3
    elif fault == 'gpu': p['kv_cache_memory_bytes'] = 1024**3
    elif fault == 'dynamic': p['kvmem']['gpu_memory_admission'] = 'incremental'
    elif fault == 'budget': p['kvmem']['gen_budget'] = 4096
    elif fault == 'batch': p['max_num_batched_tokens'] = 8192
    with pytest.raises(ValueError): d.cohort_controls(f, p, ids)
