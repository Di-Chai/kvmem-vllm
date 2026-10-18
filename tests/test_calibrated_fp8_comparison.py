import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('calibrated_fp8_comparison_test',
    ROOT/'scripts/compare_calibrated_fp8_agents.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


@pytest.mark.parametrize('failure', [None,'legacy','different_calibration','history_budget',
    'controller','production','benchmark_script','inference_profile'])
def test_comparison_requires_identical_calibrated_controls_and_only_declared_change(tmp_path, monkeypatch, failure):
    before, after = tmp_path/'before',tmp_path/'after'
    bm = {k:'same' for k in compare.IMMUTABLE}
    bm.update(host_calibration={'path':'/proof','sha256':'same'},
        runtime_sha256={compare.ARM:{k:'old' for k in compare.PRODUCTION_DELTA|compare.RUNTIME_SCRIPT_DELTA}},
        source_sha256={'scripts/runner.py':'same'},
        fp8_runtime_qualification={'path':'/qualification-before'})
    cm = copy.deepcopy(bm)
    for k in compare.PRODUCTION_DELTA|compare.RUNTIME_SCRIPT_DELTA:
        cm['runtime_sha256'][compare.ARM][k] = 'new'
    cm['fp8_runtime_qualification']['path'] = '/qualification-after'
    cm['baseline'] = dict(path=str(before),evidence_sha256={},
        production_delta={k:['old','new'] for k in compare.PRODUCTION_DELTA})
    bp,cp = {'port':18632,'model':'same'}, {'port':18636,'model':'same'}
    if failure == 'legacy':bm['host_calibration'] = cm['host_calibration'] = None
    if failure == 'different_calibration':cm['host_calibration']['sha256'] = 'other'
    if failure == 'history_budget':cm['budget'] = 'smaller'
    if failure == 'controller':cm['source_sha256']['scripts/runner.py'] = 'changed'
    if failure == 'production':cm['runtime_sha256'][compare.ARM]['src/extra.py'] = 'new'
    if failure == 'benchmark_script':cm['runtime_sha256'][compare.ARM]['scripts/extra.py'] = 'new'
    if failure == 'inference_profile':cp['model'] = 'other'
    audit = compare.Audit()
    monkeypatch.setattr(audit,'frozen_run',lambda root,**_: (bm,bp) if root==before else (cm,cp))
    monkeypatch.setattr(audit,'read',lambda _:dict(fixture_sha256='same',budget='same',
        output_quotas=[1024,4096,1024],concurrency=2,native_source='same'))
    if failure is None:
        report = audit.protocol(before,after)
        assert len(report['production_delta']) == 3 and report['controller_delta'] == {}
    else:
        with pytest.raises(ValueError):audit.protocol(before,after)


def test_ast_checker_keeps_gpu_imports_and_module_body_unexecuted(tmp_path):
    path = tmp_path/'checker.py'
    path.write_text('import missing_cuda_library\nraise RuntimeError("do not run")\n'
                    'def check(value):\n    return value+1\n')
    fn = compare.Audit().function(path,('check',),{})
    assert fn['check'](2) == 3


@pytest.mark.parametrize('failure',[None,'missing_flag','missing_setting','fraction','unknown'])
def test_calibration_normalizes_only_the_declared_cuda_policy(failure):
    base=dict(model='/same',port=18632,kvmem={'select_budget':65536,'gen_budget':32768})
    profile=copy.deepcopy(base);profile['cuda_allocator_conf']=dict(compare.ALLOCATOR_POLICY)
    manifest={'cuda_allocator_policy':'native_gc_90'}
    if failure=='missing_flag':manifest={}
    if failure=='missing_setting':del profile['cuda_allocator_conf']
    if failure=='fraction':profile['cuda_allocator_conf']['per_process_memory_fraction']=1.
    if failure=='unknown':manifest['cuda_allocator_policy']='unproven'
    if failure is None:
        assert compare.calibrated_context_profile(manifest,profile)==base
        assert profile['cuda_allocator_conf']==compare.ALLOCATOR_POLICY
        assert compare.calibrated_context_profile({},base)==base
    else:
        with pytest.raises(ValueError):compare.calibrated_context_profile(manifest,profile)


@pytest.mark.parametrize('failure', [None,'policy','profile','controller','measurement','launcher','global'])
def test_allocator_ablation_requires_exact_policy_launcher_and_unchanged_measurement(tmp_path,monkeypatch,failure):
    before,after=tmp_path/'before',tmp_path/'after'
    files=compare.PRODUCTION_DELTA|compare.ALLOCATOR_DELTA|compare.RUNTIME_SCRIPT_DELTA
    bm={k:'same' for k in compare.IMMUTABLE}
    bm.update(host_calibration={'path':'/proof','sha256':'same'},
        runtime_sha256={compare.ARM:{k:'old' for k in files}},
        source_sha256={'scripts/benchmark_aligned_fp8_agents.py':'old'},
        fp8_runtime_qualification={'path':'/before-qualification'})
    cm=copy.deepcopy(bm)
    for name in files:cm['runtime_sha256'][compare.ARM][name]='new'
    cm.update(cuda_allocator_policy='native_gc_90',
        fp8_runtime_qualification={'path':'/after-qualification'},
        source_sha256={'scripts/benchmark_aligned_fp8_agents.py':'new'},
        baseline=dict(path=str(before),evidence_sha256={},
            production_delta={k:['old','new'] for k in compare.PRODUCTION_DELTA|compare.ALLOCATOR_DELTA}))
    bp={'port':18632,'model':'same'}
    cp=dict(bp,port=18648,cuda_allocator_conf=dict(compare.ALLOCATOR_POLICY))
    source='\n'.join(f'def {name}():\n    return 1\n' for name in compare.MEASUREMENT_FUNCTIONS)
    for root in (before,after):
        path=root/'frozen/scripts/benchmark_aligned_fp8_agents.py';path.parent.mkdir(parents=True)
        path.write_text(source+'\ndef prepare():\n    return '+('1' if root==before else '2')+'\n')
    if failure=='policy':cm['cuda_allocator_policy']='other'
    if failure=='profile':cp['cuda_allocator_conf']['per_process_memory_fraction']=1.
    if failure=='controller':cm['source_sha256']['scripts/unrelated.py']='new'
    if failure=='measurement':
        path=after/'frozen/scripts/benchmark_aligned_fp8_agents.py'
        path.write_text(path.read_text().replace('def run():\n    return 1','def run():\n    return 2'))
    if failure=='global':
        path=after/'frozen/scripts/benchmark_aligned_fp8_agents.py'
        path.write_text('chain = dangerous_replacement\n'+path.read_text())
    if failure=='launcher':
        cm['runtime_sha256'][compare.ARM]['src/vllm_kvmem/serve.py']='old'
        del cm['baseline']['production_delta']['src/vllm_kvmem/serve.py']
    audit=compare.Audit()
    monkeypatch.setattr(audit,'frozen_run',lambda root,**_: (bm,bp) if root==before else (cm,cp))
    monkeypatch.setattr(audit,'read',lambda _:dict(fixture_sha256='same',budget='same',
        output_quotas=[1024,4096,1024],concurrency=2,native_source='same'))
    if failure is None:
        report=audit.protocol(before,after)
        assert len(report['production_delta'])==4
        assert report['profile_delta']['cuda_allocator_conf']==[None,compare.ALLOCATOR_POLICY]
    else:
        with pytest.raises(ValueError):audit.protocol(before,after)


def test_evidence_cannot_change_mid_comparison(tmp_path):
    path = tmp_path/'result.json';path.write_text('{"passed":true}')
    audit = compare.Audit();audit.read(path)
    path.write_text('{"passed":false}')
    with pytest.raises(ValueError,match='Evidence changed'):
        audit.read(path)


@pytest.mark.parametrize('failure',[None,'owner','runtime','receipt','gpu_live'])
def test_reused_qualification_keeps_its_original_protected_cleanup_owner(tmp_path,failure):
    owner=tmp_path/'old-run';root=owner/'qualification';(root/'process').mkdir(parents=True)
    (owner/'frozen').mkdir();(owner/'runtime'/compare.ARM).mkdir(parents=True)
    (root/'manifest.json').write_text('{}')
    m=dict(fp8_runtime_qualification=dict(path=str(root),manifest_sha256=compare.sha(root/'manifest.json')),
        runtime_sha256={compare.ARM:{}},source_sha256={})
    if failure=='owner':m['fp8_runtime_qualification']['path']='/wrong-owner/qualification'
    (owner/'manifest.json').write_text(json.dumps(m))
    stop=dict(gpu_empty=failure!='gpu_live',observed_gpu_pids=[123] if failure=='gpu_live' else [])
    (root/'process/post-stop.json').write_text(json.dumps(stop))
    binding=dict(protocol='aligned_fp8_qualification_cleanup_v1',
        qualification_manifest_sha256=compare.sha(root/'manifest.json'),
        performance_manifest_sha256=compare.sha(owner/'manifest.json'),
        post_stop_sha256=compare.sha(root/'process/post-stop.json'))
    if failure=='receipt':binding['performance_manifest_sha256']='not-the-original-owner'
    (root/'process/qualification-cleanup.json').write_text(json.dumps(binding))
    performance=dict(runtime_sha256={compare.ARM:{'src/changed.py':'different'} if failure=='runtime' else {}})
    if failure is None:compare.Audit().qualification_cleanup(root,performance)
    else:
        with pytest.raises(ValueError):compare.Audit().qualification_cleanup(root,performance)
