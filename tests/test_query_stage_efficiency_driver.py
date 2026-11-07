"""Frozen full-input workload/flag controls without a model or GPU."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_query_stage_efficiency.py'
    spec=importlib.util.spec_from_file_location('query_stage_driver_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def prepared_inputs(tmp_path):
    d=module();arm=tmp_path/'reference';runtime=arm/'runtime-v1';overlay=tmp_path/'overlay'
    for root in (runtime,overlay):
        (root/'src/vllm_kvmem').mkdir(parents=True)
        (root/'scripts').mkdir()
        (root/'pyproject.toml').write_text('[project]\nname="stub"\n')
        (root/'src/vllm_kvmem/layerwise_query.py').write_text('import os\noriginal = True\n')
        (root/'src/vllm_kvmem/transfer.py').write_text('value = 1\n')
    p=runtime/'src/vllm_kvmem/layerwise_query.py';p.write_text(p.read_text()+d.GATE)
    (overlay/'src/vllm_kvmem/layerwise_query.py').write_text('import os\noriginal = True\nprobe = False\n')
    (overlay/'src/vllm_kvmem/query_stage_profile.py').write_text('enabled = False\n')
    inputs=tmp_path/'original';inputs.mkdir();rows=[];bindings=[]
    for i,cid in enumerate(d.DEFAULT_IDS):
        origin=inputs/(cid+'.json.gz');origin.write_bytes(bytes([i]))
        binding=inputs/(cid+'.json');binding.write_text('{"seed": 123}\n')
        qb=411675 if i<2 else 263820
        rows.append(dict(id=cid,source=str(origin),source_cache_sha256=d.sha(origin),
                         input_tokens=qb+100,query_begin=qb,query_end=qb+80))
        bindings.append(dict(id=cid,request=str(binding),sha256=d.sha(binding),
                             sampling={'seed':123},max_tokens=40,thinking_token_budget=0))
    flags=dict(layerwise_prefill_pressure=True,layerwise_paired_reads=True,
               layerwise_pressure_prepared_reads=True,layerwise_pressure_submit_on_prepare=True)
    fixture=tmp_path/'fixture.json'
    fixture.write_text(json.dumps(dict(rows=rows,bindings=bindings,execution_flags=flags)))
    (arm/'fixture.json').write_text(fixture.read_text())
    profile=dict(execution_mode='compiled_graph',draft_tokens=4,max_num_seqs=8,
        max_num_batched_tokens=16424,port=1234,kvmem=dict(select_budget=36864,gen_budget=16384,
        block_tokens=32,sink_blocks=41,recent_blocks=0,gpu_memory_admission='static',request_swap_layerwise=False))
    (arm/'profile.json').write_text(json.dumps(profile))
    (arm/'runtime-proof.json').write_text(json.dumps(d.tree(runtime)))
    cache=tmp_path/'cache';cache.mkdir();(cache/'seed.json').write_text('{"tactic": 4}\n')
    return d,dict(source_arm=arm,fixture_path=fixture,overlay_source=overlay,autotune_source=cache)


def test_prepare_retains_original_budgets_prefixes_and_sampling(prepared_inputs,tmp_path):
    d,args=prepared_inputs;root=tmp_path/'diagnostic';d.prepare(root,**args)
    p=d.verify(root)
    assert [t['name'] for t in p['trials']]==['01-baseline','02-probe']
    assert p['workers']==1 and not p['formal_resume'] and not p['automatic_retry']
    assert d.read(root/'01-baseline/profile.json')==d.read(root/'02-probe/profile.json')
    assert d.read(root/'01-baseline/fixture.json')==d.read(root/'02-probe/fixture.json')
    for trial in p['trials']:
        arm=root/trial['name'];f=d.read(arm/'fixture.json');c=d.read(arm/'profile.json')
        assert c['kvmem']['select_budget']==36864 and c['kvmem']['gen_budget']==16384
        assert c['max_num_seqs']==8 and c['max_num_batched_tokens']==16424
        assert len(f['rows'])==3 and f['bindings'][0]['sampling']=={'seed':123}
        assert not f['qualification'] and not f['restore_digest_qualification']
        assert 'KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS' not in (arm/'runtime-v1/src/vllm_kvmem/layerwise_query.py').read_text()
        assert d.tree(arm/'runtime-v1')==p['source_runtime_sha256']
        assert not (arm/'process.json').exists()
    assert 'QUERY_STAGE_EXECUTION_JSON' in (root/'source-runtime/scripts/_query_stage_server.py').read_text()
    assert p['changed_runtime_files']==['src/vllm_kvmem/layerwise_query.py','src/vllm_kvmem/query_stage_profile.py']
    with pytest.raises(ValueError,match='Preserve'):d.prepare(root,**args)


def test_rejects_prefix_drift_and_recall_switches(prepared_inputs,tmp_path):
    d,args=prepared_inputs;fixture=d.read(args['fixture_path'])
    # Omitted original query establishes an earlier frozen P boundary.
    extra={**fixture['rows'][0],'id':fixture['rows'][0]['id'].replace('q0000','q0009'),
           'query_begin':400000}
    fixture['rows'].append(extra)
    with pytest.raises(ValueError,match='frozen P'):d.workload(fixture,d.DEFAULT_IDS)
    with pytest.raises(ValueError,match='execution switches'):
        d.variant_order({'b':{'KVMEM_LAYERWISE_QUERY_NO_PROBE':'1'}},('b',))
    with pytest.raises(ValueError,match='execution switches'):
        d.variant_order({'b':{'KVMEM_SELECT_BUDGET':'4096'}},('b',))
    with pytest.raises(ValueError,match='finite-owner'):
        d.ungated("import os\nos.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS']='4'\n")


def test_runtime_and_seed_are_immutable_and_abba_is_predeclared(prepared_inputs,tmp_path):
    d,args=prepared_inputs;root=tmp_path/'perf'
    variants={'baseline':{'KVMEM_QUERY_STAGE_PROFILE':'0'},
              'candidate':{'KVMEM_QUERY_STAGE_PROFILE':'0','KVMEM_QUERY_TEST_EXECUTION':'1'}}
    d.prepare(root,variants=variants,order=('baseline','candidate','candidate','baseline'),**args)
    plan=d.verify(root);assert len(plan['trials'])==4
    (root/'01-baseline/autotune/seed.json').write_text('{"tactic": 5}')
    d.verify(root)  # A trial cache is writable, immutable seed is not.
    (root/'autotune-seed/seed.json').write_text('{}')
    with pytest.raises(ValueError,match='seed changed'):d.verify(root)


def test_cuda_scorer_patch_requires_identical_packaged_copy(prepared_inputs,tmp_path):
    d,args=prepared_inputs
    runtime=args['source_arm']/'runtime-v1';overlay=args['overlay_source']
    names=['csrc/qw3_mean_k.cu','src/vllm_kvmem/cuda_sources/qw3_mean_k.cu']
    for name in names:
        for root,text in ((runtime,'old'),(overlay,'candidate')):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    with pytest.raises(ValueError,match='Declare both'):
        d.patch_runtime(runtime,overlay,names[:1],tmp_path/'partial.patch')
    (overlay/names[1]).write_text('stale')
    with pytest.raises(ValueError,match='differs'):
        d.patch_runtime(runtime,overlay,names,tmp_path/'mismatch.patch')
    assert all((runtime/name).read_text()=='old' for name in names)
    (overlay/names[1]).write_text('candidate\n')
    (overlay/names[0]).write_text('candidate\n')
    # A valid source fixture includes a complete final newline as git expects.
    for name in names:(runtime/name).write_text('old\n')
    assert d.patch_runtime(runtime,overlay,names,tmp_path/'valid.patch')==names
    assert all((runtime/name).read_text()=='candidate\n' for name in names)


def test_declared_patch_cannot_escape_runtime(prepared_inputs,tmp_path):
    d,args=prepared_inputs
    with pytest.raises(ValueError,match='explicit inference'):
        d.patch_runtime(args['source_arm']/'runtime-v1',args['overlay_source'],
                        ('../outside.py',),tmp_path/'bad.patch')


def test_server_shim_puts_probe_in_explicit_launch_not_ambient_env(monkeypatch):
    d=module();seen=[]
    serve=SimpleNamespace(build_launch=lambda *a,**k:{'environment':{'KVMEM_EXISTING':'1'}})
    serve.main=lambda:seen.append(serve.build_launch()['environment'])
    package=ModuleType('vllm_kvmem');package.serve=serve
    monkeypatch.setitem(sys.modules,'vllm_kvmem',package)
    monkeypatch.setenv('QUERY_STAGE_EXECUTION_JSON','{"KVMEM_QUERY_STAGE_PROFILE":"1"}')
    exec(d.SERVER_SHIM,{'__name__':'__main__'})
    assert seen==[{'KVMEM_EXISTING':'1','KVMEM_QUERY_STAGE_PROFILE':'1'}]


def test_multiprocessing_import_does_not_rerun_server_parser(monkeypatch):
    d=module();seen=[]
    original=lambda *a,**k:{'environment':{}}
    serve=SimpleNamespace(build_launch=original,main=lambda:seen.append('main'))
    package=ModuleType('vllm_kvmem');package.serve=serve
    monkeypatch.setitem(sys.modules,'vllm_kvmem',package)
    monkeypatch.delenv('QUERY_STAGE_EXECUTION_JSON',raising=False)
    exec(d.SERVER_SHIM,{'__name__':'__mp_main__'})
    assert seen==[] and serve.build_launch is original


def test_probe_audit_requires_actual_request_bound_complete_stages(tmp_path):
    d=module();dest=tmp_path/'trial';capture=dest/'run/server/capture';capture.mkdir(parents=True)
    case=dest/'run/answers/cases/case1';case.mkdir(parents=True)
    (case/'result.json').write_text(json.dumps({'layerwise':{'layers':[
        {'request_id':'owner','layer':i} for i in range(16)]}}))
    assert d.probe_output_audit(dest,enabled=False,case_ids=['case1'])['passed']
    with pytest.raises(ValueError,match='did not record'):
        d.probe_output_audit(dest,enabled=True,case_ids=['case1'])
    stages=[{'stage':s,'layer':i} for s in ('score_gpu_dispatch','score_download_wait','selection_cpu',
        'assembler_views_prepare','archive_report','h2d_copy','layout_and_rebuild') for i in range(16)]
    row=dict(request_id='owner',stages=stages,dropped_rows=0,dropped_gpu_events=0,unresolved_gpu_events=0)
    path=capture/'query_stage_profile.jsonl';path.write_text(json.dumps(row)+'\n')
    assert d.probe_output_audit(dest,enabled=True,case_ids=['case1'])['passed']
    with pytest.raises(ValueError,match='Disabled probe'):
        d.probe_output_audit(dest,enabled=False,case_ids=['case1'])
    row['unresolved_gpu_events']=1;path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Incomplete'):
        d.probe_output_audit(dest,enabled=True,case_ids=['case1'])
    row['unresolved_gpu_events']=0;row['stages']=stages[:-1];path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Missing per-layer'):
        d.probe_output_audit(dest,enabled=True,case_ids=['case1'])


def native_inputs(prepared_inputs, tmp_path):
    d,args=prepared_inputs;runtime=args['source_arm']/'runtime-v1'
    filename='_native'+d.sysconfig.get_config_var('EXT_SUFFIX')
    for root,contents in ((runtime,'// original\n'),(args['overlay_source'],'// added ABI-only helper\n')):
        (root/'csrc').mkdir()
        (root/'csrc/bindings.cpp').write_text(contents)
    original=runtime/'src/vllm_kvmem'/filename
    original.write_bytes(b'\x7fELF original extension')
    other=runtime/'src/vllm_kvmem/_prefill_state_native.test.so'
    other.write_bytes(b'\x7fELF unchanged prefill extension')
    build=tmp_path/'explicit-build';build.mkdir()
    supplied=build/filename;supplied.write_bytes(b'\x7fELF additive helper extension')
    (args['source_arm']/'runtime-proof.json').write_text(json.dumps(d.tree(runtime)))
    return supplied,original,other


def test_explicit_native_patch_and_binary_are_identical_in_all_arms(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,original,other=native_inputs(prepared_inputs,tmp_path)
    original_hash=d.sha(original);other_hash=d.sha(other);root=tmp_path/'native-trial'
    variants={name:{'KVMEM_QUERY_STAGE_PROFILE':'0','KVMEM_QUERY_TEST_EXECUTION':str(i)}
              for i,name in enumerate(('baseline','index','combined'))}
    d.prepare(root,native_binary=binary,overlays=(*d.OVERLAYS,'csrc/bindings.cpp'),
              variants=variants,order=tuple(variants),**args)
    plan=d.verify(root);native=plan['native_extension']
    assert native['original_sha256']==original_hash and native['sha256']==d.sha(binary)
    assert native['declared_source_sha256']['csrc/bindings.cpp']==d.sha(args['overlay_source']/'csrc/bindings.cpp')
    assert 'csrc/bindings.cpp' in plan['changed_runtime_files']
    assert 'a/csrc/bindings.cpp' in (root/'implementation.patch').read_text()
    assert plan['files']['native-extension.json']==d.sha(root/'native-extension.json')
    for trial in plan['trials']:
        runtime=root/trial['name']/'runtime-v1'
        assert d.sha(runtime/native['relative_path'])==d.sha(binary)
        assert d.sha(runtime/'src/vllm_kvmem'/other.name)==other_hash
    # Candidate preparation must not mutate the original runtime or supplied file.
    assert d.sha(original)==original_hash and d.sha(other)==other_hash
    changed=root/'02-index/runtime-v1'/native['relative_path'];changed.write_bytes(b'\x7fELF tampered')
    with pytest.raises(ValueError,match='arm runtime changed'):d.verify(root)


def test_native_source_binary_pair_is_required(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,original,_=native_inputs(prepared_inputs,tmp_path)
    runtime=args['source_arm']/'runtime-v1'
    with pytest.raises(ValueError,match='explicitly supplied binary'):
        d.freeze_native(runtime,None,['csrc/bindings.cpp'])
    with pytest.raises(ValueError,match=r'declared C\+\+ source patch'):
        d.freeze_native(runtime,binary,[])
    with pytest.raises(ValueError,match='unchanged'):
        d.freeze_native(runtime,original,['csrc/bindings.cpp'])
    with pytest.raises(ValueError,match='explicit inference'):
        d.patch_runtime(runtime,args['overlay_source'],('csrc/other.cpp',),tmp_path/'bad-native.patch')
    with pytest.raises(ValueError,match='explicit inference'):
        d.patch_runtime(runtime,args['overlay_source'],('src/vllm_kvmem/'+binary.name,),tmp_path/'binary.patch')


def test_native_replacement_rejects_other_abi_and_non_elf(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,_,_=native_inputs(prepared_inputs,tmp_path)
    runtime=args['source_arm']/'runtime-v1'
    wrong=binary.with_name('_native.wrong-abi.so');wrong.write_bytes(binary.read_bytes())
    with pytest.raises(ValueError,match='original module and Python ABI'):
        d.freeze_native(runtime,wrong,['csrc/bindings.cpp'])
    binary.write_bytes(b'not a shared object')
    with pytest.raises(ValueError,match='ELF extension'):
        d.freeze_native(runtime,binary,['csrc/bindings.cpp'])


@pytest.mark.parametrize('mismatch', ['none','path','hash'])
def test_native_identity_checked_in_actual_server_import(monkeypatch,tmp_path,mismatch):
    d=module();native=tmp_path/'_native.so';native.write_bytes(b'\x7fELF fixture')
    seen=[]
    serve=SimpleNamespace(build_launch=lambda *a,**k:{'environment':{}})
    serve.main=lambda:seen.append('started')
    package=ModuleType('vllm_kvmem');package.serve=serve
    package._native=SimpleNamespace(__file__=str(native))
    monkeypatch.setitem(sys.modules,'vllm_kvmem',package)
    expected=dict(path=str(native),sha256=d.sha(native))
    if mismatch=='path':expected['path']=str(tmp_path/'installed-shadow.so')
    if mismatch=='hash':expected['sha256']='0'*64
    audit=tmp_path/'native-runtime-identity.json'
    monkeypatch.setenv('QUERY_STAGE_EXECUTION_JSON','{}')
    monkeypatch.setenv('QUERY_STAGE_NATIVE_BINARY_JSON',json.dumps(expected))
    monkeypatch.setenv('QUERY_STAGE_NATIVE_AUDIT_OUTPUT',str(audit))
    if mismatch!='none':
        with pytest.raises(RuntimeError,match='shadowed or changed'):
            exec(d.SERVER_SHIM,{'__name__':'__main__'})
        assert seen==[] and not audit.exists()
    else:
        exec(d.SERVER_SHIM,{'__name__':'__main__'})
        assert seen==['started'] and d.read(audit)['passed']
        assert d.read(audit)['sha256']==expected['sha256']


def prefill_state_inputs(prepared_inputs, tmp_path):
    d,args=prepared_inputs;runtime=args['source_arm']/'runtime-v1'
    filename='_prefill_state_native'+d.sysconfig.get_config_var('EXT_SUFFIX')
    for root,code in ((runtime,'// original\n'),(args['overlay_source'],'// additive batch API\n')):
        (root/'csrc').mkdir(exist_ok=True)
        (root/'csrc/buffered_prefill_state.cpp').write_text(code)
        (root/'csrc/prefill_attention_commit.hpp').write_text('// unchanged arithmetic\n')
    original=runtime/'src/vllm_kvmem'/filename
    original.write_bytes(b'\x7fELF original prefill state')
    other=runtime/'src/vllm_kvmem/_native.test.so'
    other.write_bytes(b'\x7fELF untouched Store extension')
    build=tmp_path/'prefill-build';build.mkdir()
    supplied=build/filename;supplied.write_bytes(b'\x7fELF additive prefill state API')
    (args['source_arm']/'runtime-proof.json').write_text(json.dumps(d.tree(runtime)))
    return supplied,original,other


def test_prefill_state_patch_freezes_same_binary_without_rebuilding_store(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,original,other=prefill_state_inputs(prepared_inputs,tmp_path)
    old_hash,store_hash=d.sha(original),d.sha(other);root=tmp_path/'prefill-trial'
    d.prepare(root,prefill_state_binary=binary,
        overlays=(*d.OVERLAYS,'csrc/buffered_prefill_state.cpp'),**args)
    plan=d.verify(root);report=plan['prefill_state_extension']
    assert plan['native_extension'] is None
    assert report['original_sha256']==old_hash and report['sha256']==d.sha(binary)
    assert report['declared_source_sha256']=={
        name:d.sha(args['overlay_source']/name) for name in d.PREFILL_STATE_SOURCES}
    assert plan['files']['prefill-state-extension.json']==d.sha(root/'prefill-state-extension.json')
    for trial in plan['trials']:
        runtime=root/trial['name']/'runtime-v1'
        assert d.sha(runtime/report['relative_path'])==d.sha(binary)
        assert d.sha(runtime/'src/vllm_kvmem'/other.name)==store_hash
    assert d.sha(original)==old_hash
    (root/'02-probe/runtime-v1'/report['relative_path']).write_bytes(b'\x7fELF tampered')
    with pytest.raises(ValueError,match='arm runtime changed'):d.verify(root)


def test_prefill_source_and_binary_pair_is_required(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,original,_=prefill_state_inputs(prepared_inputs,tmp_path)
    runtime=args['source_arm']/'runtime-v1';changed=['csrc/buffered_prefill_state.cpp']
    with pytest.raises(ValueError,match='explicitly supplied binary'):
        d.freeze_prefill_state(runtime,None,changed)
    with pytest.raises(ValueError,match=r'declared C\+\+ source patch'):
        d.freeze_prefill_state(runtime,binary,[])
    with pytest.raises(ValueError,match='unchanged'):
        d.freeze_prefill_state(runtime,original,changed)
    bad=binary.with_name('_prefill_state_native.wrong-abi.so');bad.write_bytes(binary.read_bytes())
    with pytest.raises(ValueError,match='original module and Python ABI'):
        d.freeze_prefill_state(runtime,bad,changed)
    binary.write_bytes(b'not ELF')
    with pytest.raises(ValueError,match='ELF extension'):
        d.freeze_prefill_state(runtime,binary,changed)


def test_shared_prefill_header_cannot_change_without_rebuilding_store(prepared_inputs,tmp_path):
    d,args=prepared_inputs;binary,_,_=prefill_state_inputs(prepared_inputs,tmp_path)
    runtime=args['source_arm']/'runtime-v1';header='csrc/prefill_attention_commit.hpp'
    before=d.tree(runtime)
    (args['overlay_source']/header).write_text('// changed arithmetic also used by _native\n')
    with pytest.raises(ValueError,match='Shared prefill arithmetic header'):
        d.patch_runtime(runtime,args['overlay_source'],
                        ('csrc/buffered_prefill_state.cpp',header),tmp_path/'shared.patch')
    assert d.tree(runtime)==before
    with pytest.raises(ValueError,match='Shared prefill arithmetic header'):
        d.freeze_prefill_state(runtime,binary,[header])


@pytest.mark.parametrize('mismatch',['none','path','hash'])
def test_prefill_state_actual_server_import_identity(monkeypatch,tmp_path,mismatch):
    d=module();binary=tmp_path/'_prefill_state_native.so';binary.write_bytes(b'\x7fELF fixture')
    seen=[]
    serve=SimpleNamespace(build_launch=lambda *a,**k:{'environment':{}})
    serve.main=lambda:seen.append('started')
    package=ModuleType('vllm_kvmem');package.serve=serve
    package._prefill_state_native=SimpleNamespace(__file__=str(binary))
    monkeypatch.setitem(sys.modules,'vllm_kvmem',package)
    monkeypatch.delenv('QUERY_STAGE_NATIVE_BINARY_JSON',raising=False)
    expected=dict(path=str(binary),sha256=d.sha(binary))
    if mismatch=='path':expected['path']=str(tmp_path/'shadow.so')
    if mismatch=='hash':expected['sha256']='0'*64
    audit=tmp_path/'prefill-state-runtime-identity.json'
    monkeypatch.setenv('QUERY_STAGE_EXECUTION_JSON','{}')
    monkeypatch.setenv('QUERY_STAGE_PREFILL_STATE_BINARY_JSON',json.dumps(expected))
    monkeypatch.setenv('QUERY_STAGE_PREFILL_STATE_AUDIT_OUTPUT',str(audit))
    if mismatch!='none':
        with pytest.raises(RuntimeError,match='prefill state extension was shadowed or changed'):
            exec(d.SERVER_SHIM,{'__name__':'__main__'})
        assert seen==[] and not audit.exists()
    else:
        exec(d.SERVER_SHIM,{'__name__':'__main__'})
        assert seen==['started'] and d.read(audit)['passed']
        assert d.read(audit)['sha256']==expected['sha256']
