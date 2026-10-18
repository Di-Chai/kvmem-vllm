"""CPU-only reverse-wrapper identity, qualification and launch-order checks."""
import asyncio
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


spec = importlib.util.spec_from_file_location('rolling_reverse_test',
    Path(__file__).parents[1]/'scripts/benchmark_rolling_reverse.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def profile():
    return dict(model_format='safetensors', draft_method='mtp', draft_tokens=4,
        draft_quantization=None, execution_mode='compiled_graph', model='/models/model',
        hf_config_path='/models/model', tokenizer='/models/tokenizer', host='127.0.0.1',
        served_model_name='test', max_model_len=262144, max_num_seqs=8,
        max_num_batched_tokens=4224, query_replay_budget=16384,
        prefill_chunk_tokens=2048, gpu_memory_utilization=.81, port=18534,
        linear_backend_per_quant={'nvfp4_w4a4':'auto','fp8_w8a8':'auto'},
        kvmem={'select_budget':65536,'gen_budget':32768})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    reference=tmp_path/'reference';(reference/'frozen/scripts').mkdir(parents=True)
    (reference/'frozen/scripts/original.py').write_text('original frozen benchmark')
    for arm in bench.ARMS:
        p=profile()
        if arm=='rolling-candidate-v2':p['kvmem']['rolling_active_window']=True
        bench.cap.save(reference/(arm+'.json'),p)
        if arm!='native':
            runtime=reference/'runtime'/arm;runtime.mkdir(parents=True)
            (runtime/'exact.py').write_text('original '+arm)
    bench.cap.save(reference/'fixture.json',{'original':'complete histories and tools'})
    original=dict(source_sha256=bench.cap.hashes(reference/'frozen'),
        runtime_sha256={arm:bench.cap.hashes(reference/'runtime'/arm) for arm in bench.ARMS if arm!='native'},
        fixture_sha256=bench.cap.sha(reference/'fixture.json'),native_source={'version':'test'},
        qualifications={'pipeline-base':'passed bytes','rolling-candidate-v2':'passed bytes'},
        workload=copy.deepcopy(bench.WORKLOAD),generation_sampling={'ignore_eos':False,'temperature':.6},
        budget={'cpu_full_history':True},idle_seconds=10)
    bench.cap.save(reference/'manifest.json',original)
    control=tmp_path/'control';control.mkdir();bench.cap.save(control/'manifest.json',{'completed':True})
    native_profile=bench.profile(profile(),'native',18548)
    launch=bench.ready.native.launch.build_launch(native_profile,port=18548,async_scheduling=True,environ={})
    control_proof=dict(manifest_sha256=bench.cap.sha(control/'manifest.json'),launch_sha256='actual-launch',
        result_sha256='completed-results',argv_without_port=bench.without_port(launch['argv']),
        effective_profile=launch['native_profile'])
    install=tmp_path/'installed.json';bench.cap.save(install,{'files':{'/installed/vllm/model.py':'hash'}})
    def reference_proof(path,expected_manifest=None):
        assert path==reference
        if expected_manifest is not None:
            bench.cap.require(bench.cap.sha(path/'manifest.json')==expected_manifest,'Reference manifest changed')
        return original,{'both':'passed exact byte qualifications'}
    monkeypatch.setattr(bench,'reference_proof',reference_proof)
    monkeypatch.setattr(bench,'native_control_proof',lambda *args:control_proof)
    monkeypatch.setattr(bench,'installed_native_proof',lambda path:bench.cap.read(path))
    monkeypatch.setattr(bench.ready,'verify',lambda root:bench.cap.read(root/'manifest.json'))
    root=tmp_path/'prepared';bench.prepare(NS(output=root,reference=reference,native_control=control,installed_native=install,port=18554))
    return root,reference,original,control_proof


def test_prepare_only_adds_wrapper_and_preserves_runtimes_and_workload(prepared):
    root,reference,original,_=prepared
    manifest,_=bench.verify(root)
    assert manifest['arms']==['rolling-candidate-v2','pipeline-base','native']
    assert manifest['runtime_sha256']==original['runtime_sha256']
    assert set(manifest['source_sha256'])-set(original['source_sha256'])=={bench.ENTRY}
    assert manifest['workload']==bench.WORKLOAD
    assert (root/'fixture.json').read_bytes()==(reference/'fixture.json').read_bytes()
    for arm in bench.ARMS:
        p=bench.cap.read(root/(arm+'.json'))
        assert p==bench.profile(bench.cap.read(reference/(arm+'.json')),arm,18554)
    native=bench.cap.read(root/'native.json')
    effective=bench.ready.native.launch.build_launch(native,port=18554,async_scheduling=True,environ={})['native_profile']
    assert effective['max_num_batched_tokens']==16424
    assert effective['gpu_memory_utilization']==.90
    assert bench.cap.read(root/'status.json')['status']=='prepared'


@pytest.mark.parametrize('arm,change', [
    ('pipeline-base',{'gpu_memory_utilization':.82}),
    ('rolling-candidate-v2',{'max_num_seqs':12}),
    ('native',{'query_replay_budget':8192}),
    ('native',{'max_num_batched_tokens':32768}),
])
def test_undeclared_profile_change_rejected(prepared,arm,change):
    root,*_=prepared;p=bench.cap.read(root/(arm+'.json'));p.update(change);bench.cap.save(root/(arm+'.json'),p)
    with pytest.raises(ValueError,match='beyond declared delta'):
        bench.verify(root)


@pytest.mark.parametrize('section,value', [
    ('runtime_sha256',{'pipeline-base':{'new_overlay.py':'hash'}}),
    ('workload',dict(bench.WORKLOAD,generation_mode='fixed')),
    ('generation_sampling',{'ignore_eos':True}),
    ('budget',{'cpu_full_history':False}),
    ('qualifications',{}),
])
def test_original_semantics_and_exact_runtime_identity_are_locked(prepared,section,value):
    root,*_=prepared;m=bench.cap.read(root/'manifest.json');m[section]=value;bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='Original '+section):
        bench.verify(root)


def test_order_and_extra_source_are_rejected(prepared):
    root,*_=prepared;m=bench.cap.read(root/'manifest.json');m['arms'].reverse();bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='Reverse order'):
        bench.verify(root)
    m['arms'].reverse();m['source_sha256']['src/new_optimization.py']='hash';bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='beyond reverse wrapper'):
        bench.verify(root)


def test_native_actual_argv_cannot_hide_a_batch_or_sampling_change(prepared):
    _,_,_,proof=prepared
    changed=copy.deepcopy(proof)
    changed['argv_without_port'].extend(['--enforce-eager'])
    with pytest.raises(ValueError,match='completed .90 control beyond port'):
        bench.native_launch_identity(bench.profile(profile(),'native',18554),changed)


def test_installed_native_full_set_and_bytes_are_verified(tmp_path,monkeypatch):
    source=tmp_path/'vllm';source.mkdir();(source/'model.py').write_text('native')
    (source/'kernel.so').write_bytes(b'original library')
    path=tmp_path/'identity.json';bench.cap.save(path,dict(source_directory=str(source),files={str(p):bench.cap.sha(p) for p in source.iterdir()}))
    monkeypatch.setattr(bench.importlib.metadata,'distribution',lambda name:NS(locate_file=lambda name:source))
    bench.installed_native_proof(path)
    (source/'extra.py').write_text('unknown executable')
    with pytest.raises(ValueError,match='identity changed'):
        bench.installed_native_proof(path)
    (source/'extra.py').unlink();(source/'kernel.so').write_bytes(b'changed library')
    with pytest.raises(ValueError,match='identity changed'):
        bench.installed_native_proof(path)


def test_pending_byte_qualification_blocks_preparation_before_mutation(tmp_path,monkeypatch):
    def reject(*args):raise ValueError('GPU qualification has not completed')
    monkeypatch.setattr(bench,'reference_proof',reject)
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='has not completed'):
        bench.prepare(NS(output=out,reference=tmp_path,native_control=tmp_path,installed_native=tmp_path,port=18554))
    assert not out.exists()


@pytest.mark.parametrize('mutate', [False,True])
def test_post_idle_reverification_blocks_launch_and_restores_wait_hook(tmp_path,monkeypatch,mutate):
    bench.cap.save(tmp_path/'status.json',{'status':'prepared'});checks=[]
    def verify(root):
        checks.append(root)
        if len(checks)>2:
            if mutate:return {'changed':True},{}
            raise ValueError('Installed native changed while waiting')
        return {},{}
    monkeypatch.setattr(bench,'verify',verify)
    async def idle(*args,**kwargs):pass
    monkeypatch.setattr(bench.ready.bench.q,'wait_idle',idle)
    monkeypatch.setattr(bench.fcntl,'flock',lambda *args:None)
    async def arm(*args):
        await bench.ready.bench.q.wait_idle(tmp_path,10)
        pytest.fail('Launch after changed identity')
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    with pytest.raises(ValueError,match='changed while waiting'):
        asyncio.run(bench.run(NS(output=tmp_path)))
    assert len(checks)==3 and bench.ready.bench.q.wait_idle is idle


def test_run_calls_original_arm_in_exact_reverse_order_and_preserves_manifest(tmp_path,monkeypatch):
    bench.cap.save(tmp_path/'status.json',{'status':'prepared'});manifest={'original':'identity'};calls=[]
    monkeypatch.setattr(bench,'verify',lambda root:(manifest,{}))
    monkeypatch.setattr(bench.fcntl,'flock',lambda *args:None)
    async def idle(*args,**kwargs):pass
    monkeypatch.setattr(bench.ready.bench.q,'wait_idle',idle)
    async def arm(root,name,m):
        assert m is manifest
        await bench.ready.bench.q.wait_idle(root,10)
        calls.append(name)
        return dict(output_tokens_per_second=1,requests_completed_in_window=1,gpu={},preemptions=0)
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    monkeypatch.setattr(bench,'module',lambda *args:NS(review=lambda root:{'complete':True}))
    asyncio.run(bench.run(NS(output=tmp_path)))
    assert calls==list(bench.ARMS)
    assert bench.cap.read(tmp_path/'status.json')['status']=='completed_pending_review'
    assert bench.ready.bench.q.wait_idle is idle
