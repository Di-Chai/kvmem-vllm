"""CPU-only deployment identity and prelaunch gate tests."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('natural_native_capacity_test',
    Path(__file__).parents[1]/'scripts/benchmark_natural_native_capacity.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def profile():
    return dict(model_format='safetensors', draft_method='mtp', draft_tokens=4,
        draft_quantization=None, execution_mode='compiled_graph', model='/models/model',
        hf_config_path='/models/model', tokenizer='/models/tokenizer', host='127.0.0.1',
        served_model_name='test', max_model_len=262144, max_num_seqs=8,
        max_num_batched_tokens=4224, query_replay_budget=16384,
        prefill_chunk_tokens=2048, gpu_memory_utilization=.90, port=18534,
        linear_backend_per_quant={'nvfp4_w4a4':'auto','fp8_w8a8':'auto'},
        kvmem={'select_budget':65536,'gen_budget':32768})


def fixture(tmp_path, monkeypatch):
    reference = tmp_path/'reference'
    (reference/'frozen/scripts').mkdir(parents=True)
    (reference/'frozen/scripts/original.py').write_text('original unchanged code')
    bench.cap.save(reference/'native.json', profile())
    bench.cap.save(reference/'fixture.json', {'original':'complete input'})
    original = dict(source_sha256=bench.cap.hashes(reference/'frozen'),
        fixture_sha256=bench.cap.sha(reference/'fixture.json'), native_source={'version':'test'},
        workload=dict(concurrency=8,warmup_turns=2,epoch_turns=16,duration_seconds=300,
            mode='tools',generation_mode='natural',generation_limit=32000),
        generation_sampling={'temperature':.6,'ignore_eos':False}, budget={'complete_history':True},
        idle_seconds=10)
    bench.cap.save(reference/'manifest.json', original)
    def proof(path, expected_manifest=None):
        assert path == reference
        if expected_manifest is not None:
            bench.cap.require(bench.cap.sha(path/'manifest.json') == expected_manifest,
                              'Reference manifest changed')
        return original, {'qualified':'byte oracle'}
    monkeypatch.setattr(bench, 'reference_proof', proof)
    monkeypatch.setattr(bench.ready, 'verify', lambda root:bench.cap.read(root/'manifest.json'))
    monkeypatch.setattr(bench, 'device_memory_identity', lambda:dict(uuid='gpu-test',total_mib=97887,reserved_mib=638))
    monkeypatch.setattr(bench, 'installed_native_proof', lambda path:None)
    monkeypatch.setattr(bench, 'completed_control_identity', lambda path:{'completed':'identity'})
    lock=tmp_path/'installed.json';bench.cap.save(lock,{'locked':True})
    root = tmp_path/'prepared'
    bench.prepare(NS(reference=reference,output=root,port=18566,installed_native=lock))
    return root, reference, original


def test_prepare_preserves_frozen_workload_and_effective_native_batch(tmp_path, monkeypatch):
    root, reference, original = fixture(tmp_path, monkeypatch)
    m, _ = bench.verify(root)
    assert (root/'fixture.json').read_bytes() == (reference/'fixture.json').read_bytes()
    assert m['workload'] == original['workload'] and m['runtime_sha256'] == {}
    assert m['effective_native_profile']['max_num_batched_tokens'] == 16424
    assert m['effective_native_profile']['max_num_seqs'] == 8
    assert m['effective_native_profile']['gpu_memory_utilization'] == .915
    assert all(m['effective_native_profile'][k] for k in
        ('enable_prefix_caching','enable_chunked_prefill','async_scheduling'))
    assert set(m['source_sha256'])-set(original['source_sha256']) == {bench.ENTRY}
    assert bench.cap.read(root/'status.json')['status'] == 'prepared'


@pytest.mark.parametrize('change', [dict(max_num_seqs=12),dict(query_replay_budget=8192),
                                  dict(max_num_batched_tokens=9999)])
def test_modified_native_controls_fail_even_if_effective_work_could_be_similar(tmp_path, monkeypatch, change):
    root, _, _ = fixture(tmp_path, monkeypatch)
    p = bench.cap.read(root/'native.json');p.update(change);bench.cap.save(root/'native.json',p)
    with pytest.raises(ValueError,match='beyond declared deployment delta'):
        bench.verify(root)


@pytest.mark.parametrize('section,change', [('workload',dict(generation_mode='fixed')),
    ('generation_sampling',dict(ignore_eos=True)), ('budget',dict(complete_history=False))])
def test_workload_sampling_and_complete_history_identity_are_locked(tmp_path, monkeypatch, section, change):
    root, _, _ = fixture(tmp_path, monkeypatch)
    m = bench.cap.read(root/'manifest.json');m[section].update(change);bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='Reference '+section):
        bench.verify(root)


def test_reference_manifest_or_extra_source_cannot_change_after_prepare(tmp_path, monkeypatch):
    root, reference, _ = fixture(tmp_path, monkeypatch)
    m = bench.cap.read(root/'manifest.json');m['source_sha256']['scripts/unrelated.py']='x'
    bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='beyond deployment entry'):
        bench.verify(root)
    (reference/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='Reference manifest changed'):
        bench.verify(root)


def test_reference_proof_requires_original_qualification_before_preparation(tmp_path, monkeypatch):
    p=tmp_path/'reference';p.mkdir();bench.cap.save(p/'manifest.json',{})
    def reject(root):raise ValueError('Reference qualification invalid')
    monkeypatch.setattr(bench,'module',lambda *args:NS(verify=reject))
    with pytest.raises(ValueError,match='qualification invalid'):
        bench.prepare(NS(reference=p,output=tmp_path/'new',port=18566,installed_native=tmp_path/'unused'))
    assert not (tmp_path/'new').exists()


def test_run_rejects_gate_before_idle_or_launch(tmp_path, monkeypatch):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    def reject(root):
        raise ValueError('Reference qualification invalid')
    monkeypatch.setattr(bench,'verify',reject)
    monkeypatch.setattr(bench.ready,'run_arm',lambda *args:pytest.fail('Model launch before gate'))
    with pytest.raises(ValueError,match='qualification invalid'):
        asyncio.run(bench.run(NS(output=tmp_path)))
    assert bench.cap.read(tmp_path/'status.json')['status']=='prepared'


@pytest.mark.parametrize('mutate_manifest', [False, True])
def test_run_rechecks_identity_after_idle_wait_before_server_launch(tmp_path, monkeypatch, mutate_manifest):
    bench.cap.save(tmp_path/'status.json',dict(status='prepared'))
    checks=[]
    def verify(root):
        checks.append(root)
        if len(checks)>1:
            if mutate_manifest:
                return {'changed':True},{}
            raise ValueError('Candidate source changed while waiting')
        return {},{}
    monkeypatch.setattr(bench,'verify',verify)
    async def idle(*args,**kw):
        pass
    monkeypatch.setattr(bench.ready.bench.q,'wait_idle',idle)
    monkeypatch.setattr(bench.fcntl,'flock',lambda *args:None)
    async def arm(root, name, manifest):
        await bench.ready.bench.q.wait_idle(root,10)
        pytest.fail('Source changed during idle wait but model launched')
    monkeypatch.setattr(bench.ready,'run_arm',arm)
    with pytest.raises(ValueError,match='changed while waiting'):
        asyncio.run(bench.run(NS(output=tmp_path)))
    assert len(checks)==2 and bench.ready.bench.q.wait_idle is idle
    assert bench.cap.read(tmp_path/'status.json')['status']=='failed'


@pytest.mark.parametrize('section', ['device_memory_identity','reference_control_identity','minimum_device_free_mib','installed_native_sha256'])
def test_current_device_completed_control_and_install_locks_cannot_change(tmp_path, monkeypatch, section):
    root, _, _ = fixture(tmp_path, monkeypatch)
    m=bench.cap.read(root/'manifest.json');m[section]='changed';bench.cap.save(root/'manifest.json',m)
    with pytest.raises(ValueError,match='changed'):
        bench.verify(root)


@pytest.mark.parametrize('peak,accepted', [(93010,True),(94000,False)])
def test_headroom_uses_loading_peak_and_preserves_completed_result(tmp_path, peak, accepted):
    d=tmp_path/'native';d.mkdir()
    bench.cap.save(d/'results.json',dict(status='completed',drained=True,native_client_tokens_equal=True,preemptions=0))
    (d/'gpu_samples.jsonl').write_text(bench.json.dumps(dict(memory_mib=peak,wall_at=1))+'\n'+
        bench.json.dumps(dict(memory_mib=80000,wall_at=2))+'\n')
    (d/'native_samples.jsonl').write_text(bench.json.dumps(dict(values={'vllm:num_preemptions_total':0}))+'\n')
    m=dict(device_memory_identity=dict(total_mib=97887,reserved_mib=638))
    result=bench.deployment_acceptance(tmp_path,m)
    assert result['accepted'] is accepted and result['maximum_used_mib']==peak
    assert bench.cap.read(d/'results.json')['status']=='completed'
    if not accepted:assert result['reason']=='contract_failed_headroom'


def test_preemption_before_formal_is_not_hidden_by_zero_formal_counter(tmp_path):
    d=tmp_path/'native';d.mkdir()
    bench.cap.save(d/'results.json',dict(status='completed',drained=True,native_client_tokens_equal=True,preemptions=0))
    (d/'gpu_samples.jsonl').write_text(bench.json.dumps(dict(memory_mib=80000,wall_at=1))+'\n')
    (d/'native_samples.jsonl').write_text(bench.json.dumps(dict(values={'vllm:num_preemptions_total':1}))+'\n')
    result=bench.deployment_acceptance(tmp_path,dict(device_memory_identity=dict(total_mib=97887,reserved_mib=638)))
    assert not result['accepted'] and result['reason']=='preemptions' and result['normal_drain']
