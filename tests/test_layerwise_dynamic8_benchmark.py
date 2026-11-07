"""Validate immutable eight-owner controls without loading a model."""
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/benchmark_layerwise_dynamic8.py'
    spec = importlib.util.spec_from_file_location('dynamic8_driver_test',path)
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup(tmp_path,monkeypatch):
    d = module();source=tmp_path/'source'
    for name in ('src/vllm_kvmem','scripts','csrc','third_party'):
        (source/name).mkdir(parents=True)
    (source/'pyproject.toml').write_text('[project]\nname="test"\n')
    (source/'src/vllm_kvmem/layerwise_query.py').write_text('import os\n')
    (source/'src/vllm_kvmem/prefill_phase_gate.py').write_text('# diagnostic gate\n')
    inputs=tmp_path/'original';inputs.mkdir()
    rows=[];bindings=[]
    for i,cid in enumerate(d.DEFAULT_IDS):
        f=inputs/(cid+'.json.gz');f.write_bytes(bytes([i]))
        r=inputs/(cid+'.json');r.write_text(json.dumps(dict(seed=20260722,temperature=0.6)))
        rows.append(dict(id=cid,source=str(f),source_cache_sha256=d.sha(f),input_tokens=200000+i))
        bindings.append(dict(id=cid,request=str(r),sha256=d.sha(r),sampling={'seed':20260722},max_tokens=100))
    fixture=tmp_path/'fixture.json';fixture.write_text(json.dumps(dict(
        rows=rows,bindings=bindings,execution_flags={'layerwise_prefill_pressure':True},
        execution_kvmem_overrides={'cpu_read_cache_bytes':112000000000})))
    profile=tmp_path/'profile.json';profile.write_text(json.dumps(dict(
        execution_mode='compiled_graph',draft_tokens=4,max_num_seqs=8,max_num_batched_tokens=16424,
        kvmem=dict(select_budget=36864,gen_budget=16384,block_tokens=32,sink_blocks=41,recent_blocks=0,
                   async_refresh=True,pressure_policy='attention',prefill_physical_coalescing=True))))
    cache=tmp_path/'cache';(cache/'hash').mkdir(parents=True)
    (cache/'hash/autotune_configs.json').write_text(json.dumps({'foo':2,'bar':[1,3]}))
    seen=[]
    monkeypatch.setattr(d,'validate_profile',lambda p,*args:seen.append(p))
    args=dict(source=source,fixture_path=fixture,profile_path=profile,autotune_source=cache)
    return d,args,seen


def test_prepare_freezes_all_controls_and_only_changes_page_mode(setup,tmp_path):
    d,args,seen=setup;root=tmp_path/'bench';d.prepare(root,**args)
    p=d.verify(root)
    assert p['order']==list(d.ORDER) and len(seen)==4
    assert p['workers']==8 and p['automatic_retry'] is False and not p['external_grading']
    assert p['cpu_reservation_bytes']<p['client_credit_budget_bytes']
    base=d.read(root/'static1/profile.json')
    baseline_fixture=d.read(root/'static1/fixture.json')
    assert baseline_fixture['execution_kvmem_overrides']['cpu_read_cache_bytes']==114000000000
    assert len(baseline_fixture['rows'])==8 and baseline_fixture['bindings'][0]['sampling']=={'seed':20260722}
    for arm in d.ORDER:
        got=d.read(root/arm/'profile.json')
        assert got['kv_cache_memory_bytes']==48*1024**3
        assert got['verify_request_swap'] is False
        assert got['request_planning_batch_tokens']==16424
        assert got['kvmem']['cpu_read_cache_bytes']==114000000000
        changed={k for k in got['kvmem'] if got['kvmem'][k]!=base['kvmem'][k]}
        assert changed==({'gpu_memory_admission','request_swap_layerwise','request_checkpoint_bytes'} if arm.startswith('dynamic') else set())
        assert d.read(root/arm/'fixture.json')==baseline_fixture
        assert d.tree(root/arm/'runtime-v1')==p['source_runtime_sha256']
        assert d.canonical_cache(root/arm/'autotune')==d.canonical_cache(root/'autotune-seed')
        text=(root/arm/'runtime-v1/src/vllm_kvmem/layerwise_query.py').read_text()
        assert "KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '8'" in text
        assert "KVMEM_DIAGNOSTIC_PREFILL_GPU_COMPLETION'] = '1'" in text
        assert not (root/arm/'run').exists()
    with pytest.raises(ValueError,match='Preserve'):
        d.prepare(root,**args)
    (root/'dynamic1/runtime-v1/src/vllm_kvmem/layerwise_query.py').write_text('# mutated\n')
    with pytest.raises(ValueError,match='runtime changed'):
        d.verify(root)


def test_rejects_cpu_gate_deadlock_and_nonindependent_groups(setup,tmp_path):
    d,args,_=setup
    ids=list(d.DEFAULT_IDS);ids[1]=ids[0]
    with pytest.raises(ValueError,match='distinct'):
        d.prepare(tmp_path/'bad-ids',case_ids=ids,**args)
    fixture=d.read(args['fixture_path'])
    for row in fixture['rows']:row['input_tokens']=500000
    args['fixture_path'].write_text(json.dumps(fixture))
    with pytest.raises(ValueError,match='CPU credits'):
        d.prepare(tmp_path/'bad-capacity',**args)
    assert not (tmp_path/'bad-capacity').exists()


def test_seed_is_immutable_but_arm_autotuning_is_recordable(setup,tmp_path):
    d,args,_=setup;root=tmp_path/'bench';d.prepare(root,**args)
    p=root/'dynamic1/autotune/hash/autotune_configs.json'
    p.write_text('{"bar":[1,3],"foo":2}')
    assert d.canonical_cache(p.parents[1])==d.canonical_cache(root/'autotune-seed')
    p.write_text('{"bar":[1,3],"foo":9}')
    assert d.canonical_cache(p.parents[1])!=d.canonical_cache(root/'autotune-seed')
    d.verify(root)  # arm caches are intentionally mutable, seed never is
    (root/'autotune-seed/hash/autotune_configs.json').write_text('{}')
    with pytest.raises(ValueError,match='seed changed'):
        d.verify(root)
