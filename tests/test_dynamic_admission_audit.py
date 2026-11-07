"""No-GPU coverage of whole-operation admission and frozen qualification inputs."""
import importlib.util
import json
from pathlib import Path

import pytest


def driver():
    path = Path(__file__).resolve().parents[1]/'scripts/qualify_layerwise_incremental.py'
    spec = importlib.util.spec_from_file_location('qualification_audit_driver',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_multichunk_admission_charges_the_whole_operation_and_mtp():
    pytest.importorskip('vllm')
    from test_incremental_capacity import make_gate, request
    gate, groups, pool, controller = make_gate(blocks=14,mtp=4)
    owner, peer = request('physical'),request('peer')
    assert gate.allocate_slots(owner,8,num_lookahead_tokens=4) is not None
    owner.num_computed_tokens = 8
    assert gate.allocate_slots(peer,1,num_lookahead_tokens=4) is not None
    before = tuple(groups[0].req_to_blocks['physical'])
    assert pool.get_num_free_blocks() == 1
    # One logical 16-token piece fits. A merged 32-token operation needs two
    # additional pages including its native MTP tail; it must wait atomically.
    assert gate.can_schedule(owner,16,num_lookahead_tokens=4)
    assert not gate.can_schedule(owner,32,num_lookahead_tokens=4)
    assert gate.allocate_slots(owner,32,num_lookahead_tokens=4) is None
    assert tuple(groups[0].req_to_blocks['physical']) == before
    assert pool.get_num_free_blocks() == 1
    assert controller.pressure[-1] == ('physical',2,1)
    gate.free(owner)
    gate.free(peer)
    assert pool.get_num_free_blocks() == 13


def test_page_swap_cannot_capture_an_uncommitted_pressure_window():
    pytest.importorskip('vllm')
    from test_request_swap_control import fixture,admit
    f = fixture(blocks=6)
    controller,gate,groups,pool,scheduler,_ = f
    owner = admit(f,'owner',48)
    pending = groups[0].prepare_refresh('owner',48,32)
    before = pool.get_num_free_blocks()
    assert not controller._capture(owner)
    assert not scheduler.connector.commands
    assert pool.get_num_free_blocks() == before
    groups[0].abort_preparation(pending)
    gate.free(owner)


@pytest.fixture
def frozen_inputs(tmp_path):
    d = driver()
    source = tmp_path/'source'
    for directory in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        (source/directory).mkdir(parents=True)
    (source/'src/vllm_kvmem/layerwise_query.py').write_text('# no measurement gate\n')
    (source/'pyproject.toml').write_text('[project]\nname="test"\n')
    reference,inputs = tmp_path/'reference',tmp_path/'inputs'
    reference.mkdir();inputs.mkdir()
    profile = dict(execution_mode='compiled_graph',draft_tokens=4,max_num_batched_tokens=16424,
                   max_num_seqs=8,
                   kvmem=dict(select_budget=36864,gen_budget=16384,block_tokens=32,
                              sink_blocks=41,recent_blocks=0,async_refresh=True))
    rows,bindings = [],[]
    for index in range(2):
        case = f'{index}_cold_q0002'
        payload = inputs/(case+'.json.gz')
        payload.write_bytes(bytes([index]))
        binding = inputs/(case+'.json')
        binding.write_text(json.dumps({'temperature':0.6,'seed':20260722}))
        rows.append(dict(id=case,source=str(payload),source_cache_sha256=d.sha(payload)))
        bindings.append(dict(id=case,request=str(binding),sha256=d.sha(binding),
                             sampling={'temperature':0.6,'seed':20260722},max_tokens=40))
    fixture = dict(rows=rows,bindings=bindings,execution_flags={'layerwise_prefill_pressure':True})
    (reference/'profile.json').write_text(json.dumps(profile))
    (inputs/'fixture.json').write_text(json.dumps(fixture))
    return d,source,reference,inputs,profile,fixture


def test_prepare_freezes_inputs_without_importing_answers_or_changing_recall(frozen_inputs,tmp_path):
    d,source,reference,inputs,profile,fixture = frozen_inputs
    root = tmp_path/'qualification'
    d.prepare(root,source=source,reference=reference,inputs=inputs,
              roomy_bytes=12*1024**3,tight_bytes=3*1024**3)
    plan = d.verify(root)
    assert plan['workers'] == {'static_serial':1,'incremental_serial':1,'incremental_tight':2}
    assert plan['imported_answers'] == 0
    for kind in d.ARMS:
        got = d.read(root/kind/'fixture.json')
        assert all(got['execution_flags'][k] == v for k,v in fixture['execution_flags'].items())
        assert got['execution_flags']['layerwise_pressure_reuse_layer_views']
        assert got['bindings'][0]['sampling'] == fixture['bindings'][0]['sampling']
        assert not (root/kind/'run').exists()
        arm_profile = d.read(root/kind/'profile.json')
        assert arm_profile['max_num_seqs'] == 8
        assert arm_profile['request_planning_batch_tokens'] == 16424
        kv = arm_profile['kvmem']
        assert all(kv[k] == v for k,v in profile['kvmem'].items())
    with pytest.raises(ValueError,match='Preserve'):
        d.prepare(root,tight_bytes=3*1024**3)
    (root/'incremental_serial/runtime-v1/src/vllm_kvmem/layerwise_query.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='runtime changed'):
        d.verify(root)


def test_prepare_rejects_finite_cohort_gate_and_changed_input(frozen_inputs,tmp_path):
    d,source,reference,inputs,_,fixture = frozen_inputs
    original = Path(fixture['rows'][0]['source'])
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='input hash'):
        d.prepare(tmp_path/'bad-input',source=source,reference=reference,inputs=inputs,tight_bytes=3*1024**3)
    (source/'src/vllm_kvmem/layerwise_query.py').write_text(
        "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '6'\n")
    with pytest.raises(ValueError,match='finite-owner'):
        d.freeze(source,tmp_path/'bad-runtime')
