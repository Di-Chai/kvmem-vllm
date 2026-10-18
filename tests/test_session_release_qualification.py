"""Negative evidence checks for the real-model namespace turnover gate."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('session_release_qualification',
    Path(__file__).parents[1]/'scripts/qualify_session_release.py')
qual=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qual)


def profile():
    return qual.qualification_profile(dict(model_format='safetensors',execution_mode='compiled_graph',
        draft_method='mtp',draft_tokens=4,max_num_seqs=8,max_model_len=196608,kvmem=dict(
            select_budget=65536,gen_budget=32768,block_tokens=32,archive_backend='cpu',
            cpu_read_cache_bytes=100_000_000_000,session_capacity=32,session_checkpoint_bytes=4*1024**3,
            session_checkpoint_policy='pm',session_capture_terminal=True,rolling_active_window=True,
            cpu_history_coordination=True,active_resident_pressure_only=True)),18620)


def released():
    state=dict(session_views=[],cpu_history_admission=dict(leases=0,reserved_bytes=0),
        active_residency=dict(idle_views=0,idle_blocks=0),free_blocks=99,total_blocks=100)
    archive=dict(session_checkpoint_deferrals=0,archive_backend='cpu',repository_gc=False,
        disk_read_bytes=0,disk_write_bytes=0,session_checkpoints=0,session_checkpoint_payload_bytes=0,
        cpu_authority_bytes=0,retiring_requests=0,session_retiring_requests=0,pending_downloads=0)
    return state,archive


def test_release_proof_requires_all_cpu_and_gpu_leases_and_checkpoints_gone():
    state,archive=released()
    assert qual.completely_released(state,archive)
    mutations=[('session_views',[dict(namespace=['s','s'])]),('free_blocks',98),
        ('cpu_history_admission',dict(leases=1,reserved_bytes=0)),
        ('cpu_history_admission',dict(leases=0,reserved_bytes=1024)),
        ('active_residency',dict(idle_views=1,idle_blocks=1))]
    for key,value in mutations:
        assert not qual.completely_released(dict(state,**{key:value}),archive)
    for key in ['session_checkpoints','session_checkpoint_payload_bytes','cpu_authority_bytes',
                'retiring_requests','session_retiring_requests','pending_downloads']:
        assert not qual.completely_released(state,dict(archive,**{key:1}))
        missing=dict(archive);missing.pop(key)
        assert not qual.completely_released(state,missing)


@pytest.mark.parametrize('mutation', [dict(session_checkpoint_deferrals=1),
    dict(session_checkpoint_deferrals=None),dict(session_checkpoint_deferrals=False),
    dict(disk_write_bytes=1),dict(disk_read_bytes=1),dict(repository_gc=True),dict(archive_backend='disk')])
def test_fallback_and_disk_activity_cannot_pass_as_release(mutation):
    state,archive=released()
    with pytest.raises(ValueError):qual.completely_released(state,dict(archive,**mutation))


def test_profile_preserves_capacity_and_fixture_includes_full_long_final_wave():
    p=profile();original=copy.deepcopy(p)
    f=dict(tokens=[1]*99826,eos=0)
    budget=qual.validate_fixture(f,p)
    assert p == original and p['verify_session_restore'] is True
    assert qual.USERS*qual.WAVES > p['kvmem']['session_capacity']
    assert budget['maximum_history_tokens'] == 99826+2*qual.OUTPUT_TOKENS+qual.TOOL_TOKENS
    assert budget['archive_bytes'] > 8*99826*34816
    p['kvmem']['session_checkpoint_bytes'] *= 2
    with pytest.raises(ValueError,match='capacity'):qual.qualification_profile(p,18620)


def test_restore_proof_requires_each_request_and_mtp_and_recurrent_bytes(tmp_path):
    rows=[dict(request_id='client',response_id='response')]
    target=dict(request_id='response-0-12345678',byte_equal=True,working_attention_layers=17,
        mtp_attention_layers=1,target_kv_bytes=100,oracle_kind='native_terminal_window')
    state=dict(request_id=target['request_id'],recurrent_byte_equal=True)
    def write(t,s):
        (tmp_path/'verified_restores.jsonl').write_text(json.dumps(t)+'\n')
        (tmp_path/'verified_state_restores.jsonl').write_text(json.dumps(s)+'\n')
    write(target,state)
    assert len(qual.restore_evidence(tmp_path,rows)) == 1
    for change in [dict(request_id='other'),dict(byte_equal=False),dict(mtp_attention_layers=0),
                   dict(working_attention_layers=0),dict(target_kv_bytes=0),dict(oracle_kind='unknown')]:
        write(dict(target,**change),state)
        with pytest.raises(ValueError,match='Each continuation'):qual.restore_evidence(tmp_path,rows)
    write(target,dict(state,recurrent_byte_equal=False))
    with pytest.raises(ValueError,match='Each continuation'):qual.restore_evidence(tmp_path,rows)


def test_live_checkout_cannot_run_or_mutated_runtime_pass_verification(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='frozen'):qual.verify(tmp_path)
    frozen=tmp_path/'frozen';runtime=tmp_path/'runtime';frozen.mkdir();runtime.mkdir()
    (frozen/'entry.py').write_text('entry');(runtime/'entry.py').write_text('runtime')
    monkeypatch.setattr(qual,'ROOT',frozen)
    monkeypatch.setattr(qual.cap,'native_source_identity',lambda:{'version':'test'})
    p=profile();f=dict(tokens=[1]*99826,eos=0)
    qual.cap.save(tmp_path/'profile.json',p);qual.cap.save(tmp_path/'fixture.json',f)
    m=dict(kind='session_release_qualification_v1',source_sha256=qual.cap.hashes(frozen),
        runtime_sha256=qual.cap.hashes(runtime),profile_sha256=qual.cap.sha(tmp_path/'profile.json'),
        fixture_sha256=qual.cap.sha(tmp_path/'fixture.json'),native_source={'version':'test'},
        budget=qual.validate_fixture(f,p),concurrency=8,waves=5,short_context=qual.SHORT_CONTEXT,
        output_tokens=qual.OUTPUT_TOKENS,tool_tokens=qual.TOOL_TOKENS)
    qual.cap.save(tmp_path/'manifest.json',m)
    assert qual.verify(tmp_path) == m
    (runtime/'entry.py').write_text('mutated')
    with pytest.raises(ValueError,match='runtime'):qual.verify(tmp_path)
