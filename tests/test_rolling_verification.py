"""The qualification oracle fails on state, KV, or accepted-prefix corruption."""
from types import SimpleNamespace as NS
import json

import pytest
import torch

from vllm_kvmem.rolling_verification import RollingPressureOracle


def oracle_fixture(monkeypatch, tmp_path):
    tensors = {'a/ssm/3': torch.arange(8).reshape(2, 4), 'a/num_accepted': torch.tensor([3])}
    bundle = NS(tensors=tensors)
    generation = [7]
    def validate(value):
        assert value is bundle
        if generation[0] != 7:
            raise ValueError('Request incarnation changed')
    monkeypatch.setattr('vllm_kvmem.native_state.NativeRecurrentState',
        lambda _: NS(borrow=lambda ids: bundle, validate=validate))
    expected = torch.arange(128, dtype=torch.uint8)
    actual = expected.clone()
    old_native = torch.zeros(256, dtype=torch.uint8)
    runner = NS(_kvmem_capture_runtime=NS(last_steps={'a': [dict(history_begin=61, accepted=3, scheduled=5)]},
                                         names=['target', 'mtp'], heads=1, dim=1),
        _kvmem_sessions=NS(_rebuilt_target_bytes=lambda rid, authority: expected.clone(),
                          _target_bytes=lambda rid, n: old_native if n==64 else actual),
        _kvmem_window_adapter=NS(offsets={'a':0}),
        _kvmem_probe_output=tmp_path, _kvmem_mtp_modules={'mtp': object()})
    job = NS(command=NS(request_id='a', groups=((0, NS(replay_tokens=0)),)),
        history=64, cancelled=False, selected_tokens=32, steps=0,
        authority=NS(raw_dtype=torch.float8_e4m3fn,
            memory=NS(_store=NS(working_records=lambda: [(1,1,0,1,0)]))),
        result=(None,None,NS(remaps=[])))
    return runner, job, tensors, actual, generation


def test_rolling_oracle_requires_accepted_prefix_and_checks_target_mtp(monkeypatch, tmp_path):
    runner, job, tensors, actual, generation = oracle_fixture(monkeypatch, tmp_path)
    oracle = RollingPressureOracle(runner, job)
    oracle.verify(job)
    row = json.loads((tmp_path/'verified_rolling_pressure.jsonl').read_text())
    assert row['target_kv_byte_equal'] and row['recurrent_byte_equal']
    assert row['mtp_attention_layers'] == 1 and row['previous_accepted_step']['accepted'] == 3
    assert row['replay_tokens'] == row['private_forward_steps'] == 0


@pytest.mark.parametrize('damage', ['state', 'counter', 'kv', 'incarnation'])
def test_rolling_oracle_rejects_corruption_without_success_record(monkeypatch, tmp_path, damage):
    runner, job, tensors, actual, generation = oracle_fixture(monkeypatch, tmp_path)
    oracle = RollingPressureOracle(runner, job)
    if damage == 'state': tensors['a/ssm/3'].add_(1)
    if damage == 'counter': tensors['a/num_accepted'].fill_(1)
    if damage == 'kv': actual[0] = 99
    if damage == 'incarnation': generation[0] = 8
    with pytest.raises((RuntimeError, ValueError)):
        oracle.verify(job)
    assert not (tmp_path/'verified_rolling_pressure.jsonl').exists()


def test_oracle_rejects_stale_publication_and_cancel_never_claims_target_proof(monkeypatch, tmp_path):
    runner, job, *_ = oracle_fixture(monkeypatch, tmp_path)
    job.history = 65
    with pytest.raises(RuntimeError, match='exact accepted-prefix'):
        RollingPressureOracle(runner, job)
    job.history = 64
    oracle = RollingPressureOracle(runner, job)
    job.cancelled = True
    runner._kvmem_sessions._rebuilt_target_bytes = lambda *_: pytest.fail('Cancelled new view is not public')
    oracle.verify(job)
    row = json.loads((tmp_path/'verified_rolling_pressure.jsonl').read_text())
    assert row['cancelled'] and row['recurrent_byte_equal'] and not row['target_kv_byte_equal']


def test_stable_rows_compare_original_native_bytes_while_moved_rows_use_fp8_rebuild(monkeypatch, tmp_path):
    runner, job, _, actual, _ = oracle_fixture(monkeypatch, tmp_path)
    old = runner._kvmem_sessions._target_bytes('a',64).reshape(4,64,1)
    old[:,0] = 201  # Original rotated K can differ from a raw-FP8 reconstruction.
    actual.reshape(4,32,1)[:,0] = 201
    job.result[2].remaps = [NS(block_id=1,skip=True,working_k_resident=True,
        from_base=0,to_base=0,n_tokens=1)]
    oracle = RollingPressureOracle(runner, job)
    oracle.verify(job)
    row = json.loads((tmp_path/'verified_rolling_pressure.jsonl').read_text())
    assert row['native_stable_tokens']==1 and row['target_kv_byte_equal']
    actual[0] = 202
    with pytest.raises(RuntimeError, match='target/MTP KV'):
        oracle.verify(job)


def test_dflash_rolling_oracle_checks_draft_bytes_before_public_verification(monkeypatch,tmp_path):
    runner,job,_,_,_=oracle_fixture(monkeypatch,tmp_path)
    cache=torch.arange(8,dtype=torch.float32).reshape(4,1,1,2)
    runner._kvmem_draft_contract=NS(method='dflash',sliding_window=3)
    runner._kvmem_last_batch=NS(req_ids=['a'],num_computed_tokens_np=[3])
    runner._kvmem_sessions.model_adapter=NS(_draft_views=lambda *_:
        {'draft':(cache,torch.tensor([1,2]),torch.tensor([0,0]))})
    runner._kvmem_mtp_modules={}
    oracle=RollingPressureOracle(runner,job)
    oracle.verify(job)
    row=json.loads((tmp_path/'verified_rolling_pressure.jsonl').read_text())
    assert row['draft_byte_equal'] and row['draft_layers']==1 and row['mtp_attention_layers']==0
    cache[1].add_(1)
    with pytest.raises(RuntimeError,match='committed draft KV'):oracle.verify(job)
