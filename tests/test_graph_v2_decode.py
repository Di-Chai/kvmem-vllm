"""Independent negative gates for the final fixed-output diagnostic."""
import array
import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('graph_v2_decode',Path(__file__).parents[1]/'scripts/qualify_graph_v2_decode.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_eight_sessions_cannot_hide_a_duplicate_or_missing_native_request():
    sessions = [f's-{index}' for index in range(8)]
    responses = [dict(client_index=index,response_id=f'r-{index}') for index in range(8)]
    gate.validate_identities(sessions,responses)
    for invalid in (sessions[:-1],sessions+[sessions[0]],sessions[:-1]+[sessions[0]]):
        with pytest.raises(ValueError):gate.validate_identities(invalid,responses)
    for change in ({'response_id':None},{'response_id':'r-0'},{'client_index':0}):
        damaged = [dict(row) for row in responses]
        damaged[-1].update(change)
        with pytest.raises(ValueError):gate.validate_identities(sessions,damaged)


def test_relative_clock_requires_one_consistent_monotonic_origin():
    batch = dict(native_samples=[dict(at=2.,absolute_monotonic_at=1002.),dict(at=4.,absolute_monotonic_at=1004.)],
        steady_decode_window=dict(start=3.,end=33.))
    assert gate.absolute_window(batch)==(1003.,1033.)
    batch['native_samples'][-1]['absolute_monotonic_at'] += .01
    with pytest.raises(ValueError):gate.absolute_window(batch)
    batch['native_samples'] = [dict(at=0.,absolute_monotonic_at=float('nan'))]
    with pytest.raises(ValueError):gate.absolute_window(batch)


def test_fixture_preserves_complete_original_input_and_shared_window_gate(tmp_path):
    path = tmp_path/'input.i32'
    path.write_bytes(array.array('i',range(64)).tobytes())
    fixture = dict(schema=gate.SCHEMA,concurrency=8,output_tokens=16384,input_tokens=64,
        query_span=[60,64],min_window_seconds=30,min_native_coverage=.9,input_i32=str(path),
        input_i32_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    profile = dict(max_model_len=65536,kvmem=dict(select_budget=16,gen_budget=16384))
    # The compressed-history test must exceed K+G; this tiny fixture is rejected.
    with pytest.raises(ValueError,match='compression'):gate.validate_fixture(fixture,profile)
    fixture.update(input_tokens=16416,query_span=[16400,16416])
    path.write_bytes(array.array('i',range(16416)).tobytes())
    fixture['input_i32_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    gate.validate_fixture(fixture,profile)
    for field,value in [('min_window_seconds',2),('min_native_coverage',.1),('query_span',[0,16417]),('output_tokens',8192)]:
        with pytest.raises(ValueError):gate.validate_fixture({**fixture,field:value},profile)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError,match='input changed'):gate.validate_fixture(fixture,profile)


@pytest.mark.parametrize('failed', [False, True])
def test_completed_streams_retire_before_strict_failure_or_success(failed):
    sessions = [f's-{index}' for index in range(8)]
    responses = [dict(client_index=index,response_id=f'r-{index}') for index in range(8)]
    report = dict(status='failed' if failed else 'passed',batches=[{'steady_decode_window':{}}])
    events = []
    def retire(response_id):
        events.append(response_id)
        return dict(history_end=262101+16384-1)
    def release():
        events.append('released')
        return {'archive':{'retiring_requests':0}}
    args = (int(failed),report,sessions,responses,262101,retire,release)
    if failed:
        with pytest.raises(RuntimeError,match='Exact shared C8'):gate.retire_completed_streams(*args)
    else:
        batch,released = gate.retire_completed_streams(*args)
        assert batch is report['batches'][0] and released['archive']['retiring_requests']==0
    assert events==[f'r-{index}' for index in range(8)]+['released']


@pytest.mark.parametrize('damage', ['history','checkpoint','retire'])
def test_bad_terminal_or_retirement_still_releases_and_preserves_failure(damage):
    sessions = [f's-{index}' for index in range(8)]
    responses = [dict(client_index=index,response_id=f'r-{index}') for index in range(8)]
    released = []
    def retire(response_id):
        if damage=='retire':raise TimeoutError('Missing terminal')
        return dict(history_end=278483 if damage=='history' else 278484,
            checkpoint_tokens=32 if damage=='checkpoint' else 0)
    with pytest.raises((RuntimeError,TimeoutError)):
        gate.retire_completed_streams(0,dict(status='passed',batches=[{}]),sessions,responses,
            262101,retire,lambda: released.append(True))
    assert released==[True]
