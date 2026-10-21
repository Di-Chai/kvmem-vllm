"""CPU qualification for a bounded warm pressure diagnostic, without CUDA."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prefill_pressure_probe as probe
import profile_prefill_nsight as helper


class Profiler:
    def __init__(self):
        self.is_running = False
        self.calls = []
    def start(self):
        self.calls.append('start'); self.is_running = True
    def step(self):
        self.calls.append('step')
    def stop(self):
        self.calls.append('stop'); self.is_running = False


def batch(computed=None, *, tokens=2048, owner='owner', prefill=True):
    return NS(scheduled_new_reqs=[], num_scheduled_tokens={} if computed is None else {owner:tokens},
        scheduled_cached_reqs=NS(req_ids=[] if computed is None else [owner],
            num_computed_tokens=[] if computed is None else [computed], is_context_phase=lambda _:prefill))


def gate(tmp_path):
    clock = [1.]
    backend = Profiler()
    result = probe.PressureProfiler(backend, helper.pressure_config(40960, 61440, 15, 262045),
                                   tmp_path, clock=lambda:clock[0])
    return result, backend, clock


def test_warmup_is_not_captured_and_empty_maintenance_does_not_end_range(tmp_path):
    p, backend, clock = gate(tmp_path)
    p.before_step(batch(40960)); p.step()
    assert not backend.calls
    p.start(); p.before_step(batch(38912)); p.step()
    assert not backend.calls
    p.before_step(batch(40960)); p.step()
    assert backend.calls == ['start','step']
    for _ in range(80):
        p.before_step(batch()); p.step()
    assert p.is_running and len(p.rows) == 1
    p.before_step(batch(61440))
    assert not p.is_running and p.reason == 'logical_end'
    p.stop(); p.before_step(batch(63488)); p.step()
    assert backend.calls.count('start') == backend.calls.count('stop') == 1
    assert helper.read(tmp_path/'pressure-range.json')['reason'] == 'logical_end'


def test_deadline_stops_only_at_worker_boundary_and_marks_failure(tmp_path):
    p, backend, clock = gate(tmp_path)
    p.start(); p.before_step(batch(40960))
    clock[0] += 16
    assert p.is_running  # No unsafe background-thread CUDA call.
    with pytest.raises(RuntimeError, match='deadline'):
        p.before_step(batch())
    assert backend.calls == ['start','stop'] and p.reason == 'deadline_exceeded'


@pytest.mark.parametrize('bad', [batch(43008, owner='other'), batch(43008, prefill=False)])
def test_wrong_owner_or_decode_cannot_be_silently_profiled(tmp_path, bad):
    p, backend, clock = gate(tmp_path)
    p.start(); p.before_step(batch(40960))
    with pytest.raises(RuntimeError):
        p.before_step(bad)
    assert not p.is_running and p.reason in ('owner_changed','unexpected_batch')


def test_partial_request_stop_does_not_claim_logical_completion(tmp_path):
    p, backend, clock = gate(tmp_path)
    p.start(); p.before_step(batch(40960)); p.stop()
    assert helper.read(tmp_path/'pressure-range.json')['reason'] == 'external_stop'
    with pytest.raises(RuntimeError, match='exactly one'):
        p.start()


@pytest.mark.parametrize('args', [(0,30720,15,262045),(40960,99999,15,262045),
                                  (40960,61440,16,262045),(40960,61440,15,61440)])
def test_pressure_capture_has_bounded_valid_geometry(args):
    with pytest.raises(ValueError):
        helper.pressure_config(*args)


def test_pressure_capture_can_include_last_coalesced_block_before_first_pressure():
    assert helper.pressure_config(30720, 63456, 15, 262045) == dict(
        begin_tokens=30720, end_tokens=63456, max_seconds=15)


def test_pressure_prepare_freezes_full_warmup_and_separate_probe(tmp_path):
    source = tmp_path/'source'; runtime = source/'runtime'; runtime.mkdir(parents=True)
    (runtime/'example.py').write_text('value = 1\n')
    helper.save(source/'runtime-proof.json', {'files':{'example.py':helper.sha(runtime/'example.py')}})
    helper.save(source/'profile.json', dict(prefill_chunk_tokens=2048, max_num_batched_tokens=16424,
                                          kvmem={'select_budget':32768,'gen_budget':16384}))
    tokens = tmp_path/'tokens.i32'; tokens.write_bytes(bytes(262045*4))
    helper.save(tmp_path/'case.json', dict(input_path=str(tokens),
        row=dict(served_input_tokens=262045, served_sha256=helper.sha(tokens)), sampling={'max_tokens':128}))
    args = Namespace(source=source, case=tmp_path/'case.json', root=tmp_path/'probe', mode='kvmem',
        port=18866, delay_iterations=8, max_iterations=4, startup_timeout=1200, request_timeout=300,
        nsys='/usr/local/bin/nsys', trace_kind='pressure', begin_tokens=40960, end_tokens=61440,
        pressure_max_seconds=15)
    root = helper.prepare(args)
    plan = helper.assert_frozen(root)
    assert plan['profiler']['max_iterations'] == plan['profiler']['delay_iterations'] == 0
    assert plan['phase_scopes'] and plan['pressure_range']['end_tokens'] == 61440
    for name in ('warmup','traced'):
        case = helper.read(root/'requests'/name/'case.json')
        assert case['input_path'] == str(tokens) and sum(case['chunks']) == 262045
    (root/'prefill_pressure_probe.py').write_text('changed\n')
    with pytest.raises(ValueError, match='prefill_pressure_probe'):
        helper.assert_frozen(root)


def test_worker_creates_lazy_profiler_without_start_and_closes_on_setup_failure(tmp_path, monkeypatch):
    import types
    module = types.ModuleType('vllm.profiler.wrapper')
    module.CudaProfilerWrapper = lambda config: Profiler()
    monkeypatch.setitem(sys.modules,'vllm.profiler.wrapper',module)
    monkeypatch.setenv('PREFILL_PRESSURE_RANGE',json.dumps(helper.pressure_config(40960,61440,15,262045)))
    monkeypatch.setenv('PREFILL_PHASE_PROBE_OUTPUT',str(tmp_path))
    class Base:
        def load_model(self):
            self.profiler=None; self.profiler_config=NS(); return 'loaded'
    class Worker(probe._PressureWorkerMixin,Base):
        pass
    worker=Worker()
    assert worker.load_model() == 'loaded'
    assert worker.profiler.profiler.calls == []
    worker.profiler.start()
    def fail(_):
        raise OSError('diagnostic setup failed')
    monkeypatch.setattr(probe,'install_pressure_scopes',fail)
    with pytest.raises(OSError,match='setup failed'):
        worker.execute_model(batch(40960))
    assert worker.profiler.profiler.calls == ['start','stop']
    assert worker.profiler.reason == 'worker_exception'


def pressure_fixture(tmp_path):
    for path in ('probe','requests/warmup','requests/traced','run/server/capture'):
        (tmp_path/path).mkdir(parents=True)
    helper.save(tmp_path/'requests/traced/response.json', {'response_id':'cmpl-traced'})
    helper.save(tmp_path/'requests/warmup/evidence.json', {'refresh':[{'selection':'pressure'}]})
    stages = ['suspended','assembly_queued','assembly_started','selection_prepared','assembly_submitted','ready']
    events = [dict(stage=stage,request_id='cmpl-traced-0',selection='pressure',command_id=9) for stage in stages]
    gate = dict(reason='logical_end', request_id='cmpl-traced-0',
                public_steps=[dict(computed=40960,tokens=2048)], pressure_events=events)
    helper.save(tmp_path/'probe/pressure-range.json',gate)
    shape = dict(request_id='cmpl-traced-0',query_tokens=2048, logical_seq_len=43008, active_seq_len=26624,prefill=True)
    (tmp_path/'run/server/capture/model_step_latency.jsonl').write_text(json.dumps({'batch_geometry':[shape]})+'\n')
    report = dict(profiled_worker_shapes=[dict(scheduled_tokens=2048,context_requests=1,
                 generation_requests=0,query_tokens=2048,logical_seq_len=43008)])
    db = sqlite3.connect(tmp_path/'trace.sqlite')
    db.executescript('''
      CREATE TABLE StringIds(id INTEGER,value TEXT);
      CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,textId INTEGER,text TEXT,globalTid INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER,
                                            globalPid INTEGER,correlationId INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,
                                            globalPid INTEGER,correlationId INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
    ''')
    pid=42*(1<<24); tid=pid+5
    db.execute('INSERT INTO StringIds VALUES(1,?)',('_assemble_split',))
    db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)',[
        (i*100,i*100+1,0,'prefill_pressure::'+stage+'::9',tid) for i,stage in enumerate(stages)])
    db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)',[
        (300,350,0,'prefill_phase::transfer.upload_from_reader',tid),
        (350,400,0,'prefill_phase::rebuild.assemble_split_layer',tid)])
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(310,330,1,2097152,?,1)',(pid,))
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(370,420,1,?,2)',(pid,))
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)',[(305,309,tid,1),(360,365,tid,2)])
    db.commit(); db.close()
    return report, gate


def test_pressure_audit_requires_complete_lifecycle_actual_shapes_and_gpu_causality(tmp_path):
    report,gate = pressure_fixture(tmp_path)
    helper.verify_pressure(tmp_path,report,{})
    assert report['actual_geometry_verified']
    assert report['pressure_verified']['correlated_bulk_h2d'] == 1
    assert report['profiled_worker_shapes'][0]['active_seq_len'] == 26624
    gate['pressure_events'].pop()
    helper.save(tmp_path/'probe/pressure-range.json',gate)
    with pytest.raises(ValueError,match='complete pressure'):
        helper.verify_pressure(tmp_path,deepcopy(report),{})


def test_pressure_audit_rejects_cpu_scope_without_launch_correlation(tmp_path):
    report,gate = pressure_fixture(tmp_path)
    db=sqlite3.connect(tmp_path/'trace.sqlite'); db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME'); db.commit(); db.close()
    with pytest.raises(ValueError,match='launch correlation'):
        helper.verify_pressure(tmp_path,report,{})
