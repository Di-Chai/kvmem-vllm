"""CPU-only qualification for bounded server injection and trace accounting."""
from argparse import Namespace
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import profile_prefill_nsight as profile
import analyze_prefill_nsight as analyze


def test_capture_is_api_bounded_and_only_wraps_server():
    command = profile.nsys_command('/usr/local/bin/nsys', '/tmp/trace', ['python', 'serve.py'])
    assert '--capture-range=cudaProfilerApi' in command
    assert '--capture-range-end=stop' in command and '--kill=none' in command
    assert '--cuda-graph-trace=node' in command and '--sample=none' in command
    assert command[-2:] == ['python', 'serve.py']
    assert not any(arg.startswith(('--delay=', '--duration=')) for arg in command)
    assert profile.profiler_config(8, 4) == dict(profiler='cuda', delay_iterations=8,
        max_iterations=4, detailed_trace_annotation=True)


@pytest.mark.parametrize('delay,limit', [(-1,4),(1,0),(1,65),(True,4),(0,True)])
def test_unbounded_or_invalid_profile_is_rejected(delay, limit):
    with pytest.raises(ValueError):
        profile.profiler_config(delay, limit)


@pytest.mark.parametrize('source', [{'NSYS_SESSION_ID':'dead'}, {'CUDA_INJECTION64_PATH':'dead'},
                                    {'LD_PRELOAD':'/tmp/libToolsInjection64.so'}])
def test_injected_controller_is_rejected(source):
    with pytest.raises(ValueError, match='outside Nsight'):
        profile.clean_environment(source)


@pytest.mark.parametrize('toolkit', ['/usr/local/cuda-13','/usr/local/cuda'])
def test_clean_environment_exposes_cuda_compiler_before_server_import(monkeypatch, toolkit):
    monkeypatch.setattr(Path, 'exists', lambda path: str(path) == toolkit+'/bin/nvcc')
    original = {'PATH':'/usr/bin', 'VLLM_USE_AOT_COMPILE':'0', 'CUDA_HOME':'/stale/cuda'}
    env = profile.clean_environment(original)
    assert env['CUDA_HOME'] == toolkit
    assert env['PATH'].split(os.pathsep) == [toolkit+'/bin',str(Path(sys.executable).parent),'/usr/bin']
    assert 'VLLM_USE_AOT_COMPILE' not in env
    assert original['PATH'] == '/usr/bin' and original['CUDA_HOME'] == '/stale/cuda'


def test_clean_environment_without_cuda_preserves_executable_and_system_path(monkeypatch):
    monkeypatch.setattr(Path, 'exists', lambda path: False)
    env = profile.clean_environment({'PATH':'/usr/bin'})
    assert env['PATH'].split(os.pathsep) == [str(Path(sys.executable).parent),'/usr/bin']


def test_cpu_prepare_freezes_same_input_and_detects_mutations(tmp_path):
    source = tmp_path / 'source'
    runtime = source / 'runtime'
    runtime.mkdir(parents=True)
    module = runtime / 'sample.py'
    module.write_text('frozen = True\n')
    profile.save(source / 'runtime-proof.json', {'files':{'sample.py':profile.sha(module)}})
    original = dict(prefill_chunk_tokens=2048, max_num_batched_tokens=16424, kvmem={})
    profile.save(source / 'profile.json', original)
    input_file = tmp_path / 'tokens.i32'
    input_file.write_bytes(bytes(32768*4))
    case = dict(input_path=str(input_file), row=dict(served_input_tokens=32768,
        served_sha256=profile.sha(input_file)), sampling={'max_tokens':128})
    profile.save(tmp_path / 'case.json', case)
    args = Namespace(source=source, case=tmp_path / 'case.json', root=tmp_path / 'probe', mode='native',
        port=18866, delay_iterations=8, max_iterations=4, startup_timeout=1200, request_timeout=300,
        nsys='/usr/local/bin/nsys')
    root = profile.prepare(args)
    assert profile.read(source / 'profile.json') == original
    assert profile.assert_frozen(root)['input_sha256'] == profile.sha(input_file)
    warm, traced = (profile.read(root / 'requests' / name / 'case.json') for name in ('warmup','traced'))
    assert warm['input_path'] == traced['input_path'] and warm['sampling'] == traced['sampling']
    assert traced['chunks'] == [2048]*16
    suite = profile.read(root / 'suite.json')
    modified = deepcopy(suite)
    modified['cases'][1]['chunks'] = [1024]*32
    profile.save(root / 'suite.json', modified)
    with pytest.raises(ValueError, match='suite.json'):
        profile.assert_frozen(root)
    profile.save(root / 'suite.json', suite)
    traced['sampling']['max_tokens'] = 4
    profile.save(root / 'requests/traced/case.json', traced)
    with pytest.raises(ValueError, match='Frozen case'):
        profile.assert_frozen(root)


def database(tmp_path, *, nvtx=True):
    path = tmp_path / 'trace.sqlite'
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
            globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,
            globalPid INTEGER,correlationId INTEGER,copyKind INTEGER,bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,
            correlationId INTEGER,nameId INTEGER);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,textId INTEGER,text TEXT);
    ''')
    db.executemany('INSERT INTO StringIds VALUES (?,?)', [(1,'_row_lse'),(2,'cudaLaunchKernel'),
        (3,'BatchPrefillWithPagedKVCacheKernel<KVMemAttention>'), (4,'cudaMemcpyAsync')])
    pid = 41*(1<<24)
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)', [
        (100,200,0,pid,1,3), (300,400,0,pid,2,1)])
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)',
               (150,250,0,pid,3,2,4096))
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)', [
        (50,60,pid+7,1,2), (280,310,pid+7,2,2), (80,90,pid+7,3,4)])
    if nvtx:
        db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)', (40,410,0,
            'execute_2048_context_1(sq2048sk18432sqsq4194304sqsk37748736)_generation_0(sq0sk0sqsq0sqsk0)'))
    db.commit()
    db.close()
    return path


def test_overlap_union_gap_correlation_and_shape_use_actual_rows(tmp_path):
    result = analyze.analyze(database(tmp_path))
    gpu = result['devices']['0']
    assert gpu['observed_span_ms'] == 300/1e6
    assert gpu['activity_union_ms'] == 250/1e6
    assert gpu['gaps_ms'] == 50/1e6
    assert gpu['categories']['attention_including_any_fused_stats']['summed_ms'] == 100/1e6
    assert gpu['categories']['native_stats_auxiliary']['summed_ms'] == 100/1e6
    assert gpu['correlated_activities'] == 3
    assert gpu['largest_gaps'][0]['next_launch_not_entered_ms'] == 30/1e6
    assert result['profiled_worker_shapes'][0]['logical_seq_len'] == 18432
    assert result['all_traced_steps_single_owner_2k_prefill']


def test_missing_shapes_or_gpu_work_are_not_zero_overhead_proof(tmp_path):
    path = database(tmp_path, nvtx=False)
    assert not analyze.analyze(path)['all_traced_steps_single_owner_2k_prefill']
    db = sqlite3.connect(path)
    db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
    db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY')
    db.commit(); db.close()
    with pytest.raises(ValueError, match='No CUDA activities'):
        analyze.analyze(path)


def test_shape_audit_checks_actual_offsets_and_exact_window(tmp_path):
    (tmp_path / 'requests/traced').mkdir(parents=True)
    (tmp_path / 'probe').mkdir()
    profile.save(tmp_path / 'requests/traced/response.json', {'response_id':'cmpl-traced'})
    row = dict(batch_geometry=[dict(request_id='cmpl-traced-0', query_tokens=2048,
        logical_seq_len=18432, active_seq_len=18432, prefill=True)])
    path = tmp_path / 'probe/model_step_latency.jsonl'
    path.write_text(json.dumps(row)+'\n')
    result = dict(profiled_worker_shapes=[dict(query_tokens=2048,logical_seq_len=18432)],
                  all_traced_steps_single_owner_2k_prefill=True)
    profile.verify_shapes(tmp_path, result, 'native', 1)
    assert result['actual_geometry_verified']
    row['batch_geometry'][0]['active_seq_len'] = 16384
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='actual active-window'):
        profile.verify_shapes(tmp_path, deepcopy(result), 'native', 1)


def test_generic_copy_names_are_not_misclassified_as_archive():
    assert analyze.kernel_category('vectorized_elementwise_kernel<copy>') == 'unattributed_kernel'
    assert analyze.kernel_category('_payload') == 'named_capture_pack'
