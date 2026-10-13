import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('analyze_meank_profile',
    Path(__file__).resolve().parents[1] / 'scripts/analyze_meank_profile.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def event(category, name, start, duration, *, pid=1, tid=1, **args):
    return dict(ph='X', cat=category, name=name, ts=start, dur=duration,
                pid=pid, tid=tid, args=args)


def test_decode_clips_complete_cycles_maps_graph_launch_and_excludes_gpu_scopes():
    rows = [event('user_annotation', 'execute_context_0(0)_generation_8(32)', t, 8)
            for t in (0, 100, 200, 300)]
    rows += [event('user_annotation', 'kvmem::runner.execute', 100, 90),
             event('user_annotation', 'kvmem::target.graph', 110, 20),
             event('gpu_user_annotation', 'kvmem::target.graph', 115, 150, pid=0),
             event('cuda_runtime', 'cudaGraphLaunch', 115, 1, correlation=7),
             event('kernel', 'gemm', 120, 30, pid=0, correlation=7, device=0),
             event('kernel', 'attention', 140, 30, pid=0, tid=2, correlation=7, device=0),
             event('gpu_memcpy', 'DtoH', 290, 30, pid=0, correlation=7, device=0),
             event('cuda_runtime', 'cudaEventSynchronize', 180, 5),
             event('overhead', 'CUPTI', 190, 7)]
    result = analyzer.analyze_trace({'traceEvents': rows}, 'decode')
    assert result['window']['complete_cycles'] == 2
    assert (result['window']['start_us'], result['window']['end_us']) == (100, 300)
    assert result['gpu_timelines'] == [dict(device=0, interval_union_us=60,
        summed_activity_us=70, uncovered_interval_us=140)]
    assert result['gpu_attribution']['mapped_launch_events'] == 3
    assert result['gpu_attribution']['boundary_clipped_events'] == 1
    cpu = {row['stage']: row for row in result['cpu_stages']}
    assert cpu['kvmem::runner.execute']['exclusive_us'] == 70
    assert cpu['kvmem::target.graph']['calls'] == 1
    assert cpu['kvmem::target.graph']['inclusive_us'] == 20
    assert result['gpu_stages'][0]['stage'] == 'kvmem::target.graph'
    assert result['cuda_api_waits'][0]['summed_us'] == 5
    assert result['profiler_overhead'][0]['summed_us'] == 7


def test_prefill_limited_sample_preserves_unmapped_activity_and_thread_boundaries():
    rows = [event('user_annotation', 'execute_context_1(100)_generation_0(0)', 0, 100),
             event('user_annotation', 'kvmem::runner.execute', 0, 100),
             event('user_annotation', 'kvmem::authority.append', 10, 40, tid=2),
             event('cuda_runtime', 'cudaLaunchKernel', 20, 1, correlation=1),
             event('kernel', 'gemm', 30, 10, pid=0, correlation=1),
             event('gpu_memset', 'memset', 40, 10, pid=0, correlation=99)]
    result = analyzer.analyze_trace({'traceEvents': rows}, 'prefill')
    assert result['window']['limited_sample'] and result['window']['complete_cycles'] is None
    assert result['gpu_attribution']['unmapped_launch_events'] == 1
    assert {row['stage'] for row in result['gpu_stages']} == {'kvmem::runner.execute', 'unmapped_launch'}
    assert next(row for row in result['cpu_stages'] if row['stage'] == 'kvmem::runner.execute')['exclusive_us'] == 100
    with pytest.raises(ValueError, match='CPU kvmem'):
        analyzer.analyze_trace({'traceEvents': []})
