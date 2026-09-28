import gzip
import json

import pytest
from vllm_kvmem.profiling import native_profiler_config, summarize_trace


def test_native_profile_configuration_is_bounded_and_keeps_frontend_out(tmp_path):
    pytest.importorskip('vllm')
    from vllm.config import ProfilerConfig
    config = native_profiler_config(tmp_path, 4, 130)
    native = ProfilerConfig(**config)
    assert native.max_iterations == 4 and native.delay_iterations == 130
    assert native.ignore_frontend and not native.torch_profiler_with_stack
    assert native.torch_profiler_record_shapes
    with pytest.raises(ValueError): native_profiler_config(tmp_path, 0)
    with pytest.raises(ValueError): native_profiler_config(tmp_path, 33)


def test_profile_summary_distinguishes_overlapping_kernel_sum_from_timeline(tmp_path):
    trace = {'traceEvents': [
        {'ph': 'X', 'cat': 'kernel', 'pid': 0, 'name': 'gemm', 'ts': 0, 'dur': 10},
        {'ph': 'X', 'cat': 'kernel', 'pid': 0, 'name': 'copy_helper', 'ts': 5, 'dur': 10},
        {'ph': 'X', 'cat': 'kernel', 'pid': 0, 'name': 'gemm', 'ts': 20, 'dur': 5},
        {'ph': 'X', 'cat': 'cpu_op', 'pid': 1, 'name': 'host', 'ts': 0, 'dur': 1000},
    ]}
    path = tmp_path / 'worker.pt.trace.json.gz'
    with gzip.open(path, 'wt') as stream: json.dump(trace, stream)
    result = summarize_trace(path)
    assert not result['performance_eligible']
    assert result['kernel_count'] == 3 and result['summed_kernel_us'] == 25
    assert result['timelines'] == [dict(trace_device=0, kernel_interval_union_us=20,
                                      first_to_last_kernel_us=25, non_kernel_interval_us=5)]
    assert result['top_kernels'][0]['name'] == 'gemm' and result['top_kernels'][0]['calls'] == 2
    path = tmp_path / 'empty.json'
    path.write_text('{"traceEvents": []}')
    with pytest.raises(ValueError, match='no GPU kernels'): summarize_trace(path)
