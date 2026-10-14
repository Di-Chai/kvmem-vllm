import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('profile_elastic_decode',
    Path(__file__).resolve().parents[1] / 'scripts/profile_elastic_decode.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def config():
    return {'context_mode': 'meank', 'draft_tokens': 3, 'profiler_config': {
        'profiler': 'torch', 'torch_profiler_dir': '/tmp/trace', 'ignore_frontend': True,
        'max_iterations': 80, 'delay_iterations': 2}}


@pytest.mark.parametrize('change', [
    {'max_iterations': 0}, {'max_iterations': -1}, {'max_iterations': True},
    {'max_iterations': 2.5}, {'delay_iterations': -1}, {'delay_iterations': True},
    {'warmup_iterations': 2}, {'wait_iterations': 1}, {'ignore_frontend': False},
])
def test_unsupported_collection_fails_before_workload_or_output_directory(tmp_path, monkeypatch, change):
    cfg = config()
    cfg['profiler_config'].update(change)
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps({'config': cfg}))
    args = NS(launch=launch, output=tmp_path / 'output', concurrency=8, output_tokens=4096)
    monkeypatch.setattr(collector.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Started workload'))
    with pytest.raises(ValueError):
        collector.run(args)
    assert not args.output.exists()


def test_headroom_scales_with_delay_iterations_and_largest_accepted_draft_batch():
    cfg = config()
    cfg['draft_tokens'] = 7
    cfg['profiler_config'].update(max_iterations=100, delay_iterations=10)
    _, headroom = collector.validate_collection(cfg, NS(concurrency=8, output_tokens=4096))
    assert headroom == 512 + 112 * 8
    # This would pass the old fixed 512-token margin despite being too close
    # to completion for the configured maximum number of accepted candidates.
    progress = {'running': 8, 'output_tokens': [3000] * 8}
    assert not collector.all_decoding(progress, 8, 4096, headroom)
    progress['output_tokens'] = [1024] * 8
    assert collector.all_decoding(progress, 8, 4096, headroom)
    with pytest.raises(ValueError, match='output_tokens'):
        collector.validate_collection(cfg, NS(concurrency=8, output_tokens=1024))


@pytest.mark.parametrize('mode,method,depth', [('meank', 'mtp', 2), ('meank', 'mtp', 3),
    ('meank', 'dflash', 7), ('full', 'mtp', 3)])
def test_decode_only_accepts_native_and_bounded_methods_with_finite_headroom(mode, method, depth):
    cfg = config()
    cfg.update(context_mode=mode, draft_method=method, draft_tokens=depth, mtp_k=depth)
    _, headroom = collector.validate_collection(cfg, NS(phase='decode', concurrency=8, output_tokens=4096))
    assert headroom == 512 + 84 * (depth + 1)
    if mode == 'full':
        with pytest.raises(ValueError, match='requires MeanK'):
            collector.validate_collection(cfg, NS(phase='both', concurrency=8, output_tokens=4096))


def trace_file(tmp_path, names, *, compressed=True, with_kernel=True):
    rows = [dict(cat='user_annotation', ph='X', name=name, ts=i * 100, dur=20)
            for i, name in enumerate(names)]
    # GPU annotation duplicates must not double-count native iterations.
    rows += [dict(cat='gpu_user_annotation', ph='X', name=names[0], ts=10, dur=50)]
    if with_kernel:
        rows += [dict(cat='kernel', ph='X', name='gemm', ts=15, dur=5)]
    data = json.dumps({'traceEvents': rows}).encode()
    path = tmp_path / ('worker.pt.trace.json.gz' if compressed else 'worker.pt.trace.json')
    path.write_bytes(gzip.compress(data) if compressed else data)
    return path


@pytest.mark.parametrize('compressed', [False, True])
def test_complete_trace_checks_actual_cpu_batches_and_ignores_gpu_annotation_duplicates(tmp_path, compressed):
    path = trace_file(tmp_path, ['execute_context_0(0)_generation_8(32)',
        'execute_32_context_0(sq0sk0sqsq0sqsk0)_generation_8(sq32sk4096sqsq128sqsk16384)'],
        compressed=compressed)
    result = collector.validate_trace(path, concurrency=8, iterations=2)
    assert result['native_cpu_iterations'] == 2 and result['decode_concurrency'] == 8
    assert result['complete_json'] and result['cuda_kernels_present']
    with pytest.raises(RuntimeError, match='expected 3'):
        collector.validate_trace(path, concurrency=8, iterations=3)


@pytest.mark.parametrize('wrong_batch', [
    'execute_context_1(1024)_generation_7(28)',
    'execute_context_0(0)_generation_7(28)',
])
def test_mixed_or_dwindling_concurrency_is_rejected_even_with_complete_trace(tmp_path, wrong_batch):
    path = trace_file(tmp_path, ['execute_context_0(0)_generation_8(32)', wrong_batch])
    with pytest.raises(RuntimeError, match='prefill or a different concurrency'):
        collector.validate_trace(path, concurrency=8, iterations=2)


def test_partial_export_and_missing_cuda_data_cannot_pass(tmp_path):
    path = trace_file(tmp_path, ['execute_context_0(0)_generation_8(32)'])
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(RuntimeError, match='incomplete or invalid'):
        collector.validate_trace(path, concurrency=8, iterations=1)
    path = trace_file(tmp_path, ['execute_context_0(0)_generation_8(32)'], with_kernel=False)
    with pytest.raises(RuntimeError, match='CUDA kernels'):
        collector.validate_trace(path, concurrency=8, iterations=1)
