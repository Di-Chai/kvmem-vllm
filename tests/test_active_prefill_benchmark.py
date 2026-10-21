"""Shape fairness and input-accounting guards for native whole-model probes."""
import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
import benchmark_active_prefill as benchmark
from native_prefill_shape_probe import cap_at_boundary, case_for_request


def test_shape_families_cover_the_real_post_refresh_alignment():
    available = set()
    for chunks in benchmark.FAMILIES.values():
        end = 0
        for count in chunks:
            end += count
            available.add((count, end))
    assert (32, 32800) in available
    assert all((2048, end) in available for end in range(34848, 47137, 2048))
    assert (1952, 49088) in available
    assert (1984, 49088) in available
    assert (736, 33536) in available
    assert (22, 32790) in available
    assert (7, 32797) in available


def test_chunk_cap_never_increases_native_admission_and_decode_is_unchanged():
    chunks = benchmark.FAMILIES['steady']
    assert cap_at_boundary(32768, 2048, 49096, chunks) == 32
    assert cap_at_boundary(32768, 16, 49096, chunks) == 16
    assert cap_at_boundary(49096, 5, 49096, chunks) == 5
    assert cap_at_boundary(0, 2048, 32768, None) == 2048
    with pytest.raises(ValueError, match='full prompt'):
        cap_at_boundary(0, 2048, 49087, chunks)


def test_unknown_request_is_not_silently_timed_as_a_different_case():
    definitions = [dict(request_id='probe-a'), dict(request_id='probe-ab')]
    assert case_for_request('cmpl-probe-ab-0-uuid', definitions) == definitions[1]
    with pytest.raises(ValueError, match='Unknown or ambiguous'):
        case_for_request('cmpl-user-0-uuid', definitions)


def test_reconstruction_uses_committed_selected_length_not_total_history():
    rows = [dict(prefill_tokens=8), dict(prefill_tokens=2), dict(prefill_tokens=4)]
    events = [dict(history_begin=8, selected_tokens=4)]
    result = benchmark.reconstruct_geometry(rows, events, 14)
    assert [row['active_seq_len'] for row in result] == [8, 6, 10]
    assert [row['logical_seq_len'] for row in result] == [8, 10, 14]
    assert [row['offset'] for row in result] == [0, 4, 4]


def test_recorded_native_geometry_must_match_reconstructed_geometry():
    shape = dict(query_tokens=8, active_seq_len=8, logical_seq_len=8, offset=0, prefill=True)
    benchmark.reconstruct_geometry([dict(prefill_tokens=8, batch_geometry=[shape])], [], 8)
    wrong = dict(shape, active_seq_len=16)
    with pytest.raises(ValueError, match='disagrees'):
        benchmark.reconstruct_geometry([dict(prefill_tokens=8, batch_geometry=[wrong])], [], 8)
    with pytest.raises(ValueError, match='crossed a refresh'):
        benchmark.reconstruct_geometry([dict(prefill_tokens=8)], [dict(history_begin=4, selected_tokens=2)], 8)


def test_native_startup_passes_through_until_compile_warmup_completes(monkeypatch):
    import types
    import native_prefill_shape_probe as probe

    fake_output = types.SimpleNamespace(num_scheduled_tokens={'synthetic-0': 1, 'synthetic-1': 1})

    class NativeWorker:
        def execute_model(self, output):
            self.native_calls = getattr(self, 'native_calls', 0) + 1
            return 'native-result'

        def compile_or_warm_up_model(self):
            assert not getattr(self, '_shape_ready', False)
            assert self.execute_model(fake_output) == 'native-result'
            return 'compilation-times'

    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu_worker',
                        types.SimpleNamespace(Worker=NativeWorker))
    monkeypatch.delitem(probe.__dict__, 'NativeProbeWorker', raising=False)
    worker_type = probe.__getattr__('NativeProbeWorker')
    monkeypatch.delitem(probe.__dict__, 'NativeProbeWorker')
    worker = worker_type()
    worker._shape_timing = types.SimpleNamespace(poll=lambda: None)
    worker._shape_cases = [dict(request_id='allowed-probe')]
    assert worker.execute_model(fake_output) == 'native-result'
    assert worker.compile_or_warm_up_model() == 'compilation-times'
    assert worker._shape_ready
    assert worker.native_calls == 2
    with pytest.raises(ValueError, match='concurrent owners'):
        worker.execute_model(fake_output)
    with pytest.raises(ValueError, match='Unknown or ambiguous'):
        worker.execute_model(types.SimpleNamespace(num_scheduled_tokens={'cmpl-unexpected-0': 1}))
