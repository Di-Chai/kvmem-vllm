"""CPU-only checks: no torch import, CUDA allocation, synchronization or reset."""
import json
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.allocator_diagnostic import AllocatorDiagnostic, FIELDS, install


class Backend:
    def __init__(self):
        self.capture = False
        self.calls = 0
        self.identities = 0

    def capturing(self):
        return self.capture

    def identity(self):
        self.identities += 1
        return {'backend': 'test-native', 'device': 'no-GPU'}

    def statistics(self):
        self.calls += 1
        return {name: self.calls for name in FIELDS}


def batch(prefill=True):
    return NS(num_reqs=2, req_ids=['real-a', 'real-b'], is_prefilling_np=[prefill, prefill],
        num_scheduled_tokens=[2048, 32] if prefill else [5, 5],
        num_computed_tokens_np=[16384, 6144], num_tokens=2080 if prefill else 10,
        num_tokens_after_padding=2080 if prefill else 16)


def make(tmp_path, **kw):
    backend = Backend()
    return AllocatorDiagnostic(tmp_path/'log.jsonl', backend, **kw), backend


def records(d):
    return [json.loads(line) for line in d.output.read_text().splitlines()]


def test_default_off_does_not_construct_backend_or_touch_output(tmp_path, monkeypatch):
    monkeypatch.delenv('KVMEM_ALLOCATOR_DIAGNOSTIC', raising=False)
    monkeypatch.setattr('vllm_kvmem.allocator_diagnostic.TorchAllocator',
                        lambda *_: pytest.fail('Disabled diagnosis imported/initialized CUDA'))
    assert install(tmp_path, 'cuda') is None
    assert not list(tmp_path.iterdir())


def test_explicit_install_and_invalid_flag(tmp_path, monkeypatch):
    backend = Backend()
    monkeypatch.setenv('KVMEM_ALLOCATOR_DIAGNOSTIC', '1')
    d = install(tmp_path, 'cuda', backend=backend)
    assert d.backend is backend and d.phase_file == tmp_path/'allocator_phase.json'
    assert backend.calls == backend.identities == 0
    monkeypatch.setenv('KVMEM_ALLOCATOR_DIAGNOSTIC', 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        install(tmp_path, 'cuda', backend=backend)


def test_prefill_tokens_padding_phase_and_context_are_exact_host_values(tmp_path):
    marker = tmp_path/'phase.json'
    marker.write_text('{"phase":"cold_prepare"}')
    d, backend = make(tmp_path, phase_file=marker)
    d.begin_batch(batch(), sequence=129, offsets={'real-a': 8192})
    marker.write_text('{"phase":"warmup"}')
    d.mark('target.begin')
    d.child(request_id='real-a', owner_kind='prefill', layer='L3').mark('action.logits', rows=64, width=65536)
    d.end_batch()
    d.flush(force=True)
    data = records(d)
    row = next(r for r in data if r['event'] == 'batch')
    assert row['step_timing_sequence'] == 129 and row['num_tokens'] == row['num_tokens_after_padding'] == 2080
    assert row['local_context_cpu_upper_bound'] == [10240, 6176]
    assert row['request_ids'] == ['real-a', 'real-b']
    assert row['scheduled_tokens_cpu_upper_bound'] == [2048, 32]
    assert all(r['phase'] == 'cold_prepare' for r in data if r['event'] != 'summary')
    assert d.counts['phase_reads'] == 1 and backend.calls == 3 and backend.identities == 1


def test_graph_capture_never_reads_phase_allocator_or_writes_logs(tmp_path):
    marker = tmp_path/'phase.json'
    marker.write_text('{"phase":"warmup"}')
    d, backend = make(tmp_path, phase_file=marker)
    d.begin_batch(batch())
    d.mark('before.capture')
    before = backend.calls
    backend.capture = True
    d.mark('inside.capture')
    d.flush(force=True)
    d.end_batch()
    d.begin_batch(batch())
    assert backend.calls == before and d.counts['phase_reads'] == 1
    assert not d.output.exists()
    assert d.counts['graph_stage_skips'] == d.counts['graph_flush_skips'] == d.counts['graph_end_skips'] == d.counts['graph_batch_skips'] == 1
    backend.capture = False
    d.flush(force=True)
    assert any(r.get('stage') == 'before.capture' for r in records(d))
    assert not any(r.get('stage') == 'inside.capture' for r in records(d))


def test_dummy_inactive_and_sparse_decode_cannot_consume_prefill_records(tmp_path):
    d, backend = make(tmp_path, max_records=100, decode_interval=128)
    for _ in range(1000):
        d.mark('dummy.not_a_public_batch')
    assert backend.calls == 0 and d.retained == 0
    for _ in range(128):
        d.begin_batch(batch(False))
        d.mark('target.begin')
        d.end_batch()
    assert backend.calls == 2 and d.coverage['decode_interval_skips'] == 127
    d.begin_batch(batch(), sequence=129)
    d.mark('capture.raw.end')
    d.end_batch()
    d.flush(force=True)
    assert d.coverage['prefill_batches_selected'] == 1
    assert any(r.get('stage') == 'capture.raw.end' and r['step_timing_sequence'] == 129 for r in records(d))


def test_record_limit_skips_allocator_query_and_reports_final_drops_even_empty_buffer(tmp_path):
    d, backend = make(tmp_path, max_records=3)
    d.begin_batch(batch())
    for _ in range(4):
        d.mark('target.begin')
    assert backend.calls == 2 and d.retained == 3
    d.flush(force=True)
    for _ in range(2):
        d.mark('target.begin')
    d.flush(force=True)
    assert records(d)[-1]['counts']['record_limit_drops'] == 4
    assert records(d)[-1]['coverage']['prefill_snapshots_dropped'] == 4


def test_buffer_bound_reports_drop_and_flushes_at_batch_boundary(tmp_path):
    d, backend = make(tmp_path, buffer_records=4)
    d.begin_batch(batch())
    for _ in range(8):
        d.mark('action.softmax')
    assert len(d.buffer) == 4 and backend.calls == 3
    d.end_batch()
    assert not d.buffer and d.counts['buffer_limit_drops'] == 6
    assert d.coverage['prefill_batches_completed'] == 1


@pytest.mark.parametrize('marker', ['bad json', '{"phase":"invented"}', 'x'*4097])
def test_missing_or_invalid_phase_is_unknown_not_inferred_from_walltime(tmp_path, marker):
    p = tmp_path/'phase.json'
    p.write_text(marker)
    d, _ = make(tmp_path, phase_file=p)
    d.begin_batch(batch())
    assert d.current['phase'] == 'unknown' and d.counts['phase_unknown'] == 1


def test_unavailable_async_allocator_fields_remain_unknown_and_no_peak_delta_is_invented(tmp_path):
    d, backend = make(tmp_path)
    backend.statistics = lambda: {'allocated_bytes.all.current': 200, 'allocated_bytes.all.peak': 1000}
    d.begin_batch(batch())
    d.child(layer='L1', rows=64).child(rows=32).mark('action.logits')
    d.flush(force=True)
    row = next(r for r in records(d) if r['event'] == 'allocator')
    assert row['rows'] == 32 and row['memory']['inactive_split_bytes.all.current'] is None
    assert row['memory']['allocated_bytes.all.peak'] == 1000
    assert 'exclusive_bytes' not in row and 'gpu_idle' not in row


def test_allocator_errors_propagate_and_background_threads_cannot_sample(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    d, backend = make(tmp_path)
    d.begin_batch(batch())
    def failure():
        raise RuntimeError('allocator error')
    backend.statistics = failure
    with pytest.raises(RuntimeError, match='allocator error'):
        d.mark('target.begin')
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='engine thread'):
            pool.submit(d.mark, 'background').result()
