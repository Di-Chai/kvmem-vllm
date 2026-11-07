"""Opt-in destination byte verification catches corruption before swap ACK."""
from dataclasses import replace
import hashlib
import json
import os

import pytest
import torch

from vllm_kvmem.request_checkpoint import CacheRows, capture_exact, restore_exact, verify_exact
from vllm_kvmem.request_swap_worker import RequestSwapWorker
from test_request_checkpoint import fixture as cache_fixture, native_fixture
from test_request_swap_worker import fixture as worker_fixture, command
from test_request_swap_worker_ssd import ExactAdapter, eventually, drop_all


def byte_digest(tensor):
    view = memoryview(tensor.numpy()).cast('B')
    try:
        return hashlib.sha256(view).hexdigest()
    finally:
        view.release()


def prepared():
    bindings, tensors = cache_fixture()
    def fp8(rows, slot=0):
        values = list(bindings(rows, slot))
        for i in (0, 1):
            values[i] = replace(values[i], tensor=values[i].tensor.view(torch.float8_e4m3fn))
        return values
    for tensor in tensors[1:3]:
        raw = tensor.view(torch.uint8)
        raw.copy_(torch.arange(raw.numel()).remainder(256).to(torch.uint8).view(raw.shape))
    saved = capture_exact(('a', 2, 7), fp8((1, 5)), max_bytes=8192)
    destination = fp8((6, 2), 1)
    assert restore_exact(saved, saved.identity, destination).query()
    return saved, destination


@pytest.mark.parametrize('chunk', [1, 17, 127, 1024**2])
def test_strided_fp8_nan_and_recurrent_bytes_verify_in_bounded_logical_chunks(monkeypatch, chunk):
    saved, destination = prepared()
    allocations = []
    original = torch.empty
    def observe(size, *args, **kwargs):
        allocations.append((size, kwargs.get('device')))
        return original(size, *args, **kwargs)
    monkeypatch.setattr(torch, 'empty', observe)
    report = verify_exact(saved, saved.identity, destination, chunk_bytes=chunk)
    assert report['bytes'] == saved.bytes and report['sha256'] == byte_digest(saved.payload)
    assert len(report['schemas']) == len(saved.schemas)
    assert report['readback_peak_bytes'] == min(chunk, saved.bytes)
    assert allocations == [(min(chunk, saved.bytes), 'cpu')]
    assert report['chunks'] >= len(destination)


@pytest.mark.parametrize('binding_index', range(5))
def test_corrupted_key_value_gdn_or_accepted_counter_fails_without_mutating_source(binding_index):
    saved, destination = prepared()
    before = saved.payload.clone()
    binding = destination[binding_index]
    raw = binding.tensor[binding.rows[0]]
    raw = (raw.reshape(1) if raw.ndim == 0 else raw).view(torch.uint8)
    index = (0,)*raw.ndim
    raw[index] = int(raw[index])+1
    with pytest.raises(RuntimeError, match='verification mismatch') as failure:
        verify_exact(saved, saved.identity, destination, chunk_bytes=17)
    assert binding.name in str(failure.value) and 'payload_byte=' in str(failure.value)
    assert torch.equal(saved.payload, before)


def test_schema_validation_precedes_readback_allocation(monkeypatch):
    saved, destination = prepared()
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: pytest.fail('readback allocated before validation'))
    with pytest.raises(ValueError, match='omitted'):
        verify_exact(saved, saved.identity, destination[:-1])
    with pytest.raises(ValueError, match='Stale'):
        verify_exact(saved, ('a', 3, 7), destination)
    with pytest.raises(ValueError, match='chunk'):
        verify_exact(saved, saved.identity, destination, chunk_bytes=4*1024**2+1)


class DiagnosticAdapter(ExactAdapter):
    def verify_restore(self, snapshot, request_id, operation_id, block_ids):
        self.calls.append(('verify', request_id))
        index, generation = (0, 2) if request_id == 'a' else (1, 3)
        return verify_exact(snapshot, (request_id, generation, operation_id),
            [CacheRows('raw', self.restored, (index,))], chunk_bytes=5)


def make_worker(tmp_path, monkeypatch, *, ssd, enabled=True):
    monkeypatch.delenv('KVMEM_VERIFY_REQUEST_SWAP', raising=False)
    prior, _, _, calls = worker_fixture()
    if enabled:
        monkeypatch.setenv('KVMEM_VERIFY_REQUEST_SWAP', '1')
    prior.runner._kvmem_probe_output = tmp_path
    adapter = DiagnosticAdapter(calls)
    worker = RequestSwapWorker(prior.runner, 32, adapter=adapter,
        checkpoint_ssd_bytes=128 if ssd else 0, checkpoint_directory=tmp_path/'ssd')
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    if ssd:
        peer = worker.execute(command(request_id='b', history_end=128, command_id=20, operation_id=2))
        eventually(peer.query)
        assert worker.checkpoints['a'].stored.state == 'disk'
    return worker, adapter, calls


@pytest.mark.parametrize('ssd', [False, True])
def test_worker_verifies_after_h2d_and_before_source_release_with_one_success_receipt(tmp_path, monkeypatch, ssd):
    worker, adapter, calls = make_worker(tmp_path, monkeypatch, ssd=ssd)
    job = worker.execute(command('restore'))
    eventually(lambda: (job.query(), ('restore', 'a') in calls)[1])
    path = tmp_path/'request_swap_verified.jsonl'
    assert not path.exists() and ('verify', 'a') not in calls
    held = worker.checkpoints['a']
    source = held.stored.snapshot if ssd else held.snapshot
    digest, size = byte_digest(source.payload), source.bytes
    source = None
    adapter.restore_event.done = True
    assert job.query() and job.query()
    records = [json.loads(row) for row in path.read_text().splitlines()]
    assert len(records) == 1
    report = records[0]
    assert report['identity'] == ['a', 2, 1] and report['history_end'] == 160
    assert report['sha256'] == digest and report['bytes'] == size
    assert report['source_tier'] == ('ssd' if ssd else 'cpu')
    assert worker.snapshot()['verification_successes'] == 1
    if ssd:
        assert held.stored.snapshot is None and held.stored.disk is not None
    drop_all(worker)


@pytest.mark.parametrize('fault', ['corruption', 'log_failure'])
def test_worker_failed_verification_or_receipt_keeps_checkpoint_and_rejects_drop(tmp_path, monkeypatch, fault):
    worker, adapter, calls = make_worker(tmp_path, monkeypatch, ssd=True)
    job = worker.execute(command('restore'))
    eventually(lambda: (job.query(), ('restore', 'a') in calls)[1])
    if fault == 'corruption':
        adapter.restored[0, 0] += 1
    else:
        worker.verification_path = tmp_path  # Appending to a directory must fail.
    adapter.restore_event.done = True
    with pytest.raises((RuntimeError, OSError)):
        job.query()
    held = worker.checkpoints['a']
    assert held.phase == held.stored.state == 'restoring'
    assert held.stored.snapshot is not None and held.stored.disk is not None
    assert held.restore is not None and worker.restore_count == 0
    assert worker.snapshot()['verification_failures'] == 1
    assert not (tmp_path/'request_swap_verified.jsonl').exists()
    with pytest.raises(ValueError, match='completed'):
        worker.execute(command('drop'))
    # Data/source ownership survived failure; repair the diagnostic condition
    # and prove the same completion can verify without another table restore.
    adapter.restored[0].copy_(adapter.rows[0])
    worker.verification_path = tmp_path/'request_swap_verified.jsonl'
    assert job.query()
    assert calls.count(('restore', 'a')) == 1
    drop_all(worker)


def test_default_flag_off_skips_readback_and_creates_no_receipt(tmp_path, monkeypatch):
    worker, adapter, calls = make_worker(tmp_path, monkeypatch, ssd=False, enabled=False)
    adapter.restore_event.done = True
    assert worker.execute(command('restore')).query()
    assert ('verify', 'a') not in calls and not (tmp_path/'request_swap_verified.jsonl').exists()
    drop_all(worker)


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
                    reason='Opt-in CUDA destination byte verification')
def test_cuda_native_destination_verification_covers_fp8_mtp_and_gdn_bytes():
    adapter, pages, tensors = native_fixture('cuda')
    saved = adapter.capture('a', 4, pages, max_bytes=16384)
    saved.completion.synchronize()
    replacement = ((2, 4), (8,), (9, 10))
    for table, row in zip(adapter.runner.block_tables.block_tables, replacement):
        table.gpu[0] = torch.tensor(row, device='cuda')
    restored = adapter.restore(saved, 'a', 4, replacement)
    restored.synchronize()
    result = adapter.verify_restore(saved, 'a', 4, replacement)
    assert result['sha256'] == byte_digest(saved.payload)
    assert result['bytes'] == saved.bytes and len(result['schemas']) == 7
    tensors[1][8, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match='verification mismatch'):
        adapter.verify_restore(saved, 'a', 4, replacement)
    adapter.release('a')
