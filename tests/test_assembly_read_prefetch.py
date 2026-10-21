"""CPU scheduling qualification: actual archive jobs and fixed staging, fake CUDA."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import packed_assembly as packed
from vllm_kvmem import transfer
from vllm_kvmem import window_assembly as window
from vllm_kvmem.memory import Phase
from vllm_kvmem.memory_storage import MemoryArchive
from test_authority import setup, payload, drain


class FakeEvent:
    def __init__(self, complete=False, operations=None):
        self.complete = complete
        self.operations = operations if operations is not None else []

    def query(self):
        return self.complete

    def record(self, stream):
        self.operations.append(('event', self, stream))

    def synchronize(self):
        self.operations.append(('event_sync', self))
        self.complete = True


class FakeStream:
    def __init__(self, operations):
        self.operations = operations
        self.on_sync = None

    def wait_event(self, event):
        self.operations.append(('wait', event, self))

    def synchronize(self):
        self.operations.append(('stream_sync', self))
        if self.on_sync is not None:
            self.on_sync()


class DeviceBytes:
    device = torch.device('cuda:0')
    dtype = torch.uint8

    def __init__(self, data, operations):
        self.data, self.operations = data, operations
        self.fail_copy = False

    def __getitem__(self, item):
        return DeviceBytes(self.data[item], self.operations)

    def numel(self):
        return self.data.numel()

    def is_contiguous(self):
        return True

    def flatten(self):
        return self

    def copy_(self, source, *, non_blocking):
        self.operations.append(('copy', self, source))
        if self.fail_copy:
            raise OSError('DMA enqueue failure')
        self.data.copy_(source)
        return self

    def record_stream(self, stream):
        self.operations.append(('retain', self, stream))


@pytest.fixture
def cuda(monkeypatch):
    operations, allocations = [], []
    current = FakeStream(operations)
    state = SimpleNamespace(complete=False, operations=operations, allocations=allocations,
                            current=current)
    real_empty = torch.empty

    def allocate(*args, **kwargs):
        device = torch.device(kwargs.get('device', 'cpu'))
        if device.type == 'cuda':
            result = DeviceBytes(real_empty(*args, **{**kwargs, 'device': 'cpu'}), operations)
        else:
            kwargs.pop('pin_memory', None)
            result = real_empty(*args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **_: FakeStream(operations))
    monkeypatch.setattr(torch.cuda, 'Event', lambda: FakeEvent(state.complete, operations))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *_: current)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    state.pool = transfer.PackedTransfers(128, 2, torch.device('cuda:0'))
    return state


def ready_job(value=None, error=None):
    future = Future()
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
    return future


def test_fill_dma_and_consumer_fences_are_distinct_without_more_buffers(cuda):
    pool = cuda.pool
    first_done = Future()
    first = pool.prepare_from_reader(16, lambda buffer: first_done)
    second = pool.prepare_from_reader(16, lambda buffer: ready_job(buffer.fill_(2)))
    assert pool.prepare_from_reader(16, lambda _: pytest.fail('busy slot callback')) is None
    assert not pool.close() and len(pool.buffers) == 2
    destination = DeviceBytes(torch.empty(16, dtype=torch.uint8), cuda.operations)
    gpu_consumer = FakeEvent(False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        upload = executor.submit(pool.upload_prepared, first, destination, ready=gpu_consumer)
        with pytest.raises(TimeoutError):
            upload.result(timeout=.02)
        assert not any(op[0] == 'copy' for op in cuda.operations)
        first.payload.fill_(7)
        first_done.set_result(first.payload)
        dma = upload.result(timeout=2)
    assert torch.equal(destination.data, torch.full((16,), 7, dtype=torch.uint8))
    assert ('wait', gpu_consumer, pool.stream) in cuda.operations
    assert not gpu_consumer.complete
    # H2D, not CPU completion, controls reuse of its source pinned slot.
    assert pool.prepare_from_reader(16, lambda _: pytest.fail('pending DMA')) is None
    dma.event.complete = True
    reused = pool.prepare_from_reader(16, lambda buffer: ready_job(buffer))
    assert reused.slot == first.slot
    # CPU reuse does not grant GPU reuse: the destination's own event is still
    # explicitly passed to the copy stream on the subsequent upload.
    pool.discard_prepared(reused)
    pool.discard_prepared(second)
    assert pool.close() and not pool.preparing
    with pytest.raises(ValueError, match='Stale'):
        pool.discard_prepared(first)


@pytest.mark.parametrize('failure', ['declined', 'begin', 'fill', 'destination', 'dma'])
def test_fill_and_enqueue_failures_retire_only_after_their_own_writer(cuda, failure):
    pool = cuda.pool
    if failure == 'declined':
        assert pool.prepare_from_reader(16, lambda _: None) is None
    elif failure == 'begin':
        def begin(_):
            raise OSError('submission failed')
        with pytest.raises(OSError):
            pool.prepare_from_reader(16, begin)
    else:
        ticket = pool.prepare_from_reader(16, lambda buffer:
            ready_job(error=OSError('fill failed')) if failure == 'fill' else ready_job(buffer))
        destination = DeviceBytes(torch.empty(8 if failure == 'destination' else 16,
                                               dtype=torch.uint8), cuda.operations)
        destination.fail_copy = failure == 'dma'
        if failure == 'dma':
            pool.stream.on_sync = lambda: pytest.fail('slot released before DMA fence') if not pool.preparing else None
        with pytest.raises((ValueError, OSError)):
            pool.upload_prepared(ticket, destination)
        if failure == 'dma':
            assert any(op[0] == 'stream_sync' for op in cuda.operations)
    assert not pool.preparing and not pool._prepared and not pool.pending
    assert pool.close()


@pytest.mark.parametrize('prepared', [False, True])
def test_dma_and_failure_fence_double_failure_quarantines_both_endpoints(cuda, prepared):
    pool = cuda.pool
    destination = DeviceBytes(torch.empty(16, dtype=torch.uint8), cuda.operations)
    destination.fail_copy = True
    consumer = FakeEvent()
    def broken_fence():
        raise OSError('DMA fence also failed')
    pool.stream.on_sync = broken_fence
    with pytest.raises(OSError, match='enqueue'):
        if prepared:
            ticket = pool.prepare_from_reader(16, lambda b: ready_job(b.fill_(3)))
            pool.upload_prepared(ticket, destination, ready=consumer)
        else:
            pool.upload_from_reader(16, destination, lambda b: b.fill_(3), ready=consumer)
    assert pool.poisoned and pool.preparing and not pool.close()
    poisoned = next(iter(pool._prepared.values()))
    assert poisoned.state == 'poisoned' and destination in poisoned.owners and consumer in poisoned.owners
    assert len(pool.buffers) == 2
    with pytest.raises(RuntimeError, match='unfenced'):
        pool.prepare_from_reader(16, lambda _: pytest.fail('poisoned callback'))
    with pytest.raises(RuntimeError, match='unfenced'):
        pool.upload_from_reader(16, destination, lambda _: pytest.fail('poisoned fill'))


@pytest.mark.parametrize('prepared', [False, True])
def test_another_producer_poisoning_pool_prevents_late_dma_submission(cuda, prepared):
    pool = cuda.pool
    entered, release = Event(), Event()
    calls = 0
    def acquire(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(3)
    pool.bandwidth_limiter = SimpleNamespace(acquire=acquire)
    good = DeviceBytes(torch.empty(16,dtype=torch.uint8), cuda.operations)
    bad = DeviceBytes(torch.empty(16,dtype=torch.uint8), cuda.operations)
    bad.fail_copy = True
    def broken_fence():
        raise OSError('failed fence')
    pool.stream.on_sync = broken_fence
    def victim():
        if prepared:
            ticket = pool.prepare_from_reader(16, lambda b: ready_job(b))
            return pool.upload_prepared(ticket, good)
        return pool.upload_from_reader(16, good, lambda b: b)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(victim)
        try:
            assert entered.wait(2)
            with pytest.raises(OSError, match='enqueue'):
                pool.upload_from_reader(16, bad, lambda b: b)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='unfenced'):
            future.result(timeout=3)
    assert [op[1] for op in cuda.operations if op[0] == 'copy'] == [bad]
    assert len(pool.preparing) == len(pool._prepared) == 1
    assert not pool.close()


def memory_authority(tmp_path):
    archive = MemoryArchive(str(tmp_path), 2*4*1*2*3, 16)
    authority, _ = setup(tmp_path, archive=archive)
    assert authority.append(*payload(0, 12), history_begin=0, local_rope_begin=0)
    drain(authority)
    return authority, archive


def test_try_read_admission_does_not_wait_and_releases_tentative_leases(tmp_path, monkeypatch):
    authority, archive = memory_authority(tmp_path)
    handles = [next(iter(authority.records.values())).handle]
    monkeypatch.setattr(archive, 'read_region', lambda *_, **__: None)
    monkeypatch.setattr(archive, 'wait_for_idle', lambda *_: pytest.fail('lookahead waited for global I/O'))
    destination = torch.empty(authority.raw_record_bytes, dtype=torch.uint8)
    assert packed.begin_read_packed_records(authority, handles, destination=destination,
                                           region=(0, authority.raw_record_bytes), wait=False) is None
    assert authority.close() and archive.close()


@pytest.mark.parametrize('fault', ['short', 'different_storage', 'job'])
def test_joined_read_validation_releases_source_leases(tmp_path, monkeypatch, fault):
    authority, archive = memory_authority(tmp_path)
    handle = next(iter(authority.records.values())).handle
    destination = torch.empty(authority.record_bytes, dtype=torch.uint8)
    if fault == 'job':
        job = ready_job(error=OSError('read failed'))
    else:
        job = ready_job((bytearray(authority.record_bytes-(fault == 'short')), {}))
    monkeypatch.setattr(archive, 'read', lambda *_, **__: job)
    read = packed.begin_read_packed_records(authority, [handle], destination=destination, wait=False)
    assert read.leases
    with pytest.raises((ValueError, OSError)):
        read.result()
    assert not read.leases and not read.owners
    assert authority.close() and archive.close()


def test_read_holder_construction_failure_drains_admitted_writer(tmp_path, monkeypatch):
    authority, archive = memory_authority(tmp_path)
    handle = next(iter(authority.records.values())).handle
    reading, release = Event(), Event()
    original_gather = archive._gather_region
    def gather(*args):
        reading.set()
        assert release.wait(3)
        return original_gather(*args)
    def broken_holder(*args):
        raise ValueError('completion holder failed')
    monkeypatch.setattr(archive, '_gather_region', gather)
    monkeypatch.setattr(packed, 'PackedRead', broken_holder)
    destination = torch.empty(authority.raw_record_bytes, dtype=torch.uint8)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(packed.begin_read_packed_records, authority, [handle],
            destination=destination, region=(0,authority.raw_record_bytes), wait=False)
        try:
            assert reading.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=.02)
        finally:
            release.set()
        with pytest.raises(ValueError, match='holder failed'):
            future.result(timeout=3)
    assert authority.close() and archive.close()


def make_split(tmp_path, cuda, monkeypatch, enabled=True):
    authority, archive = memory_authority(tmp_path)
    records = list(authority.records.values())
    remaps = [SimpleNamespace(block_id=r.handle.slot, n_tokens=4, to_base=i*4,
                              from_base=(2-i)*4, skip=False, working_k_resident=True)
              for i, r in enumerate(records)]
    plan = SimpleNamespace(remaps=remaps)
    plan.release_cpu_sources = lambda: cuda.operations.append(('release_sources',))
    memory = SimpleNamespace(phase=Phase.ASSEMBLING, _plan=plan,
                             request_id=authority.memory.request_id,
                             admitted_read=authority.memory.admitted_read)
    memory.fail_transaction = lambda result: cuda.operations.append(('fail_transaction', result))
    proxy = SimpleNamespace(memory=memory, archive=archive, records=authority.records,
        shape=authority.shape, record_bytes=authority.record_bytes,
        raw_record_bytes=authority.raw_record_bytes, value_record_bytes=authority.value_record_bytes,
        raw_dtype=authority.raw_dtype, poll=authority.poll)
    layers = []
    for _ in range(2):
        key = torch.empty((4,4,1,2), dtype=torch.float8_e4m3fn)
        layers.append(window.LayerWindow(key, key.clone(), torch.zeros(16,2), torch.tensor(1.),
                                         torch.tensor([1,2,3], dtype=torch.int32)))
    assembler = window.NativeWindowAssembler(tuple(layers), 4, chunk_blocks=1, packed=True,
        direct_pinned_reads=True, reuse_resident_v=True, read_prefetch=enabled)
    assembler.device = assembler.staging.device = torch.device('cuda:0')
    monkeypatch.setattr(assembler, '_upload_metadata', lambda values: torch.tensor(values))
    from vllm_kvmem import split_assembly
    monkeypatch.setattr(split_assembly, 'gather_resident_values', lambda layers, bases, block, out: out)
    cuda.complete = True
    return assembler, proxy, plan, authority, archive, split_assembly


def test_actual_archive_read_starts_before_current_layers_and_matches_baseline(tmp_path, cuda, monkeypatch):
    assembler, proxy, plan, authority, archive, split = make_split(tmp_path, cuda, monkeypatch)
    starts, launches = [], []
    original_begin = packed.begin_read_packed_records
    def begin(a, handles, **kwargs):
        starts.append((handles[0].slot, len(launches), kwargs.get('wait', True)))
        return original_begin(a, handles, **kwargs)
    monkeypatch.setattr(packed, 'begin_read_packed_records', begin)
    seen = []
    def launch(raw, raw_indices, values, value_indices, bases, layer, **kwargs):
        launches.append((int(bases[0]), kwargs['layer_index']))
        seen.append(raw.data.clone())
    monkeypatch.setattr(split, 'assemble_split_layer', launch)
    result = assembler.assemble(proxy, plan)
    assert [entry[1:] for entry in starts] == [(0, True), (0, False), (2, False)]
    assert result.read_stats['assembly_prefetch_submitted'] == 2
    assert result.read_stats['assembly_prefetch_completed'] == 2
    assert result.read_stats['assembly_prefetch_abandoned'] == 0
    assert result.read_stats['assembly_prefetch_eligible'] == 1
    assert result.uploaded_bytes == 3*authority.raw_record_bytes
    assert len(assembler.transfers.buffers) == len(assembler.staging.buffers) == 2
    assert not assembler.transfers.preparing
    assert cuda.operations[-1] == ('release_sources',)
    expected = [authority.read_record(r.handle)[0].view(torch.uint8).flatten()
                for r in authority.records.values()]
    assert all(torch.equal(seen[2*i], expected[i]) for i in range(3))
    optimized_bytes = [data.clone() for data in seen]
    seen.clear()
    assembler.read_prefetch = False
    baseline = assembler.assemble(proxy, plan)
    assert all(torch.equal(a,b) for a,b in zip(optimized_bytes, seen))
    assert (baseline.rebuilt_tokens,baseline.uploaded_bytes,baseline.chunks) == (
        result.rebuilt_tokens,result.uploaded_bytes,result.chunks)
    assert not any(key.startswith('assembly_prefetch_') for key in baseline.read_stats)
    assert authority.close() and archive.close()


def test_layer_failure_drains_lookahead_before_transaction_or_source_release(tmp_path, cuda, monkeypatch):
    assembler, proxy, plan, authority, archive, split = make_split(tmp_path, cuda, monkeypatch)
    reading, release, layer_failed = Event(), Event(), Event()
    calls = 0
    original_gather = archive._gather_region
    def gather(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            reading.set()
            assert release.wait(3)
        return original_gather(*args)
    monkeypatch.setattr(archive, '_gather_region', gather)
    def fail(*args, **kwargs):
        assert reading.wait(2)
        layer_failed.set()
        raise OSError('layer failed with next writer alive')
    monkeypatch.setattr(split, 'assemble_split_layer', fail)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(assembler.assemble, proxy, plan)
        try:
            assert layer_failed.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=.02)
            assert not any(op[0] in ('fail_transaction', 'release_sources') for op in cuda.operations)
            assert assembler.transfers.preparing
            assert not assembler.staging.close()
        finally:
            release.set()
        with pytest.raises(OSError, match='layer failed'):
            future.result(timeout=3)
    assert not assembler.transfers.preparing
    assert assembler.read_prefetch_stats['assembly_prefetch_abandoned'] == 1
    assert [op[0] for op in cuda.operations][-2:] == ['fail_transaction', 'release_sources']
    assert authority.close() and archive.close()


@pytest.mark.parametrize('change', ['owner', 'plan', 'generation'])
def test_lookahead_cannot_be_consumed_by_changed_owner_plan_or_generation(tmp_path, cuda, monkeypatch, change):
    authority, archive = memory_authority(tmp_path)
    stats = {f'assembly_prefetch_{n}': 0 for n in ('submitted','completed','abandoned','slot_busy','admission_busy')}
    handles = [next(iter(authority.records.values())).handle]
    with pytest.raises(ValueError, match='different|match'):
        with packed.PackedReadAhead(authority, stats) as ahead:
            ahead.start(cuda.pool, handles, (0, authority.raw_record_bytes), authority.raw_record_bytes)
            consume_handles = handles
            if change == 'owner':
                monkeypatch.setattr(authority.memory, 'request_id', 'different-owner')
            elif change == 'plan':
                monkeypatch.setattr(authority.memory, '_plan', object())
            else:
                h = handles[0]
                consume_handles = [SimpleNamespace(slot=h.slot, generation=h.generation+1)]
            ahead.take(cuda.pool, consume_handles, (0, authority.raw_record_bytes),
                       DeviceBytes(torch.empty(authority.raw_record_bytes, dtype=torch.uint8), cuda.operations),
                       FakeEvent())
    assert stats['assembly_prefetch_abandoned'] == 1 and not cuda.pool.preparing
    monkeypatch.undo()
    assert authority.close() and archive.close()


def test_busy_lookahead_never_waits_and_records_distinct_admission_and_slot_counts(tmp_path, cuda, monkeypatch):
    authority, archive = memory_authority(tmp_path)
    stats = {f'assembly_prefetch_{n}': 0 for n in ('submitted','completed','abandoned','slot_busy','admission_busy')}
    handle = next(iter(authority.records.values())).handle
    a = cuda.pool.prepare_from_reader(16, lambda b: ready_job(b))
    b = cuda.pool.prepare_from_reader(16, lambda b: ready_job(b))
    with packed.PackedReadAhead(authority, stats) as ahead:
        ahead.start(cuda.pool, [handle], (0, 16), 16)
        assert stats['assembly_prefetch_slot_busy'] == 1 and ahead.pending is None
        cuda.pool.discard_prepared(a)
        monkeypatch.setattr(archive, 'read_region', lambda *_, **__: None)
        monkeypatch.setattr(archive, 'wait_for_idle', lambda *_: pytest.fail('waited for archive'))
        ahead.start(cuda.pool, [handle], (0, 16), 16)
        assert stats['assembly_prefetch_admission_busy'] == 1 and ahead.pending is None
    cuda.pool.discard_prepared(b)
    assert stats['assembly_prefetch_submitted'] == 0
    assert authority.close() and archive.close()


@pytest.mark.parametrize('reason', ['disabled', 'disk', 'raw_cache', 'not_direct'])
def test_unqualified_split_keeps_original_path(tmp_path, cuda, monkeypatch, reason):
    assembler, proxy, plan, authority, archive, _ = make_split(tmp_path, cuda, monkeypatch)
    if reason == 'disabled': assembler.read_prefetch = False
    elif reason == 'disk': proxy.archive = object()
    elif reason == 'raw_cache': assembler.raw_cache = SimpleNamespace(dtype=authority.raw_dtype)
    else: assembler.direct_pinned_reads = False
    def original(a,p,rs,pending,records,ahead):
        assert ahead is None
        return window.AssemblyResult(FakeEvent(True), (), 12, 0, 3)
    monkeypatch.setattr(assembler, '_assemble_split_chunks', original)
    result = assembler.assemble(proxy, plan)
    if reason != 'disabled':
        assert result.read_stats['assembly_prefetch_declined'] == 1
        assert result.read_stats['assembly_prefetch_submitted'] == 0
    assert authority.close() and archive.close()


def test_lookahead_drain_error_preserves_original_failure_and_releases_slot(tmp_path, cuda, monkeypatch):
    authority, archive = memory_authority(tmp_path)
    stats = {f'assembly_prefetch_{n}': 0 for n in ('submitted','completed','abandoned','slot_busy','admission_busy')}
    handle = next(iter(authority.records.values())).handle
    monkeypatch.setattr(archive, 'read_region', lambda *_, **__: ready_job(error=OSError('failed read')))
    with pytest.raises(ValueError, match='original failure') as caught:
        with packed.PackedReadAhead(authority, stats) as ahead:
            ahead.start(cuda.pool, [handle], (0,16),16)
            raise ValueError('original failure')
    assert 'failed read' in ' '.join(caught.value.__notes__)
    assert stats['assembly_prefetch_abandoned'] == 1 and not cuda.pool.preparing
    assert authority.close() and archive.close()
