"""Opt-in real CUDA→SSD→CUDA exact native-cache qualification."""
import hashlib
import os
import time
import weakref

import pytest
import torch

from vllm_kvmem.checkpoint_storage import CheckpointStore
from vllm_kvmem.request_checkpoint import checkpoint_size
from test_request_checkpoint import native_fixture


pytestmark = pytest.mark.skipif(
    os.getenv('KVMEM_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
    reason='Opt-in CUDA SSD checkpoint byte qualification')


def eventually(function):
    deadline = time.monotonic()+10
    while True:
        value = function()
        if value:
            return value
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_cuda_ssd_native_request_checkpoint_roundtrip_preserves_every_byte(tmp_path):
    adapter, original_pages, tensors = native_fixture('cuda')
    # Native FP8 K/V fixtures include every byte value, including FP8 NaNs.
    # Give the float32 GDN rows arbitrary bit patterns too, so a numerical
    # conversion/canonicalization cannot accidentally pass the byte comparison.
    for tensor in tensors[2:]:
        raw = tensor.view(torch.uint8)
        raw.copy_(torch.arange(raw.numel(), device='cuda', dtype=torch.int64)
                  .remainder(256).to(torch.uint8).reshape(raw.shape))
    original = [tensor.clone() for tensor in tensors]
    bindings = adapter.bindings('a', original_pages, capture=True)
    size = checkpoint_size(bindings)
    adapter.release('a')  # Geometry inspection above did not create a checkpoint.
    bindings = None
    operation, identity = 41, ('a', 9, 41)
    store = CheckpointStore(tmp_path/'checkpoint-ssd', size, size*2)
    assert not store.directory.exists()
    entry = store.reserve_capture(identity, size)
    saved = loaded = restored = None
    try:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            saved = adapter.capture('a', operation, original_pages, max_bytes=size)
        assert saved.payload.is_pinned() and saved.bytes == size
        assert len(saved.schemas) == 7  # Target/MTP K/V, GDN conv/SSM, accepted count.
        schema = saved.schemas
        source_owner = weakref.ref(saved.payload)
        store.install(entry, saved)
        saved = None  # Ownership transfers to the bounded store.
        assert store.snapshot()['io_pending'] == 0
        assert store.snapshot()['cpu_charged_bytes'] == size
        entry.snapshot.completion.synchronize()
        assert store.capture_ready(entry)
        eventually(store.poll)
        assert entry.state == 'disk' and entry.snapshot is None
        assert source_owner() is None
        fd = entry.disk.fd
        assert os.fstat(fd).st_size == size and list(store.directory.iterdir()) == []
        assert store.snapshot()['cpu_charged_bytes'] == 0
        assert store.snapshot()['ssd_used_bytes'] == size

        # The original allocator pages can now be used by unrelated work.
        source_rows = (original_pages[0], original_pages[1], original_pages[2], original_pages[2])
        for tensor, rows in zip(tensors, source_rows):
            for row in rows:
                tensor[row].view(torch.uint8).fill_(0xA5)
        adapter.runner.model_state.num_accepted_tokens_gpu.zero_()
        before_restore = [tensor.clone() for tensor in tensors]
        replacement = ((2, 4), (8,), (9, 10))
        for table, row in zip(adapter.runner.block_tables.block_tables, replacement):
            table.gpu[0] = torch.tensor(row, device='cuda')

        loaded = eventually(lambda: store.acquire_restore(entry))
        assert loaded.identity == identity and loaded.schemas == schema
        assert loaded.payload.is_pinned() and loaded.bytes == size
        staging_owner = weakref.ref(loaded.payload)
        view = memoryview(loaded.payload.numpy()).cast('B')
        try:
            assert hashlib.sha256(view).digest() == entry.disk.digest
        finally:
            view.release()
            view = None
        restore_stream = torch.cuda.Stream()
        restore_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(restore_stream):
            restored = adapter.restore(loaded, 'a', operation, replacement)
        loaded = None
        with pytest.raises(ValueError, match='in-flight'):
            store.drop(entry)
        assert store.reserve_capture(('other', 1, 1), size) is None
        assert store.snapshot()['cpu_resident_bytes'] == size
        assert entry.disk.fd == fd  # Disk authority survives the H2D owner.
        restored.synchronize()
        store.restore_complete(entry)
        assert staging_owner() is None
        assert store.snapshot()['cpu_charged_bytes'] == 0

        destination_rows = (replacement[0], replacement[1], replacement[2], replacement[2])
        for tensor, old, before, sources, destinations in zip(
                tensors, original, before_restore, source_rows, destination_rows):
            expected = before.clone()
            for source, destination in zip(sources, destinations):
                expected[destination].view(torch.uint8).copy_(old[source].view(torch.uint8))
            # Full-tensor comparison includes untouched neighbors and reused
            # old pages; floating NaNs are compared as raw bytes, not numbers.
            assert torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8))
        assert adapter.runner.model_state.num_accepted_tokens_gpu.item() == 3
        assert adapter._slots == {'a': (0, 9)}
        stats = store.snapshot()
        assert stats['cpu_peak_bytes'] == stats['cpu_charge_peak_bytes'] == size
        assert stats['disk_write_bytes'] == stats['disk_read_bytes'] == size
        assert stats['spills'] == stats['loads'] == 1
        assert store.drop(entry)
        adapter.release('a')
        assert adapter._slots == {}
        with pytest.raises(OSError):
            os.fstat(fd)
        stats = store.snapshot()
        assert all(stats[name] == 0 for name in ('cpu_resident_bytes', 'cpu_promised_bytes',
            'cpu_charged_bytes', 'logical_bytes', 'ssd_used_bytes', 'io_pending', 'entries'))
    finally:
        saved = loaded = None
        # Drain accepted CUDA copies even if an assertion above failed.
        torch.cuda.synchronize()
        if entry.identity in store.entries:
            if entry.state == 'restoring':
                if restored is not None:
                    restored.synchronize()
                store.restore_complete(entry)
            eventually(lambda: store.drop(entry))
        if 'a' in adapter._slots:
            adapter.release('a')
        assert store.close()
