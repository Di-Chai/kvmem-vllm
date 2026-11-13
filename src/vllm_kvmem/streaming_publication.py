"""Opt-in KVMEM_STREAM_PUBLICATION: fill a coalesced chunk's record pages during its forward.

Publication of a prefill chunk copies every layer's K/V rows into 32-token
record pages and sums each page's keys (MeanK) after the forward. Layer i's
rows exist as soon as attention layer i has projected them, so this session:

* borrows the chunk's record staging pages before the forward (the staging
  pool is independent of the native store, so no append transaction opens);
* after attention layer i (engine-thread hook), packs layer i exactly like the
  capture packet (element-wise identical FP8 conversion) and copies it to
  pinned memory on a side stream;
* on a worker, scatters layer i into every page and computes that layer's
  MeanK sums with the same native kernels (GIL released);
* registers the complete pages with the authority, whose append adopts them
  instead of copying and finalizing again.

The MTP layer is produced by the drafter and is filled at submit. Any layer
whose hook did not fire (for example a captured graph) is filled there too.
KVMEM_STREAM_PUBLICATION_VERIFY=1 compares every adopted page with the packet.

KVMEM_STREAM_PUBLICATION_ZERO_COPY=1 (direct CPU seal archives only): each page
is a new private ``bytes`` record in the archive's own layout (raw K, then V),
filled in place. Its seal adopts that object instead of packing a second copy
of pages written long before (cold in cache), which made seals slower than the
whole publication. The object stays unshared and unhashed until it is complete
(the PyBytes_FromStringAndSize(NULL, n) idiom of the native packer).
"""
import ctypes
import os
import threading
import time

import numpy as np
import torch

from . import _native
from . import settings as _settings


def enabled():
    value = _settings.get('KVMEM_STREAM_PUBLICATION', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_STREAM_PUBLICATION must be 0 or 1')
    return value == '1'


def verify_enabled():
    return _settings.get('KVMEM_STREAM_PUBLICATION_VERIFY', '0') == '1'


def start_delay():
    """KVMEM_STREAM_PUBLICATION_DELAY_MS: keep the worker idle at the start of
    the forward, while the previous chunk's publication and its promised
    refresh (layer-0 selection and upload) are on the critical path."""
    value = float(_settings.get('KVMEM_STREAM_PUBLICATION_DELAY_MS', '0'))
    if not 0 <= value <= 400:
        raise ValueError('KVMEM_STREAM_PUBLICATION_DELAY_MS must be in [0, 400]')
    return value / 1000


def zero_copy_requested():
    value = _settings.get('KVMEM_STREAM_PUBLICATION_ZERO_COPY', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_STREAM_PUBLICATION_ZERO_COPY must be 0 or 1')
    return value == '1'


def zero_copy_eligible(authority):
    archive = authority.archive
    return (zero_copy_requested() and getattr(archive, 'cpu_direct_record_seal', False) and
            getattr(archive, 'async_seal', False) and
            getattr(archive, '_native_payload_arena', None) is None and
            not getattr(authority, 'cpu_reuse_record_views', False) and
            authority.raw_dtype.itemsize == 1 and
            authority.raw_record_bytes == authority.value_record_bytes and
            getattr(archive, 'record_bytes', None) == authority.record_bytes)


_ALLOC = ctypes.pythonapi.PyBytes_FromStringAndSize
_ALLOC.restype, _ALLOC.argtypes = ctypes.py_object, (ctypes.c_void_p, ctypes.c_ssize_t)
_DATA = ctypes.pythonapi.PyBytes_AsString
_DATA.restype, _DATA.argtypes = ctypes.c_void_p, (ctypes.py_object,)


def private_record(record_bytes):
    """A new unshared bytes object and a writable uint8 view of its storage."""
    record = _ALLOC(None, record_bytes)
    storage = (ctypes.c_uint8 * record_bytes).from_address(_DATA(record))
    storage._kvmem_owner = record      # every numpy/torch view keeps the bytes alive
    return record, np.ctypeslib.as_array(storage)


def allocate_pages(authority, count, pages):
    """Append ``count`` pages (raw, values, staging slot, sums, private bytes).

    Zero-copy pages are views of a private archive record and own no staging
    slot; otherwise pages borrow the authority's ordinary staging buffers.
    """
    zero_copy = zero_copy_eligible(authority)
    half = authority.raw_record_bytes
    # One contiguous sums block (consecutive per-page views): the batched MeanK
    # division then takes one slice instead of stacking separate tensors.
    sum_block = torch.empty((count, authority.scoring_layers, authority.shape[2], authority.shape[3]),
                            dtype=torch.float32) if count > 1 else None
    for index in range(count):
        if zero_copy:
            packed, view = private_record(authority.record_bytes)
            raw = torch.from_numpy(view[:half]).view(authority.raw_dtype).reshape(authority.shape)
            values = torch.from_numpy(view[half:]).reshape(authority.shape)
            slot = None
        else:
            (raw, values, slot), packed = authority._borrow_record_buffers(), None
        sums = (sum_block[index] if sum_block is not None else
                torch.empty((authority.scoring_layers, authority.shape[2], authority.shape[3]), dtype=torch.float32))
        pages.append((raw, values, slot, sums, packed))
    return pages


class _Staging:
    """Two reusable pinned buffers; a session waits for its predecessor's worker."""
    def __init__(self):
        self.buffers = [None, None]
        self.owners = [None, None]
        self.turn = 0

    def acquire(self, session, layers, count, heads, dim):
        slot = self.turn
        self.turn ^= 1
        previous = self.owners[slot]
        if previous is not None:
            previous.finished.wait(60)
        shape = (layers, count, heads, dim)
        current = self.buffers[slot]
        if current is None or current[0].shape[1] < count or current[0].shape != (layers, current[0].shape[1], heads, dim):
            current = (torch.empty(shape, dtype=torch.uint8, pin_memory=True),
                       torch.empty(shape, dtype=torch.uint8, pin_memory=True))
            self.buffers[slot] = current
        self.owners[slot] = session
        return current[0][:, :count], current[1][:, :count]


class StreamSession:
    def __init__(self, runtime, authority, rid, history_begin, count, row_begin, captures):
        self.runtime, self.authority, self.rid = runtime, authority, rid
        self.history_begin, self.count, self.row_begin = history_begin, count, row_begin
        self.block = authority.shape[1]
        self.layers = len(captures)
        self.target_layers = len(captures) - len(getattr(runtime.runner, '_kvmem_mtp_captures', ()) or ())
        self.captures = captures
        self.scoring_layers = authority.scoring_layers
        self.simd = getattr(authority, 'cpu_mean_index_simd', False)
        staging = runtime.__dict__.setdefault('_stream_staging', _Staging())
        self.host_raw, self.host_values = staging.acquire(self, self.layers, count, runtime.heads, runtime.dim)
        streams = runtime.__dict__.setdefault('_stream_side', [])
        if not streams:
            streams.append(torch.cuda.Stream(device=runtime.runner.device))
        self.stream = streams[0]
        self.events = [None] * self.layers
        self.condition = threading.Condition()
        self.error = None
        self.aborted = False
        self.registered = False
        self.native = False
        self.finished = threading.Event()
        self.pages = None
        self.not_before = time.monotonic() + start_delay()
        self.thread = threading.Thread(target=self._run, name='kvmem-stream-publication', daemon=True)
        self.thread.start()

    # Engine thread -------------------------------------------------------
    def on_layer(self, index):
        if index >= self.layers or self.events[index] is not None or self.aborted:
            return
        ready = torch.cuda.Event()
        ready.record()
        capture = self.captures[index]
        rows = slice(self.row_begin, self.row_begin + self.count)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            # Element-wise identical to CaptureRuntime._capture_payload's
            # stack().float().clamp().to() and per-layer V scale division.
            raw = capture.key[rows].float().clamp(-448, 448).to(self.runtime.raw_dtype)
            values = (capture.value[rows].float() / self.runtime.scales[index]).clamp(-448, 448).to(
                torch.float8_e4m3fn)
            self.host_raw[index].copy_(raw.view(torch.uint8), non_blocking=True)
            self.host_values[index].copy_(values.view(torch.uint8), non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        with self.condition:
            self.events[index] = done
            self.condition.notify_all()

    def finish_forward(self):
        """At submit, after the drafter: fill MTP and any layer the hook missed.

        The next forward overwrites the persistent capture buffers, so the
        compute stream waits for every pack issued on the side stream.
        """
        for index in range(self.layers):
            if self.events[index] is None:
                self.on_layer(index)
        if not self.aborted:
            torch.cuda.current_stream().wait_event(self.events[-1])

    def abort(self):
        with self.condition:
            self.aborted = True
            self.condition.notify_all()

    # Worker ----------------------------------------------------------------
    def _run(self):
        authority = self.authority
        pages = []
        try:
            with self.condition:
                while not self.aborted and time.monotonic() < self.not_before:
                    self.condition.wait(max(0., self.not_before - time.monotonic()))
                if self.aborted:
                    return
            allocate_pages(authority, self.count // self.block, pages)
            self.pages = pages
            source_raw = self.host_raw.numpy()
            source_values = self.host_values.numpy()
            page_raw = [page[0].view(torch.uint8).numpy() for page in pages]
            page_values = [page[1].numpy() for page in pages]
            page_sums = [page[3].numpy() for page in pages]
            scratch = torch.zeros((1, authority.shape[2], authority.shape[3]), dtype=torch.float32).numpy()
            block = self.block
            # Native session object: each layer is one call that releases the GIL
            # at once (no per-layer Python job lists contending with the engine).
            native = getattr(_native, 'StreamPages', None)
            native = (native(source_raw, source_values, page_raw, page_values, page_sums, block)
                      if native is not None else None)
            self.native = native is not None
            for layer in range(self.layers):
                with self.condition:
                    while self.events[layer] is None and not self.aborted:
                        self.condition.wait(.5)
                    if self.aborted:
                        return
                    event = self.events[layer]
                event.synchronize()
                if native is not None:
                    native.layer(layer, self.simd)
                    continue
                one = slice(layer, layer + 1)
                # Copy-only native scatter of this layer into every page.
                _native.scatter_history_batch([
                    (source_raw[one], source_values[one], page_raw[p][one], page_values[p][one], None,
                     page_sums[p][one] if layer < self.scoring_layers else scratch,
                     p * block, 0, block, True) for p in range(len(pages))])
                if layer < self.scoring_layers:
                    # Same token-order FP32 sums as the page finalizer, per layer.
                    _native.finalize_mean_batch([(page_raw[p][one], page_sums[p][one], block, True)
                                                 for p in range(len(pages))], vectorized=self.simd)
            authority.register_prefilled({self.history_begin + p * block: page
                                          for p, page in enumerate(pages)})
            self.registered = True
        except BaseException as error:
            self.error = error
        finally:
            if not self.registered:
                for page in pages:
                    try:
                        if page[2] is not None:
                            authority._release_staging_slot(page[2])
                    except BaseException:
                        pass
            self.finished.set()

    def wait_registered(self, timeout=60.):
        if not self.finished.wait(timeout):
            raise TimeoutError('Streaming publication did not complete')
        if self.error is not None:
            raise RuntimeError('Streaming publication failed') from self.error
        return self.registered


def verify_page(record, raw, values, start, n):
    """Optional qualification: the adopted page equals the packet rows and sums."""
    import numpy as np
    page_raw = record.raw[:, :n].contiguous().view(torch.uint8).numpy()
    page_values = record.values[:, :n].contiguous().numpy()
    if (not np.array_equal(page_raw, raw[:, start:start+n].contiguous().view(torch.uint8).numpy()) or
            not np.array_equal(page_values, values[:, start:start+n].contiguous().numpy())):
        raise RuntimeError('Streamed record page differs from the published packet')
    check = torch.empty_like(record.key_sum).numpy()
    _native.finalize_mean_batch([(record.raw.view(torch.uint8).numpy(), check, n,
                                  record.raw.dtype == torch.float8_e4m3fn)])
    if not np.array_equal(check.view(np.uint32), record.key_sum.numpy().view(np.uint32)):
        raise RuntimeError('Streamed MeanK sums differ from the page finalizer')


def prescatter_enabled():
    value = _settings.get('KVMEM_PUBLICATION_PRESCATTER', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PUBLICATION_PRESCATTER must be 0 or 1')
    return value == '1'


class PrescatterJob:
    """Opt-in KVMEM_PUBLICATION_PRESCATTER: fill a published chunk's record pages
    span by span on a worker while the publication thread folds and prepares
    the graph for earlier spans. Same pages, sums and adoption as streaming
    publication (bit-identical), but from the downloaded packet after the
    forward, so nothing runs during the forward itself.
    """
    def __init__(self, authority, group, raw, values, simd):
        self.authority, self.block = authority, authority.shape[1]
        self.group_begin = group.history_begin
        self.spans = [(span.history_begin, span.count) for span in group.spans]
        self.raw = raw.view(torch.uint8).numpy()
        self.values = values.numpy()
        self.simd = simd
        self.events = [threading.Event() for _ in self.spans]
        self.registered = []
        self.error = None
        self.aborted = False
        self.thread = threading.Thread(target=self._run, name='kvmem-prescatter', daemon=True)
        self.thread.start()

    def _run(self):
        authority, block = self.authority, self.block
        try:
            for index, (begin, count) in enumerate(self.spans):
                if self.aborted:
                    return
                pages = []
                if begin % block == 0 and count % block == 0 and count:
                    try:
                        allocate_pages(authority, count // block, pages)
                        rows = slice(begin - self.group_begin, begin - self.group_begin + count)
                        native = _native.StreamPages(self.raw[:, rows], self.values[:, rows],
                            [p[0].view(torch.uint8).numpy() for p in pages], [p[1].numpy() for p in pages],
                            [p[3].numpy() for p in pages], block)
                        for layer in range(self.raw.shape[0]):
                            native.layer(layer, self.simd)
                        begins = [begin + i * block for i in range(len(pages))]
                        authority.register_prefilled(dict(zip(begins, pages)))
                        self.registered.extend(begins)
                        pages = []
                    finally:
                        for page in pages:       # not registered: return the slots
                            if page[2] is not None:
                                authority._release_staging_slot(page[2])
                self.events[index].set()
        except BaseException as error:
            self.error = error
        finally:
            for event in self.events:
                event.set()

    def wait_range(self, history_begin, count):
        """Block an append until the spans it covers are filled (or failed)."""
        end = history_begin + count
        for (begin, size), event in zip(self.spans, self.events):
            if begin < end and history_begin < begin + size:
                if not event.wait(60):
                    raise TimeoutError('Prescattered record pages did not complete')

    def abort(self):
        self.aborted = True
