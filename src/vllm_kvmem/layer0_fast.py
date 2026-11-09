"""Opt-in KVMEM_LAYER0_FAST: the first attention layers of a promised prefill refresh without the CPU path.

With the promise ACK (pressure_promise) the next forward starts immediately and
waits at each attention layer's guard until the producer has published chunk k,
selected that layer's window and uploaded it. The forward needs layer i about
25 + 41*i ms after the promise at G4K; the CPU producer catches up only around
layer 4. Each layer's new window is picked only from its previous window and
chunk k (layerwise_pressure ``_layer_records`` / ``extend_records``), and all of
those bytes already exist on the GPU:

* each fast layer's previous window of original-content K and V records, kept
  here in a GPU shadow (rebuilt from the CPU archive whenever this path did not
  produce that layer);
* chunk k's K and V for the layer, copied device-to-device from the capture
  payload before its download (the archive's bytes for those records);
* chunk k's attention masses, downloaded with the original fence.

For layers 0..K-1 (``KVMEM_LAYER0_FAST_LAYERS``, default 1) the fast path folds
chunk k into a copy of that layer's history (the same native batch call
publication makes), picks in history-position space (the store's pick,
``_native.pressure_pick_positions``, parity-tested), keeps exactly the plan's
skipped blocks (layer 0 only; the CPU path rebuilds every block of the other
layers), assembles with the same split-assembly kernel, and publishes each
layer's event in order. The producer runs unchanged, asserts that its own
selections, skip set and page tables are identical, and does not upload those
layers again.

Any doubt before a layer is published stops the fast path at that layer (the
original path produces it and every later layer). A mismatch after
publication is fatal: the forward has already consumed the window.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass

import torch

BLOCK = 32


def enabled():
    value = os.getenv('KVMEM_LAYER0_FAST', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYER0_FAST must be 0 or 1')
    return value == '1'


def max_owners():
    value = os.getenv('KVMEM_LAYER0_FAST_MAX_OWNERS', '2')
    if not value.isdigit() or not 1 <= int(value) <= 64:
        raise ValueError('KVMEM_LAYER0_FAST_MAX_OWNERS must be an integer in [1, 64]')
    return int(value)


def fast_layers():
    value = os.getenv('KVMEM_LAYER0_FAST_LAYERS', '1')
    if not value.isdigit() or not 1 <= int(value) <= 16:
        raise ValueError('KVMEM_LAYER0_FAST_LAYERS must be an integer in [1, 16]')
    return int(value)


@dataclass(frozen=True)
class Result:
    published: int = 0              # layers 0..published-1 were produced here
    reason: str = ''
    selected: tuple = ()            # per published layer: history begins in dense order
    skipped: frozenset = frozenset()  # layer 0's kept-resident begins
    tables: tuple = ()              # per published layer: destination page table
    timings: tuple = ()


class Shadow:
    """One owner's records of one layer on the GPU: [window (dense order) | next chunk]."""

    def __init__(self, device, window_blocks, chunk_blocks, row_bytes):
        self.window_blocks, self.chunk_blocks, self.row_bytes = window_blocks, chunk_blocks, row_bytes
        capacity = window_blocks + chunk_blocks
        self.buffers = [tuple(torch.empty((capacity, row_bytes), dtype=torch.uint8, device=device)
                              for _ in range(2)) for _ in range(2)]
        self.active = 0
        self.window = None          # history begins in dense order, valid with window_event
        self.window_event = None
        self.chunk = None           # (history_begin, blocks, event) of the captured next chunk
        self.pending = None         # (tokens, event) staged before its logical group is known
        self.lock = threading.Lock()

    def invalidate(self):
        self.window = self.window_event = self.chunk = self.pending = None


class Layer0Fast:
    def __init__(self, runner):
        self.runner = runner
        self.shadows = {}           # (request id, layer) -> Shadow
        self.owners = set()
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream(device=runner.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kvmem-layer0')
        self.counts = {}
        self.pinned = None
        self.layers = fast_layers()

    def count(self, key):
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    # -- shadow lifetime --------------------------------------------------------
    def shadow(self, rid, layer=0, create=None):
        with self.lock:
            shadow = self.shadows.get((rid, layer))
            if shadow is None and create is not None and (rid in self.owners or len(self.owners) < max_owners()):
                self.owners.add(rid)
                shadow = self.shadows[(rid, layer)] = Shadow(*create)
            return shadow

    def retire(self, rid):
        with self.lock:
            for key in [k for k in self.shadows if k[0] == rid]:
                del self.shadows[key]
            self.owners.discard(rid)

    # -- engine thread: capture --------------------------------------------------
    def stage_chunk(self, rid, raw, values, tokens):
        """Copy chunk k's K/V of each fast layer (archive bytes) before the payload download."""
        for layer in range(min(self.layers, raw.shape[0])):
            shadow = self.shadow(rid, layer)
            if shadow is None:
                continue
            with shadow.lock, torch.inference_mode():
                shadow.chunk = shadow.pending = None
                if tokens % BLOCK or tokens // BLOCK > shadow.chunk_blocks or raw.shape[1] < tokens:
                    continue
                blocks, base = tokens // BLOCK, shadow.window_blocks
                rows, vals = shadow.buffers[shadow.active]
                rows[base:base+blocks].view(-1).copy_(raw[layer, :tokens].contiguous().view(torch.uint8).view(-1))
                vals[base:base+blocks].view(-1).copy_(values[layer, :tokens].contiguous().view(torch.uint8).view(-1))
                event = torch.cuda.Event()
                event.record()
                shadow.pending = (tokens, event)

    def bind_chunk(self, packet):
        """The staged rows are exactly the packet's contiguous logical spans."""
        group = getattr(packet, 'logical_group', None)
        if group is None or len(packet.request_ids) != 1:
            return
        spans = group.spans
        if not spans:
            return
        begin = spans[0].history_begin
        contiguous = all(s.physical_offset == spans[0].physical_offset + (s.history_begin - begin)
                         for s in spans) and spans[0].physical_offset == 0
        total = sum(s.count for s in spans)
        for layer in range(self.layers):
            shadow = self.shadow(packet.request_ids[0], layer)
            if shadow is None:
                continue
            with shadow.lock:
                pending, shadow.pending = shadow.pending, None
                if pending is not None and contiguous and total == pending[0] and begin % BLOCK == 0:
                    shadow.chunk = (begin, total // BLOCK, pending[1])

    # -- engine thread: promise --------------------------------------------------
    def launch(self, pipeline, refresh, job, lease, capture):
        """Snapshot every input and start the fast path, or return None (declined)."""
        try:
            reason, inputs = self._inputs(pipeline, refresh, job, lease, capture)
        except Exception as error:      # never block the original path
            reason, inputs = 'snapshot:' + type(error).__name__, None
        job.layer0_fast_declined = reason
        if inputs is None:
            self.count('declined:' + reason)
            return None
        return self.executor.submit(self._run, lease, job, inputs)

    def _inputs(self, pipeline, refresh, job, lease, capture):
        from .layerwise_pressure import extend_records
        from .request_plan import RequestPlan, selection_config
        rid = job.command.request_id
        if self.shadow(rid, 0) is None:
            return 'no_shadow', None
        controller = pipeline.pressure
        if not (getattr(controller, 'fast_prepare', False) and getattr(controller, 'deferred_maps', False)):
            return 'controller', None
        frame = controller.resolved_maps(rid)
        if frame is None:
            return 'no_maps', None
        origin, maps = frame
        packet = None
        with capture._publication_lock():
            for candidate in capture.queue:
                group = getattr(candidate, 'logical_group', None)
                if (group is not None and candidate.request_ids == (rid,) and
                        group.spans and group.spans[-1].history_end == job.history):
                    packet = candidate
        if packet is None or not packet.confirmed:
            return 'packet', None
        confirmation = getattr(packet, 'logical_confirmation', None)
        if confirmation is None or confirmation[1] is not None:
            return 'confirmation', None
        slices = tuple(packet.logical_slices)
        if not slices or any(getattr(s, 'layerwise_pressure', None) is None for s in slices):
            return 'slices', None
        memory = job.authority.memory
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
                               selection='pressure', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=BLOCK, gen_budget=cfg.gen_budget,
                          mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        mandatory = sorted(BLOCK * b for b in fit['hard_blocks'])
        layers, reason = [], None
        for index in range(min(self.layers, len(maps), len(pipeline.names))):
            shadow = self.shadow(rid, index)
            if shadow is None:
                reason = 'no_shadow'
                break
            with shadow.lock:
                window, window_event, chunk = shadow.window, shadow.window_event, shadow.chunk
            if window is None or chunk is None:
                reason = 'shadow_state'
                break
            if chunk[0] + chunk[1] * BLOCK != job.history:
                reason = 'chunk_extent'
                break
            rows = list(maps[index])
            if (tuple(r[2] for r in rows) != window or
                    any(r[3] != BLOCK or r[4] != BLOCK * i for i, r in enumerate(rows))):
                reason = 'window_map'
                break
            mapping = extend_records(rows, origin, job.history)
            if any(r[3] != BLOCK for r in mapping):
                reason = 'partial_block'
                break
            if lease.promise_extent != BLOCK * len(window):
                reason = 'extent'
                break
            layers.append(dict(index=index, shadow=shadow, mapping=mapping, window=window,
                               window_event=window_event, chunk=chunk,
                               layer=self._layer_window(pipeline.names[index], rid)))
        if not layers:
            return reason or 'no_layers', None
        start = torch.cuda.Event()
        start.record()
        return None, dict(rid=rid, controller=controller, history=job.history, slices=slices,
            mandatory=mandatory, budget=cfg.select_budget, sink=memory.config.sink_blocks,
            limit=(getattr(memory.config, 'prefill_budget', 0) or memory.config.select_budget) // BLOCK,
            layers=layers, start=start, stop_reason=reason, promised_at=time.perf_counter())

    def _layer_window(self, name, rid):
        from .codec import native_vllm_fp8_views
        from .native_pages import kernel_table
        from .window_assembly import LayerWindow, _runner_modules
        runner = self.runner
        module = _runner_modules(runner)[name]
        attn = module.attn
        if attn.kv_cache_dtype != 'fp8_e4m3':
            raise ValueError('Layer fast path qualifies FP8 e4m3 caches only')
        groups = {n: g for g, group in enumerate(runner.kv_cache_config.kv_cache_groups) for n in group.layer_names}
        index = runner.req_states.req_id_to_index[rid]
        table = kernel_table(runner, groups[attn.layer_name], index).clone()
        key, value = native_vllm_fp8_views(attn.kv_cache, module.num_kv_heads, module.head_dim)
        return LayerWindow(key, value, module.rotary_emb.cos_sin_cache, attn._k_scale, table)

    # -- fast-path thread ---------------------------------------------------------
    def _run(self, lease, job, x):
        published, selected, tables, skipped = 0, [], [], frozenset()
        timings = [('start', time.perf_counter() - x['promised_at'])]
        reason = x['stop_reason'] or ''
        for entry in x['layers']:
            index = entry['index']
            try:
                # Shadow buffers and cache views are inference tensors (created
                # by the refresh producer); update them under the same mode.
                with torch.inference_mode():
                    picked, skip, event = self._layer(x, entry, timings)
            except _Decline as decline:
                reason = f'layer{index}:{decline.reason}'
            except Exception as error:
                reason = f'layer{index}:error:{repr(error)[:160]}'
            else:
                lease.publish_layer(index, event)
                published += 1
                selected.append(picked)
                tables.append(entry['layer'].kernel_page_table)
                if index == 0:
                    skipped = skip
                timings.append((f'published{index}', time.perf_counter() - x['promised_at']))
                continue
            break
        for entry in x['layers'][published:]:
            with entry['shadow'].lock:
                entry['shadow'].invalidate()
        self.count(f'published_layers:{published}')
        if reason:
            self.count('stopped:' + ':'.join(reason.split(':')[:2]))
        return Result(published, reason, tuple(selected), skipped, tuple(tables), tuple(timings))

    def _layer(self, x, entry, timings):
        from .buffered_prefill_attention import accept_buffered_batch
        from .layerwise_pressure import native_lookup_scores
        rid, history, index = x['rid'], x['history'], entry['index']
        shadow = entry['shadow']
        staged = copy(x['controller'].histories[rid][index])
        for part in x['slices']:
            span = part.plan.span
            if span.history_end <= staged.end:
                continue
            if part.event is not None:
                part.event.synchronize()            # the original score-download fence
            _, owner_rid, logical, masses = part.layerwise_pressure
            if owner_rid != rid:
                raise _Decline('slice_owner')
            flags = accept_buffered_batch([staged], span.history_end, span.count, span.count,
                list(part.plan.rows), part.plan.starts, logical[index:index+1], span.local_begin,
                masses[index:index+1])
            if flags != [True]:
                raise _Decline('fold')
        if staged.end != history:
            raise _Decline('fold_extent')
        mapping = entry['mapping']
        scores = native_lookup_scores(staged, mapping)
        if scores is None:
            raise _Decline('native_scores')
        scores = list(scores)
        if any(not (s >= 0) or s == float('inf') for s in scores) or not any(s > 0 for s in scores):
            raise _Decline('scores')
        window = entry['window']
        chunk_begin, chunk_blocks, chunk_event = entry['chunk']
        picked, sources, skip = plan_layer0(history, mapping, scores, window, chunk_begin, chunk_blocks,
                                            x['mandatory'], x['sink'], x['limit'], x['budget'])
        if index != 0:
            skip = frozenset()   # the CPU path rebuilds every block of the other layers
        timings.append((f'picked{index}', time.perf_counter() - x['promised_at']))
        with shadow.lock:
            raw, val = shadow.buffers[shadow.active]
            out_raw, out_val = shadow.buffers[1 - shadow.active]
        waits = [x['start'], chunk_event] + ([entry['window_event']] if entry['window_event'] is not None else [])
        event = assemble_window(raw, val, out_raw, out_val, sources, skip, entry['layer'],
                                shadow.row_bytes, self.stream, waits)
        with shadow.lock:
            shadow.active = 1 - shadow.active
            shadow.window, shadow.window_event, shadow.chunk = tuple(picked), event, None
        return tuple(picked), frozenset(picked[i] for i in skip), event

    # -- producer thread: verification and shadow rebuild -------------------------
    def verify(self, result, index, records, skip_begins, rid, name):
        selected = tuple(r.history_begin for r in records)
        if selected != result.selected[index]:
            raise RuntimeError(f'Layer {index} fast path selected a different window than the CPU path')
        if frozenset(skip_begins) != (result.skipped if index == 0 else frozenset()):
            raise RuntimeError(f'Layer {index} fast path kept different resident blocks than the CPU plan')
        # The pages the fast path wrote must be the pages the CPU path would write.
        layer = self._layer_window(name, rid)
        cpu_table = layer.kernel_page_table
        pages = -(-BLOCK * len(selected) // layer.key.shape[1])
        table = result.tables[index]
        if (cpu_table.numel() < pages or table is None or table.numel() < pages or
                not torch.equal(cpu_table[:pages], table[:pages])):
            raise RuntimeError(f'Layer {index} fast path wrote a different page table than the CPU path')

    def rebuild_layer(self, rid, index, authority, records, reader, window_blocks, chunk_blocks):
        """Producer, right after it uploaded layer ``index`` itself: refresh that layer's shadow.

        Runs while the plan is still materializing (the forward still waits for
        later layers), so the prepared reader is valid. A failure only drops the
        shadow (the next refresh takes the original path); it never fails the
        refresh it is attached to.
        """
        if index >= self.layers:
            return False
        try:
            with torch.inference_mode():
                return self._rebuild(rid, index, authority, records, reader, window_blocks, chunk_blocks)
        except Exception as error:
            self.count('rebuild_failed:' + type(error).__name__)
            shadow = self.shadow(rid, index)
            if shadow is not None:
                with shadow.lock:
                    shadow.invalidate()
            return False

    def _rebuild(self, rid, index, authority, records, reader, window_blocks, chunk_blocks):
        row_bytes = BLOCK * authority.shape[2] * authority.shape[3]
        if (authority.raw_element_bytes != 1 or reader is None or not getattr(reader, 'native', False) or
                len(records) != window_blocks or any(r.count != BLOCK or not r.archived for r in records) or
                not reader.contains(records)):
            shadow = self.shadow(rid, index)
            if shadow is not None:
                with shadow.lock:
                    shadow.invalidate()
            return False
        shadow = self.shadow(rid, index, create=(self.runner.device, window_blocks, chunk_blocks, row_bytes))
        if shadow is None:
            self.count('no_shadow_capacity')
            return False
        if self.pinned is None or self.pinned.numel() < 2 * window_blocks * row_bytes:
            self.pinned = torch.empty(2 * window_blocks * row_bytes, dtype=torch.uint8, pin_memory=True)
        total_raw = authority.shape[0] * row_bytes
        regions = ((index * row_bytes, row_bytes), (total_raw + index * row_bytes, row_bytes))
        records = list(records)
        with shadow.lock:
            # The staged chunk rows are disjoint from the window rows: keep them.
            shadow.window = shadow.window_event = None
            raw, val = shadow.buffers[shadow.active]
        # Same bounded read size as the per-layer uploads (<= 256 records).
        for begin in range(0, window_blocks, 256):
            group = records[begin:begin + 256]
            n = len(group)
            host = self.pinned[2 * begin * row_bytes:2 * (begin + n) * row_bytes]
            reader.read(group, regions, destination=host)      # [raw K x n][V x n]
            with torch.cuda.stream(self.stream):
                raw[begin:begin + n].view(-1).copy_(host[:n * row_bytes], non_blocking=True)
                val[begin:begin + n].view(-1).copy_(host[n * row_bytes:], non_blocking=True)
        with torch.cuda.stream(self.stream):
            event = torch.cuda.Event()
            event.record(self.stream)
        event.synchronize()      # the pinned staging buffer is reused next time
        with shadow.lock:
            shadow.window = tuple(r.history_begin for r in records)
            shadow.window_event = event
        self.count('rebuilt')
        return True


def assemble_window(raw, val, out_raw, out_val, sources, skip, layer, row_bytes, stream, waits=()):
    """Write layer 0's new window from shadow rows and gather the next shadow.

    ``raw``/``val`` hold one 32-token record per row (original-content FP8 K and
    stored V bytes). Block i of the new window comes from row ``sources[i]`` and
    goes to dense position 32*i, except the plan's skipped indices, which keep
    their resident cache bytes. Same split-assembly kernel as the CPU path.
    """
    from .split_assembly import assemble_split_layer
    device = raw.device
    rebuild = [(i, j) for i, j in enumerate(sources) if i not in skip]
    with torch.cuda.stream(stream):
        for event in waits:
            stream.wait_event(event)
        if rebuild:
            index = torch.tensor([j for _, j in rebuild], dtype=torch.int64, device=device)
            bases = torch.tensor([BLOCK * i for i, _ in rebuild], dtype=torch.int64, device=device)
            assemble_split_layer(raw.view(-1), index, val.view(-1), index, bases, layer,
                layer_index=0, layers=1, block_tokens=BLOCK, raw_record_bytes=row_bytes,
                value_record_bytes=row_bytes, raw_dtype=torch.float8_e4m3fn)
        gather = torch.tensor(list(sources), dtype=torch.int64, device=device)
        torch.index_select(raw, 0, gather, out=out_raw[:len(sources)])
        torch.index_select(val, 0, gather, out=out_val[:len(sources)])
        event = torch.cuda.Event()
        event.record(stream)
    return event


def plan_layer0(history, mapping, scores, window, chunk_begin, chunk_blocks, mandatory, sink, limit, budget):
    """Layer 0's selection in history positions, each block's shadow row, and
    the plan's skipped blocks (indices into the selection).

    ``window`` is the previous layer-0 window in dense order (row i holds dense
    position BLOCK*i); rows after it hold the staged chunk. The store skips a
    remap exactly when the block is GPU-resident at the same dense position
    (``baked_pos == window_pos``), i.e. previous row == new row.
    """
    from . import _native
    begins = [r[2] for r in mapping]
    picked = _native.pressure_pick_positions(history, BLOCK, sink, limit, list(mandatory),
                                             begins, list(scores), budget)
    if len(picked) != len(window) or not set(mandatory) <= set(picked):
        raise _Decline('pick_extent')
    position = {b: i for i, b in enumerate(window)}
    sources = []
    for begin in picked:
        j = position.get(begin)
        if j is None:
            c = (begin - chunk_begin) // BLOCK
            if begin % BLOCK or not 0 <= c < chunk_blocks:
                raise _Decline('source')
            j = len(window) + c
        sources.append(j)
    skip = frozenset(i for i, j in enumerate(sources) if j < len(window) and j == i)
    return list(picked), sources, skip


class _Decline(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def install(runner):
    if not enabled():
        return None
    device = getattr(runner, 'device', None)
    if device is None or torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        return None
    fast = getattr(runner, '_kvmem_layer0_fast', None)
    if fast is None:
        fast = runner._kvmem_layer0_fast = Layer0Fast(runner)
    return fast
