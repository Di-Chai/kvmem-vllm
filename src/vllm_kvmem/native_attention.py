"""Bounded native-QK side outputs for FlashInfer FA2 and SM12x XQA.

One runner, one CUDA stream and one public batch own the scratch. Each selected
layer retains compact LSE pieces at stable addresses. Normalization runs outside
the graph only for the actual sampled rows, before Action's existing D2H.
"""
from contextvars import ContextVar
from collections import Counter
from functools import wraps
import importlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from threading import Lock

import numpy as np
import torch
import triton
import triton.language as tl

from .native_attention_build import UnsupportedNativeStats, source_overlay, prefill_module, xqa_module


def piece_layout(starts, counts, length):
    """Dense sixteen-token pieces split at every KVMem logical-segment boundary."""
    cursor = pieces = 0
    result = []
    for start, count in zip(starts, counts, strict=True):
        if start != cursor or count <= 0:
            raise ValueError('Native attention segments must partition the full context')
        n = (start + count + 15) // 16 - start // 16
        result.append((start, count, pieces, n))
        cursor += count
        pieces += n
    if cursor != length or not length:
        raise ValueError('Native attention segments omit visible context')
    return result, pieces


def _layout_array(starts, counts, length):
    start, count = np.asarray(starts, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    if (start.ndim != 1 or start.shape != count.shape or not start.size or
            start[0] != 0 or np.any(count <= 0) or
            not np.array_equal(start[1:], start[:-1] + count[:-1]) or start[-1]+count[-1] != length):
        raise ValueError('Native attention segments must partition the full context')
    sizes = (start+count+15)//16 - start//16
    ends = np.cumsum(sizes)
    return np.stack((start,count,ends-sizes,sizes),axis=1).astype(np.int32), int(ends[-1])


@triton.jit
def _map_pieces(Ranges, Pieces, SEGMENTS: tl.constexpr, CONTEXT: tl.constexpr, BLOCK: tl.constexpr):
    request, segment = tl.program_id(0), tl.program_id(1)
    base = (request * SEGMENTS + segment) * 4
    start, count, piece = tl.load(Ranges + base), tl.load(Ranges + base + 1), tl.load(Ranges + base + 2)
    offset = tl.arange(0, BLOCK)
    token = start + offset
    tl.store(Pieces + request * CONTEXT + token, piece + token // 16 - start // 16, mask=offset < count)


@triton.jit
def _clear_lse(Lse, Meta, Control, HEADS: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tile * TILE + tl.arange(0, TILE)
        tl.store(Lse + (row * HEADS + head) * WIDTH + x, -float('inf'), mask=(x < width) & (x < WIDTH))


@triton.jit
def _row_lse(Lse, Meta, Control, Selected, Valid, RowLse, HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, head = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        width = tl.load(Meta + row * 3 + 1)
        x = tl.arange(0, BLOCK)
        values = tl.load(Lse + (row * HEADS + head) * WIDTH + x, mask=(x < width) & (x < WIDTH), other=-float('inf'))
        maximum = tl.max(values, 0)
        total = tl.sum(tl.exp(values - maximum), 0)
        value=maximum+tl.log(total)
        tl.store(RowLse + row * HEADS + head, value)
        if not ((value > -float('inf')) & (value < float('inf'))):
            tl.atomic_min(Valid,0)


@triton.jit
def _segment_scores(Lse, RowLse, Ranges, Meta, Control, Selected, Out,
                    HEADS: tl.constexpr, WIDTH: tl.constexpr, SEGMENTS: tl.constexpr,
                    H: tl.constexpr, B: tl.constexpr):
    row, segment = tl.load(Selected+tl.program_id(0)), tl.program_id(1)
    if row < tl.load(Control):
        request, segments = tl.load(Meta + row * 3), tl.load(Meta + row * 3 + 2)
        if segment < segments:
            row_width=tl.load(Meta+row*3+1)
            base = (request * SEGMENTS + segment) * 4
            first, count = tl.load(Ranges + base + 2), tl.load(Ranges + base + 3)
            h, p = tl.arange(0, H), tl.arange(0, B)
            normalizer = tl.load(RowLse + row * HEADS + h, mask=h < HEADS, other=0.)
            mass=0.
            for tile in range(tl.cdiv(count,B)):
                column=tile*B+p
                logits = tl.load(Lse + (row * HEADS + h[:, None]) * WIDTH + first + column[None, :],
                                 mask=(h[:, None] < HEADS) & (column[None, :] < count) &
                                      (first+column[None,:]<row_width), other=-float('inf'))
                probability = tl.exp(logits - normalizer[:, None])
                mass += tl.sum(tl.sum(probability, 1), 0) / HEADS
            tl.store(Out + row * SEGMENTS + segment, mass)


_active: ContextVar = ContextVar('kvmem_native_attention', default=None)


def _install_xqa_dispatch():
    module = importlib.import_module('flashinfer.xqa')
    original = module.get_xqa_module
    if getattr(original, '_kvmem_native_stats', False):
        return

    @wraps(original)
    def dispatch(*args, **kwargs):
        active = _active.get()
        if active is None:
            return original(*args, **kwargs)
        bank, name = active
        q_dtype, kv_dtype, page, dim, group, sliding, out_dtype, q_len = args[:8]
        ragged = args[8] if len(args) > 8 else kwargs.get('use_ragged_q', False)
        if sliding or dim > 256 or kv_dtype != torch.float8_e4m3fn:
            bank.reasons[name, 'decode'] = 'unsupported_xqa_geometry'
            return original(*args, **kwargs)
        try:
            geometry = q_dtype, kv_dtype, page, dim, group, out_dtype, q_len, ragged
            built = bank.xqa_modules.get(geometry)
            if built is None:
                if torch.cuda.is_current_stream_capturing():
                    bank.reasons[name, 'decode'] = 'xqa_specialization_not_warmed'
                    return original(*args, **kwargs)
                built = xqa_module(bank.directory, *geometry)
                bank.xqa_modules[geometry] = built
        except (UnsupportedNativeStats, RuntimeError) as error:
            bank.reasons[name, 'decode'] = 'xqa_build:' + str(error).splitlines()[0][:160]
            return original(*args, **kwargs)
        bank.reasons[name, 'decode'] = None

        def run(run_sm90, sm, kh, window, qscale, output, oscale, q, sinks,
                k, v, ksf, vsf, table, maxlen, lengths, batch, kvscale,
                semaphores, workspace, pdl, qlen, qcu, mask):
            # SM12x uses this warp-specialized path, with the original scales.
            built.xqa_wrapper(False, sm, kh, window,
                1.0 if isinstance(qscale, torch.Tensor) else qscale,
                qscale if isinstance(qscale, torch.Tensor) else None,
                output, oscale, q, sinks, k, v, ksf, vsf, table, maxlen, lengths, batch,
                1.0 if isinstance(kvscale, torch.Tensor) else kvscale,
                kvscale if isinstance(kvscale, torch.Tensor) else None,
                qlen, qcu, mask, semaphores, workspace, pdl,
                bank.rows, bank.requests, bank.pieces, bank.lses[name], bank.context, bank.width)
        return SimpleNamespace(xqa=run)

    dispatch._kvmem_native_stats = True
    module.get_xqa_module = dispatch


class NativeAttentionStats:
    def __init__(self, runner, options, output):
        from . import _native
        self.options = options
        self.directory = str(Path(output) / 'native_attention_build')
        self.output = Path(output) / 'native_attention_stats.json'
        self.mode = options.action_native_scores
        self.reasons, self.counts = {}, Counter()
        self.xqa_modules, self.prefill_modules = {}, {}
        self.plans, self.slots = {}, {}
        self.reduced = set()
        self.invalid = set()
        self.validation_failures = 0
        self.armed_rows = self.normalized_rows = 0
        self.disabled = None
        self.source_fingerprint = None
        self.report_calls = 0
        self.report_lock = Lock()
        self.allocated_bytes = 0
        self.gpu_bytes = self.host_bytes = 0
        self.verification = {'comparisons': 0, 'max_abs': 0., 'l1_max': 0.}
        self.rows = None
        self.disarm()
        modules = dict(runner.model.named_modules())
        ids = [int(re.search(r'layers\.(\d+)\.', name).group(1)) for name in runner._kvmem_layers]
        selected = _native.resolve_action_layers(options.action_layers, ids)
        layers = [modules[name] for name, number in zip(runner._kvmem_layers, ids) if number in selected]
        self.names = [layer.attn.layer_name for layer in layers]
        if torch.cuda.get_device_capability()[0] != 12:
            self.disabled = 'unsupported_architecture_requires_sm12x'
            return
        try:
            self.source_fingerprint = source_overlay(self.directory).name
        except UnsupportedNativeStats as error:
            self.disabled = str(error)
            return
        geometry = {(layer.num_heads, layer.head_dim) for layer in layers}
        if len(geometry) != 1:
            self.disabled = 'nonuniform_head_geometry'
            return
        self.heads, _ = geometry.pop()
        self.max_rows = options.action_native_max_rows
        self.context = options.memory.select_budget + options.memory.gen_budget + runner.max_num_tokens
        self.max_segments = math.ceil(self.context / options.memory.block_tokens) + 2 * runner.max_num_reqs + 136
        self.width = math.ceil(self.context / 16) + self.max_segments
        shapes = {'rows': ((runner.max_num_tokens,), torch.int32),
            'requests': ((self.max_rows,), torch.int32),
            'pieces': ((runner.max_num_reqs, self.context), torch.int32),
            'ranges': ((runner.max_num_reqs, self.max_segments, 4), torch.int32),
            'meta': ((self.max_rows, 3), torch.int32), 'control': ((1,), torch.int32),
            'selected': ((self.max_rows,),torch.int32),
            'row_lse': ((self.max_rows, self.heads), torch.float32),
            'valid_flags': ((len(layers),), torch.int32)}
        self.allocated_bytes = sum(math.prod(shape) * 4 for shape, _ in shapes.values())
        self.allocated_bytes += len(layers) * self.max_rows * self.max_segments * 4
        self.allocated_bytes += len(layers) * self.max_rows * self.heads * self.width * 4
        self.gpu_bytes = self.allocated_bytes
        host_fields = ('rows', 'requests', 'ranges', 'meta', 'valid_flags', 'selected')
        self.host_bytes = sum(math.prod(shapes[field][0])*4 for field in host_fields)
        self.allocated_bytes += self.host_bytes
        if self.allocated_bytes > options.action_native_max_bytes:
            self.disabled = 'startup_buffer_capacity'
            self.allocated_bytes = 0
            self.gpu_bytes = self.host_bytes = 0
            return
        device = layers[0].q_norm.weight.device
        for name, (shape, dtype) in shapes.items():
            setattr(self, name, torch.empty(shape, dtype=dtype, device=device))
        self.host = {name: torch.empty(shapes[name][0],dtype=torch.int32,pin_memory=True) for name in host_fields}
        self.upload_event = None
        self.masses = {name: torch.empty((self.max_rows, self.max_segments), dtype=torch.float32, device=device)
                       for name in self.names}
        self.lses = {name: torch.empty((self.max_rows, self.heads, self.width), dtype=torch.float32, device=device)
                     for name in self.names}
        self.ranges.zero_()
        self.disarm()
        _install_xqa_dispatch()
        for layer in layers:
            self._wrap(layer.attn.impl, layer.attn.layer_name)

    def disarm(self):
        if self.rows is not None:
            self.rows.fill_(-1)
            self.control.zero_()
        self.plans, self.slots = {}, {}
        self.reduced = set()
        self.invalid = set()
        self.validation_dirty = False
        self.frame_invalid = False
        self.frame_success = Counter()
        self.selected_slots,self.request_reasons={},{}
        if self.rows is not None:
            self.valid_flags.fill_(1)

    def arm(self, plans):
        self.disarm()
        if self.disabled:
            return
        total = sum(len(plan['rows']) for plan in plans.values())
        if total > self.max_rows:
            self.batch_reason = 'sample_row_capacity'
            return
        self.batch_reason = None
        meta, requests, row_indices, sample_ids = [], [], [], []
        # Pinned metadata has one bounded owner. Its previous DMA must finish
        # before the CPU modifies it; graph/input tensor addresses never change.
        if self.upload_event is not None and not self.upload_event.query():
            self.upload_event.synchronize()
        self.host['ranges'].zero_()
        self.host['rows'].fill_(-1)
        for request_id, plan in plans.items():
            if not plan['rows']:
                continue
            if '_layout' in plan:
                layout,pieces=plan['_layout'],plan['_width']
            else:
                layout, pieces = _layout_array(plan['starts'], plan['counts'], plan['local_begin'] + plan['count'])
            if (len(layout) > self.max_segments or pieces > self.width or
                    plan['local_begin'] + plan['count'] > self.context):
                self.disarm()
                self.batch_reason = 'segment_or_context_capacity'
                return
            i, first, n = plan['index'], len(meta), len(plan['rows'])
            self.host['ranges'].numpy()[i, :len(layout)] = layout
            segments=plan.get('_segments',len(layout))
            self.slots[request_id] = first, n, segments
            for row in plan['rows']:
                sample_ids.append(len(meta))
                row_indices.append(plan['begin'] + row)
                meta.append((i, pieces, segments))
                requests.append(i)
        if meta:
            self.armed_rows += len(meta)
            self.host['meta'].numpy()[:len(meta)] = meta
            self.host['requests'].numpy()[:len(meta)] = requests
            self.host['rows'].numpy()[row_indices] = sample_ids
            for name in ('rows','ranges'):
                getattr(self,name).copy_(self.host[name],non_blocking=True)
            for name in ('meta','requests'):
                getattr(self,name)[:len(meta)].copy_(self.host[name][:len(meta)],non_blocking=True)
            self.upload_event = torch.cuda.Event()
            self.upload_event.record()
            self.control.fill_(len(meta))
            _map_pieces[(max(plan['index'] for plan in plans.values())+1, max(m[2] for m in meta))](self.ranges, self.pieces,
                self.max_segments, self.context, triton.next_power_of_2(self.options.memory.block_tokens))
            self.active_rows, self.active_segments = len(meta), max(m[2] for m in meta)
            self.reduction_rows=len(meta)
            self.host['selected'].numpy()[:len(meta)]=np.arange(len(meta),dtype=np.int32)
            self.selected[:len(meta)].copy_(self.host['selected'][:len(meta)],non_blocking=True)
            self.upload_event.record()
            for lse in self.lses.values():
                _clear_lse[(len(meta), self.heads, triton.cdiv(max(m[1] for m in meta), 256))](
                    lse, self.meta, self.control, self.heads, self.width, 256)
        self.plans = plans

    def resolve(self, plans):
        """Bind exact sampled rows/logical segments after native forward."""
        if self.disabled or getattr(self,'batch_reason',None):
            return
        if self.upload_event is not None and not self.upload_event.query():
            self.upload_event.synchronize()
        selected=[];maximum=0
        for request_id,plan in plans.items():
            if not plan['rows']:
                continue
            if request_id not in self.slots:
                self.request_reasons[request_id]='candidate_not_prepared'
                continue
            try:
                candidate=self.plans[request_id]
                if '_layout' in candidate:
                    extent=plan['local_begin']+plan['count']
                    end=int(np.searchsorted(candidate['starts'],extent-1,side='right'))
                    expected=np.stack((candidate['starts'][:end],
                        np.minimum(candidate['counts'][:end],extent-candidate['starts'][:end]),
                        candidate['history'][:end]),axis=1)
                    actual=np.stack((plan['starts'],plan['counts'],plan['history']),axis=1)
                    if not np.array_equal(expected,actual):
                        raise ValueError('candidate logical history changed')
                else:
                    raise ValueError('candidate cached layout missing')
                first,n,_=self.slots[request_id]
                source_rows=self.plans[request_id]['rows']
                row_ids=[first+source_rows.index(row) for row in plan['rows']]
            except (ValueError,IndexError) as error:
                self.request_reasons[request_id]='candidate_domain:'+str(error)
                continue
            self.selected_slots[request_id]=(len(selected),len(row_ids))
            selected.extend(row_ids)
            segments=len(plan['starts'])
            self.slots[request_id]=(first,n,segments)
            maximum=max(maximum,self.plans[request_id].get('_segments',segments))
        self.reduction_rows=len(selected)
        self.active_segments=maximum
        if selected:
            self.host['selected'].numpy()[:len(selected)]=selected
            self.selected[:len(selected)].copy_(self.host['selected'][:len(selected)],non_blocking=True)
            self.upload_event=torch.cuda.Event();self.upload_event.record()

    def _wrap(self, impl, name):
        original = impl.forward

        @wraps(original)
        def forward(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            if attn_metadata is None:
                return original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            wrapper = getattr(getattr(attn_metadata, 'prefill', None), 'wrapper', None)
            saved = None
            if wrapper is not None:
                # FA2 applies q_scale even to unquantized Q; the legacy scorer
                # only applies it for FP8 Q. That is a domain change, rather
                # than tensor-core rounding, and must use the legacy fallback.
                if (attn_metadata.q_data_type_prefill in (torch.bfloat16,torch.float16) and
                        getattr(layer,'_q_scale_float',1.0) != 1.0):
                    self.reasons[name, 'prefill'] = 'q_scale_domain_mismatch'
                elif (wrapper._backend != 'fa2' or wrapper._custom_mask_buf is not None or
                        wrapper._pos_encoding_mode != 'NONE' or not wrapper._causal or
                        wrapper._window_left != -1 or wrapper._logits_soft_cap not in (0, None) or
                        query.shape[-1] > 256):
                    self.reasons[name, 'prefill'] = 'unsupported_prefill_backend_or_mask'
                else:
                    try:
                        geometry = (attn_metadata.q_data_type_prefill, torch.float8_e4m3fn,
                                    output.dtype, wrapper._paged_kv_indptr_buf.dtype, query.shape[-1])
                        built = self.prefill_modules.get(geometry)
                        if built is None:
                            if torch.cuda.is_current_stream_capturing():
                                raise UnsupportedNativeStats('prefill_specialization_not_warmed')
                            built = prefill_module(self.directory, *geometry)
                            self.prefill_modules[geometry] = built
                        saved = wrapper._cached_module

                        def paged_run(*run_args):
                            # Original FA2 ABI: 16 common parameters, then optional
                            # mask/ALiBi fields, softcap and the effective QK scale.
                            built.paged_run(*run_args[:16], self.rows, self.requests, self.pieces, self.lses[name],
                                self.context, self.width, attn_metadata.num_decode_tokens, run_args[23])
                        wrapper._cached_module = SimpleNamespace(paged_run=paged_run)
                        self.reasons[name, 'prefill'] = None
                    except (UnsupportedNativeStats, RuntimeError) as error:
                        self.reasons[name, 'prefill'] = 'prefill_build:' + str(error).splitlines()[0][:160]
            token = _active.set((self, name))
            try:
                result = original(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            finally:
                _active.reset(token)
                if saved is not None:
                    wrapper._cached_module = saved
            return result
        impl.forward = forward

    def validate(self):
        """Fence once after all owners' scoring and downloads are enqueued."""
        if self.validation_dirty:
            self.host['valid_flags'].copy_(self.valid_flags, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            event.synchronize()
            self.validation_dirty = False
            self.invalid.update(name for name, valid in zip(self.names, self.host['valid_flags'].tolist())
                                if not valid)
            if self.invalid:
                self.frame_invalid = True
                self.validation_failures += 1
                self.counts.subtract(self.frame_success)
                self.frame_success.clear()
        return not self.frame_invalid

    def score(self, name, request_id, prefill, *, defer_validation=False):
        phase = 'prefill' if prefill else 'decode'
        reason = self.disabled or getattr(self, 'batch_reason', None)
        if not reason:
            reason=self.request_reasons.get(request_id)
        if not reason and request_id not in self.slots:
            reason = 'unprepared_forward'
        if not reason:
            reason = self.reasons.get((name, phase), 'backend_not_observed')
        if not reason and self.frame_invalid:
            reason = 'invalid_native_mass'
        if reason:
            self.counts['fallback:' + phase + ':' + reason] += 1
            return None
        if name not in self.reduced:
            self.normalized_rows += self.reduction_rows
            lse = self.lses[name]
            _row_lse[(self.reduction_rows, self.heads)](lse, self.meta, self.control,self.selected,
                self.valid_flags[self.names.index(name)],self.row_lse,
                self.heads, self.width, triton.next_power_of_2(self.width), num_warps=8)
            _segment_scores[(self.reduction_rows, self.active_segments)](lse, self.row_lse, self.ranges,
                self.meta, self.control,self.selected,self.masses[name], self.heads, self.width, self.max_segments,
                triton.next_power_of_2(self.heads), triton.next_power_of_2(math.ceil(self.options.memory.block_tokens / 16) + 1))
            self.reduced.add(name)
            # GPU-only check here: a host readback inside the request/layer
            # loop would prevent enqueueing the remaining scoring work.
            self.validation_dirty = True
        if not defer_validation:
            self.validate()
        first, rows, segments = self.slots[request_id]
        result = self.masses[name][first:first+rows, :segments]
        if request_id in self.selected_slots:
            selected_begin,selected_count=self.selected_slots[request_id]
            result=self.masses[name][:,:segments].index_select(0,
                self.selected[selected_begin:selected_begin+selected_count].long())
        # A missed producer (including a graph captured before instrumentation)
        # cannot publish uninitialized or stale mass. prepare() clears LSE before
        # every public forward, including graph replays; the initial empty rows
        # disable statistics during private forwards and warmup.
        if self.frame_invalid:
            self.counts['fallback:' + phase + ':invalid_native_mass'] += 1
            return None
        self.counts[self.mode + ':' + phase] += 1
        self.frame_success[self.mode + ':' + phase] += 1
        return result

    def compare(self, native, reference):
        difference = (native - reference).abs()
        self.verification['comparisons'] += 1
        self.verification['max_abs'] = max(self.verification['max_abs'], float(difference.max().item()))
        self.verification['l1_max'] = max(self.verification['l1_max'], float(difference.sum(-1).max().item()))

    def report(self, *, force=False):
        self.report_calls += 1
        if not force and self.report_calls != 1 and self.report_calls % 128:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.report_lock:
            temporary = self.output.with_suffix('.tmp')
            temporary.write_text(json.dumps({'requested_mode': self.mode, 'allocated_bytes': self.allocated_bytes,
                'persistent_gpu_bytes': self.gpu_bytes, 'persistent_pinned_host_bytes': self.host_bytes,
                'disabled_reason': self.disabled, 'layer_phase_counts': dict(self.counts),
                'source_fingerprint': self.source_fingerprint, 'verification': self.verification,
                'validation_failures': self.validation_failures,
                'armed_candidate_rows': self.armed_rows,
                'normalized_layer_sample_rows': self.normalized_rows}, indent=2) + '\n')
            temporary.replace(self.output)


def install_native_attention(runner, options, output):
    if getattr(options, 'action_native_scores', 'off') == 'off':
        return None
    return NativeAttentionStats(runner, options, output)
