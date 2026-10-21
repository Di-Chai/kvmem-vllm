"""Opt-in GPU component experiment; production capture is not modified.

Compare the current full-batch FP8 capture packing chain with a two-kernel
candidate. Timings include output allocation, exclude input/pointer-table setup,
DMA, CPU archive publication and model execution. A gain here is not an
end-to-end throughput claim. This candidate intentionally excludes split lanes.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch
import triton
import triton.language as tl


@triton.jit
def _header(Starts, Global, Local, Rejected, Out, N: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    begin = tl.load(Starts + i, i < N + 1, 0).to(tl.int64)
    tl.store(Out + i, begin, i < N + 1)
    glob = tl.load(Global + begin, i < N, 0)
    local = tl.load(Local + begin, i < N, 0)
    rejected = tl.load(Rejected + i, i < N, 0).to(tl.int64)
    tl.store(Out + N + 1 + i, glob, i < N)
    tl.store(Out + 2 * N + 1 + i, local, i < N)
    tl.store(Out + 3 * N + 1 + i, rejected, i < N)


@triton.jit
def _fp8_payload(Pointers, Scales, Out, TOKENS, WIDTH: tl.constexpr,
                 LAYERS: tl.constexpr, BLOCK: tl.constexpr):
    layer = tl.program_id(1)
    kind = tl.program_id(2)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pointer = tl.load(Pointers + layer * 4 + kind)
    stride = tl.load(Pointers + layer * 4 + 2 + kind)
    src = pointer.to(tl.pointer_type(tl.bfloat16))
    x = tl.load(src + offset // WIDTH * stride + offset % WIDTH,
                offset < TOKENS * WIDTH, 0).to(tl.float32)
    if kind == 1:
        scale = tl.load(Scales + layer).to(tl.float32)
        # The reference divides in FP32 before clamping and FP8 conversion.
        # Do not replace this with approximate reciprocal multiplication.
        x = tl.div_rn(x, scale)
    x = tl.minimum(tl.maximum(x, -448., propagate_nan=tl.PropagateNan.ALL),
                   448., propagate_nan=tl.PropagateNan.ALL)
    byte = x.to(tl.float8e4nv, fp_downcast_rounding='rtne').to(tl.uint8, bitcast=True)
    tl.store(Out + (kind * LAYERS + layer) * TOKENS * WIDTH + offset,
             byte, offset < TOKENS * WIDTH)


class Candidate:
    """Stable source addresses are owned by this component fixture, not serving."""
    def __init__(self, captured, scales):
        if not captured:
            raise ValueError('Empty capture')
        geometry = captured[0].key.shape[1:]
        self.width = geometry[0] * geometry[1]
        pointers = []
        for item in captured:
            for value in (item.key, item.value):
                if (value.dtype != torch.bfloat16 or value.device.type != 'cuda' or
                        value.ndim != 3 or value.shape[1:] != geometry or
                        value.stride(-1) != 1 or value.stride(1) != geometry[1]):
                    raise ValueError('Candidate requires stable BF16 rows with contiguous heads')
            pointers.append([item.key.data_ptr(), item.value.data_ptr(),
                             item.key.stride(0), item.value.stride(0)])
        self.sources = tuple(captured)
        self.scales = scales
        self.pointers = torch.tensor(pointers, device=scales.device, dtype=torch.int64)

    def __call__(self, batch, rejected):
        n, tokens = batch.num_reqs, batch.num_tokens
        layers = len(self.sources)
        header_bytes = (4 * n + 1) * 8
        result = torch.empty(header_bytes + 2 * layers * tokens * self.width,
                             dtype=torch.uint8, device=self.scales.device)
        _header[(1,)](batch.query_start_loc, batch.positions,
            self.sources[0].positions, rejected, result[:header_bytes].view(torch.int64),
            n, triton.next_power_of_2(n + 1))
        _fp8_payload[(triton.cdiv(tokens * self.width, 1024), layers, 2)](
            self.pointers, self.scales, result[header_bytes:], tokens, self.width,
            layers, 1024, enable_fp_fusion=False)
        return result


def fixture(tokens, layers, heads, dim, requests, strided):
    from vllm_kvmem.capture_runtime import CaptureRuntime
    if tokens < requests:
        raise ValueError('Each owner must contain at least one input row')
    stride = 2 if strided else 1
    shape = (tokens * stride, heads, dim)
    scales = torch.tensor([(.125, .3, 1., 1.75, 3.25, 7.)[i % 6]
                           for i in range(layers)], device='cuda', dtype=torch.float32)
    starts = torch.tensor([i * tokens // requests for i in range(requests + 1)],
                          device='cuda', dtype=torch.int32)
    local = torch.arange(tokens, device='cuda', dtype=torch.int64) + 49152
    global_positions = local + 65536
    captures = []
    for i in range(layers):
        key = (torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 20)[::stride]
        value = (torch.randn(shape, device='cuda', dtype=torch.bfloat16) * (20 + i))[::stride]
        # Saturation, signed zero, subnormal and conversion boundaries matter
        # as much as normal model values. Nonfinite behavior is also compared.
        edges = torch.tensor([0., -0., 448., -448., 464., -464., .0009765625,
                              -.0009765625, float('inf'), -float('inf'),
                              float('nan'), -float('nan')], device='cuda', dtype=torch.bfloat16)
        key[0].flatten()[:len(edges)].copy_(edges)
        value[0].flatten()[:len(edges)].copy_(edges)
        captures.append(NS(key=key, value=value, positions=local))
    batch = NS(num_reqs=requests, num_tokens=tokens, query_start_loc=starts,
               positions=global_positions)
    rejected = torch.arange(requests, device='cuda', dtype=torch.int32).remainder(2)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype, runtime.scales = torch.float8_e4m3fn, scales
    candidate = Candidate(captures, scales)
    def reference():
        return runtime._capture_payload(captures, batch, rejected, tuple(range(requests)))[0]
    return reference, lambda: candidate(batch, rejected), candidate


def measure(call, repeats):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gpu, submit = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        started = time.perf_counter()
        result = call()
        submit.append((time.perf_counter() - started) * 1000)
        end.record()
        end.synchronize()
        gpu.append(begin.elapsed_time(end))
        del result
    return dict(gpu_interval_ms=gpu, host_submit_ms=submit,
                peak_temporary_allocated_bytes=torch.cuda.max_memory_allocated() - base)


def run(args):
    torch.manual_seed(71)
    cases = []
    for tokens in args.tokens:
        reference, candidate, _owner = fixture(tokens, args.layers, args.heads, args.dim,
                                                args.requests, args.strided)
        actual, expected = candidate(), reference()
        torch.cuda.synchronize()
        mismatch = int((actual != expected).sum().item())
        if mismatch:
            offsets = (actual != expected).nonzero().flatten()[:20].cpu().tolist()
            raise RuntimeError(f'Fused pack changed {mismatch} bytes; first offsets={offsets}')
        size = actual.numel()
        del actual, expected
        for _ in range(args.warmup):
            reference(); candidate()
        torch.cuda.synchronize()
        rounds = []
        combined = defaultdict(list)
        for mode in ('reference', 'candidate', 'candidate', 'reference'):
            values = measure(reference if mode == 'reference' else candidate, args.repeats)
            rounds.append(dict(mode=mode, **values))
            combined[mode].extend(values['gpu_interval_ms'])
        medians = {mode: statistics.median(values) for mode, values in combined.items()}
        cases.append(dict(tokens=tokens, bytes=size, byte_equal=True, rounds=rounds,
            median_gpu_interval_ms=medians,
            component_speedup=medians['reference']/medians['candidate']))
    return dict(scope=__doc__, gpu=torch.cuda.get_device_name(), layers=args.layers,
        heads=args.heads, dim=args.dim, requests=args.requests, strided=args.strided,
        warmup=args.warmup, repeats_per_round=args.repeats, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu', action='store_true', help='Explicit opt-in; reserve the GPU externally')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', default=[128, 1024, 2048])
    parser.add_argument('--layers', type=int, default=17)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--requests', type=int, default=1)
    parser.add_argument('--strided', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error('GPU component experiments require --run-gpu and external GPU ownership')
    args.output.write_text(json.dumps(run(args), indent=2) + '\n')
