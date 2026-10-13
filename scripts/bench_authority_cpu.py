#!/usr/bin/env python3
"""Isolate authority CPU work; no GPU, NVMe I/O or model forward is measured.

Compare an earlier authority.py with the working tree using --baseline-source.
History-depth cases use small tensors and the real native repository to isolate
directory overhead; sealed-flush cases use the model's 6 MiB record geometry.
Both use a synchronous discard-only archive, so results are not serving TPS.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig


class _Completed:
    def done(self):
        return True

    def result(self, timeout=None):
        return None


class _DiscardArchive:
    def __init__(self, record_bytes):
        self.record_bytes = record_bytes
        self.admit = True

    def write(self, namespace, leases, payload):
        assert len(payload) == len(leases) * self.record_bytes
        return _Completed() if self.admit else None

    def reclaim(self, namespace, entry):
        return True


def _make(cls, history, extra, geometry):
    layers, heads, dim = geometry
    config = KVMemConfig(block_tokens=128, select_budget=max(128, history),
                        gen_budget=extra, repository_blocks=(max(128, history)+extra)//128+16,
                        sink_blocks=1, recent_blocks=0)
    archive = _DiscardArchive(layers * 128 * heads * dim * 3)
    authority = cls('cpu-bench', config, archive, layers, heads, dim, resident_records=16)
    return authority, archive


def _summary(samples):
    ordered = sorted(samples)
    return dict(median_us=statistics.median(samples), min_us=min(samples),
                max_us=max(samples), samples_us=ordered)


def history_case(cls, history, flush_loops, decode_steps):
    # Tiny tensor geometry keeps historical seeding inexpensive. Directory and
    # native lease/Store behavior still use actual accepted records/generations.
    authority, _ = _make(cls, history, ((decode_steps*3+127)//128+1)*128, (2, 1, 2))
    raw = torch.ones((2, 16*128, 1, 2), dtype=torch.bfloat16)
    values = torch.zeros_like(raw, dtype=torch.uint8)
    for begin in range(0, history, 16*128):
        count = min(16*128, history-begin)
        assert authority.append(raw[:, :count], values[:, :count], history_begin=begin,
                                local_rope_begin=begin)
        authority.poll()
    for _ in range(50):
        assert authority.flush()
    start = time.perf_counter_ns()
    for _ in range(flush_loops):
        assert authority.flush()
    empty_flush_us = (time.perf_counter_ns()-start)/flush_loops/1000
    raw = torch.ones((2, 3, 1, 2), dtype=torch.bfloat16)
    values = torch.zeros_like(raw, dtype=torch.uint8)
    start = time.perf_counter_ns()
    for step in range(decode_steps):
        begin = history + step*3
        assert authority.append(raw, values, history_begin=begin, local_rope_begin=begin)
    append_us = (time.perf_counter_ns()-start)/decode_steps/1000
    assert authority.history_end == history + decode_steps*3
    authority.poll()
    assert authority.close()
    return empty_flush_us, append_us


def sealed_case(cls, blocks, raw, values):
    authority, archive = _make(cls, 128, max(128, blocks*128), (16, 4, 256))
    archive.admit = False
    assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
    archive.admit = True
    start = time.perf_counter_ns()
    assert authority.flush()
    elapsed = (time.perf_counter_ns()-start)/1000
    assert all(record.submitted and record.digest for record in authority.records.values())
    authority.poll()
    assert authority.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--flush-loops', type=int, default=2000)
    parser.add_argument('--decode-steps', type=int, default=256)
    args = parser.parse_args()
    if min(args.repeats, args.flush_loops, args.decode_steps) <= 0:
        parser.error('all repetition counts must be positive')
    torch.set_num_threads(1)
    implementations = {'current': RequestAuthority}
    sources = {'current': Path(sys.modules[RequestAuthority.__module__].__file__)}
    if args.baseline_source:
        name = 'vllm_kvmem._cpu_benchmark_baseline_authority'
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        implementations['baseline'] = module.RequestAuthority
        sources['baseline'] = args.baseline_source
    result = dict(scope='CPU authority only, no GPU or storage performance claims',
                  archive='synchronous discard-only', torch_threads=1,
                  source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in sources.items()},
                  history_geometry=dict(layers=2, heads=1, head_dim=2, block_tokens=128),
                  sealed_geometry=dict(layers=16, heads=4, head_dim=256,
                                       block_tokens=128, record_bytes=6291456),
                  repeats=args.repeats, flush_loops=args.flush_loops,
                  decode_steps=args.decode_steps, decode_tokens_per_append=3,
                  history={}, sealed_flush={})
    names = list(implementations)
    for history in (65536, 262144):
        cases = {name: {'empty_flush': [], 'append_3_tokens': []} for name in names}
        for repeat in range(args.repeats):
            for name in names[::1 if repeat % 2 else -1]:
                empty, append = history_case(implementations[name], history,
                                             args.flush_loops, args.decode_steps)
                cases[name]['empty_flush'].append(empty)
                cases[name]['append_3_tokens'].append(append)
        result['history'][str(history)] = {name: {stage: _summary(samples)
            for stage, samples in stages.items()} for name, stages in cases.items()}
    for blocks in (1, 8):
        raw = torch.ones((16, blocks*128, 4, 256), dtype=torch.bfloat16)
        values = torch.zeros_like(raw, dtype=torch.uint8)
        cases = {name: [] for name in names}
        for repeat in range(args.repeats):
            for name in names[::1 if repeat % 2 else -1]:
                cases[name].append(sealed_case(implementations[name], blocks, raw, values))
        result['sealed_flush'][str(blocks)] = {name: _summary(samples) for name, samples in cases.items()}
    formatted = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted)
    print(formatted, end='')


if __name__ == '__main__':
    main()
