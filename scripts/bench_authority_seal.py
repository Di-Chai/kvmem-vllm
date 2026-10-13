#!/usr/bin/env python3
"""Split current sealed-record views/join/SHA/temporary cleanup on CPU only.

Run only while serving measurements are idle: even a single CPU thread shares
memory bandwidth with the model's host work. This does not benchmark model TPS,
archive admission/cache copies/NVMe, MeanK sums, or generation lease operations.
Source tensors remain preallocated; cleanup measures only temporary snapshot
bytes and buffer views, not freeing authoritative record tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch


RECORD_BYTES = 16 * 128 * 4 * 256 * 3


def split_seal(records):
    """Same payload and SHA expressions as RequestAuthority.flush, with clocks."""
    started = time.perf_counter_ns()
    views = []
    for record in records:
        views.extend((memoryview(record.raw.view(torch.uint8).numpy()).cast('B'),
                      memoryview(record.values.numpy()).cast('B')))
    viewed = time.perf_counter_ns()
    payload = b''.join(views)
    joined = time.perf_counter_ns()
    view = memoryview(payload)
    for i, record in enumerate(records):
        begin = i * RECORD_BYTES
        record.digest = hashlib.sha256(view[begin:begin+RECORD_BYTES]).hexdigest()
    hashed = time.perf_counter_ns()
    del view, payload, views
    cleaned = time.perf_counter_ns()
    return dict(views=(viewed-started)/1000, join=(joined-viewed)/1000,
                sha=(hashed-joined)/1000, cleanup=(cleaned-hashed)/1000,
                split_total=(cleaned-started)/1000)


def combined_seal(records):
    """Uninterrupted operation; returning releases temporary snapshot/views."""
    views = []
    for record in records:
        views.extend((memoryview(record.raw.view(torch.uint8).numpy()).cast('B'),
                      memoryview(record.values.numpy()).cast('B')))
    payload = b''.join(views)
    view = memoryview(payload)
    for i, record in enumerate(records):
        begin = i * RECORD_BYTES
        record.digest = hashlib.sha256(view[begin:begin+RECORD_BYTES]).hexdigest()


def summary(samples):
    return dict(median_us=statistics.median(samples), min_us=min(samples),
                max_us=max(samples), samples_us=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('repeats must be positive')
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1] / 'src/vllm_kvmem/authority.py'
    result = dict(scope='CPU sealed-payload diagnosis; not serving throughput',
        authority_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch_threads=1, repeats=args.repeats, warmup_rounds=1,
        geometry=dict(layers=16, block_tokens=128, heads=4, head_dim=256,
                      raw_dtype='bfloat16', values_dtype='uint8', record_bytes=RECORD_BYTES),
        exclusions=['source allocation/free', 'MeanK sums', 'native pin/generation bookkeeping',
                    'archive admission', 'CPU cache admission copies', 'NVMe I/O', 'GPU work'],
        cleanup_scope='temporary immutable snapshot and buffer views only',
        cache_regime='preallocated reused sources; no CPU-cache flush; fresh joined bytes each call',
        cases={})
    for blocks in (1, 8):
        records = [SimpleNamespace(
            raw=torch.full((16, 128, 4, 256), i+1, dtype=torch.bfloat16),
            values=torch.full((16, 128, 4, 256), (i*31+7) % 256, dtype=torch.uint8),
            digest=None) for i in range(blocks)]
        # Independent legacy encoding validates the layout and every timed SHA.
        expected = [hashlib.sha256(record.raw.view(torch.uint8).numpy().tobytes()
                                  + record.values.numpy().tobytes()).hexdigest()
                    for record in records]

        def verify():
            assert [record.digest for record in records] == expected

        split_seal(records)
        verify()
        combined_seal(records)
        verify()
        samples = {name: [] for name in ('views', 'join', 'sha', 'cleanup', 'split_total', 'combined')}
        for repeat in range(args.repeats):
            # Alternate ordering so one case does not always inherit the same
            # allocator/cache state from its preceding neighbor.
            for mode in (('split', 'combined') if repeat % 2 else ('combined', 'split')):
                if mode == 'split':
                    for name, duration in split_seal(records).items():
                        samples[name].append(duration)
                else:
                    started = time.perf_counter_ns()
                    combined_seal(records)
                    samples['combined'].append((time.perf_counter_ns()-started)/1000)
                verify()
        result['cases'][str(blocks)] = dict(blocks=blocks, payload_bytes=blocks*RECORD_BYTES,
            all_digests_verified=True, digest_sha256=expected,
            phases={name: summary(values) for name, values in samples.items()})
    formatted = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted)
    print(formatted, end='')


if __name__ == '__main__':
    main()
