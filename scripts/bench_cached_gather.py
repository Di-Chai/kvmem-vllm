"""Measure only immutable CPU-cache gather, with the serving record/slab sizes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from vllm_kvmem.cached_storage import CachedArchive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pinned', action='store_true')
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    record_bytes, records = 6291456, 32
    sources = [bytes([i]) * record_bytes for i in range(records)]
    total = records * record_bytes
    if args.pinned:
        import torch
        owner = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        destination = memoryview(owner.numpy())
    else:
        owner = bytearray(total)
        destination = memoryview(owner)
    pools = {workers: ThreadPoolExecutor(max_workers=workers) for workers in (1, 2, 4, 8)}
    archives = {workers: SimpleNamespace(record_bytes=record_bytes, gather_workers=workers,
        _gather_executor=pool) for workers, pool in pools.items()}

    def copy(name):
        if name == 'python_serial':
            for i, source in enumerate(sources):
                destination[i*record_bytes:(i+1)*record_bytes] = source
        else:
            CachedArchive._gather(archives[int(name)], destination, sources)

    names = ['python_serial', '1', '2', '4', '8']
    samples = {name: [] for name in names}
    try:
        for name in names:
            copy(name)  # First touch and lazy worker startup outside measurement.
        for repeat in range(args.repeats):
            for name in names[::1 if repeat % 2 else -1]:
                start = time.perf_counter()
                copy(name)
                samples[name].append(time.perf_counter()-start)
        for i, source in enumerate(sources):
            assert destination[i*record_bytes:(i+1)*record_bytes] == source
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    report = dict(scope='CPU copy only; warm source/destination; no H2D, model or archive I/O',
        pinned=args.pinned, record_bytes=record_bytes, records=records, repeats=args.repeats,
        results={name: dict(median_seconds=statistics.median(rows),
            median_gib_per_second=total/1024**3/statistics.median(rows), samples_seconds=rows)
            for name, rows in samples.items()}, byte_equal=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
