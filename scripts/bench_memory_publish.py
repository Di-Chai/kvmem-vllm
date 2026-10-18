"""Bounded CPU-only immutable publication benchmark; no model/GPU inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from threading import Event, Thread
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=64)
    parser.add_argument('--record-bytes', type=int, default=34816 * 32)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    total = args.records * args.record_bytes
    if min(args.records, args.record_bytes, args.repeats) < 1 or total > 128 * 1024**2 or args.repeats > 31:
        raise ValueError('Benchmark is bounded to 128 MiB per snapshot and 31 repeats')
    import vllm_kvmem
    native = load_module('vllm_kvmem._native', args.native.resolve())
    vllm_kvmem._native = native
    if not hasattr(native, 'copy_record_slices'):
        raise ValueError('Benchmark requires the new isolated host extension')
    from vllm_kvmem import memory_storage
    old = load_module('vllm_kvmem._publish_baseline', args.baseline_source.resolve())
    modules = {'baseline': old, 'native_copy': memory_storage}
    source = b'x' * total
    samples = {name: [] for name in modules}
    heartbeat_samples = {name: [] for name in modules}

    def run(name):
        module = modules[name]
        tier = module.MemoryTier(args.record_bytes, args.records)
        spans = [module.MemorySpan(i, i * args.record_bytes, args.record_bytes) for i in range(args.records)]
        for i in range(args.records):
            tier.allocate(i)
        begin = time.perf_counter()
        tier.io(spans, source, True)
        end = time.perf_counter()
        # Copy correctness is outside timing; no retained larger owner is used.
        assert len(tier.records) == args.records
        assert all(type(value) is bytes and len(value) == args.record_bytes and
                   value == b'x' * args.record_bytes for value in tier.records.values())
        return begin, end

    for name in modules:
        run(name)
    for repeat in range(args.repeats):
        for name in list(modules)[::1 if repeat % 2 else -1]:
            begin, end = run(name)
            samples[name].append(end - begin)

    stop = Event()
    ticks = []

    def heartbeat():
        while not stop.wait(0.0005):
            ticks.append(time.perf_counter())

    worker = Thread(target=heartbeat)
    worker.start()
    try:
        for repeat in range(args.repeats):
            for name in list(modules)[::1 if repeat % 2 else -1]:
                begin, end = run(name)
                observed = [tick for tick in ticks if begin < tick < end]
                times = [begin, *observed, end]
                heartbeat_samples[name].append(dict(seconds=end - begin, ticks=len(observed),
                    max_gap_seconds=max(b - a for a, b in zip(times, times[1:]))))
    finally:
        stop.set()
        worker.join()
    report = dict(scope='CPU tier immutable write only; source snapshot already packed; no GPU or archive hashing',
        note='Independent bytes still require a copy; new path releases the GIL and publication lock during memcpy.',
        native=str(args.native.resolve()), native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),
        baseline_source=str(args.baseline_source.resolve()),
        baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(Path(memory_storage.__file__).read_bytes()).hexdigest(),
        records=args.records, record_bytes=args.record_bytes, snapshot_bytes=total,
        repeats=args.repeats, python_switch_interval=sys.getswitchinterval(), byte_equal=True,
        results={name: dict(median_seconds=statistics.median(values),
            median_gib_per_second=total / 1024**3 / statistics.median(values),
            copy_seconds=values,
            heartbeat_median_max_gap_seconds=statistics.median(x['max_gap_seconds'] for x in heartbeat_samples[name]),
            heartbeat_median_ticks=statistics.median(x['ticks'] for x in heartbeat_samples[name]),
            heartbeat=heartbeat_samples[name]) for name, values in samples.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
