"""CPU-only, fixed-source pressure read/gather worker and affinity diagnostic.

All immutable authority bytes and both destination buffers are first-touched on
NUMA node 0 once. Arms change execution affinity/work count, never allocation
policy or data. Destinations are pre-touched pageable CPU storage, not CUDA
pinned memory; no DMA or model/GIL competitor is represented by this component.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
from threading import Barrier, get_native_id
import time
from types import SimpleNamespace as NS

import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import record_key
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.packed_assembly import read_packed_records

RAW_BYTES = 17 * 32 * 4 * 256
RECORD_BYTES = RAW_BYTES * 2
CHUNK = 128


def cpulist(text):
    result = []
    for part in text.strip().split(','):
        a, *b = part.split('-')
        result.extend(range(int(a), int(b[0]) + 1) if b else [int(a)])
    return result


def affinity(cpus):
    for task in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(task.name), cpus)
    actual = {task.name: sorted(os.sched_getaffinity(int(task.name)))
              for task in Path('/proc/self/task').iterdir()}
    assert all(set(value) == set(cpus) for value in actual.values())
    return actual


def numa_pages():
    totals = {}
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        for item in line.split():
            if item.startswith('N') and '=' in item and item[1:item.index('=')].isdigit():
                node, count = item.split('='); totals[node] = totals.get(node, 0) + int(count)
    return totals


def proof(root):
    files = [Path(__file__), root / 'csrc/bindings.cpp', Path(_native.__file__),
             *(root / 'src/vllm_kvmem' / name for name in
               ('memory_storage.py', 'storage.py', 'packed_assembly.py'))]
    return {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def setup(args, node0):
    affinity(node0)
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    row = next(row for row in rows if row['command_id'] == 16)
    selected = row['selected_history_ranges']
    skipped = row['assembly_skipped_tokens'] // 32
    assert selected[:skipped] == [[i * 32, 32] for i in range(skipped)]
    order = [start // 32 for start, count in selected[skipped:] if count == 32]
    assert len(order) == 931 and row['assembly_uploaded_bytes'] == len(order) * RAW_BYTES
    extent = row['history_begin'] // 32
    archive = MemoryArchive(str(args.output.parent / 'unused-cpu-only'), RECORD_BYTES, extent + 2,
        gather_workers=8, cpu_direct_record_seal=True, cpu_seal_hash_workers=1)
    repo = _native.Repository(extent + 2, 32)
    leases = [repo.reserve(i * 32, 32).commit() for i in range(extent)]
    for begin in range(0, extent, CHUNK):
        chunk = list(range(begin, min(begin + CHUNK, extent)))
        views = []
        for i in chunk:
            views.extend((struct.pack('<Q', i) * (RAW_BYTES // 8),
                          struct.pack('<Q', (1 << 64) - 1 - i) * (RAW_BYTES // 8)))
        job = archive.write_from_views('bench', leases[begin:begin + len(chunk)], views)
        assert job is not None
        job.result()
    views = None
    sources = archive._region_sources([archive.key('bench', leases[i]) for i in order])
    buffers = [bytearray(CHUNK * RAW_BYTES) for _ in range(2)]
    for buffer in buffers:
        buffer[::4096] = b'\x5a' * len(buffer[::4096])
    targets = [torch.frombuffer(buffer, dtype=torch.uint8) for buffer in buffers]
    authority = NS(archive=archive, shape=(17, 32, 4, 256), record_bytes=RECORD_BYTES,
        poll=lambda: None, records={record_key(lease.handle): NS(archived=True, count=32)
                                  for lease in leases},
        memory=NS(request_id='bench', admitted_read=lambda handle: (repo.pin(handle), None)))
    chunks = [(order[start:start + CHUNK], sources[start:start + CHUNK])
              for start in range(0, len(order), CHUNK)]
    # Force all eight persistent worker threads to exist before changing masks.
    gate = Barrier(8)
    def arrive():
        gate.wait(timeout=30)
        return get_native_id()
    jobs = [archive._gather_executor.submit(arrive) for _ in range(8)]
    worker_tids = sorted(job.result() for job in jobs)

    def run(kind, verify=False):
        stats = {}; began = time.perf_counter()
        for index, (ids, values) in enumerate(chunks):
            target = targets[index % 2][:len(ids) * RAW_BYTES]
            if kind == 'full_read':
                _, got = read_packed_records(authority, [leases[i].handle for i in ids],
                    destination=target, region=(0, RAW_BYTES))
                for key, value in got.items():
                    stats[key] = stats.get(key, 0) + value
            else:
                jobs = archive._gather(memoryview(target.numpy()), values, 0, RAW_BYTES)
                stats['cache_gather_tasks'] = stats.get('cache_gather_tasks', 0) + jobs
            if verify:
                actual = memoryview(target.numpy())
                for at, source in enumerate(values):
                    assert actual[at * RAW_BYTES:(at + 1) * RAW_BYTES].tobytes() == source[:RAW_BYTES]
        elapsed = (time.perf_counter() - began) * 1000
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        return dict(wall_ms=elapsed, stats=stats)
    return run, archive, dict(history_records=extent, selected_records=len(selected),
        skipped_records=skipped, copied_records=len(order), chunks=[len(c[0]) for c in chunks],
        record_bytes=RECORD_BYTES, raw_region_bytes=RAW_BYTES, total_copied_bytes=len(order) * RAW_BYTES,
        source_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        source_policy='One public archive publication; every allocation/first touch on node0; same bytes for every arm.',
        destination_bytes=2 * CHUNK * RAW_BYTES, destination_kind='Two pre-touched pageable CPU buffers; no CUDA/pinning/DMA.',
        worker_native_tids=worker_tids, node_pages_after_setup=numa_pages())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, default=Path('artifacts/prefill-pressure-phases-20261022-r1/run/server/capture/bounded_refresh.jsonl'))
    parser.add_argument('--repeats', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evidence output must be new')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    original = sorted(os.sched_getaffinity(0))
    node0 = sorted(set(cpulist(Path('/sys/devices/system/node/node0/cpulist').read_text())) & set(original))
    before = proof(root)
    run, archive, geometry = setup(args, node0)
    evidence = dict(scope='CPU component only; ready-record authority poll is a no-op; real repository generation pins, archive admission, parallel gather, destination checks and lease cleanup are executed.',
        source_hashes=before, geometry=geometry, affinity_masks={'node0': node0, 'all64': original},
        numa_balancing=Path('/proc/sys/kernel/numa_balancing').read_text().strip(),
        python_threads_requested=1, samples=[], affinity_observations=[], validation=[])
    configurations = [(mask, workers) for mask in ('node0', 'all64') for workers in (1, 2, 4, 8)]
    try:
        for mask, workers in configurations:
            affinity(evidence['affinity_masks'][mask]); archive.gather_workers = workers
            for kind in ('full_read', 'gather'):
                run(kind, verify=True)
                run(kind)
            evidence['validation'].append(dict(affinity=mask, workers=workers, both_paths_byte_equal=True))
        for repetition in range(args.repeats):
            order = configurations[repetition % len(configurations):] + configurations[:repetition % len(configurations)]
            if repetition % 2:
                order = order[::-1]
            for mask, workers in order:
                snapshot = affinity(evidence['affinity_masks'][mask]); archive.gather_workers = workers
                if repetition == 0:
                    evidence['affinity_observations'].append(dict(affinity=mask, workers=workers, threads=snapshot))
                kinds = ('full_read', 'gather') if repetition % 2 == 0 else ('gather', 'full_read')
                for kind in kinds:
                    evidence['samples'].append(dict(repetition=repetition, affinity=mask, workers=workers, kind=kind, **run(kind)))
        evidence['summary'] = []
        for mask, workers in configurations:
            for kind in ('full_read', 'gather'):
                values = [s['wall_ms'] for s in evidence['samples'] if (s['affinity'], s['workers'], s['kind']) == (mask, workers, kind)]
                evidence['summary'].append(dict(affinity=mask, workers=workers, kind=kind,
                    mean_ms=statistics.mean(values), median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values)))
        evidence['node_pages_after_measurement'] = numa_pages()
        evidence['final_io_usage'] = archive.io_usage()
        after = proof(root)
        evidence['final_source_hashes'] = after
        evidence['sources_unchanged'] = before == after
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence['summary'], indent=2), flush=True)
        assert before == after, 'Loaded extension/source changed during the diagnostic'
    finally:
        assert archive.close()
        affinity(original)


if __name__ == '__main__':
    main()
