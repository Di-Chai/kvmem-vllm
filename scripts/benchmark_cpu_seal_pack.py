"""Bounded CPU record sealing: one versus eight pack workers, same frozen binary.

The primary interval is write_from_views through Future.result: admission,
allocation, worker dispatch, native packing, SHA256 and atomic publication.
Every invocation also performs a real read, exact byte/SHA verification and
repository retirement. Those costs are reported separately and in a full
lifecycle interval; they are not silently charged to production sealing.

Preparation imports no numeric/GPU libraries. Execution needs --run-cpu and
external coordination with model/performance jobs. Only generated inputs and
this experiment's records are touched; there is no user-data archive.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import sys
import time

LAYERS, TOKENS, HEADS, DIM = 17, 32, 4, 256
HALF_BYTES = LAYERS * TOKENS * HEADS * DIM
RECORD_BYTES = 2 * HALF_BYTES
COUNTS = (32, 64)
WORKERS = (1, 8)
SEED = 22022


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def abba_pairs(pairs):
    if type(pairs) is not int or pairs < 12 or pairs % 2:
        raise ValueError('Use an even number of at least 12 matched AB/BA pairs')
    return tuple(WORKERS if i % 2 == 0 else WORKERS[::-1] for i in range(pairs))


def validate_counts(counts):
    if tuple(counts) != COUNTS or any(type(n) is not int for n in counts):
        raise ValueError('The qualified record counts are exactly 32 and 64')
    return COUNTS


def prepare(directory, source_root):
    if directory.exists():
        raise ValueError('Prepare requires a new experiment directory')
    source_root = source_root.resolve()
    package = source_root / 'src/vllm_kvmem'
    binaries = list(package.glob('_native*.so'))
    if len(binaries) != 1:
        raise ValueError('Freeze requires exactly one compiled _native library')
    directory.mkdir(parents=True)
    shutil.copytree(package, directory / 'runtime/vllm_kvmem',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), directory / Path(__file__).name)
    proof = directory / 'source-proof'
    proof.mkdir()
    for name in ('csrc/bindings.cpp', 'tests/test_cpu_seal_pack.py',
                 'tests/test_cpu_seal_hash.py', 'tests/test_cpu_direct_record_seal.py'):
        source = source_root / name
        if not source.is_file():
            raise ValueError('Missing source/qualification dependency: ' + name)
        destination = proof / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    manifest = dict(schema=1, source_root=str(source_root), scope=__doc__,
        files={str(path.relative_to(directory)): sha256(path) for path in files},
        native_binary=str((Path('runtime/vllm_kvmem') / binaries[0].name)),
        geometry=dict(layers=LAYERS, tokens_per_record=TOKENS, kv_heads=HEADS,
            head_dim=DIM, bytes_per_raw_k_or_v=HALF_BYTES,
            record_bytes=RECORD_BYTES, records=list(COUNTS)),
        fixed_options=dict(cpu_direct_record_seal=True, cpu_seal_hash_workers=8,
            gather_workers=8, archive_workers=2, max_pending=2, native_payloads=False),
        arms=list(WORKERS), source_kind='Deterministic generated nonzero uint8 FP8 storage bytes',
        preparation_initializes_cuda=False)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_sources(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Frozen source changed or missing: ' + name)
    if Path(__file__).resolve() != (directory / 'benchmark_cpu_seal_pack.py').resolve():
        raise ValueError('Execute the frozen benchmark script')
    return manifest


def memory_snapshot():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            values[key + '_bytes'] = int(value.split()[0]) * 1024
        elif key == 'Threads':
            values['threads'] = int(value)
    values['process_lifetime_ru_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return values


def thread_affinity():
    return {task.name: sorted(os.sched_getaffinity(int(task.name)))
            for task in sorted(Path('/proc/self/task').iterdir())}


def make_archive(directory, count, workers):
    from vllm_kvmem.memory_storage import MemoryArchive
    return MemoryArchive(str(directory / 'unused-memory-only'), RECORD_BYTES, count,
        workers=2, max_pending=2, io_max_bytes=2 * count * RECORD_BYTES,
        cpu_direct_record_seal=True, cpu_seal_pack_workers=workers,
        cpu_seal_hash_workers=8, gather_workers=8)


def repository(count):
    from vllm_kvmem import _native
    repo = _native.Repository(count, TOKENS)
    leases = [repo.reserve(i * TOKENS, TOKENS).commit() for i in range(count)]
    handles = [_native.Handle(lease.handle.slot, lease.handle.generation) for lease in leases]
    return repo, leases, handles


def release(repo, leases, handles, archive, namespace):
    from vllm_kvmem import _native
    for lease, handle in zip(leases, handles):
        assert repo.retire(handle)
        lease.reset()
        result = repo.reclaim(handle, lambda record: archive.reclaim(namespace, record))
        assert result == _native.ReclaimResult.RECLAIMED, result
    assert archive.namespace_allocated_records(namespace) == 0
    assert archive.io_usage()['archive_io_pending_bytes'] == 0


def failure_qualification(directory, sources):
    """Real pack/hash work, then injected boundary failures; never timed.

    Fine-grained parallel submit/worker-drain failure tests are separately
    frozen in source-proof/tests/test_cpu_seal_pack.py. Here each new experiment
    must prove batch rollback, byte-credit release and subsequent reuse with
    its actual loaded library before collecting any performance samples.
    """
    results = []
    for workers in WORKERS:
        for failure in ('after_pack', 'after_hash', 'publication_partial_insert'):
            archive = make_archive(directory, 2, workers)
            repo, leases, handles = repository(2)
            namespace = f'qualification-{workers}-{failure}'
            if failure in ('after_pack', 'after_hash'):
                name = '_pack_records' if failure == 'after_pack' else '_record_digests'
                original = getattr(archive, name)

                def failing(values, original=original):
                    original(values)
                    raise OSError('injected post-work rollback qualification')

                setattr(archive, name, failing)
            else:
                original = archive.tier.records

                class PartialInsert(dict):
                    def update(self, values):
                        key = next(iter(values))
                        self[key] = values[key]
                        raise OSError('injected partial publication qualification')

                archive.tier.records = PartialInsert()
            job = archive.write_from_views(namespace, leases, sources[:4])
            assert job is not None
            try:
                job.result(30)
            except OSError as error:
                assert 'injected' in str(error)
            else:
                raise AssertionError('Injected failure did not reach the real archive job')
            assert not archive._entries and not archive.tier.records
            assert archive.tier.used_slots == 0 and len(archive._free_ids) == 2
            usage = archive.io_usage()
            assert usage['archive_io_pending_bytes'] == usage['seal_batches'] == usage['seal_direct_bytes'] == 0
            try:
                archive.read(namespace, leases)
            except KeyError:
                pass
            else:
                raise AssertionError('Failed records became readable')
            if failure == 'publication_partial_insert':
                archive.tier.records = original
            else:
                setattr(archive, name, original)
            # Reuse the same slot generations only after failed job cleanup.
            good = archive.write_from_views(namespace, leases, sources[:4])
            assert good is not None
            stats = good.result(30)
            assert len(stats['record_digests']) == 2
            release(repo, leases, handles, archive, namespace)
            assert archive.close()
            results.append(dict(workers=workers, failure=failure, rollback=True,
                credits_released=True, successful_reuse=True, failed_future_retained_until_reuse=True))
            del job, good, archive
    return results


def verify_read(payload, sources, expected, record_bytes=RECORD_BYTES):
    """Zero-copy byte comparison plus independent SHA; outside seal timing."""
    import numpy as np
    view = memoryview(payload).cast('B')
    for index, digest in enumerate(expected):
        record = view[index * record_bytes:(index + 1) * record_bytes]
        if hashlib.sha256(record).hexdigest() != digest:
            raise AssertionError('Archive read SHA differs from input SHA')
        at = 0
        for source in sources[2 * index:2 * index + 2]:
            part = memoryview(source).cast('B')
            if not np.array_equal(np.frombuffer(record[at:at + len(part)], dtype=np.uint8),
                                  np.frombuffer(part, dtype=np.uint8)):
                raise AssertionError('Archive read bytes differ from the sealed sources')
            at += len(part)
        assert at == record_bytes


def trial(archive, workers, count, sources, expected, destination, namespace):
    repo, leases, handles = repository(count)  # Caller ownership, outside sealing.
    before_usage = archive.io_usage()
    rss_before = memory_snapshot()
    cpu = time.process_time_ns()
    started = time.perf_counter_ns()
    job = archive.write_from_views(namespace, leases, sources)
    assert job is not None, 'An independently drained owner must fit its exact byte budget'
    stats = job.result(30)
    sealed = time.perf_counter_ns()
    seal_cpu = time.process_time_ns() - cpu
    rss_sealed = memory_snapshot()
    assert stats['record_digests'] == expected
    read_began = time.perf_counter_ns()
    read = archive.read(namespace, leases, destination=destination)
    assert read is not None
    payload, read_stats = read.result(30)
    read_done = time.perf_counter_ns()
    verify_read(payload, sources, expected)
    verified = time.perf_counter_ns()
    rss_verified = memory_snapshot()
    usage = archive.io_usage()
    assert usage['cpu_authority_bytes'] == count * RECORD_BYTES
    assert usage['archive_io_peak_bytes'] == 2 * count * RECORD_BYTES
    assert usage['archive_io_deferrals'] == usage['archive_io_pending_bytes'] == 0
    assert usage['seal_direct_bytes'] - before_usage['seal_direct_bytes'] == count * RECORD_BYTES
    assert usage['seal_hash_parallel_batches_completed'] - before_usage['seal_hash_parallel_batches_completed'] == 1
    assert usage['seal_hash_parallel_shards_completed'] - before_usage['seal_hash_parallel_shards_completed'] == 8
    assert usage['seal_pack_parallel_batches_completed'] - before_usage['seal_pack_parallel_batches_completed'] == (workers == 8)
    assert usage['seal_pack_parallel_shards_completed'] - before_usage['seal_pack_parallel_shards_completed'] == (8 if workers == 8 else 0)
    release_began = time.perf_counter_ns()
    release(repo, leases, handles, archive, namespace)
    released = time.perf_counter_ns()
    assert archive.tier.used_slots == 0 and archive.io_usage()['cpu_authority_bytes'] == 0
    rss_after = memory_snapshot()
    stats = {key: value for key, value in stats.items() if key != 'record_digests'}
    return dict(workers=workers, owner=namespace,
        seal_wall_ms=(sealed - started) / 1e6, seal_process_cpu_ms=seal_cpu / 1e6,
        read_wall_ms=(read_done - read_began) / 1e6,
        verify_wall_ms=(verified - read_done) / 1e6,
        release_wall_ms=(released - release_began) / 1e6,
        seal_plus_release_ms=(sealed - started + released - release_began) / 1e6,
        lifecycle_wall_ms=(released - started) / 1e6,
        seal_stats=stats, read_stats=read_stats,
        io_peak_bytes=usage['archive_io_peak_bytes'], resident_published_bytes=count * RECORD_BYTES,
        rss=dict(before=rss_before, sealed=rss_sealed, verified=rss_verified, released=rss_after),
        actual_pack_parallel_shards=8 if workers == 8 else 0,
        byte_equal=True, sha256_equal=True, released=True)


def summarize(samples):
    fields = ('seal_wall_ms', 'seal_process_cpu_ms', 'seal_plus_release_ms',
              'read_wall_ms', 'verify_wall_ms', 'release_wall_ms', 'lifecycle_wall_ms')
    result = {}
    for workers in WORKERS:
        rows = [row for row in samples if row['workers'] == workers]
        result[str(workers)] = {name: dict(mean=statistics.mean(row[name] for row in rows),
            median=statistics.median(row[name] for row in rows), min=min(row[name] for row in rows),
            max=max(row[name] for row in rows)) for name in fields}
        result[str(workers)]['seal_phase_mean_ms'] = {name: 1000 * statistics.mean(
            row['seal_stats'].get(name, 0.) for row in rows) for name in
            ('seal_pack_seconds', 'seal_hash_seconds', 'seal_record_publish_seconds')}
    result['seal_wall_mean_change_percent'] = (result['8']['seal_wall_ms']['mean'] /
        result['1']['seal_wall_ms']['mean'] - 1) * 100
    return result


def run(directory, pairs):
    manifest = verify_sources(directory)
    orders = abba_pairs(pairs)
    output = directory / 'results.json'
    if output.exists():
        raise ValueError('Use a fresh experiment: results already exist')
    sys.path.insert(0, str(directory / 'runtime'))
    import numpy as np
    from vllm_kvmem import _native
    if Path(_native.__file__).resolve() != (directory / manifest['native_binary']).resolve():
        raise ValueError('The frozen compiled library is not active')
    if not callable(getattr(_native, 'pack_record_pairs', None)):
        raise ValueError('The native pack-pairs API must be present in both arms')
    if 'torch' in sys.modules:
        raise ValueError('This CPU benchmark must not initialize torch or CUDA')
    evidence = dict(schema=1, complete=False, scope=__doc__, manifest_sha256=sha256(directory / 'manifest.json'),
        native_binary_sha256=sha256(_native.__file__), python=sys.version, numpy=np.__version__,
        machine=platform.machine(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        thread_affinity_before=thread_affinity(), cases=[],
        rss_note='Process RSS snapshots are lower bounds on per-call peaks. VmHWM/ru_maxrss are cumulative process-lifetime high-water marks, not arm-isolated peaks; shared pages can count in RSS.',
        budget_note='Exact I/O peak includes sealed source plus immutable record snapshot (2x payload). Source inputs and reusable read destination are also listed separately; logical capacity is not physical RSS.',
        allocation_note='Sources and destination are generated/first-touched once per geometry on the unchanged process affinity. Source pages are pageable CPU memory; no CUDA pinned allocator, DMA or GPU competitor is represented.',
        verification_note='Input generation/SHA and repository setup excluded from seal interval. Real read, exact bytes/SHA, diagnostics and release included in lifecycle; each phase separately reported.',
        limits=['Component only: no capture scatter, MeanK, graph work, GPU DMA, model or concurrent owner contention.',
                'One persistent archive per arm, one active unique owner at a time; worker threads prewarmed equally.',
                'Injected boundary rollback is mandatory; fine-grained worker/submit failure qualifications are frozen separately.'])
    for count in validate_counts(COUNTS):
        cold = time.perf_counter()
        generator = np.random.default_rng(SEED + count)
        sources = [generator.integers(1, 256, HALF_BYTES, dtype=np.uint8) for _ in range(2 * count)]
        expected = []
        for index in range(count):
            digest = hashlib.sha256()
            digest.update(sources[2 * index]); digest.update(sources[2 * index + 1])
            expected.append(digest.hexdigest())
        expected = tuple(expected)
        destination = np.zeros(count * RECORD_BYTES, dtype=np.uint8)
        destination.fill(0xA5)  # First-touch every destination page outside timing.
        preparation_seconds = time.perf_counter() - cold
        cold = time.perf_counter()
        qualification = failure_qualification(directory, sources)
        qualification_seconds = time.perf_counter() - cold
        arms = {workers: make_archive(directory, count, workers) for workers in WORKERS}
        result = dict(records=count, history_tokens=count * TOKENS, record_bytes=RECORD_BYTES,
            input_bytes=count * RECORD_BYTES, read_destination_bytes=destination.nbytes,
            io_max_bytes=2 * count * RECORD_BYTES, records_sha256=expected,
            input_recipe=dict(seed=SEED + count, shape_per_view=[LAYERS, TOKENS, HEADS, DIM],
                independent_contiguous_views=2 * count, dtype='uint8', range_inclusive=[1, 255]),
            input_and_destination_preparation_seconds=preparation_seconds,
            mandatory_failure_qualification=qualification, qualification_seconds=qualification_seconds,
            samples=[], complete=False)
        try:
            cold = time.perf_counter()
            for iteration in range(2):
                for workers in (WORKERS if iteration % 2 == 0 else WORKERS[::-1]):
                    trial(arms[workers], workers, count, sources, expected, destination,
                          f'warm-{count}-{iteration}-{workers}')
            result['warmup_seconds'] = time.perf_counter() - cold
            result['thread_affinity_after_warmup'] = thread_affinity()
            evidence['cases'].append(result)
            for pair, order in enumerate(orders):
                for workers in order:
                    sample = trial(arms[workers], workers, count, sources, expected, destination,
                                   f'measured-{count}-{pair}-{workers}')
                    sample['pair'] = pair
                    result['samples'].append(sample)
            result['summary'] = summarize(result['samples'])
            result['usage'] = {str(workers): archive.io_usage() for workers, archive in arms.items()}
            result['complete'] = True
        finally:
            for archive in arms.values():
                assert archive.close(), 'All admitted work must finish before teardown'
            output.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(dict(records=count, summary=result['summary'])), flush=True)
    verify_sources(directory)
    evidence.update(complete=True, thread_affinity_after=thread_affinity(), final_memory=memory_snapshot())
    output.write_text(json.dumps(evidence, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    setup = commands.add_parser('prepare')
    setup.add_argument('--directory', type=Path, required=True)
    setup.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    execute = commands.add_parser('run')
    execute.add_argument('--directory', type=Path, required=True)
    execute.add_argument('--pairs', type=int, default=12)
    execute.add_argument('--run-cpu', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.directory, args.source_root)
    else:
        if not args.run_cpu:
            parser.error('run requires --run-cpu and external performance-job coordination')
        run(args.directory, args.pairs)


if __name__ == '__main__':
    main()
