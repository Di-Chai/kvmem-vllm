#!/usr/bin/env python3
"""Reproducible CPU-only cost/capacity probe for accepted block dependencies.

This is a sparse synthetic host benchmark, not a model quality or GPU throughput
measurement. Every graph row is built through the production acceptance API.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

from vllm_kvmem.block_graph import BlockGraph, GraphBudget, GraphSample
from vllm_kvmem.graph_scoring import score_graph


class MeterBudget(GraphBudget):
    def __init__(self, maximum):
        super().__init__(maximum)
        self.peak = 0

    def reserve(self, owner, amount, limit):
        charge = super().reserve(owner, amount, limit)
        self.peak = max(self.peak, self.used)
        return charge


def samples(source, block_tokens, positions, kind='prefill'):
    previous = tuple(range(max(0, source - 4 * block_tokens), source, block_tokens))
    targets = previous + (source,)
    weights = ((.8 / len(previous),) * len(previous) + (.2,)) if previous else (1.,)
    return tuple(GraphSample(position, targets, weights, kind) for position in positions)


def accept(graph, end, rows=()):
    with graph.prepare(graph.accepted_through, end, rows) as update:
        update.commit()


def build(graph, history_tokens, chunk_tokens):
    block = graph.block_tokens
    # P intentionally retains one partial source; M finishes it with a second
    # sample. This exercises the exact accumulator copied by real checkpoints.
    p_end = history_tokens - block // 2
    while graph.accepted_through < p_end:
        end = min(graph.accepted_through + chunk_tokens, p_end)
        rows = []
        for source in range(graph.accepted_through // block * block, end, block):
            positions = [p for p in (source, source + block // 2)
                         if graph.accepted_through <= p < end]
            rows.extend(samples(source, block, positions))
        accept(graph, end, rows)
    p = graph.snapshot()
    accept(graph, history_tokens, samples(history_tokens - block, block,
                                         (history_tokens - block // 2,)))
    return p, graph.snapshot()


def distribution(elapsed):
    ordered = sorted(elapsed)
    return {'repetitions': len(elapsed), 'mean_us': statistics.mean(elapsed) * 1e6,
            'median_us': statistics.median(elapsed) * 1e6,
            'p95_us': ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] * 1e6,
            'min_us': min(elapsed) * 1e6, 'max_us': max(elapsed) * 1e6}


def timed(function, repeats):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    return distribution(elapsed)


def run(history_tokens=262144, block_tokens=32, repeats=50,
        session_bytes=33554432, total_bytes=268435456):
    if (history_tokens < 4 * block_tokens or history_tokens % block_tokens or
            block_tokens < 2 or repeats <= 0):
        raise ValueError('History must contain aligned blocks; repetitions must be positive')
    pool = MeterBudget(total_bytes)
    graphs, checkpoints = [], []
    report = {'schema': 1, 'kind': 'CPU synthetic graph-state/propagation benchmark',
              'quality_claim': False, 'gpu_used': False,
              'config': {'history_tokens': history_tokens, 'block_tokens': block_tokens,
                         'source_blocks': history_tokens // block_tokens,
                         'samples_per_source': 2, 'topk': 4, 'sessions': 2,
                         'max_bytes_per_session': session_bytes, 'max_total_bytes': total_bytes,
                         'repeats': repeats},
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'available_cpu_count': len(os.sched_getaffinity(0))},
              'measurements': {}, 'memory_phases': []}

    def memory(label):
        report['memory_phases'].append({'phase': label, 'accounted_global_bytes': pool.used,
             'accounted_peak_bytes': pool.peak, 'sessions': [graph.stats() for graph in graphs]})

    try:
        tracemalloc.start()
        started = time.perf_counter()
        for index in range(2):
            graph = BlockGraph(('host-probe', index), block_tokens, budget=pool,
                               max_bytes_per_session=session_bytes)
            graphs.append(graph)
            graph.update_message_spans(tuple((begin, min(begin + 2048, history_tokens), 'tool')
                                             for begin in range(0, history_tokens, 2048)))
            checkpoints.append(build(graph, history_tokens, max(1024, block_tokens)))
        report['build_seconds_including_allocation_tracing'] = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report['python_allocation_probe'] = {'retained_bytes': current, 'peak_bytes': peak,
            'scope': 'Both graphs, P/M checkpoints, benchmark setup temporaries; imported modules excluded',
            'tracing_disabled_during_timing': True}
        memory('two_sessions_each_with_P_and_M')
        graph = graphs[0]
        p, m = checkpoints[0]
        measurements = report['measurements']
        measurements['no_sample_accepted_progress'] = timed(
            lambda: accept(graph, graph.accepted_through + 1), repeats)

        def sampled_append():
            source = (graph.accepted_through + block_tokens - 1) // block_tokens * block_tokens
            accept(graph, source + block_tokens, samples(source, block_tokens,
                   (source, source + block_tokens // 2), 'output'))
        measurements['accepted_sample_aggregation_and_topk'] = timed(sampled_append, repeats)

        def capture_snapshot():
            snap = graph.snapshot()
            snap.close()
        measurements['snapshot_capture_release'] = timed(capture_snapshot, repeats)
        measurements['restore_M_with_exact_P_preservation'] = timed(
            lambda: graph.restore(m, preserve_future=True, preserved_prefix=p,
                                  until=m.accepted_through), repeats)
        memory('M_restored_with_preserved_P')
        starts = list(range(0, history_tokens, block_tokens))
        raw = [1. / (index + 1) for index in range(len(starts))]
        eligible = [True] * len(starts)
        for mode in ('locality', 'attention', 'multi'):
            with graph.snapshot() as snap:
                memory('inflight_' + mode)
                measurements['score_' + mode] = timed(
                    lambda: score_graph(raw, starts, eligible, snap, mode=mode), repeats)
        with ThreadPoolExecutor(max_workers=2) as workers:
            measurements['two_sessions_concurrent_no_sample_callbacks'] = timed(
                lambda: list(workers.map(lambda owned: accept(owned, owned.accepted_through + 1), graphs)),
                repeats)
        memory('completed')
        report['status'] = 'passed'
    except BufferError as error:
        report['status'] = 'capacity_rejected'
        report['error'] = str(error)
        memory('capacity_rejected')
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        for pair in checkpoints:
            for snapshot in pair:
                snapshot.close()
        for graph in graphs:
            graph.close()
        report['global_bytes_after_release'] = pool.used
        report['peak_accounted_global_bytes'] = pool.peak
        if pool.used:
            raise AssertionError('Graph ownership was not fully released')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-tokens', type=int, default=262144)
    parser.add_argument('--block-tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--session-bytes', type=int, default=33554432)
    parser.add_argument('--total-bytes', type=int, default=268435456)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path to preserve prior measurements')
    report = run(args.history_tokens, args.block_tokens, args.repeats,
                 args.session_bytes, args.total_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'peak_accounted_global_bytes': report['peak_accounted_global_bytes'],
                      'global_bytes_after_release': report['global_bytes_after_release']}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
