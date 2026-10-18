"""CPU-only dense graph acceptance probe, using the original scalar reference.

This measures maintenance of a full active-window attention matrix, not model
throughput. Both paths consume identical float32 buffers and accepted positions.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from vllm_kvmem.block_graph import BlockGraph, GraphSample


def run(calls=100, active_tokens=50176, chunk_tokens=1024, block_tokens=32):
    if calls <= 0 or active_tokens <= 0 or any(v % block_tokens for v in (active_tokens, chunk_tokens)):
        raise ValueError('Require positive repetitions and aligned active/chunk lengths')
    targets = tuple(range(0, active_tokens, block_tokens))
    masses = np.random.default_rng(20261018).random(
        (chunk_tokens//block_tokens, len(targets)), dtype=np.float32)
    masses /= masses.sum(axis=1, keepdims=True)
    report = dict(kind='CPU-only dense attention acceptance', gpu_used=False,
        calls=calls, active_tokens=active_tokens, chunk_tokens=chunk_tokens,
        sample_rows=len(masses), target_columns=len(targets), shape=list(masses.shape),
        includes='Capacity preflight, mass validation/normalization, source aggregation, top-k, publication.',
        excludes='Model, attention capture, GPU transfers and final Query scoring.', measurements={})
    reference = None
    for backend in ('python_reference', 'native_dense'):
        graph = BlockGraph(backend, block_tokens, initial_position=active_tokens)
        elapsed = []
        try:
            for _ in range(calls):
                begin = graph.accepted_through
                positions = tuple(range(begin+block_tokens-1, begin+chunk_tokens, block_tokens))
                start = time.perf_counter()
                update = (graph.prepare_accept(begin, begin+chunk_tokens, positions, targets, masses, 'prefill')
                    if backend == 'native_dense' else graph.prepare(begin, begin+chunk_tokens,
                        tuple(GraphSample(p, targets, v, 'prefill') for p, v in zip(positions, masses))))
                with update:
                    update.commit()
                elapsed.append(time.perf_counter()-start)
            report['measurements'][backend] = dict(seconds=sum(elapsed),
                mean_ms=statistics.mean(elapsed)*1000,
                p95_ms=sorted(elapsed)[min(len(elapsed)-1,int(.95*len(elapsed)))]*1000)
            if backend == 'python_reference':
                reference = graph.rows, graph.partial
            elif (graph.rows, graph.partial) != reference:
                raise AssertionError('Dense acceptance changed graph rows or partial sums')
        finally:
            graph.close()
            if graph.budget.used:
                raise AssertionError('Graph maintenance leaked its capacity reservation')
    report['bit_exact_equal'] = True
    report['speedup'] = (report['measurements']['python_reference']['seconds'] /
                         report['measurements']['native_dense']['seconds'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path to retain prior evidence')
    report = run(args.calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
