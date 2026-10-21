"""CPU-only sealed-index microbenchmark; excludes model and archive publication."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from vllm_kvmem import _native


def benchmark():
    rng = np.random.default_rng(17)
    raw = [rng.integers(0, 127, (17, 32, 4, 256), dtype=np.uint8) for _ in range(32)]
    sums = [np.empty((16, 4, 256), dtype=np.float32) for _ in raw]
    jobs = [(a, b, 32, True) for a, b in zip(raw, sums)]
    results = []
    for mode in ('single', 'batch', 'single', 'batch'):
        spans = []
        for i in range(35):
            started = time.perf_counter()
            if mode == 'single':
                for a, b, count, _ in jobs:
                    _native.finalize_mean_fp8(a, b, count)
            else:
                _native.finalize_mean_batch(jobs)
            if i >= 5:
                spans.append(time.perf_counter()-started)
        results.append(dict(mode=mode, median_seconds=statistics.median(spans), seconds=spans))
    return dict(results=results, records=32, raw_shape=[17, 32, 4, 256],
        scoring_layers=16, precision='FP8 e4m3 to token-order FP32 sum', warmup_iterations=5,
        scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(), indent=2)+'\n')
