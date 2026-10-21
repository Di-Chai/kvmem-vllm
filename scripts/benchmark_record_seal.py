"""CPU-only copy/hash comparison; excludes publication and model concurrency."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from vllm_kvmem import _native


def benchmark():
    record_bytes, count = 1114112, 32
    sources = [bytes([i])*int(record_bytes/2) for i in range(count*2)]
    offsets = [i*record_bytes for i in range(count)]
    rows = []
    expected = None
    for mode in ('slab_copy', 'direct_single', 'direct_batch')*2:
        times = []
        for i in range(23):
            records = snapshot = None
            start = time.perf_counter()
            if mode == 'slab_copy':
                snapshot = _native.pack_record_views(sources)
                view = memoryview(snapshot)
                digests = [hashlib.sha256(view[offset:offset+record_bytes]).hexdigest() for offset in offsets]
                records = _native.copy_record_slices(view, offsets, record_bytes)
                view = None
            else:
                records = (_native.pack_record_pairs(sources, record_bytes) if mode == 'direct_batch'
                           else [_native.pack_record_views(sources[j:j+2]) for j in range(0, count*2, 2)])
                digests = [hashlib.sha256(record).hexdigest() for record in records]
            elapsed = time.perf_counter()-start
            if i >= 3:
                times.append(elapsed)
            expected = digests if expected is None else expected
            assert digests == expected
        rows.append(dict(mode=mode, median_seconds=statistics.median(times), seconds=times))
    return dict(results=rows, records=count, record_bytes=record_bytes,
                payload_bytes=count*record_bytes, warmup_iterations=3, scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(), indent=2)+'\n')
